"""WASAPI loopback capture for Windows render endpoints.

Unlike a Stereo Mix device, which taps a single physical codec, WASAPI loopback
can capture the audio rendered to *any* output endpoint, including Bluetooth
headsets such as AirPods. This is the backend to use when the meeting audio goes
to a device that Stereo Mix cannot see.

Requires ``pyaudiowpatch`` (a PyAudio fork with WASAPI loopback support).
"""

from __future__ import annotations

import importlib.util
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from core.audio_recorder import AudioChunkWriter, AudioRecorderError, load_resampled_mono

LOOPBACK_SUFFIX = "[Loopback]"


@dataclass(frozen=True)
class LoopbackDevice:
    """A capturable Windows render endpoint exposed as a loopback input."""

    index: int
    name: str
    channels: int
    sample_rate: int
    is_default: bool = False

    @property
    def clean_name(self) -> str:
        """Return the endpoint name without the synthetic loopback suffix."""

        return self.name.replace(LOOPBACK_SUFFIX, "").strip()


def pyaudiowpatch_available() -> bool:
    """Return whether the WASAPI loopback backend can be used."""

    return importlib.util.find_spec("pyaudiowpatch") is not None


def _import_pyaudio():
    try:
        import pyaudiowpatch as pyaudio
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise AudioRecorderError(
            "WASAPI loopback requires the pyaudiowpatch package. Install it with "
            "`pip install pyaudiowpatch`."
        ) from exc
    return pyaudio


def _pyaudio_errors(pyaudio) -> tuple[type[BaseException], ...]:
    """Return the exception types PyAudio raises for device and stream failures.

    PyAudio builds differ in which error classes they expose (pyaudiowpatch does
    not define ``PyAudioError``), so the standard failures are always included
    and the library-specific ones are added when present.
    """

    errors: list[type[BaseException]] = [OSError, RuntimeError, ValueError, KeyError, AttributeError]
    for name in ("PyAudioError", "PyAudioManagerError"):
        candidate = getattr(pyaudio, name, None)
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            errors.append(candidate)
    return tuple(errors)


def list_loopback_devices() -> list[LoopbackDevice]:
    """Return every loopback-capable render endpoint, marking the system default."""

    pyaudio = _import_pyaudio()
    audio_errors = _pyaudio_errors(pyaudio)
    engine = pyaudio.PyAudio()
    try:
        try:
            wasapi = engine.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_name = str(engine.get_device_info_by_index(wasapi["defaultOutputDevice"])["name"])
        except audio_errors:  # the host API lookup is best-effort
            default_name = ""

        devices: list[LoopbackDevice] = []
        for info in engine.get_loopback_device_info_generator():
            name = str(info["name"])
            devices.append(
                LoopbackDevice(
                    index=int(info["index"]),
                    name=name,
                    channels=int(info["maxInputChannels"]) or 2,
                    sample_rate=int(info["defaultSampleRate"]),
                    is_default=bool(default_name) and name.startswith(default_name),
                )
            )
        return devices
    finally:
        engine.terminate()


def format_loopback_devices(devices: list[LoopbackDevice]) -> str:
    """Return a printable list of loopback devices."""

    if not devices:
        return "No se encontraron dispositivos de loopback WASAPI."
    lines = ["Dispositivos WASAPI loopback:"]
    for device in devices:
        marker = " | POR DEFECTO" if device.is_default else ""
        lines.append(
            f"[{device.index}] {device.clean_name} | "
            f"in={device.channels} | sr={device.sample_rate}{marker}"
        )
    return "\n".join(lines)


class WasapiLoopbackRecorder:
    """Record the audio rendered to any Windows output endpoint, including Bluetooth."""

    def __init__(
        self,
        output_dir: Path,
        target_sample_rate: int = 16_000,
        device_name_hint: str | None = None,
        device_index: int | None = None,
        blocksize: int = 2048,
    ) -> None:
        self.output_dir = output_dir
        self.target_sample_rate = target_sample_rate
        self.device_name_hint = device_name_hint
        self.device_index = device_index
        self.blocksize = blocksize
        self.device: LoopbackDevice | None = None
        self._raw_path = output_dir / "system_loopback_raw.wav"
        self._mono_path = output_dir / "system_loopback.wav"
        self._writer: AudioChunkWriter | None = None
        self._stream = None
        self._engine = None
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._started_at: float | None = None
        self._level_lock = threading.Lock()
        self._level = 0.0
        self._session_peak = 0.0
        self._audio_errors: tuple[type[BaseException], ...] = (OSError, RuntimeError, ValueError)

    @staticmethod
    def list_devices() -> str:
        """Return a printable list of loopback devices."""

        return format_loopback_devices(list_loopback_devices())

    def resolve_device(self) -> LoopbackDevice:
        """Return the endpoint to capture, honouring the configured hint or index."""

        devices = list_loopback_devices()
        if not devices:
            raise AudioRecorderError(
                "No hay dispositivos de loopback WASAPI disponibles. "
                "Verifica que pyaudiowpatch este instalado y que exista una salida activa."
            )

        if self.device_index is not None:
            for device in devices:
                if device.index == self.device_index:
                    return device
            raise AudioRecorderError(f"El dispositivo loopback {self.device_index} no existe.")

        if self.device_name_hint:
            hint = self.device_name_hint.casefold()
            for device in devices:
                if hint in device.clean_name.casefold():
                    return device
            raise AudioRecorderError(f"Ningun dispositivo loopback coincide con '{self.device_name_hint}'.")

        for device in devices:
            if device.is_default:
                return device
        return devices[0]

    def start(self) -> LoopbackDevice:
        """Open the loopback stream and begin writing chunks to disk."""

        if self._started_at is not None:
            raise AudioRecorderError("La grabacion ya esta en curso.")

        pyaudio = _import_pyaudio()
        self._audio_errors = _pyaudio_errors(pyaudio)
        self.device = self.resolve_device()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._engine = pyaudio.PyAudio()
        try:
            self._stream = self._engine.open(
                format=pyaudio.paFloat32,
                channels=self.device.channels,
                rate=self.device.sample_rate,
                input=True,
                input_device_index=self.device.index,
                frames_per_buffer=self.blocksize,
            )
        except self._audio_errors as exc:
            self._terminate_engine()
            raise AudioRecorderError(
                f"No se pudo abrir el loopback de '{self.device.clean_name}': {exc}"
            ) from exc

        self._writer = AudioChunkWriter(self._raw_path, self.device.sample_rate, self.device.channels)
        self._writer.start()

        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        self._started_at = time.monotonic()
        return self.device

    def _pump(self) -> None:
        """Read blocks from the loopback stream and forward them to the writer."""

        assert self._stream is not None and self._writer is not None and self.device is not None
        channels = self.device.channels
        while not self._stop_flag.is_set():
            try:
                raw = self._stream.read(self.blocksize, exception_on_overflow=False)
            except self._audio_errors:
                # Device removed, driver reset, or the endpoint went away.
                break
            if not raw:
                continue
            audio = np.frombuffer(raw, dtype=np.float32)
            if channels > 1:
                audio = audio.reshape(-1, channels)
            self._writer.write(audio)

            mono = audio.mean(axis=1) if audio.ndim > 1 else audio
            if mono.size:
                peak = float(np.max(np.abs(mono)))
                with self._level_lock:
                    if peak > self._level:
                        self._level = peak
                    if peak > self._session_peak:
                        self._session_peak = peak

    def levels(self) -> float:
        """Return the peak amplitude since the last call, then reset it."""

        with self._level_lock:
            value = self._level
            self._level = 0.0
        return value

    def session_peak(self) -> float:
        """Return the highest amplitude seen since the recording started."""

        with self._level_lock:
            return self._session_peak

    def elapsed_seconds(self) -> float:
        """Return how long the recording has been running."""

        if self._started_at is None:
            return 0.0
        return time.monotonic() - self._started_at

    @property
    def raw_path(self) -> Path:
        """Path of the raw capture at the device sample rate."""

        return self._raw_path

    @property
    def mono_path(self) -> Path:
        """Path of the resampled mono capture."""

        return self._mono_path

    def stop(self) -> tuple[Path, Path | None]:
        """Stop recording and return ``(raw_wav, mono_16k_wav)``.

        The mono file is ``None`` when the capture produced no samples, so the
        caller can decide whether silence is fatal (a diagnostic) or acceptable
        (mixing with a microphone track).
        """

        if self._started_at is None:
            raise AudioRecorderError("La grabacion no esta en curso.")

        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            finally:
                self._stream = None
        self._terminate_engine()

        if self._writer is not None:
            self._writer.stop()
            self._writer = None
        self._started_at = None

        mono = load_resampled_mono(self._raw_path, self.target_sample_rate)
        mono_out: Path | None = None
        if mono.size:
            sf.write(self._mono_path, mono, self.target_sample_rate, subtype="PCM_16")
            mono_out = self._mono_path
        return self._raw_path, mono_out

    def _terminate_engine(self) -> None:
        if self._engine is not None:
            try:
                self._engine.terminate()
            finally:
                self._engine = None
