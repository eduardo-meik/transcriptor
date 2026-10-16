"""Diagnostics for the Meik Labs meeting assistant audio capture.

Two modes:

``level``
    Records a short sample from the system-audio (loopback/stereo-mix) device
    and from the microphone, then reports signal levels. Use this to detect the
    most common failure: a *silent* system capture because Teams is playing
    through a different output (for example a Bluetooth headset) than the one
    Stereo Mix is attached to.

``inspect``
    Reads an existing WAV file (for example ``meeting_audio.wav``) and reports
    its format, duration and signal levels.

Usage::

    python check_capture.py level --seconds 10
    python check_capture.py level --system-device 14 --mic-device 1
    python check_capture.py inspect outputs/20260101_120000/meeting_audio.wav
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from core.audio_recorder import (
    AudioChunkWriter,
    AudioDeviceSelection,
    AudioRecorderError,
    MeetingAudioRecorder,
)
from core.wasapi_loopback import (
    WasapiLoopbackRecorder,
    format_loopback_devices,
    list_loopback_devices,
    pyaudiowpatch_available,
)

# Peak amplitude thresholds (linear), used to classify capture quality.
SILENCE_PEAK = 0.0005  # -66 dBFS
VERY_LOW_PEAK = 0.01  # -40 dBFS
LOW_PEAK = 0.05  # -26 dBFS


def _db(level: float) -> str:
    """Return a dBFS label for a linear amplitude."""

    if level <= 0:
        return "-inf dBFS"
    return f"{20 * np.log10(level):.1f} dBFS"


def _verdict(peak: float) -> str:
    """Classify a peak amplitude in a human-readable way."""

    if peak < SILENCE_PEAK:
        return "SILENCIO (no se capturo audio)"
    if peak < VERY_LOW_PEAK:
        return "MUY BAJO (probablemente inaudible para Whisper)"
    if peak < LOW_PEAK:
        return "BAJO (revisa volumen; puede transcribir mal)"
    return "OK (nivel adecuado)"


def _per_second_levels(mono: np.ndarray, sample_rate: int) -> list[float]:
    """Return the peak amplitude for each whole second of audio."""

    levels: list[float] = []
    for start in range(0, len(mono), sample_rate):
        window = mono[start : start + sample_rate]
        if window.size == 0:
            continue
        levels.append(float(np.max(np.abs(window))))
    return levels


def _capture(device: int, seconds: float, label: str, quiet: bool = False) -> tuple[np.ndarray, int]:
    """Record ``seconds`` from ``device`` and return mono float32 audio + rate."""

    sample_rate, channels = MeetingAudioRecorder.find_supported_input_settings(device)
    info = sd.query_devices(device)
    if not quiet:
        print(f"  {label}: device {device} ({info['name']}) @ {sample_rate} Hz, {channels} ch")

    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        device=device,
        dtype="float32",
        blocking=True,
        extra_settings=MeetingAudioRecorder.wasapi_settings(device),
    )
    frames = np.asarray(audio, dtype=np.float32).reshape(-1, channels)
    return frames.mean(axis=1), sample_rate


def _report(label: str, mono: np.ndarray, sample_rate: int, out_path: Path) -> float:
    """Print level information for a capture and save it to disk."""

    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
    print(f"  {label}: pico {_db(peak)} | RMS {_db(rms)} | {len(mono) / sample_rate:.1f} s")
    print(f"  {label} -> {_verdict(peak)}")

    sf.write(out_path, mono, sample_rate, subtype="PCM_16")
    print(f"  {label} guardado en: {out_path}")

    levels = _per_second_levels(mono, sample_rate)
    if levels:
        timeline = " ".join(f"{level:.3f}" for level in levels)
        print(f"  {label} pico por segundo: {timeline}")
    return peak


class _LiveSource:
    """Common interface for a live, level-reporting capture source."""

    description = ""

    def start(self) -> None:
        raise NotImplementedError

    def level(self) -> float:
        raise NotImplementedError

    def elapsed(self) -> float:
        raise NotImplementedError

    def stop(self) -> Path:
        raise NotImplementedError


class _LoopbackSource(_LiveSource):
    """WASAPI loopback capture: works with Bluetooth and any render endpoint."""

    def __init__(self, output_dir: Path, device_index: int | None = None, name_hint: str | None = None) -> None:
        self._recorder = WasapiLoopbackRecorder(
            output_dir,
            device_name_hint=name_hint,
            device_index=device_index,
        )

    @property
    def description(self) -> str:
        try:
            device = self._recorder.resolve_device()
        except AudioRecorderError as exc:
            return f"WASAPI loopback (error: {exc})"
        return f"WASAPI loopback -> {device.clean_name} ({device.sample_rate} Hz, {device.channels} ch)"

    def start(self) -> None:
        self._recorder.start()

    def level(self) -> float:
        return self._recorder.levels()

    def elapsed(self) -> float:
        return self._recorder.elapsed_seconds()

    def stop(self) -> Path:
        raw, mono = self._recorder.stop()
        return mono or raw


class _StereoMixSource(_LiveSource):
    """Stereo Mix capture through sounddevice, for comparison only."""

    def __init__(self, output_dir: Path, device_index: int | None = None) -> None:
        self._output_dir = output_dir
        self._device_index = device_index
        self._path = output_dir / "system_check.wav"
        self._stream: sd.InputStream | None = None
        self._writer: AudioChunkWriter | None = None
        self._lock = threading.Lock()
        self._peak = 0.0
        self._started_at: float | None = None

    def _resolve(self) -> int:
        if self._device_index is not None:
            return self._device_index
        recorder = MeetingAudioRecorder(
            output_dir=self._output_dir,
            devices=AudioDeviceSelection(),
        )
        return recorder.resolve_system_capture_device()

    @property
    def description(self) -> str:
        try:
            device = self._resolve()
            sample_rate, channels = MeetingAudioRecorder.find_supported_input_settings(device)
        except AudioRecorderError as exc:
            return f"Stereo Mix (error: {exc})"
        return f"Stereo Mix -> device {device} ({sample_rate} Hz, {channels} ch)"

    def start(self) -> None:
        device = self._resolve()
        sample_rate, channels = MeetingAudioRecorder.find_supported_input_settings(device)
        self._writer = AudioChunkWriter(self._path, sample_rate, channels)
        self._writer.start()

        def callback(indata: np.ndarray, _frames: int, _time_info: object, _status: object) -> None:
            array = np.asarray(indata, dtype=np.float32)
            self._writer.write(array)  # type: ignore[union-attr]
            mono = array.mean(axis=1) if array.ndim > 1 else array
            if mono.size:
                peak = float(np.max(np.abs(mono)))
                with self._lock:
                    if peak > self._peak:
                        self._peak = peak

        self._stream = sd.InputStream(
            device=device,
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            callback=callback,
            extra_settings=MeetingAudioRecorder.wasapi_settings(device),
        )
        self._stream.start()
        self._started_at = time.monotonic()

    def level(self) -> float:
        with self._lock:
            value = self._peak
            self._peak = 0.0
        return value

    def elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        return time.monotonic() - self._started_at

    def stop(self) -> Path:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._writer is not None:
            self._writer.stop()
            self._writer = None
        self._started_at = None
        return self._path


def _build_source(args: argparse.Namespace, output_dir: Path) -> _LiveSource:
    if args.backend == "stereo-mix":
        return _StereoMixSource(output_dir, device_index=args.system_device)
    return _LoopbackSource(output_dir, device_index=args.system_device, name_hint=args.device_name)


def run_devices(_args: argparse.Namespace) -> int:
    """List the loopback endpoints and Stereo Mix candidates available for capture."""

    print("=" * 68)
    print(" DISPOSITIVOS DE CAPTURA DE SISTEMA")
    print("=" * 68)
    print("")

    if not pyaudiowpatch_available():
        print(" WASAPI loopback: NO disponible (falta el paquete pyaudiowpatch).")
    else:
        print(" " + format_loopback_devices(list_loopback_devices()).replace("\n", "\n "))

    print("")
    candidates = [
        line
        for line in MeetingAudioRecorder.list_devices().splitlines()
        if "system-capture-candidate" in line
    ]
    if candidates:
        print(" Candidatos Stereo Mix (sounddevice):")
        for line in candidates:
            print(f"   {line.strip()}")
    else:
        print(" Candidatos Stereo Mix (sounddevice): ninguno detectado.")

    print("")
    print(" Recomendacion: usa --backend loopback si tus reuniones suenan por AirPods/Bluetooth.")
    return 0


def run_mics(args: argparse.Namespace) -> int:
    """Record a short sample from every input device and rank them by level.

    Useful when the expected microphone is silent: it shows at a glance which
    capture endpoint is actually picking up sound.
    """

    host_apis = sd.query_hostapis()
    candidates: dict[str, tuple[int, str, str, bool]] = {}
    for index, device in enumerate(sd.query_devices()):
        if int(device["max_input_channels"]) <= 0:
            continue
        name = " ".join(str(device["name"]).split())
        api = str(host_apis[device["hostapi"]]["name"])
        is_wasapi = api.upper().endswith("WASAPI")
        current = candidates.get(name.casefold())
        if current is None or (is_wasapi and not current[3]):
            candidates[name.casefold()] = (index, name, api, is_wasapi)

    ordered = sorted(candidates.values(), key=lambda item: item[0])
    seconds = args.seconds
    print("=" * 68)
    print(f" PRUEBA DE MICROFONOS ({len(ordered)} dispositivos, {seconds:.0f}s cada uno)")
    print("=" * 68)
    print(" HABLA de forma continua mientras corre la prueba.")
    print("")

    results: list[tuple[float, int, str, str]] = []
    for position, (index, name, api, _is_wasapi) in enumerate(ordered, start=1):
        print(f" [{position}/{len(ordered)}] {name} | {api}")
        try:
            mono, _sample_rate = _capture(index, seconds, "    ", quiet=True)
        except (AudioRecorderError, sd.PortAudioError, OSError, RuntimeError, ValueError) as exc:
            print(f"     error: {exc}")
            continue
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        results.append((peak, index, name, api))
        print(f"     pico {_db(peak)} -> {_verdict(peak)}")

    print("")
    print("=" * 68)
    print(" RESULTADO (mejor primero)")
    print("=" * 68)
    for peak, index, name, api in sorted(results, reverse=True):
        print(f"  [{index:3d}] {_db(peak):>9}  {name} | {api}")

    working = [item for item in results if item[0] >= VERY_LOW_PEAK]
    print("")
    if not working:
        print(" Ningun microfono capturo voz. Revisa el mute del equipo y el nivel de entrada.")
        return 2

    best = max(working)
    print(f" RECOMENDADO: device {best[1]} ({best[2]}) con pico {_db(best[0])}")
    print(f"   main.py --mic-device {best[1]}")
    return 0


def run_test(args: argparse.Namespace) -> int:
    """Guided capture test with countdown, live level bar and explicit end.

    This command is intentionally verbose. Once capture is confirmed working it
    can be dropped in favour of ``main.py``, which keeps a compact live status
    line during real recordings.
    """

    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    source = _build_source(args, output_dir)

    print("=" * 68)
    print(" TEST DE CAPTURA DE AUDIO")
    print("=" * 68)
    print(f" Backend : {args.backend}")
    print(f" Fuente  : {source.description}")
    print("")
    print(" INSTRUCCIONES")
    print("   1. Deja los AirPods puestos, con el audio de la reunion activo.")
    print("   2. Cuando aparezca HABLA AHORA, habla en voz alta un momento")
    print("      o pide a alguien de la reunion que hable.")
    print("   3. Mira la barra. Si se queda vacia, NO se esta capturando.")
    print("")

    for remaining in (3, 2, 1):
        sys.stdout.write(f" Comenzando en {remaining}...")
        sys.stdout.write("\r")
        sys.stdout.flush()
        time.sleep(1)

    try:
        source.start()
    except AudioRecorderError as exc:
        print(" " * 40, end="\r")
        print(f"Error: {exc}")
        return 1

    print(" " * 40, end="\r")
    print("=" * 68)
    print(" >>>>>>>>>>>>>>>>>>  HABLA AHORA  <<<<<<<<<<<<<<<<<<")
    print("=" * 68)

    total = args.seconds
    peak = 0.0
    spinner = "|/-\\"
    frames = 0
    interrupted = False
    try:
        while True:
            time.sleep(0.15)
            elapsed = source.elapsed()
            level = source.level()
            peak = max(peak, level)
            dbfs = 20 * np.log10(level) if level > 0 else -100.0
            filled = int(max(0.0, min(1.0, (dbfs + 60) / 60)) * 30)
            bar = "#" * filled + "." * (30 - filled)
            sys.stdout.write(
                f"\r REC {spinner[frames % 4]} [{bar}] {elapsed:5.1f}s/{total:.0f}s "
                f" nivel {dbfs:6.1f} dBFS  pico {_db(peak)}   "
            )
            sys.stdout.flush()
            frames += 1
            if elapsed >= total:
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        stop_error = None
        try:
            saved = source.stop()
        except AudioRecorderError as exc:
            saved = None
            stop_error = exc

    print("")
    if stop_error is not None:
        print("=" * 68)
        print(" FIN DEL TEST")
        print("=" * 68)
        print(f" Error al cerrar la captura: {stop_error}")
        return 1

    print("=" * 68)
    print(" FIN DEL TEST" + ("  (detenido con Ctrl+C)" if interrupted else ""))
    print("=" * 68)
    print(f" Pico alcanzado : {_db(peak)}")
    print(f" Veredicto      : {_verdict(peak)}")
    print(f" Archivo        : {saved}")

    peak_verdict = _verdict(peak)
    if peak < SILENCE_PEAK:
        print("")
        print(" SIGUIENTE PASO: no se capturo audio del sistema.")
        print("   - Prueba con --backend stereo-mix para comparar.")
        print("   - Revisa que la salida elegida este reproduciendo audio.")
        return 2

    print("")
    print(" SIGUIENTE PASO: la captura funciona.")
    print("   Prueba el flujo completo con:")
    print("     python main.py --no-summary")
    if peak_verdict.startswith("BAJO"):
        print("   (Sube el volumen de Teams para mejorar la transcripcion.)")
    return 0


def run_level(args: argparse.Namespace) -> int:

    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Diagnostico de captura de audio")
    print("Reproduce el audio de la reunion mientras corre este test.")
    print("")

    default_output = sd.query_devices(kind="output")
    print(f"Dispositivo de salida por defecto: {default_output['name']}")
    print("  Si Teams suena por Bluetooth/HDMI, Stereo Mix grabara silencio.")
    print("")

    peaks: dict[str, float] = {}

    system_device = args.system_device
    if system_device is None:
        system_device = MeetingAudioRecorder(output_dir=output_dir).resolve_system_capture_device()
    print(f"Capturando sistema ({args.seconds:.0f} s)...")
    system_mono, system_rate = _capture(system_device, args.seconds, "sistema")
    peaks["sistema"] = _report("sistema", system_mono, system_rate, output_dir / "system_check.wav")
    print("")

    if not args.skip_mic:
        mic_device = args.mic_device
        if mic_device is None:
            mic_device = sd.default.device[0]
        if mic_device is None or mic_device < 0:
            print("Sin microfono por defecto; se omite.")
        else:
            print(f"Capturando microfono ({args.seconds:.0f} s)...")
            mic_mono, mic_rate = _capture(int(mic_device), args.seconds, "microfono")
            peaks["microfono"] = _report("microfono", mic_mono, mic_rate, output_dir / "mic_check.wav")
            print("")

    print("Resumen")
    for name, peak in peaks.items():
        print(f"  {name}: pico {_db(peak)} -> {_verdict(peak)}")

    if "sistema" in peaks and peaks["sistema"] < SILENCE_PEAK:
        print("")
        print("El canal de sistema esta en silencio. Revisa:")
        print("  1. Salida de audio de Teams == dispositivo que alimenta Mezcla estereo.")
        print("  2. Volumen del sistema y de Teams por encima de 0.")
        print("  3. Si usas auriculares Bluetooth, Stereo Mix no lo captura.")
        return 2

    if "sistema" in peaks and peaks["sistema"] < VERY_LOW_PEAK:
        print("")
        print("Aviso: el canal de sistema capturo senal muy debil.")
        print("  Sube el volumen de Teams/sistema y repite el test antes de grabar la reunion.")

    print("")
    print(f"Archivos de diagnostico en: {output_dir}")
    return 0


def _find_output_device(name: str) -> int | None:
    """Return the output device index whose name contains ``name``, preferring WASAPI."""

    host_apis = sd.query_hostapis()
    fallback: int | None = None
    for index, device in enumerate(sd.query_devices()):
        if int(device["max_output_channels"]) <= 0:
            continue
        device_name = " ".join(str(device["name"]).split())
        if name.casefold() not in device_name.casefold():
            continue
        if str(host_apis[device["hostapi"]]["name"]).upper().endswith("WASAPI"):
            return index
        if fallback is None:
            fallback = index
    return fallback


def _play_tone(name: str, tone: np.ndarray, sink: dict[str, str]) -> None:
    """Play a tone to the speaker matching ``name``, falling back to the default.

    Uses sounddevice rather than soundcard: the latter is pure Python but has
    proven unstable on this setup and can terminate the whole process.
    """

    try:
        index = _find_output_device(name)
        if index is None:
            index = sd.default.device[1]
        if index is None or int(index) < 0:
            sink["error"] = "no hay dispositivo de salida disponible."
            return

        device_index = int(index)
        info = sd.query_devices(device_index)
        rate = int(info["default_samplerate"])
        channels = min(2, int(info["max_output_channels"]))
        sink["device"] = " ".join(str(info["name"]).split())

        block = np.repeat(tone[:, None], channels, axis=1) if channels > 1 else tone
        sd.play(np.ascontiguousarray(block, dtype=np.float32), samplerate=rate, device=device_index)
        sd.wait()
    except (sd.PortAudioError, OSError, RuntimeError, ValueError) as exc:
        sink["error"] = f"no se pudo reproducir el tono: {exc}"


def run_loopback(args: argparse.Namespace) -> int:
    """Play a tone to the captured endpoint and record it back through loopback.

    This is the end-to-end proof of the capture path: the tone is rendered to the
    same endpoint that is being captured, so it is recorded if and only if the
    loopback chain works. With AirPods the tone is only audible to you, not to
    the other meeting participants.
    """

    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print(" TEST DE LOOPBACK - tono de ida y vuelta (WASAPI)")
    print("=" * 68)

    recorder = WasapiLoopbackRecorder(
        output_dir,
        device_name_hint=args.device_name,
        device_index=args.system_device,
    )
    try:
        device = recorder.resolve_device()
    except AudioRecorderError as exc:
        print(f" Error: {exc}")
        return 1

    print(f" Endpoint a capturar : {device.clean_name}")
    print(f" Formato             : {device.sample_rate} Hz, {device.channels} ch")
    print("")
    print(" INSTRUCCIONES")
    print(f"   1. Escucharas un tono de 440 Hz durante {max(1.0, args.seconds):.0f}s en ese dispositivo.")
    print("   2. Con AirPods solo lo oyes tu; los demas participantes NO lo oyen.")
    print("   3. Sube el volumen del sistema a un nivel comodo (50% o mas).")
    print("")

    for remaining in (3, 2, 1):
        sys.stdout.write(f" Comenzando en {remaining}...\r")
        sys.stdout.flush()
        time.sleep(1)
    print(" " * 40, end="\r")

    try:
        recorder.start()
    except AudioRecorderError as exc:
        print(f" Error: {exc}")
        return 1

    tone_seconds = max(1.0, args.seconds)
    timeline = np.arange(int(tone_seconds * device.sample_rate)) / device.sample_rate
    tone = (0.3 * np.sin(2 * np.pi * 440.0 * timeline)).astype(np.float32)

    print("=" * 68)
    print(" >>>>>>>>>>>>  REPRODUCIENDO TONO AHORA  <<<<<<<<<<<<")
    print("=" * 68)

    sink: dict[str, str] = {}
    player = threading.Thread(
        target=_play_tone,
        args=(device.clean_name, tone, sink),
        daemon=True,
    )
    player.start()

    total = tone_seconds + 1.0
    peak = 0.0
    spinner = "|/-\\"
    frame = 0
    while True:
        time.sleep(0.15)
        elapsed = recorder.elapsed_seconds()
        level = recorder.levels()
        peak = max(peak, level)
        dbfs = 20 * np.log10(level) if level > 0 else -100.0
        filled = int(max(0.0, min(1.0, (dbfs + 60) / 60)) * 30)
        bar = "#" * filled + "." * (30 - filled)
        sys.stdout.write(
            f"\r REC {spinner[frame % 4]} [{bar}] {elapsed:5.1f}s/{total:.0f}s "
            f" nivel {dbfs:6.1f} dBFS  pico {_db(peak)}   "
        )
        sys.stdout.flush()
        frame += 1
        if elapsed >= total:
            break
    player.join(timeout=2)

    saved: Path | None = None
    stop_error: AudioRecorderError | None = None
    try:
        _raw, saved = recorder.stop()
    except AudioRecorderError as exc:
        stop_error = exc

    print("")
    print("=" * 68)
    print(" FIN DEL TEST")
    print("=" * 68)
    if sink.get("device"):
        print(f" Tono reproducido en  : {sink['device']}")
    if sink.get("error"):
        print(f" Aviso de reproduccion: {sink['error']}")
    print(f" Pico capturado       : {_db(peak)}")
    if saved is not None:
        print(f" Archivo              : {saved}")
    if stop_error is not None:
        print(f" Error al cerrar      : {stop_error}")
    print("")

    if peak < SILENCE_PEAK:
        print(" RESULTADO: el tono NO se capturo.")
        print("   El loopback de ese endpoint no esta entregando audio.")
        print("   Revisa: volumen del sistema en 0, endpoint en silencio,")
        print("   o un realzador de audio (Nahimic/Realtek) que redirige la salida.")
        return 2

    print(" RESULTADO: el tono SI se capturo. La ruta de grabacion FUNCIONA.")
    print("   Ya puedes grabar la reunion con:")
    print("     python main.py --backend loopback --no-summary")
    return 0


def run_monitor(args: argparse.Namespace) -> int:
    """Show a live level meter so you can confirm capture while the meeting runs."""

    device = args.system_device
    if device is None:
        device = MeetingAudioRecorder(output_dir=Path("outputs/_diag")).resolve_system_capture_device()

    sample_rate, channels = MeetingAudioRecorder.find_supported_input_settings(device)
    info = sd.query_devices(device)
    print(f"Monitoreando device {device} ({info['name']}) @ {sample_rate} Hz, {channels} ch")
    print("Pide a alguien que hable en la reunion. Ctrl+C para salir.")
    print("")

    window_peak = {"value": 0.0}
    overall_peak = {"value": 0.0}

    def callback(indata: np.ndarray, _frames: int, _time_info: object, _status: object) -> None:
        mono = np.asarray(indata, dtype=np.float32).reshape(-1, channels).mean(axis=1)
        if mono.size:
            value = float(np.max(np.abs(mono)))
            window_peak["value"] = max(window_peak["value"], value)
            overall_peak["value"] = max(overall_peak["value"], value)

    interval = 0.2
    bars = 40
    stream = sd.InputStream(
        device=device,
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        callback=callback,
    )
    started = time.monotonic()
    try:
        stream.start()
        while True:
            time.sleep(interval)
            value = window_peak["value"]
            window_peak["value"] = 0.0
            dbfs = 20 * np.log10(value) if value > 0 else -100.0
            filled = int(max(0.0, min(1.0, (dbfs + 60) / 60)) * bars)
            bar = "#" * filled + "." * (bars - filled)
            elapsed = time.monotonic() - started
            print(f"[{bar}] {dbfs:6.1f} dBFS  {elapsed:6.1f}s", end="\r", flush=True)
            if args.seconds and elapsed >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()

    print("")
    print("")
    print(f"Pico global: {_db(overall_peak['value'])} -> {_verdict(overall_peak['value'])}")
    return 0


def run_inspect(args: argparse.Namespace) -> int:

    path = Path(args.wav_path)
    if not path.exists():
        print(f"Error: no existe {path}")
        return 1

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    duration = len(mono) / sample_rate
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0

    print(f"Archivo: {path}")
    print(f"  Formato: {audio.shape[1]} canal(es), {sample_rate} Hz")
    print(f"  Duracion: {duration:.1f} s ({duration / 60:.1f} min)")
    print(f"  Pico: {_db(peak)} | RMS: {_db(rms)}")

    if len(mono) == 0:
        print("  Veredicto: VACIO (sin muestras)")
        return 2
    if peak < SILENCE_PEAK:
        print("  Veredicto: SILENCIO")
        return 2

    silent_seconds = sum(1 for level in _per_second_levels(mono, sample_rate) if level < SILENCE_PEAK)
    print(f"  Segundos en silencio: {silent_seconds}")
    print("  Veredicto: OK, listo para transcribir")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audio capture diagnostics for the meeting assistant.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    level = subparsers.add_parser("level", help="Record a short sample and report signal levels.")
    level.add_argument("--seconds", type=float, default=10.0, help="Seconds to record per source.")
    level.add_argument("--system-device", type=int, help="System-audio capture device index.")
    level.add_argument("--mic-device", type=int, help="Microphone device index.")
    level.add_argument("--skip-mic", action="store_true", help="Only test the system-audio channel.")
    level.add_argument("--output-dir", default="outputs/_diag", help="Folder for diagnostic WAV files.")
    level.set_defaults(func=run_level)

    test = subparsers.add_parser("test", help="Guided capture test with countdown and live level feedback.")
    test.add_argument(
        "--backend",
        choices=["loopback", "stereo-mix"],
        default="loopback",
        help="loopback = WASAPI loopback (captures Bluetooth/AirPods); stereo-mix = Mezcla estereo.",
    )
    test.add_argument("--seconds", type=float, default=10.0, help="Test duration in seconds.")
    test.add_argument("--system-device", type=int, help="Device index (loopback index or Stereo Mix index).")
    test.add_argument("--device-name", help="Loopback endpoint name substring, e.g. 'AirPods'.")
    test.add_argument("--output-dir", default="outputs/_diag", help="Folder for the test WAV.")
    test.set_defaults(func=run_test)

    devices = subparsers.add_parser("devices", help="List WASAPI loopback endpoints and Stereo Mix candidates.")
    devices.set_defaults(func=run_devices)

    mics = subparsers.add_parser("mics", help="Test every input device and rank them by captured level.")
    mics.add_argument("--seconds", type=float, default=2.5, help="Seconds recorded per device.")
    mics.set_defaults(func=run_mics)

    inspect = subparsers.add_parser("inspect", help="Inspect an existing WAV file.")
    inspect.add_argument("wav_path", help="Path to the WAV file to inspect.")
    inspect.set_defaults(func=run_inspect)

    monitor = subparsers.add_parser("monitor", help="Live level meter for the system-audio device.")
    monitor.add_argument("--system-device", type=int, help="System-audio capture device index.")
    monitor.add_argument("--seconds", type=float, default=0.0, help="Stop after N seconds (0 = until Ctrl+C).")
    monitor.set_defaults(func=run_monitor)

    loopback = subparsers.add_parser(
        "loopback-test",
        help="Play a tone through the speakers and record it back from Stereo Mix.",
    )
    loopback.add_argument("--system-device", type=int, help="Loopback endpoint index to capture.")
    loopback.add_argument("--device-name", help="Loopback endpoint name substring, e.g. 'AirPods'.")
    loopback.add_argument("--seconds", type=float, default=3.0, help="Tone duration in seconds.")
    loopback.add_argument("--output-dir", default="outputs/_diag", help="Folder for the recorded WAV.")
    loopback.set_defaults(func=run_loopback)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return int(args.func(args))
    except AudioRecorderError as exc:
        print(f"Error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrumpido.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
