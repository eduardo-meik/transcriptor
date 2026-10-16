"""Post-processing helpers for reducing stored meeting audio size."""

from __future__ import annotations

import logging
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import av

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioOptimizationReport:
    converted: tuple[Path, ...]
    warnings: tuple[str, ...]


class AudioOptimizationError(RuntimeError):
    """Raised when a WAV cannot be safely replaced with a validated MP3."""


def optimize_wav_files(meeting_dir: Path) -> AudioOptimizationReport:
    """Convert completed meeting WAV artifacts to MP3, removing only validated sources."""

    wav_paths = tuple(sorted(path for path in meeting_dir.glob("*.wav") if path.is_file()))
    if not wav_paths:
        return AudioOptimizationReport(converted=(), warnings=())

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return AudioOptimizationReport(
            converted=(),
            warnings=(f"FFmpeg no está disponible; se conservaron {len(wav_paths)} archivo(s) WAV.",),
        )

    converted: list[Path] = []
    warnings: list[str] = []
    for wav_path in wav_paths:
        try:
            converted.append(_convert_and_remove_wav(wav_path, ffmpeg))
        except (
            AudioOptimizationError,
            OSError,
            EOFError,
            wave.Error,
            subprocess.SubprocessError,
            av.error.FFmpegError,
            ValueError,
        ) as exc:
            warning = f"{wav_path.name}: no se pudo optimizar; se conserva el WAV ({exc})."
            logger.warning("%s", warning)
            warnings.append(warning)

    return AudioOptimizationReport(converted=tuple(converted), warnings=tuple(warnings))


def read_mp3_duration(path: Path) -> float:
    """Return the duration of a readable MP3 in seconds."""

    try:
        with av.open(str(path)) as audio_file:
            if not audio_file.streams.audio:
                raise AudioOptimizationError(f"El archivo no contiene audio: {path}")
            if audio_file.duration is not None:
                return audio_file.duration / av.time_base

            stream = audio_file.streams.audio[0]
            if stream.duration is None or stream.time_base is None:
                raise AudioOptimizationError(f"No se pudo determinar la duración del MP3: {path}")
            return float(stream.duration * stream.time_base)
    except (OSError, av.error.FFmpegError, ValueError) as exc:
        raise AudioOptimizationError(f"No se pudo leer la duración del MP3 {path}: {exc}") from exc


def _convert_and_remove_wav(wav_path: Path, ffmpeg: str) -> Path:
    with wave.open(str(wav_path), "rb") as source:
        frame_rate = source.getframerate()
        if frame_rate <= 0 or source.getnframes() <= 0:
            raise AudioOptimizationError("el archivo WAV está vacío o no tiene una frecuencia válida")
        source_duration = source.getnframes() / frame_rate

    mp3_path = wav_path.with_suffix(".mp3")
    temporary_mp3 = wav_path.with_name(f".{wav_path.stem}.converting.mp3")
    timeout = max(120.0, source_duration * 2)
    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(wav_path),
                "-vn",
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "4",
                str(temporary_mp3),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or f"FFmpeg terminó con código {result.returncode}"
            raise AudioOptimizationError(detail)
        if not temporary_mp3.is_file() or temporary_mp3.stat().st_size == 0:
            raise AudioOptimizationError("FFmpeg no generó un MP3 válido")

        mp3_duration = _validate_mp3(temporary_mp3)
        tolerance = max(0.5, source_duration * 0.01)
        if abs(mp3_duration - source_duration) > tolerance:
            raise AudioOptimizationError(
                f"la duración del MP3 ({mp3_duration:.2f}s) no coincide con la del WAV ({source_duration:.2f}s)"
            )

        temporary_mp3.replace(mp3_path)
        wav_path.unlink()
        return mp3_path
    finally:
        if temporary_mp3.exists():
            try:
                temporary_mp3.unlink()
            except OSError:
                logger.warning("Could not remove temporary MP3 file %s", temporary_mp3, exc_info=True)


def _validate_mp3(path: Path) -> float:
    with av.open(str(path)) as audio_file:
        if not audio_file.streams.audio:
            raise AudioOptimizationError("el MP3 no contiene una pista de audio")
        if audio_file.duration is not None:
            duration = audio_file.duration / av.time_base
        else:
            stream = audio_file.streams.audio[0]
            if stream.duration is None or stream.time_base is None:
                raise AudioOptimizationError("no se pudo determinar la duración del MP3")
            duration = float(stream.duration * stream.time_base)

        has_audio_frames = False
        for _frame in audio_file.decode(audio=0):
            has_audio_frames = True
        if not has_audio_frames or duration <= 0:
            raise AudioOptimizationError("el MP3 está vacío o no se puede decodificar")
        return duration
