"""Local transcription with faster-whisper and optional WhisperX diarization."""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import os
from pathlib import Path
import re
from typing import Any

from faster_whisper import WhisperModel

# Transcripts are opened on Windows by double-click (Notepad, Word, editors that
# default to the ANSI code page). A UTF-8 BOM makes those apps detect UTF-8 so
# Spanish accents (á, é, ñ, ¿, ¡) are not shown as mojibake like "Ã¡".
OUTPUT_ENCODING = "utf-8-sig"

# Whisper follows the style of its prompt. Without one it sometimes drops
# accents and inverted punctuation in Spanish; a short accented prompt fixes it.
DEFAULT_SPANISH_PROMPT = (
    "Reunión en español. Transcripción con tildes y signos: ¿qué?, ¡sí!, información, también."
)


class TranscriptionError(RuntimeError):
    """Raised when local transcription fails."""


@dataclass(frozen=True)
class TranscriptionConfig:
    model_size: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "es"
    allow_cpu_fallback: bool = True
    initial_prompt: str | None = None
    use_whisperx: bool = False
    diarize: bool = False
    hf_token: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    text_path: Path
    markdown_path: Path
    segment_path: Path
    text: str


class FasterWhisperTranscriber:
    """Transcribe meeting audio locally with faster-whisper and optional WhisperX diarization."""

    def __init__(self, config: TranscriptionConfig) -> None:
        self.config = config

    def transcribe(self, audio_path: Path, output_dir: Path) -> TranscriptionResult:
        """Transcribe a WAV file and save text, Markdown, and timestamped segments."""

        if not audio_path.exists():
            raise TranscriptionError(f"Audio file does not exist: {audio_path}")

        if self.config.use_whisperx or self.config.diarize:
            return self._transcribe_with_whisperx(audio_path, output_dir)
        return self._transcribe_faster(audio_path, output_dir)

    def transcribe_tracks(
        self,
        tracks: list[tuple[str, Path | None]],
        output_dir: Path,
        fallback_audio: Path | None = None,
    ) -> TranscriptionResult:
        """Transcribe each speaker track separately and merge them by time.

        The recorder keeps the microphone (the local user) and the system audio
        (remote participants) as separate WAVs, so transcribing them one by one
        labels every segment with who spoke without needing diarization models.
        """

        usable = [(label, path) for label, path in tracks if path is not None and _has_audio(path)]
        if self.config.use_whisperx or self.config.diarize or not usable:
            if fallback_audio is None:
                raise TranscriptionError("No audio tracks available to transcribe.")
            return self.transcribe(fallback_audio, output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        model = self._load_model()
        merged: list[dict[str, Any]] = []
        language = self.config.language
        for label, path in usable:
            model, segments, info = self._run_model(model, path)
            language = getattr(info, "language", language) or language
            for segment in segments:
                text = _get_segment_text(segment)
                if not text:
                    continue
                merged.append(
                    {
                        "start": float(_get_value(segment, "start", 0.0)),
                        "end": float(_get_value(segment, "end", 0.0)),
                        "text": text,
                        "speaker": label,
                    }
                )

        merged.sort(key=lambda item: (item["start"], item["end"]))
        merged = _drop_echo_segments(merged)
        audio_path = fallback_audio or usable[0][1]
        return self._write_outputs(output_dir, merged, audio_path, language)

    def _transcribe_faster(self, audio_path: Path, output_dir: Path) -> TranscriptionResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        model = self._load_model()
        _, segments, info = self._run_model(model, audio_path)
        return self._write_outputs(output_dir, segments, audio_path, info.language)

    def _transcribe_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "language": self.config.language,
            "task": "transcribe",
            "vad_filter": True,
        }
        prompt = self._effective_prompt()
        if prompt:
            kwargs["initial_prompt"] = prompt
        return kwargs

    def _effective_prompt(self) -> str | None:
        if self.config.initial_prompt:
            return self.config.initial_prompt
        if (self.config.language or "").lower().startswith("es"):
            return DEFAULT_SPANISH_PROMPT
        return None

    def _run_model(self, model: WhisperModel, audio_path: Path) -> tuple[WhisperModel, list[Any], Any]:
        transcribe_kwargs = self._transcribe_kwargs()
        try:
            segments, info = model.transcribe(str(audio_path), **transcribe_kwargs)
            return model, list(segments), info
        except (RuntimeError, ValueError, OSError) as exc:
            # CUDA libraries (for example cublas64_12.dll) can fail lazily on the
            # first real inference rather than when the model is constructed, so
            # the CPU fallback has to be applied here as well.
            if not self.config.allow_cpu_fallback or self.config.device == "cpu":
                raise TranscriptionError(f"Transcription failed: {exc}") from exc
            print(f"CUDA transcription failed: {exc}")
            print("Retrying on CPU with int8 compute.")
            model = WhisperModel(self.config.model_size, device="cpu", compute_type="int8")
            try:
                segments, info = model.transcribe(str(audio_path), **transcribe_kwargs)
                return model, list(segments), info
            except (RuntimeError, ValueError, OSError) as retry_exc:
                raise TranscriptionError(f"Transcription failed on CPU as well: {retry_exc}") from retry_exc

    def _write_outputs(
        self,
        output_dir: Path,
        segments: list[Any],
        audio_path: Path,
        language: str,
    ) -> TranscriptionResult:
        lines: list[str] = []
        for segment in segments:
            text = _get_segment_text(segment)
            if not text:
                continue
            speaker = _get_segment_speaker(segment)
            lines.append(f"[{speaker}] {text}" if speaker else text)
        text = "\n".join(lines)

        text_path = output_dir / "transcript_raw.txt"
        markdown_path = output_dir / "transcript_raw.md"
        segment_path = output_dir / "transcript_segments.md"

        text_path.write_text(text + "\n", encoding=OUTPUT_ENCODING)
        markdown_path.write_text(self._format_markdown(text, audio_path, language), encoding=OUTPUT_ENCODING)
        segment_path.write_text(self._format_segments(segments), encoding=OUTPUT_ENCODING)

        return TranscriptionResult(
            text_path=text_path,
            markdown_path=markdown_path,
            segment_path=segment_path,
            text=text,
        )

    def _transcribe_with_whisperx(self, audio_path: Path, output_dir: Path) -> TranscriptionResult:
        try:
            import whisperx
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise TranscriptionError(
                "WhisperX is not installed. Install it with `pip install whisperx` and configure HF_TOKEN for diarization."
            ) from exc

        output_dir.mkdir(parents=True, exist_ok=True)
        token = self.config.hf_token or os.getenv("HF_TOKEN")
        audio = whisperx.load_audio(str(audio_path))

        try:
            model = whisperx.load_model(
                self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
            result = model.transcribe(
                audio,
                batch_size=8,
                language=self.config.language,
                task="transcribe",
                initial_prompt=self._effective_prompt(),
            )
            segments = result["segments"]

            if self.config.diarize:
                if not token:
                    raise TranscriptionError(
                        "HF_TOKEN is required for WhisperX diarization. Set HF_TOKEN in your environment or config."
                    )
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=token,
                    device=self.config.device,
                )
                diarize_segments = diarize_model(
                    audio,
                    min_speakers=self.config.min_speakers,
                    max_speakers=self.config.max_speakers,
                )
                align_model, align_metadata = whisperx.load_align_model(
                    language_code=self.config.language,
                    device=self.config.device,
                )
                aligned = whisperx.align(
                    segments,
                    align_model,
                    align_metadata,
                    audio,
                    device=self.config.device,
                    return_char_alignments=False,
                )
                result = whisperx.assign_word_speakers(diarize_segments, aligned)
                segments = result["segments"]

        except (RuntimeError, ValueError, OSError, AttributeError) as exc:
            raise TranscriptionError(f"WhisperX transcription failed: {exc}") from exc

        return self._write_outputs(output_dir, segments, audio_path, self.config.language)

    def _load_model(self) -> WhisperModel:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        try:
            return WhisperModel(
                self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            if not self.config.allow_cpu_fallback or self.config.device == "cpu":
                raise TranscriptionError(f"Could not load Whisper model on {self.config.device}: {exc}") from exc
            print(f"Could not load CUDA Whisper model: {exc}")
            print("Falling back to CPU with int8 compute.")
            return WhisperModel(self.config.model_size, device="cpu", compute_type="int8")

    @staticmethod
    def _format_markdown(text: str, audio_path: Path, language: str) -> str:
        return f"""# Raw Meeting Transcript

**Audio:** `{audio_path.name}`
**Detected language:** `{language}`

---

{text}
"""

    @staticmethod
    def _format_segments(segments: list[object] | list[dict[str, Any]]) -> str:
        lines = ["# Timestamped Transcript", ""]
        for segment in segments:
            start = float(_get_value(segment, "start", 0.0))
            end = float(_get_value(segment, "end", 0.0))
            text = _get_segment_text(segment)
            speaker = _get_segment_speaker(segment)
            if text:
                label = f" **[{speaker}]**" if speaker else ""
                lines.append(f"**{_format_timestamp(start)} - {_format_timestamp(end)}**{label} {text}")
                lines.append("")
        return "\n".join(lines)


def _get_value(segment: Any, key: str, default: Any = None) -> Any:
    if isinstance(segment, dict):
        return segment.get(key, default)
    return getattr(segment, key, default)


def _get_segment_text(segment: Any) -> str:
    return str(_get_value(segment, "text", "") or "").strip()


def _get_segment_speaker(segment: Any) -> str | None:
    speaker = _get_value(segment, "speaker")
    return str(speaker) if speaker else None


def _has_audio(path: Path) -> bool:
    """Return True when a WAV exists and holds more than its 44-byte header."""

    try:
        return path.exists() and path.stat().st_size > 44
    except OSError:
        return False


def _normalize_for_compare(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _drop_echo_segments(segments: list[dict[str, Any]], similarity: float = 0.6) -> list[dict[str, Any]]:
    """Drop microphone segments that are just the speakers' audio leaking in.

    Without headphones the microphone also picks up the remote participants, so
    the same sentence appears twice: once as "Participantes" and once as "Yo".
    A "Yo" segment overlapping a remote one with very similar text is an echo.
    """

    remote = [segment for segment in segments if segment.get("speaker") != "Yo"]
    kept: list[dict[str, Any]] = []
    for segment in segments:
        if segment.get("speaker") == "Yo":
            mine = _normalize_for_compare(segment["text"])
            is_echo = False
            for other in remote:
                overlaps = other["start"] < segment["end"] and segment["start"] < other["end"]
                if not overlaps:
                    continue
                ratio = difflib.SequenceMatcher(None, mine, _normalize_for_compare(other["text"])).ratio()
                if ratio >= similarity:
                    is_echo = True
                    break
            if is_echo:
                continue
        kept.append(segment)
    return kept


def _format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
