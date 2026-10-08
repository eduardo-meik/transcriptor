"""Local transcription with faster-whisper and optional WhisperX diarization."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel


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

    def _transcribe_faster(self, audio_path: Path, output_dir: Path) -> TranscriptionResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        model = self._load_model()

        transcribe_kwargs: dict[str, Any] = {
            "language": self.config.language,
            "task": "transcribe",
            "vad_filter": True,
        }
        if self.config.initial_prompt:
            transcribe_kwargs["initial_prompt"] = self.config.initial_prompt

        try:
            segments, info = model.transcribe(str(audio_path), **transcribe_kwargs)
            collected_segments = list(segments)
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
                collected_segments = list(segments)
            except (RuntimeError, ValueError, OSError) as retry_exc:
                raise TranscriptionError(f"Transcription failed on CPU as well: {retry_exc}") from retry_exc

        text = "\n".join(_get_segment_text(segment) for segment in collected_segments if _get_segment_text(segment))
        text_path = output_dir / "transcript_raw.txt"
        markdown_path = output_dir / "transcript_raw.md"
        segment_path = output_dir / "transcript_segments.md"

        text_path.write_text(text + "\n", encoding="utf-8")
        markdown_path.write_text(self._format_markdown(text, audio_path, info.language), encoding="utf-8")
        segment_path.write_text(self._format_segments(collected_segments), encoding="utf-8")

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
                initial_prompt=self.config.initial_prompt,
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

        text = "\n".join(_get_segment_text(segment) for segment in segments if _get_segment_text(segment))
        text_path = output_dir / "transcript_raw.txt"
        markdown_path = output_dir / "transcript_raw.md"
        segment_path = output_dir / "transcript_segments.md"

        text_path.write_text(text + "\n", encoding="utf-8")
        markdown_path.write_text(self._format_markdown(text, audio_path, self.config.language), encoding="utf-8")
        segment_path.write_text(self._format_segments(segments), encoding="utf-8")

        return TranscriptionResult(
            text_path=text_path,
            markdown_path=markdown_path,
            segment_path=segment_path,
            text=text,
        )

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
            if isinstance(segment, dict):
                start = float(segment.get("start", 0.0))
                end = float(segment.get("end", 0.0))
                text = str(segment.get("text", "")).strip()
                speaker = segment.get("speaker")
            else:
                start = getattr(segment, "start")
                end = getattr(segment, "end")
                text = getattr(segment, "text").strip()
                speaker = getattr(segment, "speaker", None)
            if text:
                label = f" **[{speaker}]**" if speaker else ""
                lines.append(f"**{_format_timestamp(start)} - {_format_timestamp(end)}**{label} {text}")
                lines.append("")
        return "\n".join(lines)


def _get_segment_text(segment: Any) -> str:
    if isinstance(segment, dict):
        text = segment.get("text", "")
    else:
        text = getattr(segment, "text", "")
    return str(text).strip()


def _format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
