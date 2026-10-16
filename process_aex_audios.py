"""Resume transcription, cleanup, and summary generation for AEX audio files."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.config import load_config
from core.summarizer import MeetingSummarizer, SummaryConfig
from core.transcriber import FasterWhisperTranscriber, TranscriptionConfig


def _has_content(path: Path) -> bool:
    """Return whether a generated text artifact contains meaningful content."""
    return path.exists() and bool(path.read_text(encoding="utf-8").strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process AEX audio files with local transcription and LM Studio.")
    parser.add_argument("--audio-dir", type=Path, default=Path("AEX_audios"))
    parser.add_argument("--model", default="large-v3", help="faster-whisper model size.")
    parser.add_argument("--lmstudio-model", default="google/gemma-4-e4b")
    parser.add_argument("--lmstudio-url", default="http://localhost:1234/v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    audio_paths = sorted(path for path in args.audio_dir.glob("*.m4a") if path.is_file())
    if not audio_paths:
        print(f"No .m4a files found in {args.audio_dir}")
        return 1

    transcriber = FasterWhisperTranscriber(
        TranscriptionConfig(
            model_size=args.model,
            device="cpu",
            compute_type="int8",
            language=config.language,
            allow_cpu_fallback=False,
            initial_prompt=config.initial_prompt,
        )
    )
    assistant = MeetingSummarizer(
        SummaryConfig(
            provider="lmstudio",
            lmstudio_base_url=args.lmstudio_url,
            lmstudio_model=args.lmstudio_model,
            cleanup_pass=True,
            meeting_type="technical",
        )
    )

    for index, audio_path in enumerate(audio_paths, start=1):
        output_dir = args.audio_dir / f"{audio_path.stem}_opt"
        raw_path = output_dir / "transcript_raw.txt"
        cleaned_path = args.audio_dir / f"{audio_path.stem}_opt_cleaned.txt"
        summary_path = args.audio_dir / f"{audio_path.stem}_opt_summary.md"
        print(f"[{index}/{len(audio_paths)}] {audio_path.name}")

        if _has_content(raw_path):
            raw_text = raw_path.read_text(encoding="utf-8").strip()
            print("  Reusing existing transcription.")
        else:
            transcript = transcriber.transcribe(audio_path, output_dir)
            raw_text = transcript.text
            print(f"  Transcription saved: {raw_path.name}")

        if _has_content(cleaned_path):
            cleaned_text = cleaned_path.read_text(encoding="utf-8").strip()
            print("  Reusing existing cleaned transcript.")
        else:
            cleaned_text = assistant.cleanup_transcript(raw_text)
            cleaned_path.write_text(cleaned_text.strip() + "\n", encoding="utf-8")
            print(f"  Cleanup saved: {cleaned_path.name}")

        if _has_content(summary_path):
            print("  Reusing existing summary.")
        else:
            summary = assistant._create_provider().summarize(cleaned_text)
            summary_path.write_text(summary.strip() + "\n", encoding="utf-8")
            print(f"  Summary saved: {summary_path.name}")

    print("AEX batch processing complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())