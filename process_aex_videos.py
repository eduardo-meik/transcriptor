"""Resume transcription, cleanup, and summary generation for AEX video files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from core.config import load_config
from core.summarizer import MeetingSummarizer, SummaryConfig
from core.transcriber import FasterWhisperTranscriber, TranscriptionConfig


def _has_content(path: Path) -> bool:
    """Return whether a generated text artifact contains meaningful content."""
    return path.exists() and bool(path.read_text(encoding="utf-8").strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process AEX video files with local transcription and LM Studio.")
    parser.add_argument("--video-dir", type=Path, default=Path("AEX_videoCooke"))
    parser.add_argument("--model", default="large-v3", help="faster-whisper model size.")
    parser.add_argument("--device", default="cuda", help="Whisper device (cuda or cpu).")
    parser.add_argument("--compute-type", default="float16", help="Whisper compute type.")
    parser.add_argument("--lmstudio-model", default="google/gemma-4-e4b")
    parser.add_argument("--lmstudio-url", default="http://localhost:1234/v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    video_paths = sorted(path for path in args.video_dir.glob("*.mov") if path.is_file())
    if not video_paths:
        print(f"No .mov files found in {args.video_dir}")
        return 1

    transcriber = FasterWhisperTranscriber(
        TranscriptionConfig(
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=config.language,
            allow_cpu_fallback=True,
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

    for index, video_path in enumerate(video_paths, start=1):
        output_dir = args.video_dir / f"{video_path.stem}_opt"
        raw_path = output_dir / "transcript_raw.txt"
        cleaned_path = args.video_dir / f"{video_path.stem}_opt_cleaned.txt"
        summary_path = args.video_dir / f"{video_path.stem}_opt_summary.md"
        print(f"[{index}/{len(video_paths)}] {video_path.name}", flush=True)

        if _has_content(raw_path):
            raw_text = raw_path.read_text(encoding="utf-8").strip()
            print("  Reusing existing transcription.")
        else:
            transcript = transcriber.transcribe(video_path, output_dir)
            raw_text = transcript.text
            print(f"  Transcription saved: {raw_path.name}", flush=True)

        if not raw_text.strip():
            print("  Empty transcription (no speech detected); skipping cleanup and summary.")
            continue

        if _has_content(cleaned_path):
            cleaned_text = cleaned_path.read_text(encoding="utf-8").strip()
            print("  Reusing existing cleaned transcript.")
        else:
            cleaned_text = assistant.cleanup_transcript(raw_text)
            cleaned_path.write_text(cleaned_text.strip() + "\n", encoding="utf-8")
            print(f"  Cleanup saved: {cleaned_path.name}", flush=True)

        if _has_content(summary_path):
            print("  Reusing existing summary.")
        else:
            summary = assistant._create_provider().summarize(cleaned_text)
            summary_path.write_text(summary.strip() + "\n", encoding="utf-8")
            print(f"  Summary saved: {summary_path.name}", flush=True)

    print("AEX video batch processing complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
