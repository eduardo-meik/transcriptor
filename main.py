"""Interactive CLI for the Meik Labs local meeting assistant."""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from core.audio_recorder import AudioDeviceSelection, AudioRecorderError, MeetingAudioRecorder
from core.config import load_config
from core.summarizer import MeetingSummarizer, SummaryConfig, SummaryError
from core.transcriber import FasterWhisperTranscriber, TranscriptionConfig, TranscriptionError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record, transcribe, and summarize Teams meetings locally.")
    parser.add_argument("--list-devices", action="store_true", help="Print audio devices and exit.")
    parser.add_argument("--check-devices", action="store_true", help="Validate selected audio devices and exit.")
    parser.add_argument("--system-device", type=int, help="System-audio capture device index, e.g. Stereo Mix.")
    parser.add_argument("--mic-device", type=int, help="Microphone input device index.")
    parser.add_argument("--no-summary", action="store_true", help="Skip external AI summarization.")
    parser.add_argument("--provider", choices=["anthropic", "google"], help="Summary provider.")
    parser.add_argument("--model", help="faster-whisper model size, e.g. large-v3, medium, small.")
    parser.add_argument("--prompt", help="Optional technical glossary/context to bias Whisper transcription.")
    parser.add_argument("--whisperx", action="store_true", help="Use WhisperX and optional diarization for multi-speaker meetings.")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization when using WhisperX.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()

    if args.list_devices:
        print(MeetingAudioRecorder.list_devices())
        return 0

    output_dir = config.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    recorder = MeetingAudioRecorder(
        output_dir=output_dir,
        target_sample_rate=config.sample_rate,
        target_channels=config.channels,
        devices=AudioDeviceSelection(
            system_output_device=args.system_device,
            microphone_device=args.mic_device,
        ),
    )

    if args.check_devices:
        try:
            print(recorder.check_devices())
            return 0
        except AudioRecorderError as exc:
            print(f"Error: {exc}")
            return 1

    transcriber_config = TranscriptionConfig(
        model_size=args.model or config.whisper_model,
        device=config.whisper_device,
        compute_type=config.whisper_compute_type,
        language=config.language,
        allow_cpu_fallback=config.allow_cpu_fallback,
        initial_prompt=args.prompt or config.initial_prompt,
        use_whisperx=args.whisperx,
        diarize=args.diarize or config.enable_diarization,
        hf_token=os.getenv("HF_TOKEN"),
    )

    try:
        input("Press Enter to start recording...")
        recorder.start()
        print("Recording. Press Enter again to stop.")
        input()
        recording = recorder.stop()
        print(f"Saved mixed audio: {recording.mixed_wav}")

        transcriber = FasterWhisperTranscriber(transcriber_config)
        if transcriber_config.use_whisperx or transcriber_config.diarize:
            print("Transcribing with WhisperX and optional diarization...")
        else:
            print("Transcribing locally with faster-whisper...")
        transcript = transcriber.transcribe(recording.mixed_wav, output_dir)
        print(f"Saved transcript: {transcript.markdown_path}")

        if not args.no_summary:
            print("Generating optional AI meeting report...")
            summarizer = MeetingSummarizer(
                SummaryConfig(
                    provider=args.provider or config.summary_provider,
                    anthropic_model=config.anthropic_model,
                    google_model=config.google_model,
                )
            )
            report_path = summarizer.summarize_to_file(transcript.text, output_dir)
            print(f"Saved report: {report_path}")

        print(f"Done. Output folder: {output_dir}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Stopping recorder if needed...")
        try:
            recorder.stop()
        except AudioRecorderError:
            pass
        return 130
    except (AudioRecorderError, TranscriptionError, SummaryError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
