"""Interactive CLI for the Meik Labs local meeting assistant."""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from datetime import datetime

from core.audio_recorder import AudioDeviceSelection, AudioRecorderError, MeetingAudioRecorder
from core.config import load_config
from core.summarizer import MeetingSummarizer, SummaryConfig, SummaryError
from core.transcriber import FasterWhisperTranscriber, TranscriptionConfig, TranscriptionError

SPINNER = "|/-\\"

# Peak-amplitude thresholds used to warn about a channel that captured nothing.
SILENCE_PEAK = 0.0005  # -66 dBFS
VERY_LOW_PEAK = 0.01  # -40 dBFS


def _db(level: float) -> str:
    """Return a short dBFS label for a linear amplitude."""

    if level <= 0:
        return "  -inf"
    return f"{20 * math.log10(level):6.1f}"


def _db_label(level: float) -> str:
    """Return a dBFS label including the unit, for post-recording summaries."""

    if level <= 0:
        return "-inf dBFS"
    return f"{20 * math.log10(level):.1f} dBFS"


def _run_recording_status(
    recorder: MeetingAudioRecorder,
    stop_event: threading.Event,
    max_seconds: float = 0.0,
) -> None:
    """Show a live recording indicator until the user stops the capture.

    The indicator makes it obvious that recording is in progress and whether each
    source is actually receiving signal, which is the fastest way to notice a
    muted endpoint or a wrong capture device.
    """

    frame = 0
    try:
        while not stop_event.is_set():
            time.sleep(0.25)
            elapsed = recorder.elapsed_seconds()
            levels = recorder.levels()
            stamp = f"{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}"
            remaining = f" / {max_seconds:.0f}s" if max_seconds else ""
            sys.stdout.write(
                f"\r REC {SPINNER[frame % 4]} {stamp}{remaining}  "
                f"sistema {_db(levels.get('system', 0.0))} dBFS  "
                f"microfono {_db(levels.get('microphone', 0.0))} dBFS  "
                f"[Enter para detener]   "
            )
            sys.stdout.flush()
            frame += 1
            if max_seconds and elapsed >= max_seconds:
                break
    except KeyboardInterrupt:
        pass
    sys.stdout.write("\r" + " " * 84 + "\r")
    sys.stdout.flush()


def _report_capture_quality(recorder: MeetingAudioRecorder, system_only: bool) -> None:
    """Report the recorded signal level and warn when a channel captured nothing.

    A silent channel is otherwise invisible: the WAV is still produced and the
    transcript simply comes back empty, so the problem is only noticed after the
    meeting is over. Surfacing it here keeps that failure mode visible.
    """

    peaks = recorder.session_peaks()
    system_peak = peaks.get("system", 0.0)
    microphone_peak = peaks.get("microphone", 0.0)

    warnings: list[str] = []
    if system_peak < SILENCE_PEAK:
        warnings.append(
            "El canal de SISTEMA no capturo audio. Causas probables: nadie mas hablo durante "
            "la grabacion, o Teams cambio de dispositivo de salida."
        )
    elif system_peak < VERY_LOW_PEAK:
        warnings.append("El canal de SISTEMA capturo muy bajo. Revisa el volumen de salida.")

    if not system_only:
        if microphone_peak < SILENCE_PEAK:
            warnings.append(
                "El canal de MICROFONO no capturo audio. Revisa el dispositivo de entrada con "
                "`python main.py --check-devices`."
            )
        elif microphone_peak < VERY_LOW_PEAK:
            warnings.append("El canal de MICROFONO capturo muy bajo. Revisa el nivel de entrada.")

    print(f"Niveles grabados: sistema {_db_label(system_peak)} | microfono {_db_label(microphone_peak)}")
    if not warnings:
        return

    print("")
    print("=" * 68)
    print(" AVISO: la grabacion puede estar incompleta")
    print("=" * 68)
    for warning in warnings:
        print(f" - {warning}")
    print("")
    print(" Diagnostico: python check_capture.py test")
    print("=" * 68)
    print("")


def _start_stop_watcher(stop_event: threading.Event) -> None:
    """Set ``stop_event`` once the user presses Enter."""

    def _wait() -> None:
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            stop_event.set()

    threading.Thread(target=_wait, daemon=True).start()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record, transcribe, and summarize Teams meetings locally.")
    parser.add_argument("--list-devices", action="store_true", help="Print audio devices and exit.")
    parser.add_argument("--check-devices", action="store_true", help="Validate selected audio devices and exit.")
    parser.add_argument("--system-device", type=int, help="System-audio capture device index, e.g. Stereo Mix.")
    parser.add_argument("--mic-device", type=int, help="Microphone input device index.")
    parser.add_argument(
        "--backend",
        choices=["wasapi-loopback", "stereo-mix"],
        default="wasapi-loopback",
        help="System-audio capture backend. wasapi-loopback captures any output, including Bluetooth/AirPods.",
    )
    parser.add_argument("--loopback-device", help="Loopback endpoint name substring, e.g. 'AirPods'.")
    parser.add_argument(
        "--mic-name",
        help="Microphone name substring. Defaults to the microphone of the captured headset.",
    )
    parser.add_argument("--no-summary", action="store_true", help="Skip external AI summarization.")
    parser.add_argument("--provider", choices=["anthropic", "google", "lmstudio"], help="Summary provider.")
    parser.add_argument("--model", help="faster-whisper model size, e.g. large-v3, medium, small.")
    parser.add_argument("--prompt", help="Optional technical glossary/context to bias Whisper transcription.")
    parser.add_argument("--whisperx", action="store_true", help="Use WhisperX and optional diarization for multi-speaker meetings.")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization when using WhisperX.")
    parser.add_argument("--cleanup", action="store_true", help="Run a conservative transcription cleanup pass with the selected LLM.")
    parser.add_argument("--meeting-type", choices=["technical", "business", "general"], default="technical", help="Meeting type for analysis prompts.")
    parser.add_argument("--analysis-only", action="store_true", help="Generate meeting analysis without standard summary.")
    parser.add_argument("--lmstudio-url", default="http://localhost:1234/v1", help="LM Studio OpenAI-compatible base URL.")
    parser.add_argument("--lmstudio-model", default="local-model", help="LM Studio model name.")
    parser.add_argument(
        "--system-only",
        action="store_true",
        help="Record only the system audio (what others say) and skip the microphone.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Auto-start and stop recording after N seconds (0 = manual Enter control).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()

    if args.list_devices:
        from core.wasapi_loopback import format_loopback_devices, list_loopback_devices, pyaudiowpatch_available

        print("WASAPI loopback devices (recommended for Bluetooth/AirPods):")
        if pyaudiowpatch_available():
            print(format_loopback_devices(list_loopback_devices()))
        else:
            print("  pyaudiowpatch is not installed; run `pip install pyaudiowpatch`.")
        print("")
        print(MeetingAudioRecorder.list_devices())
        return 0

    output_dir = config.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")

    # The microphone defaults to the system input device rather than being paired
    # with the captured headset: on Bluetooth headsets the microphone (HFP) and
    # the loopback-captured output (A2DP) cannot be active at the same time, so
    # pairing them silently breaks one of the two channels.
    mic_name = args.mic_name

    recorder = MeetingAudioRecorder(
        output_dir=output_dir,
        target_sample_rate=config.sample_rate,
        target_channels=config.channels,
        devices=AudioDeviceSelection(
            system_output_device=args.system_device,
            microphone_device=args.mic_device,
        ),
        backend=args.backend,
        loopback_device_name=args.loopback_device,
        loopback_device_index=args.system_device if args.backend == "wasapi-loopback" else None,
        microphone_device_name=mic_name,
        system_only=args.system_only,
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
        if args.duration > 0:
            print(f"Auto-start: recording for {args.duration:.0f} seconds...")
        else:
            input("Press Enter to start recording...")
        loopback_device = recorder.start()
        if loopback_device is not None:
            print(f"Recording system audio from loopback: {loopback_device.clean_name}")
        else:
            print("Recording system audio from Stereo Mix / loopback input.")
        if args.duration > 0:
            print(f"Recording {args.duration:.0f}s. Habla ahora.")
        else:
            print("Recording. Press Enter to stop (Ctrl+C also stops and continues).")

        stop_event = threading.Event()
        _start_stop_watcher(stop_event)
        _run_recording_status(recorder, stop_event, args.duration)

        recording = recorder.stop()
        print(f"Saved mixed audio: {recording.mixed_wav}")
        _report_capture_quality(recorder, args.system_only)

        transcriber = FasterWhisperTranscriber(transcriber_config)
        if transcriber_config.use_whisperx or transcriber_config.diarize:
            print("Transcribing with WhisperX and optional diarization...")
        else:
            print("Transcribing locally with faster-whisper...")
        transcript = transcriber.transcribe(recording.mixed_wav, output_dir)
        print(f"Saved transcript: {transcript.markdown_path}")

        if not args.no_summary:
            print("Generating AI meeting outputs...")
            summarizer = MeetingSummarizer(
                SummaryConfig(
                    provider=args.provider or config.summary_provider,
                    anthropic_model=config.anthropic_model,
                    google_model=config.google_model,
                    lmstudio_base_url=args.lmstudio_url,
                    lmstudio_model=args.lmstudio_model,
                    cleanup_pass=args.cleanup,
                    meeting_type=args.meeting_type,
                )
            )

            cleaned_transcript = summarizer.cleanup_transcript(transcript.text)
            if cleaned_transcript != transcript.text:
                cleaned_path = output_dir / "transcript_cleaned.md"
                cleaned_path.write_text(cleaned_transcript.strip() + "\n", encoding="utf-8")
                print(f"Saved cleaned transcript: {cleaned_path}")

            if args.analysis_only:
                analysis_path = output_dir / "meeting_analysis.md"
                analysis_path.write_text(summarizer.build_meeting_analysis(cleaned_transcript).strip() + "\n", encoding="utf-8")
                print(f"Saved analysis: {analysis_path}")
            else:
                report_path = summarizer.summarize_to_file(cleaned_transcript, output_dir)
                print(f"Saved report: {report_path}")
                analysis_path = output_dir / "meeting_analysis.md"
                analysis_path.write_text(summarizer.build_meeting_analysis(cleaned_transcript).strip() + "\n", encoding="utf-8")
                print(f"Saved analysis: {analysis_path}")

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
