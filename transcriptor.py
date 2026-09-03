#!/usr/bin/env python3
"""
Audio & Video Transcription Script
Transcribes audio and video files to Spanish text using OpenAI's Whisper model.
Supports MP4, MOV (iPhone), and other common formats.
"""

import argparse
import os
import sys
import subprocess
import tempfile
from pathlib import Path
import whisper
from datetime import datetime


# Supported formats
VIDEO_FORMATS = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.flv', '.wmv', '.webm', '.3gp'}
AUDIO_FORMATS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.wma', '.aac', '.alac'}


def is_video_file(file_path):
    """Check if the file is a video file."""
    extension = Path(file_path).suffix.lower()
    return extension in VIDEO_FORMATS


def is_audio_file(file_path):
    """Check if the file is an audio file."""
    extension = Path(file_path).suffix.lower()
    return extension in AUDIO_FORMATS


def extract_audio_from_video(video_path, output_audio_path=None):
    """
    Extract audio from a video file using FFmpeg.
    
    Args:
        video_path: Path to the video file
        output_audio_path: Path where to save the extracted audio (default: temp file)
    
    Returns:
        Path to the extracted audio file
    """
    if output_audio_path is None:
        # Create a temporary audio file
        fd, output_audio_path = tempfile.mkstemp(suffix=".wav", prefix="transcriptor_audio_")
        os.close(fd)
    
    print(f"Extracting audio from video: {video_path}")
    print(f"Saving to: {output_audio_path}")
    
    try:
        # Use FFmpeg to extract audio
        # -i: input file
        # -q:a: audio quality (0=best)
        # -map a: select only audio stream
        command = [
            "ffmpeg",
            "-i", str(video_path),
            "-q:a", "0",
            "-map", "a",
            "-y",  # Overwrite output file without asking
            str(output_audio_path)
        ]
        
        # Run FFmpeg with minimal output
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        print(f"✓ Audio extracted successfully")
        return output_audio_path
        
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e.stderr}", file=sys.stderr)
        if output_audio_path and os.path.exists(output_audio_path):
            os.remove(output_audio_path)
        raise
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg.", file=sys.stderr)
        print("  Windows: choco install ffmpeg", file=sys.stderr)
        print("  macOS: brew install ffmpeg", file=sys.stderr)
        print("  Linux: sudo apt install ffmpeg", file=sys.stderr)
        raise


def transcribe_audio(audio_path, model_size="base", output_dir=None, output_format="txt", output_stem=None):
    """
    Transcribe an audio file to Spanish text using Whisper.
    
    Args:
        audio_path: Path to the audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        output_dir: Directory to save transcription (default: same as audio file)
        output_format: Output format (txt, srt, vtt, json)
    
    Returns:
        Dictionary with transcription result
    """
    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)
    
    print(f"Transcribing audio file: {audio_path}")
    print("This may take a few minutes depending on the file size and model...")
    
    # Transcribe with Spanish language setting
    result = model.transcribe(
        audio_path,
        language="es",  # Spanish
        task="transcribe",
        verbose=True
    )
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path(audio_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename
    audio_filename = output_stem or Path(audio_path).stem
    
    # Save transcription based on format
    if output_format == "txt":
        output_file = output_dir / f"{audio_filename}_transcription.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"\n✓ Transcription saved to: {output_file}")
    
    elif output_format == "srt":
        output_file = output_dir / f"{audio_filename}_transcription.srt"
        write_srt(result["segments"], output_file)
        print(f"\n✓ SRT subtitle file saved to: {output_file}")
    
    elif output_format == "vtt":
        output_file = output_dir / f"{audio_filename}_transcription.vtt"
        write_vtt(result["segments"], output_file)
        print(f"\n✓ VTT subtitle file saved to: {output_file}")
    
    elif output_format == "json":
        import json
        output_file = output_dir / f"{audio_filename}_transcription.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n✓ JSON file saved to: {output_file}")
    
    return result


def write_srt(segments, output_file):
    """Write segments to SRT subtitle format."""
    with open(output_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start_time = format_timestamp(segment["start"], always_include_hours=True, decimal_marker=",")
            end_time = format_timestamp(segment["end"], always_include_hours=True, decimal_marker=",")
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment['text'].strip()}\n\n")


def write_vtt(segments, output_file):
    """Write segments to WebVTT subtitle format."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for segment in segments:
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment['text'].strip()}\n\n")


def format_timestamp(seconds, always_include_hours=False, decimal_marker="."):
    """Convert seconds to timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    
    if always_include_hours or hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    else:
        return f"{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio and video files to Spanish text using Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Formats:
  Video: MP4, MOV (iPhone), M4V, AVI, MKV, FLV, WMV, WebM, 3GP
  Audio: MP3, WAV, M4A, FLAC, OGG, Opus, WMA, AAC, ALAC

Examples:
  # Transcribe video file
  python transcriptor.py video.mp4
  
  # Transcribe iPhone video
  python transcriptor.py iphone_video.mov
  
  # Use a larger model for better accuracy
  python transcriptor.py media.mp4 --model medium
  
  # Save as SRT subtitle file
  python transcriptor.py media.mp4 --format srt
  
  # Specify output directory
  python transcriptor.py media.mp4 --output ./transcriptions

Model sizes (from fastest/least accurate to slowest/most accurate):
  tiny, base, small, medium, large
        """
    )
    
    parser.add_argument(
        "media_file",
        help="Path to the audio or video file to transcribe"
    )
    
    parser.add_argument(
        "-m", "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: base)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output directory for transcription (default: same as media file)"
    )
    
    parser.add_argument(
        "-f", "--format",
        choices=["txt", "srt", "vtt", "json"],
        default="txt",
        help="Output format (default: txt)"
    )
    
    args = parser.parse_args()
    
    # Check if media file exists
    if not os.path.exists(args.media_file):
        print(f"Error: Media file not found: {args.media_file}", file=sys.stderr)
        sys.exit(1)
    
    # Determine if it's a video or audio file
    media_path = args.media_file
    temp_audio_file = None
    
    try:
        if is_video_file(media_path):
            print(f"Video file detected: {Path(media_path).name}")
            # Extract audio from video
            temp_audio_file = extract_audio_from_video(media_path)
            audio_to_transcribe = temp_audio_file
            
        elif is_audio_file(media_path):
            print(f"Audio file detected: {Path(media_path).name}")
            audio_to_transcribe = media_path
            
        else:
            print(f"Error: Unsupported file format", file=sys.stderr)
            print(f"Supported video formats: {', '.join(VIDEO_FORMATS)}", file=sys.stderr)
            print(f"Supported audio formats: {', '.join(AUDIO_FORMATS)}", file=sys.stderr)
            sys.exit(1)
        
        print()
        result = transcribe_audio(
            audio_to_transcribe,
            model_size=args.model,
            output_dir=args.output or Path(media_path).parent,
            output_format=args.format,
            output_stem=Path(media_path).stem
        )
        
        print("\n" + "="*50)
        print("TRANSCRIPTION PREVIEW:")
        print("="*50)
        print(result["text"][:500] + ("..." if len(result["text"]) > 500 else ""))
        print("\n✓ Transcription complete!")
        
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        sys.exit(1)
        
    finally:
        # Clean up temporary audio file if it was created
        if temp_audio_file and os.path.exists(temp_audio_file):
            try:
                os.remove(temp_audio_file)
                print(f"Cleaned up temporary files")
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_audio_file}: {e}")


if __name__ == "__main__":
    main()
