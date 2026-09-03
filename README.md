# Audio & Video Transcriptor / Meik Labs Meeting Assistant

This workspace now contains two related workflows:

- `transcriptor.py`: the original file-based audio/video transcription CLI.
- `main.py`: an interactive local meeting assistant for recording, transcribing, and optionally summarizing Microsoft Teams meetings on Windows.

## Meeting Assistant

The meeting assistant records Windows system audio through an available loopback/stereo-mix capture device and records your microphone as a second stream. When stopped, it mixes both sources into a 16 kHz mono WAV file, transcribes it locally with `faster-whisper`, and can optionally generate a Markdown report through Anthropic or Google GenAI.

### Meeting Assistant Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

For NVIDIA/CUDA acceleration, install a PyTorch/CUDA stack compatible with your driver and GPU before using `faster-whisper`. The app defaults to `device="cuda"` and `compute_type="float16"`, with CPU fallback enabled through configuration.

Create a `.env` file only if you want API-based summaries:

```env
SUMMARY_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key_here
# or
# SUMMARY_PROVIDER=google
# GOOGLE_API_KEY=your_key_here

WHISPER_MODEL=large-v3
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
WHISPER_ALLOW_CPU_FALLBACK=true
MEIK_LANGUAGE=es
HF_HUB_DISABLE_SYMLINKS_WARNING=1
# Optional, only for higher Hugging Face download limits:
# HF_TOKEN=your_huggingface_token_here
```

The first transcription may download the `faster-whisper` model from Hugging Face. An unauthenticated HF Hub warning is normal; configure `HF_TOKEN` only if downloads are rate-limited or slow. The Windows symlink cache warning is harmless and disabled by default in the app.

List devices:

```bash
python main.py --list-devices
```

Look for a device marked `system-capture-candidate`, such as `Mezcla estéreo` / `Stereo Mix`. If Windows does not expose one, enable Stereo Mix in the Windows sound control panel or install a virtual loopback device.

Record and transcribe, skipping external summarization:

```bash
python main.py --no-summary
```

Choose devices explicitly:

```bash
python main.py --system-device 12 --mic-device 3 --no-summary
```

Validate selected devices without starting a recording:

```bash
python main.py --system-device 20 --mic-device 14 --check-devices
```

Run full recording, transcription, and summary:

```bash
python main.py --provider anthropic
```

Outputs are saved under `outputs/<timestamp>/`:

- `meeting_audio.wav`
- `system_raw.wav`
- `microphone_raw.wav`
- `transcript_raw.txt`
- `transcript_raw.md`
- `transcript_segments.md`
- `meeting_report.md` when summarization is enabled

> Privacy note: transcription is local. Summarization through Anthropic or Google sends transcript text to an external API, so use `--no-summary` for fully local processing.

---

# Original File Transcriptor

A Python script to transcribe audio and video files into Spanish text using OpenAI's Whisper model. Supports MP4, MOV (iPhone videos), and other common formats.

## Features

- 🎯 Transcribe audio and video files to Spanish text
- 📹 Video support: MP4, MOV (iPhone), M4V, AVI, MKV, and more
- 🎵 Audio support: MP3, WAV, M4A, FLAC, OGG, and more
- 📝 Multiple output formats: TXT, SRT, VTT, JSON
- 🎚️ Multiple model sizes for different accuracy/speed tradeoffs
- ⏱️ Includes timestamps in subtitle formats
- 🔄 Automatic audio extraction from video files

## Prerequisites

- Python 3.8 or higher
- FFmpeg (required by Whisper and for video processing)

### Installing FFmpeg

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update && sudo apt install ffmpeg
```

## Setup

1. **Create a virtual environment:**
```bash
python -m venv venv
```

2. **Activate the virtual environment:**

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Transcribe an audio or video file to Spanish text:
```bash
# Audio file
python transcriptor.py audio.mp3

# Video file (automatic audio extraction)
python transcriptor.py video.mp4

# iPhone video
python transcriptor.py iphone_video.mov
```

This will create a file named `<filename>_transcription.txt` in the same directory as your media file.

### Advanced Options

**Use a different model size:**
```bash
python transcriptor.py video.mp4 --model medium
```

Available models (from fastest to most accurate):
- `tiny` - Fastest, least accurate
- `base` - Good balance (default)
- `small` - Better accuracy
- `medium` - High accuracy
- `large` - Best accuracy, slowest

**Save as subtitle format:**
```bash
# SRT format (for most video players)
python transcriptor.py video.mp4 --format srt

# VTT format (for web)
python transcriptor.py video.mp4 --format vtt

# JSON format (with detailed metadata)
python transcriptor.py video.mp4 --format json
```

**Specify output directory:**
```bash
python transcriptor.py video.mp4 --output ./transcriptions
```

### Complete Examples

```bash
# Transcribe iPhone video with high accuracy
python transcriptor.py iphone_video.mov --model medium --format srt --output ./transcriptions

# Transcribe MP4 video as subtitle file
python transcriptor.py my_video.mp4 --model medium --format srt --output ./transcriptions

# Transcribe audio file
python transcriptor.py my_podcast.mp3 --model medium --format srt --output ./transcriptions
```

## Supported Formats

### Video Formats
- MP4
- MOV (iPhone videos)
- M4V
- AVI
- MKV
- FLV
- WMV
- WebM
- 3GP

### Audio Formats
Whisper supports most common audio formats:
- MP3
- WAV
- M4A
- FLAC
- OGG
- Opus
- WMA
- AAC
- ALAC
- And many more...

## Model Sizes & Performance

| Model  | Parameters | Size | Speed | Accuracy |
|--------|-----------|------|-------|----------|
| tiny   | 39M       | ~75 MB | Fastest | Basic |
| base   | 74M       | ~140 MB | Fast | Good |
| small  | 244M      | ~460 MB | Moderate | Better |
| medium | 769M      | ~1.5 GB | Slow | High |
| large  | 1550M     | ~3 GB | Slowest | Best |

**Note:** The first time you use a model, it will be downloaded automatically.

## Output Formats

### TXT (Plain Text)
Simple text file with the complete transcription.

### SRT (SubRip Subtitle)
Standard subtitle format with timestamps, compatible with most video players.

### VTT (WebVTT)
Web-compatible subtitle format, ideal for HTML5 video.

### JSON
Complete transcription data including:
- Full text
- Segmented text with timestamps
- Language detection
- Metadata

## Tips

1. **For better accuracy:** Use the `medium` or `large` model
2. **For faster processing:** Use the `tiny` or `base` model
3. **For long recordings:** Consider using `base` or `small` to balance speed and accuracy
4. **Audio quality matters:** Clear audio with minimal background noise produces better results

## Troubleshooting

**"FFmpeg not found" error:**
- Make sure FFmpeg is installed and available in your PATH
- Restart your terminal after installing FFmpeg

**Out of memory error:**
- Try using a smaller model (`tiny` or `base`)
- Process shorter audio segments

**Slow transcription:**
- Use a smaller model for faster processing
- Consider using a GPU-enabled version of PyTorch if you have an NVIDIA GPU

## License

This project uses OpenAI's Whisper model. Please check [OpenAI's license](https://github.com/openai/whisper) for more information.

## Credits

- [OpenAI Whisper](https://github.com/openai/whisper) - The underlying transcription model
