from __future__ import annotations

import math
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from core.audio_optimizer import optimize_wav_files, read_mp3_duration


class AudioOptimizerTests(unittest.TestCase):
    def test_missing_ffmpeg_preserves_all_wav_files(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            meeting_dir = Path(temporary_dir)
            wav_paths = (meeting_dir / "meeting_audio.wav", meeting_dir / "microphone_raw.wav")
            for wav_path in wav_paths:
                self._write_wav(wav_path)

            with mock.patch("core.audio_optimizer.shutil.which", return_value=None):
                report = optimize_wav_files(meeting_dir)

            self.assertEqual(report.converted, ())
            self.assertEqual(len(report.warnings), 1)
            self.assertTrue(all(wav_path.exists() for wav_path in wav_paths))

    def test_ffmpeg_failure_preserves_source_and_cleans_temporary_output(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            meeting_dir = Path(temporary_dir)
            wav_path = meeting_dir / "meeting_audio.wav"
            temporary_mp3 = meeting_dir / ".meeting_audio.converting.mp3"
            self._write_wav(wav_path)
            failure = subprocess.CompletedProcess(
                args=["ffmpeg"],
                returncode=1,
                stdout="",
                stderr="encoder failed",
            )

            with (
                mock.patch("core.audio_optimizer.shutil.which", return_value="ffmpeg.exe"),
                mock.patch("core.audio_optimizer.subprocess.run", return_value=failure),
            ):
                report = optimize_wav_files(meeting_dir)

            self.assertEqual(report.converted, ())
            self.assertIn("encoder failed", report.warnings[0])
            self.assertTrue(wav_path.exists())
            self.assertFalse(temporary_mp3.exists())

    def test_invalid_mp3_is_rejected_and_wav_is_retained(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            meeting_dir = Path(temporary_dir)
            wav_path = meeting_dir / "meeting_audio.wav"
            self._write_wav(wav_path)

            def write_invalid_mp3(command, **_kwargs):
                Path(command[-1]).write_bytes(b"not an mp3")
                return subprocess.CompletedProcess(
                    args=command,
                    returncode=0,
                    stdout="",
                    stderr="",
                )

            with (
                mock.patch("core.audio_optimizer.shutil.which", return_value="ffmpeg.exe"),
                mock.patch("core.audio_optimizer.subprocess.run", side_effect=write_invalid_mp3),
            ):
                report = optimize_wav_files(meeting_dir)

            self.assertEqual(report.converted, ())
            self.assertTrue(report.warnings)
            self.assertTrue(wav_path.exists())
            self.assertFalse((meeting_dir / "meeting_audio.mp3").exists())

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_success_converts_and_removes_all_wav_files(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            meeting_dir = Path(temporary_dir)
            wav_paths = (meeting_dir / "meeting_audio.wav", meeting_dir / "microphone_raw.wav")
            for wav_path in wav_paths:
                self._write_wav(wav_path)

            report = optimize_wav_files(meeting_dir)

            self.assertEqual(report.warnings, ())
            self.assertEqual({path.name for path in report.converted}, {"meeting_audio.mp3", "microphone_raw.mp3"})
            self.assertTrue(all(not wav_path.exists() for wav_path in wav_paths))
            self.assertTrue(all(path.exists() for path in report.converted))
            self.assertAlmostEqual(read_mp3_duration(meeting_dir / "meeting_audio.mp3"), 1.0, delta=0.5)

    @staticmethod
    def _write_wav(path: Path) -> None:
        sample_rate = 16_000
        with wave.open(str(path), "wb") as audio_file:
            audio_file.setparams((1, 2, sample_rate, sample_rate, "NONE", "not compressed"))
            audio_file.writeframes(
                b"".join(
                    struct.pack("<h", int(4000 * math.sin(2 * math.pi * 440 * index / sample_rate)))
                    for index in range(sample_rate)
                )
            )


if __name__ == "__main__":
    unittest.main()
