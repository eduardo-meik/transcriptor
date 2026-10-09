from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import main
from core.capture_quality import capture_warnings
from meeting_gui import (
    MeetingControlApp,
    derive_meeting_title,
    format_elapsed,
    format_meeting_duration,
    list_meetings,
)


class MeetingGuiTests(unittest.TestCase):
    def test_formats_elapsed_time_with_hours(self):
        self.assertEqual(format_elapsed(0), "00:00:00")
        self.assertEqual(format_elapsed(65), "00:01:05")
        self.assertEqual(format_elapsed(3661), "01:01:01")

    def test_gui_flag_is_available_without_changing_cli_default(self):
        with mock.patch("sys.argv", ["main.py"]):
            self.assertFalse(main.parse_args().gui)
        with mock.patch("sys.argv", ["main.py", "--gui"]):
            self.assertTrue(main.parse_args().gui)

    def test_gui_styles_use_a_light_palette_and_segoe_ui(self):
        app = MeetingControlApp.__new__(MeetingControlApp)
        app.root = mock.Mock()
        style = mock.Mock()

        with mock.patch("meeting_gui.ttk.Style", return_value=style):
            app._configure_styles()

        style.theme_use.assert_called_once_with("clam")
        style.configure.assert_any_call("App.TFrame", background="#f4f6f8")
        style.configure.assert_any_call("Card.TFrame", background="#ffffff")
        style.configure.assert_any_call(
            "Treeview",
            background="#ffffff",
            fieldbackground="#ffffff",
            foreground="#344054",
            rowheight=28,
            borderwidth=0,
            font=("Segoe UI", 10),
        )
        style.configure.assert_any_call(
            "TNotebook.Tab",
            background="#e4e7ec",
            foreground="#344054",
            padding=(14, 8),
            font=("Segoe UI", 10, "bold"),
        )

    def test_lists_completed_meetings_by_date_with_wav_duration(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_root = Path(temporary_dir)
            old_meeting = output_root / "20260101_120000"
            new_meeting = output_root / "20260203_140000"
            incomplete_meeting = output_root / "20260304_150000"
            for meeting_dir in (old_meeting, new_meeting, incomplete_meeting):
                meeting_dir.mkdir()
                (meeting_dir / "transcript_raw.md").write_text("Transcripción", encoding="utf-8")

            self._write_wav(old_meeting / "meeting_audio.wav", seconds=65)
            self._write_wav(new_meeting / "meeting_audio.wav", seconds=3661)

            meetings = list_meetings(output_root)

        self.assertEqual([entry.meeting_date.strftime("%Y%m%d") for entry in meetings], [
            "20260304",
            "20260203",
            "20260101",
        ])
        self.assertEqual([format_meeting_duration(entry.duration_seconds) for entry in meetings], [
            "—",
            "01:01:01",
            "00:01:05",
        ])
        self.assertEqual(meetings[1].transcript_path.name, "transcript_raw.md")

    def test_ignores_output_folders_without_a_transcript(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            meeting_dir = Path(temporary_dir) / "20260101_120000"
            meeting_dir.mkdir()
            (meeting_dir / "meeting_audio.wav").touch()

            self.assertEqual(list_meetings(Path(temporary_dir)), [])

    def test_derives_a_short_title_locally_from_transcript_markdown(self):
        transcript = (
            "# Raw Meeting Transcript\n\n"
            "**Audio:** `meeting_audio.wav`\n"
            "**Detected language:** `es`\n\n"
            "---\n\n"
            "Ya. ¿Me escuchas?\n"
            "Sí.\n"
            "Todo excepto la Cato.\n"
        )

        self.assertEqual(derive_meeting_title(transcript), "Todo excepto la Cato")

    def test_meeting_entry_includes_summary_only_when_report_exists(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            meeting_dir = Path(temporary_dir) / "20260101_120000"
            meeting_dir.mkdir()
            (meeting_dir / "transcript_raw.md").write_text(
                "# Raw Meeting Transcript\n\n---\n\nSe revisaron las ventas del trimestre.",
                encoding="utf-8",
            )
            (meeting_dir / "meeting_report.md").write_text("# Resumen", encoding="utf-8")

            meetings = list_meetings(Path(temporary_dir))

        self.assertEqual(meetings[0].title, "Se revisaron las ventas del trimestre")
        self.assertEqual(meetings[0].summary_path.name, "meeting_report.md")

    @staticmethod
    def _write_wav(path: Path, seconds: int) -> None:
        with wave.open(str(path), "wb") as audio_file:
            audio_file.setparams((1, 2, 16_000, seconds * 16_000, "NONE", "not compressed"))
            audio_file.writeframes(b"\x00\x00" * seconds * 16_000)


class CaptureWarningTests(unittest.TestCase):
    def test_warns_when_both_channels_are_silent(self):
        warnings = capture_warnings({"system": 0.0, "microphone": 0.0})
        self.assertEqual(len(warnings), 2)
        self.assertIn("SISTEMA no capturo audio", warnings[0])
        self.assertIn("MICROFONO no capturo audio", warnings[1])

    def test_system_only_mode_does_not_warn_about_microphone(self):
        warnings = capture_warnings({"system": 0.005, "microphone": 0.0}, system_only=True)
        self.assertEqual(warnings, ["El canal de SISTEMA capturo muy bajo. Revisa el volumen de salida."])

    def test_does_not_warn_when_both_channels_have_healthy_levels(self):
        self.assertEqual(capture_warnings({"system": 0.3, "microphone": 0.25}), [])


if __name__ == "__main__":
    unittest.main()
