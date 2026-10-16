from __future__ import annotations

import queue
import tempfile
import unittest
import wave
from datetime import datetime
from pathlib import Path
from unittest import mock

import main
import meeting_gui
from core.capture_quality import capture_warnings
from meeting_gui import (
    MeetingControlApp,
    MeetingEntry,
    OperationFailed,
    derive_meeting_title,
    format_elapsed,
    format_meeting_duration,
    list_meetings,
)


class MeetingGuiTests(unittest.TestCase):
    def test_windows_identity_is_set_before_creating_tk(self):
        calls = mock.Mock()
        config = mock.Mock()
        with (
            mock.patch("meeting_gui._configure_windows_app_identity", calls.identity),
            mock.patch("meeting_gui.tk.Tk", calls.create_root),
            mock.patch("meeting_gui.MeetingControlApp", calls.create_app),
        ):
            self.assertEqual(meeting_gui.run_gui(config), 0)

        self.assertEqual(calls.mock_calls, [
            mock.call.identity(),
            mock.call.create_root(),
            mock.call.create_app(calls.create_root.return_value, config),
            mock.call.create_root().mainloop(),
        ])

    def test_windows_identity_uses_explicit_app_id(self):
        with (
            mock.patch("meeting_gui.os.name", "nt"),
            mock.patch("meeting_gui.ctypes.WinDLL", create=True) as load_library,
        ):
            set_app_id = load_library.return_value.SetCurrentProcessExplicitAppUserModelID
            set_app_id.return_value = 0
            meeting_gui._configure_windows_app_identity()

        load_library.assert_called_once_with("shell32")
        set_app_id.assert_called_once_with("MeikLabs.Transcriptor")

    def test_windows_identity_failure_is_explicit(self):
        with (
            mock.patch("meeting_gui.os.name", "nt"),
            mock.patch("meeting_gui.ctypes.WinDLL", create=True) as load_library,
        ):
            load_library.return_value.SetCurrentProcessExplicitAppUserModelID.return_value = -1
            with self.assertRaisesRegex(OSError, "Windows application identity"):
                meeting_gui._configure_windows_app_identity()

    def test_non_windows_startup_skips_windows_identity(self):
        with (
            mock.patch("meeting_gui.os.name", "posix"),
            mock.patch("meeting_gui.ctypes.WinDLL", create=True) as load_library,
        ):
            meeting_gui._configure_windows_app_identity()

        load_library.assert_not_called()

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

    def test_processing_error_is_shown_in_a_dialog(self):
        app = MeetingControlApp.__new__(MeetingControlApp)
        app.root = mock.Mock()
        app._messages = queue.Queue()
        app._messages.put(OperationFailed("OpenRouter request failed: 401 - Unauthorized"))
        app.message_label = mock.Mock()
        app.start_button = mock.Mock()
        app.stop_button = mock.Mock()
        app._loopback_devices = {}
        app._microphone_devices = {}
        app._set_status = mock.Mock()
        app._recorder = mock.Mock()

        with mock.patch("meeting_gui.messagebox.showerror") as show_error:
            app._poll_messages()

        show_error.assert_called_once_with(
            "Error de procesamiento",
            "Error: OpenRouter request failed: 401 - Unauthorized",
            parent=app.root,
        )
        app.message_label.configure.assert_called_once_with(
            text="Error: OpenRouter request failed: 401 - Unauthorized"
        )
        self.assertIsNone(app._recorder)

    def test_recording_popup_is_shown_only_while_recording_and_minimized(self):
        app = MeetingControlApp.__new__(MeetingControlApp)
        app.root = mock.Mock()
        app.root.state.return_value = "iconic"
        app._state = "recording"
        app._recording_popup = None

        with (
            mock.patch.object(app, "_show_recording_popup") as show_popup,
            mock.patch.object(app, "_close_recording_popup") as close_popup,
        ):
            app._sync_recording_popup()

        show_popup.assert_called_once_with()
        close_popup.assert_not_called()

    def test_recording_popup_is_closed_after_recording_or_restore(self):
        app = MeetingControlApp.__new__(MeetingControlApp)
        app.root = mock.Mock()
        app.root.state.return_value = "iconic"
        app._state = "transcribing"
        app._recording_popup = None

        with mock.patch.object(app, "_close_recording_popup") as close_popup:
            app._sync_recording_popup()

        close_popup.assert_called_once_with()

        app._close_recording_popup = mock.Mock()
        app._restore_main_window()
        app._close_recording_popup.assert_called_once_with()
        app.root.deiconify.assert_called_once_with()
        app.root.lift.assert_called_once_with()
        app.root.focus_force.assert_called_once_with()

    def test_root_minimize_event_schedules_popup_sync(self):
        app = MeetingControlApp.__new__(MeetingControlApp)
        app.root = mock.Mock()
        event = mock.Mock()
        event.widget = app.root

        app._on_root_unmapped(event)

        app.root.after_idle.assert_called_once_with(app._sync_recording_popup)

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

    def test_reads_duration_from_mp3_when_wav_is_not_present(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            meeting_dir = Path(temporary_dir) / "20260101_120000"
            meeting_dir.mkdir()
            (meeting_dir / "transcript_raw.md").write_text(
                "# Raw Meeting Transcript\n\n---\n\nSe revisaron las ventas del trimestre.",
                encoding="utf-8",
            )
            (meeting_dir / "meeting_audio.mp3").touch()

            with mock.patch("meeting_gui.read_mp3_duration", return_value=125.0) as read_duration:
                meetings = list_meetings(Path(temporary_dir))

        read_duration.assert_called_once_with(meeting_dir / "meeting_audio.mp3")
        self.assertEqual(meetings[0].duration_seconds, 125.0)

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

    def test_manual_meeting_title_overrides_transcript_derived_title(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            meeting_dir = Path(temporary_dir) / "20260101_120000"
            meeting_dir.mkdir()
            (meeting_dir / "transcript_raw.md").write_text(
                "# Raw Meeting Transcript\n\n---\n\nSe revisaron las ventas del trimestre.",
                encoding="utf-8",
            )
            (meeting_dir / "meeting_title.txt").write_text("Planificación Q4\n", encoding="utf-8")

            meetings = list_meetings(Path(temporary_dir))

        self.assertEqual(meetings[0].title, "Planificación Q4")

    def test_rename_selected_meeting_saves_override_and_updates_library(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            meeting_dir = Path(temporary_dir) / "20260101_120000"
            meeting_dir.mkdir()
            transcript_path = meeting_dir / "transcript_raw.md"
            transcript_path.write_text(
                "# Raw Meeting Transcript\n\n---\n\nSe revisaron las ventas del trimestre.",
                encoding="utf-8",
            )
            entry = MeetingEntry(
                title="Se revisaron las ventas del trimestre",
                meeting_date=datetime(2026, 1, 1, 12, 0),
                duration_seconds=60,
                transcript_path=transcript_path,
                summary_path=None,
            )
            app = MeetingControlApp.__new__(MeetingControlApp)
            app.root = mock.Mock()
            app.meetings_table = mock.Mock()
            app.meetings_table.selection.return_value = ("0",)
            app.meetings_table.item.return_value = (
                entry.title,
                "01/01/2026 12:00",
                "00:01:00",
                "Ver",
                "—",
            )
            app._meeting_entries = {"0": entry}
            app.preview_title_label = mock.Mock()

            with mock.patch("meeting_gui.simpledialog.askstring", return_value="  Planificación Q4  "):
                app._rename_selected_meeting()

            self.assertEqual(
                (meeting_dir / "meeting_title.txt").read_text(encoding="utf-8").strip(),
                "Planificación Q4",
            )
            self.assertEqual(app._meeting_entries["0"].title, "Planificación Q4")
            app.meetings_table.item.assert_called_with(
                "0",
                values=("Planificación Q4", "01/01/2026 12:00", "00:01:00", "Ver", "—"),
            )
            app.preview_title_label.configure.assert_called_once_with(text="Planificación Q4")
            self.assertEqual(list_meetings(Path(temporary_dir))[0].title, "Planificación Q4")

    def test_rename_selected_meeting_rejects_empty_title(self):
        app = MeetingControlApp.__new__(MeetingControlApp)
        app.root = mock.Mock()
        app.meetings_table = mock.Mock()
        app.meetings_table.selection.return_value = ("0",)
        entry = MeetingEntry(
            title="Nombre original",
            meeting_date=datetime(2026, 1, 1, 12, 0),
            duration_seconds=None,
            transcript_path=Path("meeting/transcript_raw.md"),
            summary_path=None,
        )
        app._meeting_entries = {"0": entry}

        with (
            mock.patch("meeting_gui.simpledialog.askstring", return_value="   "),
            mock.patch("meeting_gui.messagebox.showwarning") as show_warning,
        ):
            app._rename_selected_meeting()

        show_warning.assert_called_once()
        self.assertEqual(app._meeting_entries["0"].title, "Nombre original")
        app.meetings_table.item.assert_not_called()

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
