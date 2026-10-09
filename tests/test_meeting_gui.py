from __future__ import annotations

import unittest
from unittest import mock

import main
from core.capture_quality import capture_warnings
from meeting_gui import format_elapsed


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
