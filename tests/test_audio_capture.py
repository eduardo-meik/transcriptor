"""Tests for the audio capture helpers.

These cover the behaviours that were expensive to get right on real hardware:
the WASAPI ``auto_convert`` flag (without it a Realtek microphone-array endpoint
records near-silence), the device-resolution priority, and the mixing maths.
Nothing here touches a real audio device.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import sounddevice as sd
import soundfile as sf

import main
from core.audio_recorder import (
    AudioDeviceSelection,
    MeetingAudioRecorder,
    endpoint_match_key,
    find_input_device_by_name,
    load_resampled_mono,
    looks_like_system_capture_device,
    pad_to_length,
)

HOST_APIS = [
    {"name": "MME", "default_input_device": 1, "default_output_device": 4},
    {"name": "Windows WASAPI", "default_input_device": 15, "default_output_device": 13},
]


def make_device(name: str, in_channels: int = 1, out_channels: int = 0, hostapi: int = 1, samplerate: int = 48000):
    """Build the subset of a sounddevice device description that the code reads."""

    return {
        "name": name,
        "max_input_channels": in_channels,
        "max_output_channels": out_channels,
        "hostapi": hostapi,
        "default_samplerate": samplerate,
    }


def query_hostapis(*args):
    """Mirror sounddevice: no argument returns the list, an index returns one dict."""

    if args:
        return HOST_APIS[args[0]]
    return HOST_APIS


class WasapiSettingsTests(unittest.TestCase):
    """Guard auto_convert: losing it silently turns the microphone into silence."""

    def test_wasapi_device_requests_auto_convert(self):
        device = make_device("Varios microfonos (Realtek)", hostapi=1)
        with (
            mock.patch("core.audio_recorder.sd.query_devices", return_value=device),
            mock.patch("core.audio_recorder.sd.query_hostapis", side_effect=query_hostapis),
            mock.patch("core.audio_recorder.sd.WasapiSettings") as settings,
        ):
            MeetingAudioRecorder.wasapi_settings(15)

        settings.assert_called_once_with(auto_convert=True)

    def test_non_wasapi_device_gets_no_extra_settings(self):
        device = make_device("Varios microfonos (Realtek)", hostapi=0)
        with (
            mock.patch("core.audio_recorder.sd.query_devices", return_value=device),
            mock.patch("core.audio_recorder.sd.query_hostapis", side_effect=query_hostapis),
            mock.patch("core.audio_recorder.sd.WasapiSettings") as settings,
        ):
            extra = MeetingAudioRecorder.wasapi_settings(1)

        self.assertIsNone(extra)
        settings.assert_not_called()

    def test_unknown_device_index_is_tolerated(self):
        with mock.patch(
            "core.audio_recorder.sd.query_devices",
            side_effect=sd.PortAudioError("Invalid device"),
        ):
            self.assertIsNone(MeetingAudioRecorder.wasapi_settings(99))


class MicrophoneResolutionTests(unittest.TestCase):
    """The priority order matters: sd.default.device gives the MME endpoint, not WASAPI."""

    def build(self, device_index: int | None = None, name: str | None = None) -> MeetingAudioRecorder:
        return MeetingAudioRecorder(
            output_dir=Path("."),
            devices=AudioDeviceSelection(microphone_device=device_index),
            microphone_device_name=name,
        )

    def test_explicit_device_wins(self):
        with mock.patch("core.audio_recorder.sd.query_hostapis", return_value=HOST_APIS):
            self.assertEqual(self.build(device_index=7).resolve_microphone_device(), 7)

    def test_name_match_wins_over_defaults(self):
        recorder = self.build(name="Auriculares (AirPods Pro)")
        with (
            mock.patch("core.audio_recorder.sd.query_hostapis", return_value=HOST_APIS),
            mock.patch("core.audio_recorder.find_input_device_by_name", return_value=14) as finder,
        ):
            self.assertEqual(recorder.resolve_microphone_device(), 14)

        finder.assert_called_once_with("(AirPods Pro)")

    def test_wasapi_default_wins_over_mme_default(self):
        with (
            mock.patch("core.audio_recorder.sd.query_hostapis", return_value=HOST_APIS),
            mock.patch("core.audio_recorder.sd.default.device", [1, 4]),
        ):
            self.assertEqual(self.build().resolve_microphone_device(), 15)

    def test_falls_back_to_system_default_without_wasapi(self):
        host_apis = [{"name": "MME", "default_input_device": 1, "default_output_device": 4}]
        with (
            mock.patch("core.audio_recorder.sd.query_hostapis", return_value=host_apis),
            mock.patch("core.audio_recorder.sd.default.device", [3, 4]),
        ):
            self.assertEqual(self.build().resolve_microphone_device(), 3)

    def test_raises_when_nothing_is_configured(self):
        host_apis = [{"name": "MME", "default_input_device": 1, "default_output_device": 4}]
        with (
            mock.patch("core.audio_recorder.sd.query_hostapis", return_value=host_apis),
            mock.patch("core.audio_recorder.sd.default.device", [-1, -1]),
        ):
            with self.assertRaises(Exception):
                self.build().resolve_microphone_device()


class EndpointMatchKeyTests(unittest.TestCase):
    """A headset exposes differently named playback and capture endpoints."""

    def test_extracts_the_parenthesised_part(self):
        self.assertEqual(
            endpoint_match_key("Auriculares (AirPods Pro - Find My)"),
            "(AirPods Pro - Find My)",
        )

    def test_pairs_playback_and_capture_names(self):
        playback = endpoint_match_key("Auriculares (AirPods Pro - Find My)")
        capture = "Auriculares con microfono (AirPods Pro - Find My)"
        self.assertIn(playback, capture)

    def test_without_parentheses_returns_the_name(self):
        self.assertEqual(endpoint_match_key("Varios microfonos"), "Varios microfonos")

    def test_collapses_embedded_newlines(self):
        self.assertEqual(endpoint_match_key("Auriculares\n(AirPods)"), "(AirPods)")


class FindInputDeviceTests(unittest.TestCase):
    def test_prefers_wasapi_and_ignores_output_only_endpoints(self):
        devices = [
            make_device("Auriculares (AirPods)", hostapi=0),
            make_device("Altavoces (AirPods)", in_channels=0, out_channels=2, hostapi=1),
            make_device("Auriculares (AirPods)", hostapi=1),
        ]
        with (
            mock.patch("core.audio_recorder.sd.query_devices", return_value=devices),
            mock.patch("core.audio_recorder.sd.query_hostapis", return_value=HOST_APIS),
        ):
            self.assertEqual(find_input_device_by_name("AirPods"), 2)

    def test_returns_fallback_when_only_non_wasapi_matches(self):
        devices = [make_device("Auriculares (AirPods)", hostapi=0)]
        with (
            mock.patch("core.audio_recorder.sd.query_devices", return_value=devices),
            mock.patch("core.audio_recorder.sd.query_hostapis", return_value=HOST_APIS),
        ):
            self.assertEqual(find_input_device_by_name("AirPods"), 0)

    def test_returns_none_without_a_match(self):
        devices = [make_device("Varios microfonos (Realtek)")]
        with (
            mock.patch("core.audio_recorder.sd.query_devices", return_value=devices),
            mock.patch("core.audio_recorder.sd.query_hostapis", return_value=HOST_APIS),
        ):
            self.assertIsNone(find_input_device_by_name("AirPods"))

    def test_empty_hint_returns_none(self):
        self.assertIsNone(find_input_device_by_name("   "))


class SystemCaptureDetectionTests(unittest.TestCase):
    def test_detects_spanish_stereo_mix(self):
        self.assertTrue(looks_like_system_capture_device("Mezcla estereo (Realtek HD Audio Stereo input)"))

    def test_detects_english_stereo_mix(self):
        self.assertTrue(looks_like_system_capture_device("Stereo Mix (Realtek(R) Audio)"))

    def test_detects_loopback_monitor(self):
        self.assertTrue(looks_like_system_capture_device("Monitor of Speakers"))

    def test_rejects_a_plain_microphone(self):
        self.assertFalse(looks_like_system_capture_device("Varios microfonos (Realtek(R) Audio)"))

    def test_rejects_speakers(self):
        self.assertFalse(looks_like_system_capture_device("Altavoces (Realtek(R) Audio)"))


class DeviceListingTests(unittest.TestCase):
    def test_keeps_one_line_per_device_for_names_with_newlines(self):
        devices = [make_device("Auriculares con microfono\n(Hands-Free)", hostapi=1)]
        with (
            mock.patch("core.audio_recorder.sd.query_devices", return_value=devices),
            mock.patch("core.audio_recorder.sd.query_hostapis", return_value=HOST_APIS),
        ):
            listing = MeetingAudioRecorder.list_devices()

        device_lines = [line for line in listing.splitlines() if line.startswith("[")]
        self.assertEqual(len(device_lines), 1)

    def test_marks_system_capture_candidates(self):
        devices = [
            make_device("Mezcla estereo (Realtek)", in_channels=2, hostapi=1),
            make_device("Varios microfonos (Realtek)", in_channels=2, hostapi=1),
        ]
        with (
            mock.patch("core.audio_recorder.sd.query_devices", return_value=devices),
            mock.patch("core.audio_recorder.sd.query_hostapis", return_value=HOST_APIS),
        ):
            listing = MeetingAudioRecorder.list_devices()

        candidates = [line for line in listing.splitlines() if "system-capture-candidate" in line]
        self.assertEqual(len(candidates), 1)


class AudioMathTests(unittest.TestCase):
    def test_missing_file_loads_as_empty(self):
        self.assertEqual(load_resampled_mono(Path("does-not-exist.wav"), 16000).size, 0)

    def test_none_loads_as_empty(self):
        self.assertEqual(load_resampled_mono(None, 16000).size, 0)

    def test_resamples_to_the_target_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            timeline = np.arange(48000, dtype=np.float32) / 48000
            sf.write(path, np.sin(2 * np.pi * 440 * timeline), 48000, subtype="PCM_16")

            mono = load_resampled_mono(path, 16000)

        self.assertAlmostEqual(len(mono), 16000, delta=50)

    def test_downsamples_stereo_to_mono(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stereo.wav"
            left = np.full(16000, 0.5, dtype=np.float32)
            right = np.full(16000, -0.5, dtype=np.float32)
            sf.write(path, np.stack([left, right], axis=1), 16000, subtype="PCM_16")

            mono = load_resampled_mono(path, 16000)

        self.assertLess(float(np.abs(mono).max()), 0.05)

    def test_pad_to_length_trims(self):
        audio = np.arange(10, dtype=np.float32)
        self.assertEqual(len(pad_to_length(audio, 4)), 4)

    def test_pad_to_length_extends_with_zeros(self):
        padded = pad_to_length(np.ones(3, dtype=np.float32), 5)
        self.assertEqual(len(padded), 5)
        self.assertEqual(float(padded[-1]), 0.0)


class MixToTargetWavTests(unittest.TestCase):
    def test_mix_resamples_and_pads_the_shorter_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            sf.write(output_dir / "system_raw.wav", np.full(48000, 0.3, dtype=np.float32), 48000, subtype="PCM_16")
            sf.write(output_dir / "microphone_raw.wav", np.full(16000, 0.2, dtype=np.float32), 16000, subtype="PCM_16")

            recorder = MeetingAudioRecorder(output_dir=output_dir, target_sample_rate=16000)
            mixed_path = output_dir / "meeting_audio.wav"
            recorder.mix_to_target_wav(mixed_path)

            audio, rate = sf.read(mixed_path)

        self.assertEqual(rate, 16000)
        self.assertAlmostEqual(len(audio), 16000, delta=50)
        self.assertAlmostEqual(float(np.abs(audio).max()), 0.5, delta=0.05)

    def test_mix_raises_when_both_sources_are_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            recorder = MeetingAudioRecorder(output_dir=output_dir, target_sample_rate=16000)

            with self.assertRaises(Exception):
                recorder.mix_to_target_wav(output_dir / "meeting_audio.wav")


class SessionPeakTests(unittest.TestCase):
    """Session peaks back the post-recording silent-channel warning."""

    def test_records_the_highest_peak_per_source(self):
        recorder = MeetingAudioRecorder(output_dir=Path("."))
        recorder.record_level("system", np.array([[0.2], [0.1]], dtype=np.float32))
        recorder.record_level("system", np.array([[0.05]], dtype=np.float32))
        recorder.record_level("microphone", np.array([[0.4]], dtype=np.float32))

        peaks = recorder.session_peaks()

        self.assertAlmostEqual(peaks["system"], 0.2)
        self.assertAlmostEqual(peaks["microphone"], 0.4)

    def test_handles_mono_and_multichannel_input(self):
        recorder = MeetingAudioRecorder(output_dir=Path("."))
        recorder.record_level("system", np.array([0.3, -0.6], dtype=np.float32))
        self.assertAlmostEqual(recorder.session_peaks()["system"], 0.6)

    def test_ignores_empty_buffers(self):
        recorder = MeetingAudioRecorder(output_dir=Path("."))
        recorder.record_level("system", np.array([], dtype=np.float32))
        self.assertEqual(recorder.session_peaks()["system"], 0.0)


class _StubRecorder:
    """Minimal duck type for the capture-quality reporter."""

    def __init__(self, peaks: dict[str, float]) -> None:
        self._peaks = peaks

    def session_peaks(self) -> dict[str, float]:
        return self._peaks


class CaptureQualityWarningTests(unittest.TestCase):
    def report(self, peaks: dict[str, float], system_only: bool = False) -> str:
        stream = io.StringIO()
        with mock.patch("sys.stdout", stream):
            main.report_capture_quality(_StubRecorder(peaks), system_only)  # type: ignore[arg-type]
        return stream.getvalue()

    def test_warns_when_both_channels_are_silent(self):
        text = self.report({"system": 0.0, "microphone": 0.0})
        self.assertIn("AVISO", text)
        self.assertIn("SISTEMA", text)
        self.assertIn("MICROFONO", text)

    def test_warns_when_a_channel_is_very_low(self):
        text = self.report({"system": 0.2, "microphone": 0.001})
        self.assertIn("muy bajo", text)

    def test_is_quiet_when_both_channels_have_signal(self):
        text = self.report({"system": 0.2, "microphone": 0.1})
        self.assertNotIn("AVISO", text)
        self.assertIn("Niveles grabados", text)

    def test_system_only_does_not_warn_about_the_microphone(self):
        text = self.report({"system": 0.2, "microphone": 0.0}, system_only=True)
        self.assertNotIn("AVISO", text)
        self.assertNotIn("MICROFONO", text)


if __name__ == "__main__":
    unittest.main()
