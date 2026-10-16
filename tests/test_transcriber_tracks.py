import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core import transcriber as tr


def _seg(start, end, text):
    return SimpleNamespace(start=start, end=end, text=text)


class FakeModel:
    def __init__(self, by_name):
        self.by_name = by_name

    def transcribe(self, path, **kwargs):
        return iter(self.by_name[Path(path).name]), SimpleNamespace(language="es")


class TranscribeTracksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.mic = self.dir / "mic.wav"
        self.sys = self.dir / "sys.wav"
        self.mixed = self.dir / "mixed.wav"
        for path in (self.mic, self.sys, self.mixed):
            path.write_bytes(b"\0" * 100)

    def tearDown(self):
        self.tmp.cleanup()

    def _transcriber(self):
        return tr.FasterWhisperTranscriber(tr.TranscriptionConfig(device="cpu"))

    def test_labels_speakers_drops_echo_and_keeps_accents(self):
        model = FakeModel(
            {
                "mic.wav": [_seg(0.0, 2.0, "Hola, ¿cómo están?"), _seg(5.0, 7.0, "La reunión empieza mañana.")],
                "sys.wav": [_seg(3.0, 4.5, "Muy bien, gracias."), _seg(5.1, 7.1, "La reunión empieza mañana")],
            }
        )
        with mock.patch.object(tr, "WhisperModel", return_value=model):
            result = self._transcriber().transcribe_tracks(
                [("Yo", self.mic), ("Participantes", self.sys)], self.dir / "out", fallback_audio=self.mixed
            )

        self.assertEqual(
            result.text.splitlines(),
            [
                "[Yo] Hola, ¿cómo están?",
                "[Participantes] Muy bien, gracias.",
                "[Participantes] La reunión empieza mañana",
            ],
        )
        raw = result.text_path.read_bytes()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        self.assertIn("¿cómo están?", raw.decode("utf-8-sig"))
        self.assertIn("**[Yo]**", result.segment_path.read_text(encoding="utf-8-sig"))

    def test_falls_back_to_mixed_when_tracks_empty(self):
        empty = self.dir / "empty.wav"
        empty.write_bytes(b"\0" * 44)
        model = FakeModel({"mixed.wav": [_seg(0.0, 1.0, "Sí, información.")]})
        with mock.patch.object(tr, "WhisperModel", return_value=model):
            result = self._transcriber().transcribe_tracks(
                [("Yo", empty), ("Participantes", None)], self.dir / "out", fallback_audio=self.mixed
            )
        self.assertEqual(result.text, "Sí, información.")

    def test_default_spanish_prompt(self):
        self.assertEqual(self._transcriber()._transcribe_kwargs()["initial_prompt"], tr.DEFAULT_SPANISH_PROMPT)


if __name__ == "__main__":
    unittest.main()
