import os
import unittest
from unittest import mock

from core.config import DEFAULT_OPENROUTER_MODEL, AppConfig
from core.summarizer import MeetingSummarizer, SummaryConfig, SummaryError


class MeetingAssistantTests(unittest.TestCase):
    def test_lmstudio_summary_config_is_supported(self):
        config = SummaryConfig(
            provider="lmstudio",
            lmstudio_base_url="http://localhost:1234/v1",
            lmstudio_model="local-model",
        )
        self.assertEqual(config.provider, "lmstudio")
        self.assertEqual(config.lmstudio_base_url, "http://localhost:1234/v1")
        self.assertEqual(config.lmstudio_model, "local-model")

    def test_openrouter_is_the_default_summary_provider_and_model(self):
        self.assertEqual(AppConfig().summary_provider, "openrouter")
        self.assertEqual(AppConfig().openrouter_model, "nvidia/nemotron-3-ultra-550b-a55b")
        self.assertEqual(SummaryConfig().provider, "openrouter")
        self.assertEqual(SummaryConfig().openrouter_model, DEFAULT_OPENROUTER_MODEL)

    def test_openrouter_uses_configured_model_and_bearer_key(self):
        response = mock.Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": "Resumen generado"}}]}
        with (
            mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-openrouter-key"}, clear=True),
            mock.patch("core.summarizer.requests.post", return_value=response) as post,
        ):
            summary = MeetingSummarizer(
                SummaryConfig(provider="openrouter", openrouter_model="google/gemma-4-26b-a4b-it:free")
            ).build_meeting_analysis("Se acordó revisar el plan.")

        self.assertEqual(summary, "Resumen generado")
        post.assert_called_once()
        self.assertEqual(post.call_args.args[0], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer test-openrouter-key")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "google/gemma-4-26b-a4b-it:free")
        self.assertIn("Decisiones clave", post.call_args.kwargs["json"]["messages"][1]["content"])

    def test_openrouter_uses_nemotron_default_model_in_request(self):
        response = mock.Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": "Resumen generado"}}]}
        with (
            mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-openrouter-key"}, clear=True),
            mock.patch("core.summarizer.requests.post", return_value=response) as post,
        ):
            MeetingSummarizer(SummaryConfig()).build_meeting_analysis("Se acordó revisar el plan.")

        self.assertEqual(post.call_args.kwargs["json"]["model"], "nvidia/nemotron-3-ultra-550b-a55b")

    def test_openrouter_reports_missing_api_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(SummaryError, "OPENROUTER_API_KEY is not configured"):
                MeetingSummarizer(SummaryConfig(provider="openrouter")).build_meeting_analysis("Transcript")

    def test_openrouter_reports_api_errors(self):
        response = mock.Mock(status_code=401, text="Unauthorized")
        with (
            mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-openrouter-key"}, clear=True),
            mock.patch("core.summarizer.requests.post", return_value=response),
        ):
            with self.assertRaisesRegex(SummaryError, "OpenRouter request failed: 401"):
                MeetingSummarizer(SummaryConfig(provider="openrouter")).build_meeting_analysis("Transcript")

    def test_meeting_tools_exist(self):
        summarizer = MeetingSummarizer(SummaryConfig(provider="lmstudio"))
        self.assertTrue(hasattr(summarizer, "cleanup_transcript"))
        self.assertTrue(hasattr(summarizer, "build_meeting_analysis"))
        self.assertTrue(hasattr(summarizer, "compare_meetings"))


if __name__ == "__main__":
    unittest.main()
