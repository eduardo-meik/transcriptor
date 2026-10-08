import unittest

from core.summarizer import MeetingSummarizer, SummaryConfig


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

    def test_meeting_tools_exist(self):
        summarizer = MeetingSummarizer(SummaryConfig(provider="lmstudio"))
        self.assertTrue(hasattr(summarizer, "cleanup_transcript"))
        self.assertTrue(hasattr(summarizer, "build_meeting_analysis"))
        self.assertTrue(hasattr(summarizer, "compare_meetings"))


if __name__ == "__main__":
    unittest.main()
