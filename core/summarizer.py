"""Optional meeting summarization through external LLM APIs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class SummaryError(RuntimeError):
    """Raised when a meeting summary cannot be generated."""


class _Provider(Protocol):
    def summarize(self, transcript: str) -> str:
        """Return a Markdown meeting report."""


@dataclass(frozen=True)
class SummaryConfig:
    provider: str = "anthropic"
    anthropic_model: str = "claude-3-5-sonnet-latest"
    google_model: str = "gemini-1.5-pro"


class MeetingSummarizer:
    """Generate a professional Markdown report from a raw transcript."""

    def __init__(self, config: SummaryConfig) -> None:
        self.config = config

    def summarize_to_file(self, transcript: str, output_dir: Path) -> Path:
        """Summarize transcript and save `meeting_report.md`."""

        if not transcript.strip():
            raise SummaryError("Transcript is empty; skipping summary generation.")

        output_dir.mkdir(parents=True, exist_ok=True)
        provider = self._create_provider()
        report = provider.summarize(transcript)
        report_path = output_dir / "meeting_report.md"
        report_path.write_text(report.strip() + "\n", encoding="utf-8")
        return report_path

    def _create_provider(self) -> _Provider:
        provider = self.config.provider.lower().strip()
        if provider == "anthropic":
            return _AnthropicProvider(self.config.anthropic_model)
        if provider == "google":
            return _GoogleProvider(self.config.google_model)
        raise SummaryError(f"Unsupported summary provider: {self.config.provider}")


class _AnthropicProvider:
    def __init__(self, model: str) -> None:
        self.model = model

    def summarize(self, transcript: str) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise SummaryError("ANTHROPIC_API_KEY is not configured.")

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise SummaryError("Install the `anthropic` package to use Anthropic summaries.") from exc

        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=3000,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(transcript)}],
        )
        return "\n".join(block.text for block in response.content if getattr(block, "type", "") == "text")


class _GoogleProvider:
    def __init__(self, model: str) -> None:
        self.model = model

    def summarize(self, transcript: str) -> str:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SummaryError("GOOGLE_API_KEY or GEMINI_API_KEY is not configured.")

        try:
            from google import genai
        except ImportError as exc:
            raise SummaryError("Install the `google-genai` package to use Google summaries.") from exc

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=f"{_SYSTEM_PROMPT}\n\n{_build_user_prompt(transcript)}",
        )
        if not response.text:
            raise SummaryError("Google GenAI returned an empty summary.")
        return response.text


_SYSTEM_PROMPT = """You are a senior technical meeting analyst for Meik Labs. Write concise, professional Markdown in Spanish unless the transcript clearly uses another language. Do not invent decisions, owners, dates, or commitments that are not supported by the transcript."""


def _build_user_prompt(transcript: str) -> str:
    return f"""Create a meeting report with exactly these sections:

# Executive Summary
# Key Decisions
# Action Items / To-Do
# Technical Risks / Critical Points
# Open Questions
# Follow-up Recommendations

Transcript:

{transcript}
"""
