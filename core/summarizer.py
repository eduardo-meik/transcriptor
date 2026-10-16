"""Optional meeting summarization through external LLM APIs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests


class SummaryError(RuntimeError):
    """Raised when a meeting summary cannot be generated."""


class _Provider(Protocol):
    def summarize(self, transcript: str) -> str:
        """Return a Markdown meeting report."""


@dataclass(frozen=True)
class SummaryConfig:
    provider: str = "openrouter"
    anthropic_model: str = "claude-3-5-sonnet-latest"
    google_model: str = "gemini-1.5-pro"
    openrouter_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_model: str = "local-model"
    cleanup_pass: bool = False
    meeting_type: str = "technical"


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
        report_path.write_text(report.strip() + "\n", encoding="utf-8-sig")
        return report_path

    def cleanup_transcript(self, transcript: str) -> str:
        """Optional conservative cleanup pass for raw transcripts."""
        if not transcript.strip():
            raise SummaryError("Transcript is empty; skipping cleanup.")
        if not self.config.cleanup_pass:
            return transcript
        provider = self._create_provider()
        if not hasattr(provider, "cleanup"):
            return transcript
        return provider.cleanup(transcript)

    def build_meeting_analysis(self, transcript: str) -> str:
        """Generate dedicated analysis sections for technical meetings."""
        if not transcript.strip():
            raise SummaryError("Transcript is empty; skipping analysis.")
        provider = self._create_provider()
        return provider.summarize(
            transcript,
            task="analysis",
            meeting_type=self.config.meeting_type,
        )

    def compare_meetings(self, transcripts: list[str], meeting_names: list[str] | None = None) -> str:
        """Compare multiple meeting transcripts and extract patterns and deltas."""
        if not transcripts:
            raise SummaryError("No meeting transcripts were provided for comparison.")
        provider = self._create_provider()
        return provider.summarize(
            "\n\n---MEETING-BREAK---\n\n".join(
                f"## {meeting_names[idx] if meeting_names and idx < len(meeting_names) else f'Meeting {idx + 1}'}\n{transcript}"
                for idx, transcript in enumerate(transcripts)
            ),
            task="compare",
            meeting_type=self.config.meeting_type,
        )

    def _create_provider(self) -> _Provider:
        provider = self.config.provider.lower().strip()
        if provider == "openrouter":
            return _OpenRouterProvider(self.config.openrouter_model)
        if provider == "anthropic":
            return _AnthropicProvider(self.config.anthropic_model)
        if provider == "google":
            return _GoogleProvider(self.config.google_model)
        if provider == "lmstudio":
            return _LMStudioProvider(
                base_url=self.config.lmstudio_base_url,
                model=self.config.lmstudio_model,
            )
        raise SummaryError(f"Unsupported summary provider: {self.config.provider}")


class _OpenRouterProvider:
    def __init__(self, model: str) -> None:
        self.model = model

    def summarize(self, transcript: str, task: str = "summary", meeting_type: str = "technical", **_: Any) -> str:
        chunks = _split_transcript(transcript, max_characters=4_000)
        if len(chunks) > 1:
            transcript = "\n\n".join(
                self._chat(
                    _SYSTEM_PROMPT,
                    "Extrae solo hechos, decisiones, tareas, responsables y riesgos del siguiente fragmento:\n\n" + chunk,
                    temperature=0.1,
                    max_tokens=600,
                )
                for chunk in chunks
            )
        return self._chat(
            _SYSTEM_PROMPT,
            _build_user_prompt(transcript, task=task, meeting_type=meeting_type),
            temperature=0.2,
            max_tokens=1500,
        )

    def cleanup(self, transcript: str) -> str:
        return "\n\n".join(
            self._chat(
                "Corrige solo la transcripción. Mantén el contenido original. Corrige puntuación, "
                "mayúsculas, fillers y términos técnicos. No inventes hechos ni decisiones. "
                "Devuelve solo el texto corregido. La reunión puede tratar de acuicultura, ingeniería "
                "naval o desarrollo de software; usa ese contexto y el glosario técnico para resolver "
                "ambigüedades.",
                chunk,
                temperature=0.1,
                max_tokens=1000,
            )
            for chunk in _split_transcript(transcript)
        )

    def _chat(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise SummaryError("OPENROUTER_API_KEY is not configured.")

        payload: dict[str, str | float | int | list[dict[str, str]]] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=600,
            )
        except requests.RequestException as exc:
            raise SummaryError(f"OpenRouter request failed: {exc}") from exc
        if response.status_code >= 400:
            raise SummaryError(f"OpenRouter request failed: {response.status_code} - {response.text[:400]}")

        try:
            response.encoding = "utf-8"
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise SummaryError("OpenRouter returned an empty summary.")
            return content.strip()
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise SummaryError("OpenRouter returned an unexpected response.") from exc


class _AnthropicProvider:
    def __init__(self, model: str) -> None:
        self.model = model

    def summarize(self, transcript: str, **_: Any) -> str:
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
            messages=[{"role": "user", "content": _build_user_prompt(transcript, task="summary")}],
        )
        return "\n".join(block.text for block in response.content if getattr(block, "type", "") == "text")

    def cleanup(self, transcript: str) -> str:
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
            max_tokens=2000,
            system="Corrige solo la transcripción. Conserva el significado, corrige puntuación y términos técnicos. No inventes información. Devuelve solo el texto limpio.",
            messages=[{"role": "user", "content": transcript}],
        )
        return "\n".join(block.text for block in response.content if getattr(block, "type", "") == "text")


class _GoogleProvider:
    def __init__(self, model: str) -> None:
        self.model = model

    def summarize(self, transcript: str, **_: Any) -> str:
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
            contents=f"{_SYSTEM_PROMPT}\n\n{_build_user_prompt(transcript, task='summary')}",
        )
        if not response.text:
            raise SummaryError("Google GenAI returned an empty summary.")
        return response.text


class _LMStudioProvider:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model or os.getenv("LMSTUDIO_MODEL", "local-model")

    def summarize(self, transcript: str, task: str = "summary", meeting_type: str = "technical", **_: Any) -> str:
        chunks = _split_transcript(transcript, max_characters=4_000)
        if len(chunks) > 1:
            transcript = "\n\n".join(
                self._chat(
                    _SYSTEM_PROMPT,
                    "Extrae solo hechos, decisiones, tareas, responsables y riesgos del siguiente fragmento:\n\n" + chunk,
                    temperature=0.1,
                    max_tokens=600,
                )
                for chunk in chunks
            )
        return self._chat(
            _SYSTEM_PROMPT,
            _build_user_prompt(transcript, task=task, meeting_type=meeting_type),
            temperature=0.2,
            max_tokens=1500,
        )

    def cleanup(self, transcript: str) -> str:
        return "\n\n".join(
            self._chat(
                "Corrige solo la transcripción. Mantén el contenido original. Corrige puntuación, "
                "mayúsculas, fillers y términos técnicos. No inventes hechos ni decisiones. "
                "Devuelve solo el texto corregido. La reunión puede tratar de acuicultura, ingeniería "
                "naval o desarrollo de software; usa ese contexto y el glosario técnico para resolver "
                "ambigüedades.",
                chunk,
                temperature=0.1,
                max_tokens=1000,
            )
            for chunk in _split_transcript(transcript)
        )

    def _chat(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": "none",
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=600,
        )
        if response.status_code >= 400:
            raise SummaryError(f"LM Studio request failed: {response.status_code} - {response.text[:400]}")
        response.encoding = "utf-8"
        body = response.json()
        try:
            return body["choices"][0]["message"]["content"].strip()
        except (KeyError, TypeError, IndexError) as exc:
            raise SummaryError(f"LM Studio returned an unexpected payload: {body}") from exc


_SYSTEM_PROMPT = """You are a senior technical meeting analyst for Meik Labs. Write concise, professional Markdown in Spanish unless the transcript clearly uses another language. Do not invent decisions, owners, dates, or commitments that are not supported by the transcript. Most of the meeting are related to engineering for aquaculture, naval engineering and software development, so use this categories and whisper_glossary.txt to solve ambiguity."""


def _split_transcript(transcript: str, max_characters: int = 2_000) -> list[str]:
    """Split text at line boundaries for local models with limited context windows."""
    chunks: list[str] = []
    current_words: list[str] = []
    current_length = 0
    for word in transcript.split():
        word_length = len(word) + (1 if current_words else 0)
        if current_words and current_length + word_length > max_characters:
            chunks.append(" ".join(current_words))
            current_words = []
            current_length = 0
        current_words.append(word)
        current_length += word_length
    if current_words:
        chunks.append(" ".join(current_words))
    return chunks or [transcript]


def _build_user_prompt(transcript: str, task: str = "summary", meeting_type: str = "technical") -> str:
    meeting_label = {
        "technical": "técnica",
        "business": "de negocio",
        "general": "general",
    }.get(meeting_type, "general")
    if task == "analysis":
        return f"""Analiza esta reunión {meeting_label} y responde en español con estas secciones:

# Resumen ejecutivo
# Decisiones clave
# Riesgos y bloqueos
# Tareas / compromisos detectados
# Seguidores y preguntas abiertas
# Síntesis de sentimiento o nivel de acuerdo

Reunión:

{transcript}
"""

    if task == "compare":
        return f"""Compara estas reuniones y responde en español con estas secciones:

# Tendencias generales
# Diferencias clave entre reuniones
# Problemas recurrentes
# Tareas o decisiones que se repiten
# Recomendaciones de mejora

Transcripciones:

{transcript}
"""

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
