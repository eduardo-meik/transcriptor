"""Configuration helpers for the Meik Labs meeting assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv


SummaryProvider = Literal["anthropic", "google"]

DEFAULT_WHISPER_GLOSSARY_PATH = Path(__file__).resolve().parent.parent / "whisper_glossary.txt"


def _load_whisper_glossary() -> str:
    """Return the central glossary to improve transcription quality.

    Priority order:
    1) WHISPER_GLOSSARY env var (fast override for ad-hoc additions)
    2) WHISPER_GLOSSARY_PATH file path (single editable source file)
    3) built-in fallback list
    """

    override = os.getenv("WHISPER_GLOSSARY")
    if override and override.strip():
        return override.strip()

    glossary_path = os.getenv("WHISPER_GLOSSARY_PATH")
    if glossary_path:
        path = Path(glossary_path).expanduser()
    else:
        path = DEFAULT_WHISPER_GLOSSARY_PATH

    if path.exists():
        terms = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            terms.append(line)
        if terms:
            return ", ".join(terms) + "."

    return (
        "Reunión técnica: Kubernetes, Docker, AWS, GCP, microservicios, API, backend, frontend, "
        "pull request, merge, sprint, deploy, CI/CD, refactor, observabilidad, log, tracing."
    )


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration loaded from environment variables."""

    output_root: Path = Path("outputs")
    sample_rate: int = 16_000
    channels: int = 1
    language: str = "es"
    whisper_model: str = "large"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    allow_cpu_fallback: bool = True
    initial_prompt: str | None = _load_whisper_glossary()
    enable_diarization: bool = False
    summary_provider: SummaryProvider = "anthropic"
    anthropic_model: str = "claude-3-5-sonnet-latest"
    google_model: str = "gemini-1.5-pro"


def load_config() -> AppConfig:
    """Load configuration from `.env` and process environment variables."""

    load_dotenv()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    prompt = os.getenv("WHISPER_INITIAL_PROMPT") or _load_whisper_glossary()
    return AppConfig(
        output_root=Path(os.getenv("MEIK_OUTPUT_ROOT", "outputs")),
        sample_rate=int(os.getenv("MEIK_SAMPLE_RATE", "16000")),
        channels=int(os.getenv("MEIK_CHANNELS", "1")),
        language=os.getenv("MEIK_LANGUAGE", "es"),
        whisper_model=os.getenv("WHISPER_MODEL", "large"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cuda"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "float16"),
        allow_cpu_fallback=os.getenv("WHISPER_ALLOW_CPU_FALLBACK", "true").lower()
        in {"1", "true", "yes", "y"},
        initial_prompt=prompt,
        enable_diarization=os.getenv("WHISPER_ENABLE_DIARIZATION", "false").lower()
        in {"1", "true", "yes", "y"},
        summary_provider=os.getenv("SUMMARY_PROVIDER", "anthropic"),  # type: ignore[arg-type]
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        google_model=os.getenv("GOOGLE_MODEL", "gemini-1.5-pro"),
    )
