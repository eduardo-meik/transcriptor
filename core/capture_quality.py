"""Shared signal-level checks for recorded meeting audio."""

from __future__ import annotations


SILENCE_PEAK = 0.0005  # -66 dBFS
VERY_LOW_PEAK = 0.01  # -40 dBFS


def capture_warnings(peaks: dict[str, float], system_only: bool = False) -> list[str]:
    """Describe channels whose recorded signal may be missing or too quiet."""

    warnings: list[str] = []
    system_peak = peaks.get("system", 0.0)
    microphone_peak = peaks.get("microphone", 0.0)

    if system_peak < SILENCE_PEAK:
        warnings.append(
            "El canal de SISTEMA no capturo audio. Causas probables: nadie mas hablo durante "
            "la grabacion, o Teams cambio de dispositivo de salida."
        )
    elif system_peak < VERY_LOW_PEAK:
        warnings.append("El canal de SISTEMA capturo muy bajo. Revisa el volumen de salida.")

    if not system_only:
        if microphone_peak < SILENCE_PEAK:
            warnings.append(
                "El canal de MICROFONO no capturo audio. Revisa el dispositivo de entrada con "
                "`python main.py --check-devices`."
            )
        elif microphone_peak < VERY_LOW_PEAK:
            warnings.append("El canal de MICROFONO capturo muy bajo. Revisa el nivel de entrada.")

    return warnings
