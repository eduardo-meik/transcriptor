"""Desktop control panel for recording and processing meeting audio."""

from __future__ import annotations

import logging
import math
import os
import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import TypeAlias

import sounddevice as sd

from core.audio_recorder import AudioDeviceSelection, AudioRecorderError, MeetingAudioRecorder
from core.capture_quality import SILENCE_PEAK, VERY_LOW_PEAK, capture_warnings
from core.config import AppConfig
from core.summarizer import MeetingSummarizer, SummaryConfig
from core.transcriber import FasterWhisperTranscriber, TranscriptionConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordingStarted:
    system_name: str


@dataclass(frozen=True)
class CaptureSaved:
    audio_path: Path
    warnings: list[str]


@dataclass(frozen=True)
class ProcessingPhase:
    title: str
    detail: str
    state: str


@dataclass(frozen=True)
class MeetingComplete:
    output_dir: Path
    transcript: Path
    report: Path | None
    analysis: Path | None


@dataclass(frozen=True)
class OperationFailed:
    message: str


GuiMessage: TypeAlias = RecordingStarted | CaptureSaved | ProcessingPhase | MeetingComplete | OperationFailed


def format_elapsed(seconds: float) -> str:
    """Format elapsed time as an hours-aware meeting timer."""

    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def _dbfs(level: float) -> str:
    if level <= 0:
        return "-inf dBFS"
    return f"{20 * math.log10(level):.1f} dBFS"


def _device_label(index: int, name: str) -> str:
    return f"{' '.join(name.split())}  [{index}]"


class MeetingControlApp:
    """Tkinter UI that runs recording and post-processing off the UI thread."""

    def __init__(self, root: tk.Tk, config: AppConfig) -> None:
        self.root = root
        self.config = config
        self._messages: queue.Queue[GuiMessage] = queue.Queue()
        self._recorder: MeetingAudioRecorder | None = None
        self._output_dir: Path | None = None
        self._state = "idle"
        self._close_after_processing = False
        self._loopback_devices: dict[str, int] = {}
        self._microphone_devices: dict[str, int] = {}
        self._selected_system_label = ""
        self._session_summary_enabled = True
        self._summary_enabled = tk.BooleanVar(value=True)

        self.root.title("Meik Labs | Asistente de reuniones")
        self.root.geometry("900x720")
        self.root.minsize(760, 660)
        self.root.configure(background="#10151f")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._configure_styles()
        self._build_interface()
        self._refresh_devices()
        self.root.after(120, self._poll_messages)
        self.root.after(250, self._update_live_status)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background="#10151f")
        style.configure("Card.TFrame", background="#192231")
        style.configure("TLabel", background="#10151f", foreground="#e7edf7", font=("Segoe UI", 10))
        style.configure("Card.TLabel", background="#192231", foreground="#e7edf7", font=("Segoe UI", 10))
        style.configure("Muted.Card.TLabel", background="#192231", foreground="#9ba9bd", font=("Segoe UI", 9))
        style.configure("Title.Card.TLabel", background="#192231", foreground="#f5f7fb", font=("Segoe UI", 12, "bold"))
        style.configure("Timer.TLabel", background="#192231", foreground="#f5f7fb", font=("Segoe UI", 32, "bold"))
        style.configure("Status.TLabel", background="#192231", foreground="#73d6a3", font=("Segoe UI", 10, "bold"))
        style.configure("Warning.TLabel", background="#192231", foreground="#f4bd63", font=("Segoe UI", 10, "bold"))
        style.configure("TCombobox", fieldbackground="#222d3e", background="#222d3e", foreground="#e7edf7")
        style.configure("TCheckbutton", background="#192231", foreground="#e7edf7")
        style.map("TCheckbutton", background=[("active", "#192231")])
        style.configure("System.Horizontal.TProgressbar", troughcolor="#263447", background="#54a9ff")
        style.configure("Microphone.Horizontal.TProgressbar", troughcolor="#263447", background="#50d19a")
        style.configure("Start.TButton", font=("Segoe UI", 11, "bold"), padding=(18, 12))
        style.configure("Stop.TButton", font=("Segoe UI", 11, "bold"), padding=(18, 12))
        style.configure("Refresh.TButton", padding=(10, 5))

    def _build_interface(self) -> None:
        container = ttk.Frame(self.root, style="App.TFrame", padding=(24, 20))
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container, style="App.TFrame")
        header.pack(fill="x", pady=(0, 18))
        ttk.Label(
            header,
            text="ASISTENTE DE REUNIONES",
            font=("Segoe UI", 18, "bold"),
            foreground="#f5f7fb",
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="Control local de captura y procesamiento. Revisa los medidores antes de empezar.",
            foreground="#9ba9bd",
        ).pack(anchor="w", pady=(4, 0))

        devices_card = self._card(container)
        devices_card.pack(fill="x", pady=(0, 14))
        top_row = ttk.Frame(devices_card, style="Card.TFrame")
        top_row.pack(fill="x", padx=18, pady=(15, 8))
        ttk.Label(top_row, text="FUENTES DE AUDIO", style="Title.Card.TLabel").pack(side="left")
        ttk.Button(top_row, text="Actualizar dispositivos", style="Refresh.TButton", command=self._refresh_devices).pack(
            side="right"
        )
        device_grid = ttk.Frame(devices_card, style="Card.TFrame")
        device_grid.pack(fill="x", padx=18, pady=(2, 16))
        device_grid.columnconfigure((0, 1), weight=1)

        ttk.Label(device_grid, text="Audio de la reunion", style="Muted.Card.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 12), pady=(3, 6)
        )
        ttk.Label(device_grid, text="Tu microfono", style="Muted.Card.TLabel").grid(
            row=0, column=1, sticky="w", padx=(12, 0), pady=(3, 6)
        )
        self.system_combo = ttk.Combobox(device_grid, state="readonly")
        self.system_combo.grid(row=1, column=0, sticky="ew", padx=(0, 12))
        self.microphone_combo = ttk.Combobox(device_grid, state="readonly")
        self.microphone_combo.grid(row=1, column=1, sticky="ew", padx=(12, 0))

        self.system_meter, self.system_level_text, self.system_signal_text = self._build_meter_card(
            container, "AUDIO DE LA REUNION", "Lo que se escucha por la salida seleccionada", "System"
        )
        self.microphone_meter, self.microphone_level_text, self.microphone_signal_text = self._build_meter_card(
            container, "TU MICROFONO", "Tu voz y el dispositivo de entrada seleccionado", "Microphone"
        )

        status_card = self._card(container)
        status_card.pack(fill="x", pady=(0, 14))
        status_row = ttk.Frame(status_card, style="Card.TFrame")
        status_row.pack(fill="x", padx=18, pady=(14, 12))
        self.timer_label = ttk.Label(status_row, text="00:00:00", style="Timer.TLabel")
        self.timer_label.pack(side="left")
        status_copy = ttk.Frame(status_row, style="Card.TFrame")
        status_copy.pack(side="left", fill="x", expand=True, padx=(22, 0))
        self.status_label = ttk.Label(status_copy, text="LISTO PARA GRABAR", style="Status.TLabel")
        self.status_label.pack(anchor="w")
        self.phase_label = ttk.Label(status_copy, text="Selecciona los dispositivos y comienza.", style="Muted.Card.TLabel")
        self.phase_label.pack(anchor="w", pady=(5, 0))
        self.summary_check = ttk.Checkbutton(
            status_card,
            text=f"Generar resumen y analisis al terminar ({self.config.summary_provider})",
            variable=self._summary_enabled,
        )
        self.summary_check.pack(anchor="w", padx=18, pady=(0, 13))

        controls = ttk.Frame(container, style="App.TFrame")
        controls.pack(fill="x", pady=(0, 12))
        self.start_button = ttk.Button(controls, text="Iniciar grabacion", style="Start.TButton", command=self._start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(
            controls, text="Detener y procesar", style="Stop.TButton", command=self._stop, state="disabled"
        )
        self.stop_button.pack(side="left", padx=(10, 0))
        self.open_folder_button = ttk.Button(
            controls, text="Abrir carpeta de resultados", command=self._open_output_folder, state="disabled"
        )
        self.open_folder_button.pack(side="right")

        self.message_label = ttk.Label(container, text="", wraplength=830, justify="left", foreground="#f4bd63")
        self.message_label.pack(fill="x", pady=(0, 8))
        ttk.Label(
            container,
            text="La transcripcion se procesa localmente. El resumen, si esta activado, envia el texto al proveedor configurado.",
            foreground="#8290a4",
            wraplength=830,
        ).pack(anchor="w", side="bottom", pady=(8, 0))

    @staticmethod
    def _card(parent: tk.Widget) -> ttk.Frame:
        return ttk.Frame(parent, style="Card.TFrame")

    def _build_meter_card(
        self, parent: tk.Widget, title: str, description: str, style_name: str
    ) -> tuple[ttk.Progressbar, ttk.Label, ttk.Label]:
        card = self._card(parent)
        card.pack(fill="x", pady=(0, 12))
        row = ttk.Frame(card, style="Card.TFrame")
        row.pack(fill="x", padx=18, pady=(13, 5))
        ttk.Label(row, text=title, style="Title.Card.TLabel").pack(side="left")
        signal_label = ttk.Label(row, text="SIN SEÑAL", style="Muted.Card.TLabel")
        signal_label.pack(side="right")
        ttk.Label(card, text=description, style="Muted.Card.TLabel").pack(anchor="w", padx=18)
        meter_row = ttk.Frame(card, style="Card.TFrame")
        meter_row.pack(fill="x", padx=18, pady=(8, 15))
        meter = ttk.Progressbar(
            meter_row,
            maximum=1.0,
            mode="determinate",
            style=f"{style_name}.Horizontal.TProgressbar",
        )
        meter.pack(side="left", fill="x", expand=True)
        level_label = ttk.Label(meter_row, text="-inf dBFS", style="Muted.Card.TLabel", width=13, anchor="e")
        level_label.pack(side="right", padx=(12, 0))
        return meter, level_label, signal_label

    def _refresh_devices(self) -> None:
        from core.wasapi_loopback import list_loopback_devices

        self.message_label.configure(text="")
        errors: list[str] = []
        try:
            loopback_devices = list_loopback_devices()
        except (AudioRecorderError, OSError, RuntimeError, ValueError) as exc:
            loopback_devices = []
            errors.append(f"No se pudieron consultar salidas WASAPI: {exc}")

        self._loopback_devices = {
            _device_label(device.index, device.clean_name): device.index for device in loopback_devices
        }
        loopback_values = list(self._loopback_devices)
        self.system_combo.configure(values=loopback_values)
        if loopback_values:
            default_device = next((device for device in loopback_devices if device.is_default), loopback_devices[0])
            self.system_combo.set(_device_label(default_device.index, default_device.clean_name))
        else:
            self.system_combo.set("")
            errors.append("No hay una salida WASAPI disponible para capturar el audio de la reunion.")

        try:
            devices = sd.query_devices()
            self._microphone_devices = {
                _device_label(index, str(device["name"])): index
                for index, device in enumerate(devices)
                if int(device["max_input_channels"]) > 0
            }
            microphone_values = list(self._microphone_devices)
        except (sd.PortAudioError, OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            self._microphone_devices = {}
            microphone_values = []
            errors.append(f"No se pudieron consultar los microfonos: {exc}")

        self.microphone_combo.configure(values=microphone_values)
        if microphone_values:
            try:
                default_index = MeetingAudioRecorder.default_wasapi_input()
            except (sd.PortAudioError, OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                default_index = None
                errors.append(f"No se pudo detectar el microfono WASAPI predeterminado: {exc}")
            default_label = next(
                (label for label, index in self._microphone_devices.items() if index == default_index),
                microphone_values[0],
            )
            self.microphone_combo.set(default_label)
        else:
            self.microphone_combo.set("")
            errors.append("No hay microfonos de entrada disponibles.")

        self._set_status("LISTO PARA GRABAR", "Elige las fuentes de audio antes de iniciar.", "ready")
        self.start_button.configure(state="normal" if loopback_values and microphone_values else "disabled")
        if errors:
            self.message_label.configure(text="\n".join(errors))

    def _set_status(self, title: str, detail: str, state: str) -> None:
        self._state = state
        self.status_label.configure(text=title)
        self.phase_label.configure(text=detail)
        self._set_device_controls_enabled(state in {"idle", "ready", "complete", "error"})

    def _set_device_controls_enabled(self, enabled: bool) -> None:
        device_state = "readonly" if enabled else "disabled"
        self.system_combo.configure(state=device_state)
        self.microphone_combo.configure(state=device_state)
        self.summary_check.configure(state="normal" if enabled else "disabled")

    def _start(self) -> None:
        if self.system_combo.get() not in self._loopback_devices:
            messagebox.showerror("Falta la salida de audio", "Selecciona una salida de audio valida.", parent=self.root)
            return
        if self.microphone_combo.get() not in self._microphone_devices:
            messagebox.showerror("Falta el microfono", "Selecciona un microfono valido.", parent=self.root)
            return

        output_dir = self.config.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
        self._output_dir = output_dir
        self._selected_system_label = self.system_combo.get()
        self._recorder = MeetingAudioRecorder(
            output_dir=output_dir,
            target_sample_rate=self.config.sample_rate,
            target_channels=self.config.channels,
            devices=AudioDeviceSelection(microphone_device=self._microphone_devices[self.microphone_combo.get()]),
            backend="wasapi-loopback",
            loopback_device_index=self._loopback_devices[self.system_combo.get()],
        )
        self.message_label.configure(text="")
        self.timer_label.configure(text="00:00:00")
        self._reset_meters()
        self._set_status("INICIANDO CAPTURA", "Abriendo las fuentes de audio seleccionadas...", "starting")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        threading.Thread(target=self._start_recording_worker, daemon=True).start()

    def _start_recording_worker(self) -> None:
        assert self._recorder is not None
        try:
            device = self._recorder.start()
            name = device.clean_name if device is not None else self._selected_system_label
            self._messages.put(RecordingStarted(name))
        except Exception as exc:
            logger.exception("Could not start meeting recording")
            self._messages.put(OperationFailed(str(exc)))

    def _stop(self) -> None:
        if self._recorder is None or self._state != "recording":
            return
        self._session_summary_enabled = self._summary_enabled.get()
        self._set_status("GUARDANDO AUDIO", "Cerrando la captura y preparando el archivo de audio...", "stopping")
        self.stop_button.configure(state="disabled")
        self.start_button.configure(state="disabled")
        threading.Thread(target=self._process_recording_worker, daemon=True).start()

    def _process_recording_worker(self) -> None:
        recorder = self._recorder
        output_dir = self._output_dir
        summary_enabled = self._session_summary_enabled
        if recorder is None or output_dir is None:
            self._messages.put(OperationFailed("No hay una grabacion activa para procesar."))
            return
        try:
            recording = recorder.stop()
            warnings = capture_warnings(recorder.session_peaks())
            self._messages.put(CaptureSaved(recording.mixed_wav, warnings))
            transcriber = FasterWhisperTranscriber(
                TranscriptionConfig(
                    model_size=self.config.whisper_model,
                    device=self.config.whisper_device,
                    compute_type=self.config.whisper_compute_type,
                    language=self.config.language,
                    allow_cpu_fallback=self.config.allow_cpu_fallback,
                    initial_prompt=self.config.initial_prompt,
                    hf_token=os.getenv("HF_TOKEN"),
                )
            )
            self._messages.put(
                ProcessingPhase("TRANSCRIBIENDO", "Transcripcion local con faster-whisper...", "transcribing")
            )
            transcript = transcriber.transcribe(recording.mixed_wav, output_dir)

            report_path: Path | None = None
            analysis_path: Path | None = None
            if summary_enabled:
                self._messages.put(
                    ProcessingPhase(
                        "GENERANDO INFORMES",
                        "Creando resumen y analisis de la reunion...",
                        "summarizing",
                    )
                )
                summarizer = MeetingSummarizer(
                    SummaryConfig(
                        provider=self.config.summary_provider,
                        anthropic_model=self.config.anthropic_model,
                        google_model=self.config.google_model,
                        openrouter_model=self.config.openrouter_model,
                    )
                )
                cleaned_transcript = summarizer.cleanup_transcript(transcript.text)
                report_path = summarizer.summarize_to_file(cleaned_transcript, output_dir)
                analysis_path = output_dir / "meeting_analysis.md"
                analysis_path.write_text(
                    summarizer.build_meeting_analysis(cleaned_transcript).strip() + "\n",
                    encoding="utf-8",
                )

            self._messages.put(MeetingComplete(output_dir, transcript.markdown_path, report_path, analysis_path))
        except Exception as exc:
            logger.exception("Meeting recording or post-processing failed")
            self._messages.put(OperationFailed(str(exc)))

    def _poll_messages(self) -> None:
        while True:
            try:
                message = self._messages.get_nowait()
            except queue.Empty:
                break
            if isinstance(message, RecordingStarted):
                self._set_status("GRABANDO", f"Capturando audio del sistema: {message.system_name}", "recording")
                self.stop_button.configure(state="normal")
            elif isinstance(message, CaptureSaved):
                warning_text = "\n".join(message.warnings)
                detail = f"Audio guardado: {message.audio_path}"
                self.message_label.configure(text=warning_text)
                self.open_folder_button.configure(state="normal")
                self._set_status("AUDIO GUARDADO", detail, "transcribing")
            elif isinstance(message, ProcessingPhase):
                self._set_status(message.title, message.detail, message.state)
            elif isinstance(message, MeetingComplete):
                details = [f"Transcripcion: {message.transcript}"]
                if message.report is not None:
                    details.append(f"Resumen: {message.report}")
                if message.analysis is not None:
                    details.append(f"Analisis: {message.analysis}")
                self.message_label.configure(text="\n".join(details))
                self._output_dir = message.output_dir
                self.open_folder_button.configure(state="normal")
                self.start_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self._set_status("REUNION PROCESADA", "La captura, transcripcion y tareas seleccionadas terminaron.", "complete")
                self._recorder = None
                if self._close_after_processing:
                    self.root.destroy()
                    return
            else:
                self.message_label.configure(text=f"Error: {message.message}")
                self.start_button.configure(state="normal" if self._loopback_devices and self._microphone_devices else "disabled")
                self.stop_button.configure(state="disabled")
                self._set_status("OCURRIO UN ERROR", "Revisa el mensaje para ver el motivo y corregirlo.", "error")
                self._recorder = None
        self.root.after(120, self._poll_messages)

    def _update_live_status(self) -> None:
        recorder = self._recorder
        if recorder is not None and self._state in {"recording", "stopping"}:
            elapsed = recorder.elapsed_seconds()
            self.timer_label.configure(text=format_elapsed(elapsed))
            levels = recorder.levels()
            peaks = recorder.session_peaks()
            self._update_meter(
                self.system_meter,
                self.system_level_text,
                self.system_signal_text,
                levels.get("system", 0.0),
                peaks.get("system", 0.0),
            )
            self._update_meter(
                self.microphone_meter,
                self.microphone_level_text,
                self.microphone_signal_text,
                levels.get("microphone", 0.0),
                peaks.get("microphone", 0.0),
            )
        self.root.after(250, self._update_live_status)

    @staticmethod
    def _update_meter(
        meter: ttk.Progressbar,
        level_label: ttk.Label,
        signal_label: ttk.Label,
        current_level: float,
        session_peak: float,
    ) -> None:
        meter.configure(value=min(max(current_level, 0.0), 1.0))
        level_label.configure(text=_dbfs(current_level))
        if session_peak < SILENCE_PEAK:
            signal_label.configure(text="SIN SEÑAL")
        elif session_peak < VERY_LOW_PEAK:
            signal_label.configure(text="SEÑAL BAJA")
        else:
            signal_label.configure(text="RECIBIENDO AUDIO")

    def _reset_meters(self) -> None:
        for meter, level_label, signal_label in (
            (self.system_meter, self.system_level_text, self.system_signal_text),
            (self.microphone_meter, self.microphone_level_text, self.microphone_signal_text),
        ):
            meter.configure(value=0.0)
            level_label.configure(text="-inf dBFS")
            signal_label.configure(text="SIN SEÑAL")

    def _open_output_folder(self) -> None:
        if self._output_dir is not None and self._output_dir.exists():
            os.startfile(self._output_dir)  # type: ignore[attr-defined]

    def _on_close(self) -> None:
        if self._state == "recording":
            should_stop = messagebox.askyesno(
                "Detener grabacion",
                "Hay una grabacion activa. ¿Quieres detenerla y procesarla antes de cerrar?",
                parent=self.root,
            )
            if should_stop:
                self._close_after_processing = True
                self._stop()
            return
        if self._state in {"starting", "stopping", "transcribing", "summarizing"}:
            messagebox.showwarning(
                "Procesamiento en curso",
                "Espera a que termine el procesamiento para cerrar y conservar los resultados.",
                parent=self.root,
            )
            return
        self.root.destroy()


def run_gui(config: AppConfig) -> int:
    """Create the desktop application and run its Tk event loop."""

    root = tk.Tk()
    MeetingControlApp(root, config)
    root.mainloop()
    return 0
