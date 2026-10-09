"""Desktop control panel for recording and processing meeting audio."""

from __future__ import annotations

import logging
import math
import os
import queue
import re
import threading
import tkinter as tk
import wave
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


@dataclass(frozen=True)
class MeetingEntry:
    title: str
    meeting_date: datetime
    duration_seconds: float | None
    transcript_path: Path
    summary_path: Path | None


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


def list_meetings(output_root: Path) -> list[MeetingEntry]:
    """Find completed meetings and derive their duration from the saved WAV."""

    if not output_root.exists():
        return []

    entries: list[MeetingEntry] = []
    for meeting_dir in output_root.iterdir():
        if not meeting_dir.is_dir():
            continue
        transcript_path = meeting_dir / "transcript_raw.md"
        if not transcript_path.is_file():
            continue

        try:
            title = derive_meeting_title(transcript_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError) as exc:
            logger.warning("Could not read meeting transcript %s to derive its title: %s", transcript_path, exc)
            title = meeting_dir.name

        try:
            meeting_date = datetime.strptime(meeting_dir.name, "%Y%m%d_%H%M%S")
        except ValueError:
            meeting_date = datetime.fromtimestamp(meeting_dir.stat().st_mtime)

        audio_path = meeting_dir / "meeting_audio.wav"
        duration_seconds: float | None = None
        if audio_path.is_file():
            try:
                with wave.open(str(audio_path), "rb") as audio_file:
                    frame_rate = audio_file.getframerate()
                    if frame_rate > 0:
                        duration_seconds = audio_file.getnframes() / frame_rate
                    else:
                        logger.warning("Could not determine meeting duration from %s: invalid sample rate", audio_path)
            except (OSError, wave.Error) as exc:
                logger.warning("Could not read meeting audio duration from %s: %s", audio_path, exc)

        summary_path = meeting_dir / "meeting_report.md"
        entries.append(
            MeetingEntry(
                title=title,
                meeting_date=meeting_date,
                duration_seconds=duration_seconds,
                transcript_path=transcript_path,
                summary_path=summary_path if summary_path.is_file() else None,
            )
        )

    return sorted(entries, key=lambda entry: entry.meeting_date, reverse=True)


def derive_meeting_title(markdown: str, max_length: int = 72) -> str:
    """Create a short local title from the first substantial transcript sentence."""

    transcript = markdown.split("---", maxsplit=1)[-1]
    fallback = "Reunión sin título"
    for line in transcript.splitlines():
        candidate = line.strip()
        if not candidate or candidate.startswith("#") or candidate.startswith("**"):
            continue
        candidate = re.sub(r"[*_`>#]", "", candidate)
        candidate = re.sub(r"^(?:(?:ya|sí|si|eh|mmm|mm|bueno)[,.\s]+)+", "", candidate, flags=re.IGNORECASE)
        if not candidate:
            continue
        fallback = candidate
        for sentence in re.split(r"(?<=[.!?])\s+", candidate):
            words = re.findall(r"[^\W_]+", sentence, flags=re.UNICODE)
            if len(words) >= 4:
                candidate = sentence.strip(" \t\"'“”")
                break
        else:
            words = candidate.split()
            if len(words) < 4:
                continue
            candidate = " ".join(words[:10]).strip(" \t\"'“”")

        candidate = candidate.rstrip(" .!?")
        if len(candidate) > max_length:
            candidate = candidate[: max_length - 1].rsplit(" ", 1)[0].rstrip(" ,;:-") + "…"
        return candidate[:1].upper() + candidate[1:]
    return fallback[:max_length]


def format_meeting_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    return format_elapsed(seconds)


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
        self._meeting_entries: dict[str, MeetingEntry] = {}
        self._selected_transcript_text = ""
        self._selected_system_label = ""
        self._session_summary_enabled = True
        self._summary_enabled = tk.BooleanVar(value=True)
        self._recording_popup: tk.Toplevel | None = None
        self._popup_timer_label: tk.Label | None = None

        self.root.title("Meik Labs | Asistente de reuniones")
        self.root.iconbitmap(str(Path(__file__).resolve().with_name("icon.ico")))
        self.root.geometry("900x720")
        self.root.minsize(760, 660)
        self.root.configure(background="#f4f6f8")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Unmap>", self._on_root_unmapped, add="+")
        self.root.bind("<Map>", self._on_root_mapped, add="+")
        self._configure_styles()
        self._build_interface()
        self._refresh_meetings()
        self._refresh_devices()
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.root.after(120, self._poll_messages)
        self.root.after(250, self._update_live_status)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background="#f4f6f8")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("TLabel", background="#f4f6f8", foreground="#344054", font=("Segoe UI", 10))
        style.configure("Card.TLabel", background="#ffffff", foreground="#344054", font=("Segoe UI", 10))
        style.configure("Muted.Card.TLabel", background="#ffffff", foreground="#667085", font=("Segoe UI", 9))
        style.configure("Title.Card.TLabel", background="#ffffff", foreground="#101828", font=("Segoe UI", 12, "bold"))
        style.configure("Timer.TLabel", background="#ffffff", foreground="#101828", font=("Segoe UI", 32, "bold"))
        style.configure("Status.TLabel", background="#ffffff", foreground="#067647", font=("Segoe UI", 10, "bold"))
        style.configure("Warning.TLabel", background="#ffffff", foreground="#b54708", font=("Segoe UI", 10, "bold"))
        style.configure("TCombobox", fieldbackground="#ffffff", background="#ffffff", foreground="#344054")
        style.configure("TCheckbutton", background="#ffffff", foreground="#344054")
        style.map("TCheckbutton", background=[("active", "#ffffff")])
        style.configure("TButton", background="#ffffff", foreground="#344054", padding=(10, 6))
        style.map(
            "TButton",
            background=[("active", "#f2f4f7"), ("disabled", "#f9fafb")],
            foreground=[("disabled", "#98a2b3")],
        )
        style.configure("TNotebook", background="#f4f6f8", borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background="#e4e7ec",
            foreground="#344054",
            padding=(14, 8),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#ffffff"), ("active", "#eaecf0")],
            foreground=[("selected", "#101828")],
        )
        style.configure(
            "Treeview",
            background="#ffffff",
            fieldbackground="#ffffff",
            foreground="#344054",
            rowheight=28,
            borderwidth=0,
            font=("Segoe UI", 10),
        )
        style.map("Treeview", background=[("selected", "#d1e9ff")], foreground=[("selected", "#1849a9")])
        style.configure(
            "Treeview.Heading",
            background="#f2f4f7",
            foreground="#344054",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
        )
        style.map("Treeview.Heading", background=[("active", "#eaecf0")])
        style.configure("System.Horizontal.TProgressbar", troughcolor="#e4e7ec", background="#1570ef")
        style.configure("Microphone.Horizontal.TProgressbar", troughcolor="#e4e7ec", background="#12b76a")
        style.configure("Start.TButton", font=("Segoe UI", 11, "bold"), padding=(18, 12))
        style.configure("Stop.TButton", font=("Segoe UI", 11, "bold"), padding=(18, 12))
        style.configure("Refresh.TButton", padding=(10, 5))

    def _build_interface(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)
        capture_tab = ttk.Frame(self.notebook, style="App.TFrame")
        meetings_tab = ttk.Frame(self.notebook, style="App.TFrame")
        self.notebook.add(capture_tab, text="Captura")
        self.notebook.add(meetings_tab, text="Reuniones")

        container = ttk.Frame(capture_tab, style="App.TFrame", padding=(24, 20))
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container, style="App.TFrame")
        header.pack(fill="x", pady=(0, 18))
        ttk.Label(
            header,
            text="ASISTENTE DE REUNIONES",
            font=("Segoe UI", 18, "bold"),
            foreground="#101828",
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="Control local de captura y procesamiento. Revisa los medidores antes de empezar.",
            foreground="#667085",
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

        self.message_label = ttk.Label(container, text="", wraplength=830, justify="left", foreground="#b54708")
        self.message_label.pack(fill="x", pady=(0, 8))
        ttk.Label(
            container,
            text="La transcripcion se procesa localmente. El resumen, si esta activado, envia el texto al proveedor configurado.",
            foreground="#667085",
            wraplength=830,
        ).pack(anchor="w", side="bottom", pady=(8, 0))

        self._build_meetings_interface(meetings_tab)

    def _build_meetings_interface(self, parent: ttk.Frame) -> None:
        container = ttk.Frame(parent, style="App.TFrame", padding=(24, 20))
        container.pack(fill="both", expand=True)
        header = ttk.Frame(container, style="App.TFrame")
        header.pack(fill="x", pady=(0, 18))
        ttk.Label(
            header,
            text="REUNIONES",
            font=("Segoe UI", 18, "bold"),
            foreground="#101828",
        ).pack(side="left")
        ttk.Button(
            header,
            text="Actualizar",
            style="Refresh.TButton",
            command=self._refresh_meetings,
        ).pack(side="right")

        card = self._card(container)
        card.pack(fill="both", expand=True)
        ttk.Label(
            card,
            text="Selecciona una reunión para previsualizar su transcripción. El título se extrae localmente del texto.",
            style="Muted.Card.TLabel",
        ).pack(anchor="w", padx=16, pady=(14, 10))

        table_frame = ttk.Frame(card, style="Card.TFrame")
        table_frame.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        columns = ("title", "date", "duration", "transcript", "summary")
        self.meetings_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        self.meetings_table.heading("title", text="Reunión")
        self.meetings_table.heading("date", text="Fecha")
        self.meetings_table.heading("duration", text="Duración")
        self.meetings_table.heading("transcript", text="Archivo")
        self.meetings_table.heading("summary", text="Resumen")
        self.meetings_table.column("title", width=280, anchor="w")
        self.meetings_table.column("date", width=140, anchor="w")
        self.meetings_table.column("duration", width=90, anchor="center")
        self.meetings_table.column("transcript", width=100, anchor="center")
        self.meetings_table.column("summary", width=100, anchor="center")
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.meetings_table.yview)
        self.meetings_table.configure(yscrollcommand=scrollbar.set)
        self.meetings_table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.meetings_table.bind("<<TreeviewSelect>>", self._on_meeting_selected)
        self.meetings_table.bind("<ButtonRelease-1>", self._on_meeting_table_click)
        self.meetings_table.bind("<Double-1>", self._on_meeting_table_double_click)

        preview_card = self._card(container)
        preview_card.pack(fill="both", expand=True, pady=(8, 0))
        preview_header = ttk.Frame(preview_card, style="Card.TFrame")
        preview_header.pack(fill="x", padx=16, pady=(12, 8))
        self.preview_title_label = ttk.Label(
            preview_header,
            text="Previsualización de la transcripción",
            style="Title.Card.TLabel",
        )
        self.preview_title_label.pack(side="left")
        self.copy_transcript_button = ttk.Button(
            preview_header,
            text="Copiar",
            command=self._copy_selected_transcript,
            state="disabled",
        )
        self.copy_transcript_button.pack(side="right")
        preview_frame = ttk.Frame(preview_card, style="Card.TFrame")
        preview_frame.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        self.transcript_preview = tk.Text(
            preview_frame,
            wrap="word",
            height=8,
            state="disabled",
            background="#ffffff",
            foreground="#344054",
            insertbackground="#344054",
            relief="flat",
            padx=12,
            pady=10,
            font=("Segoe UI", 10),
        )
        self.transcript_preview.tag_configure("h1", font=("Segoe UI", 16, "bold"), foreground="#101828")
        self.transcript_preview.tag_configure("h2", font=("Segoe UI", 13, "bold"), foreground="#101828")
        self.transcript_preview.tag_configure("h3", font=("Segoe UI", 11, "bold"), foreground="#101828")
        self.transcript_preview.tag_configure("bold", font=("Segoe UI", 10, "bold"))
        self.transcript_preview.tag_configure("italic", font=("Segoe UI", 10, "italic"))
        self.transcript_preview.tag_configure("code", font=("Consolas", 10), foreground="#175cd3")
        self.transcript_preview.tag_configure("muted", foreground="#667085")
        self.transcript_preview.tag_configure("bullet", foreground="#067647")
        preview_scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=self.transcript_preview.yview)
        self.transcript_preview.configure(yscrollcommand=preview_scrollbar.set)
        self.transcript_preview.pack(side="left", fill="both", expand=True)
        preview_scrollbar.pack(side="right", fill="y")

        footer = ttk.Frame(container, style="App.TFrame")
        footer.pack(fill="x", pady=(12, 0))
        self.meetings_count_label = ttk.Label(footer, text="0 reuniones", foreground="#667085")
        self.meetings_count_label.pack(side="left")
        self.open_transcript_button = ttk.Button(
            footer,
            text="Abrir archivo MD",
            command=self._open_selected_transcript,
            state="disabled",
        )
        self.open_transcript_button.pack(side="right")
        self.open_summary_button = ttk.Button(
            footer,
            text="Ver resumen",
            command=self._open_selected_summary,
            state="disabled",
        )
        self.open_summary_button.pack(side="right", padx=(0, 8))

    def _refresh_meetings(self) -> None:
        entries = list_meetings(self.config.output_root)
        for item in self.meetings_table.get_children():
            self.meetings_table.delete(item)
        self._meeting_entries.clear()
        for index, entry in enumerate(entries):
            item_id = str(index)
            self._meeting_entries[item_id] = entry
            self.meetings_table.insert(
                "",
                "end",
                iid=item_id,
                values=(
                    entry.title,
                    entry.meeting_date.strftime("%d/%m/%Y %H:%M"),
                    format_meeting_duration(entry.duration_seconds),
                    "Ver",
                    "Ver" if entry.summary_path is not None else "—",
                ),
            )
        count = len(entries)
        noun = "reunión" if count == 1 else "reuniones"
        self.meetings_count_label.configure(text=f"{count} {noun}")
        self.open_transcript_button.configure(state="disabled")
        self.open_summary_button.configure(state="disabled")
        self.copy_transcript_button.configure(state="disabled")
        self.preview_title_label.configure(text="Previsualización de la transcripción")
        self._set_transcript_preview("")

    def _on_tab_changed(self, _event: tk.Event[tk.Misc]) -> None:
        self._refresh_meetings()

    def _on_meeting_selected(self, _event: tk.Event[tk.Misc]) -> None:
        selection = self.meetings_table.selection()
        entry = self._meeting_entries.get(selection[0]) if selection else None
        if entry is None:
            self._selected_transcript_text = ""
            self._set_transcript_preview("")
            self.open_transcript_button.configure(state="disabled")
            self.open_summary_button.configure(state="disabled")
            self.copy_transcript_button.configure(state="disabled")
            return
        try:
            self._selected_transcript_text = entry.transcript_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            self._selected_transcript_text = ""
            logger.exception("Could not load meeting transcript %s", entry.transcript_path)
            self._set_transcript_preview(f"No se pudo cargar la transcripción:\n{exc}")
            self.copy_transcript_button.configure(state="disabled")
        else:
            self._set_transcript_preview(self._selected_transcript_text)
            self.copy_transcript_button.configure(state="normal")
        self.preview_title_label.configure(text=entry.title)
        self.open_transcript_button.configure(state="normal")
        self.open_summary_button.configure(state="normal" if entry.summary_path is not None else "disabled")

    def _set_transcript_preview(self, content: str) -> None:
        self.transcript_preview.configure(state="normal")
        self.transcript_preview.delete("1.0", "end")
        for line in content.splitlines():
            heading = re.match(r"^(#{1,6})\s+(.*)$", line)
            if heading:
                tag = "h1" if len(heading.group(1)) == 1 else "h2" if len(heading.group(1)) == 2 else "h3"
                self._insert_markdown_inline(heading.group(2), (tag,))
            elif re.match(r"^\s*(?:---+|\*\*\*+|___+)\s*$", line):
                self.transcript_preview.insert("end", "────────────────────────────\n", ("muted",))
                continue
            else:
                bullet = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", line)
                if bullet:
                    self.transcript_preview.insert("end", f"{bullet.group(1)}{bullet.group(2)} ", ("bullet",))
                    self._insert_markdown_inline(bullet.group(3))
                else:
                    quote = re.match(r"^\s*>\s?(.*)$", line)
                    if quote:
                        self.transcript_preview.insert("end", "│ ", ("muted",))
                        self._insert_markdown_inline(quote.group(1), ("muted",))
                    else:
                        self._insert_markdown_inline(line)
            self.transcript_preview.insert("end", "\n")
        self.transcript_preview.configure(state="disabled")

    def _insert_markdown_inline(self, text: str, base_tags: tuple[str, ...] = ()) -> None:
        pattern = re.compile(r"(\*\*.+?\*\*|__.+?__|`[^`]+`|\*[^*]+\*|_[^_]+_)")
        offset = 0
        for match in pattern.finditer(text):
            if match.start() > offset:
                self.transcript_preview.insert("end", text[offset : match.start()], base_tags)
            token = match.group()
            if token.startswith(("**", "__")):
                rendered, tags = token[2:-2], (*base_tags, "bold")
            elif token.startswith("`"):
                rendered, tags = token[1:-1], (*base_tags, "code")
            else:
                rendered, tags = token[1:-1], (*base_tags, "italic")
            self.transcript_preview.insert("end", rendered, tags)
            offset = match.end()
        if offset < len(text):
            self.transcript_preview.insert("end", text[offset:], base_tags)

    def _on_meeting_table_click(self, event: tk.Event[tk.Misc]) -> None:
        item_id = self.meetings_table.identify_row(event.y)
        if item_id:
            self.meetings_table.selection_set(item_id)
            column = self.meetings_table.identify_column(event.x)
            if column == "#4":
                self._open_selected_transcript()
            elif column == "#5":
                self._open_selected_summary()

    def _on_meeting_table_double_click(self, _event: tk.Event[tk.Misc]) -> None:
        self._on_meeting_selected(_event)

    def _open_selected_transcript(self) -> None:
        selection = self.meetings_table.selection()
        if not selection:
            return
        entry = self._meeting_entries.get(selection[0])
        if entry is None or not entry.transcript_path.is_file():
            messagebox.showerror("Archivo no encontrado", "La transcripción seleccionada ya no existe.", parent=self.root)
            self._refresh_meetings()
            return
        try:
            os.startfile(entry.transcript_path)  # type: ignore[attr-defined]
        except OSError as exc:
            logger.exception("Could not open meeting transcript %s", entry.transcript_path)
            messagebox.showerror("No se pudo abrir el archivo", str(exc), parent=self.root)

    def _open_selected_summary(self) -> None:
        selection = self.meetings_table.selection()
        if not selection:
            return
        entry = self._meeting_entries.get(selection[0])
        if entry is None or entry.summary_path is None or not entry.summary_path.is_file():
            messagebox.showerror("Resumen no disponible", "Esta reunión no tiene un archivo de resumen.", parent=self.root)
            self._refresh_meetings()
            return
        try:
            os.startfile(entry.summary_path)  # type: ignore[attr-defined]
        except OSError as exc:
            logger.exception("Could not open meeting summary %s", entry.summary_path)
            messagebox.showerror("No se pudo abrir el resumen", str(exc), parent=self.root)

    def _copy_selected_transcript(self) -> None:
        selection = self.meetings_table.selection()
        if not selection or not self._selected_transcript_text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self._selected_transcript_text)
        entry = self._meeting_entries.get(selection[0])
        if entry is not None:
            self.preview_title_label.configure(text=f"{entry.title} — copiada")

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
        self._sync_recording_popup()

    def _on_root_unmapped(self, event: tk.Event[tk.Misc]) -> None:
        if event.widget is self.root:
            self.root.after_idle(self._sync_recording_popup)

    def _on_root_mapped(self, event: tk.Event[tk.Misc]) -> None:
        if event.widget is self.root:
            self.root.after_idle(self._sync_recording_popup)

    def _sync_recording_popup(self) -> None:
        if self._state != "recording" or self.root.state() != "iconic":
            self._close_recording_popup()
            return
        if self._recording_popup is None:
            try:
                self._show_recording_popup()
            except tk.TclError:
                logger.exception("Could not display the minimized recording status window")
                self._close_recording_popup()

    def _show_recording_popup(self) -> None:
        popup = tk.Toplevel(self.root)
        self._recording_popup = popup
        popup.overrideredirect(True)
        popup.attributes("-topmost", True)
        popup.configure(background="#ffffff")
        popup.resizable(False, False)
        popup.update_idletasks()
        x_position = max(0, popup.winfo_screenwidth() - 330)
        y_position = max(0, popup.winfo_screenheight() - 180)
        popup.geometry(f"310x132+{x_position}+{y_position}")

        card = tk.Frame(
            popup,
            background="#ffffff",
            highlightbackground="#d0d5dd",
            highlightthickness=1,
            padx=14,
            pady=12,
        )
        card.pack(fill="both", expand=True)
        tk.Label(
            card,
            text="●  GRABACIÓN ACTIVA",
            background="#ffffff",
            foreground="#b42318",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        self._popup_timer_label = tk.Label(
            card,
            text=self.timer_label.cget("text"),
            background="#ffffff",
            foreground="#101828",
            font=("Segoe UI", 20, "bold"),
        )
        self._popup_timer_label.pack(anchor="w", pady=(3, 4))
        tk.Button(
            card,
            text="Volver a la ventana",
            command=self._restore_main_window,
            font=("Segoe UI", 9),
        ).pack(anchor="e")

    def _close_recording_popup(self) -> None:
        popup = self._recording_popup
        self._recording_popup = None
        self._popup_timer_label = None
        if popup is not None and popup.winfo_exists():
            popup.destroy()

    def _restore_main_window(self) -> None:
        self._close_recording_popup()
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

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
                self._refresh_meetings()
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
            formatted_elapsed = format_elapsed(elapsed)
            self.timer_label.configure(text=formatted_elapsed)
            if self._popup_timer_label is not None:
                self._popup_timer_label.configure(text=formatted_elapsed)
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
