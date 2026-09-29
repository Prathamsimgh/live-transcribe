"""Tkinter window: single-click toggle, VU meter, live transcript.

The view owns no application logic. It calls into a controller and exposes
thread-safe update methods, since results arrive on a worker thread and
hotkeys arrive on a pynput thread -- both marshal onto the Tk loop.
"""

from __future__ import annotations

import os
import queue
import subprocess
import tkinter as tk
from tkinter import scrolledtext
from typing import Optional

BG = "#1e1e24"
FG = "#e8e8ef"
MUTED = "#8b8b9a"
ACCENT = "#4ec9b0"
DANGER = "#e05561"
PANEL = "#26262e"


def to_pynput_hotkey(spec: str) -> str:
    """'ctrl+alt+c' -> '<ctrl>+<alt>+c'"""
    parts = []
    for token in spec.lower().split("+"):
        token = token.strip()
        if not token:
            continue
        parts.append(token if len(token) == 1 else f"<{token}>")
    return "+".join(parts)


class TranscribeWindow:
    def __init__(self, controller, settings) -> None:
        self.controller = controller
        self.s = settings
        self._events: "queue.Queue" = queue.Queue()

        self.root = tk.Tk()
        self.root.title("Live Transcribe")
        self.root.configure(bg=BG)
        self.root.geometry("560x460")
        self.root.minsize(420, 320)
        if settings.always_on_top:
            self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build()
        self._hotkeys = None
        self.root.after(50, self._drain)
        self.root.after(33, self._tick_meter)
        self._level = 0.0

    # --------------------------------------------------------------- layout

    def _build(self) -> None:
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=12, pady=(12, 4))
        tk.Label(header, text="Live Transcribe", bg=BG, fg=FG,
                 font=("Segoe UI", 13, "bold")).pack(side="left")
        self.badge = tk.Label(header, text="", bg=BG, fg=MUTED,
                              font=("Segoe UI", 9))
        self.badge.pack(side="right")

        self.toggle = tk.Button(
            self.root, text="START", command=self._on_toggle,
            bg=ACCENT, fg="#10221e", activebackground=ACCENT,
            font=("Segoe UI", 14, "bold"), relief="flat", cursor="hand2",
            height=2, bd=0,
        )
        self.toggle.pack(fill="x", padx=12, pady=6)

        meter_row = tk.Frame(self.root, bg=BG)
        meter_row.pack(fill="x", padx=12, pady=(0, 6))
        self.meter = tk.Canvas(meter_row, height=10, bg=PANEL,
                               highlightthickness=0)
        self.meter.pack(side="left", fill="x", expand=True)
        self.status = tk.Label(meter_row, text="idle", bg=BG, fg=MUTED,
                               font=("Segoe UI", 9), width=22, anchor="e")
        self.status.pack(side="right")

        self.view = scrolledtext.ScrolledText(
            self.root, bg=PANEL, fg=FG, insertbackground=FG,
            font=("Consolas", 10), relief="flat", wrap="word",
            padx=10, pady=8, state="disabled",
        )
        self.view.pack(fill="both", expand=True, padx=12, pady=6)
        self.view.tag_configure("time", foreground=MUTED)
        self.view.tag_configure("note", foreground="#d7a65f")

        footer = tk.Frame(self.root, bg=BG)
        footer.pack(fill="x", padx=12, pady=(0, 6))
        for label, cmd in (("Copy All", self._on_copy),
                           ("Open in Notepad", self._on_open)):
            tk.Button(footer, text=label, command=cmd, bg=PANEL, fg=FG,
                      activebackground="#33333d", activeforeground=FG,
                      relief="flat", cursor="hand2", bd=0,
                      font=("Segoe UI", 9), padx=12, pady=5).pack(side="left",
                                                                  padx=(0, 6))
        self.path_label = tk.Label(self.root, text="", bg=BG, fg=MUTED,
                                   font=("Segoe UI", 8), anchor="w")
        self.path_label.pack(fill="x", padx=12, pady=(0, 10))

    # ------------------------------------------------------------- hotkeys

    def install_hotkeys(self) -> None:
        try:
            from pynput import keyboard
        except Exception:
            return
        mapping = {
            to_pynput_hotkey(self.s.hotkey_toggle): lambda: self.post("hotkey_toggle"),
            to_pynput_hotkey(self.s.hotkey_copy): lambda: self.post("hotkey_copy"),
        }
        try:
            self._hotkeys = keyboard.GlobalHotKeys(mapping)
            self._hotkeys.daemon = True
            self._hotkeys.start()
        except Exception:
            self._hotkeys = None

    # -------------------------------------------------- thread-safe updates

    def post(self, kind: str, payload=None) -> None:
        """Called from any thread. Marshals onto the Tk loop."""
        self._events.put((kind, payload))

    def set_level(self, level: float) -> None:
        self._level = level

    def _drain(self) -> None:
        try:
            while True:
                kind, payload = self._events.get_nowait()
                self._handle(kind, payload)
        except queue.Empty:
            pass
        self.root.after(50, self._drain)

    def _handle(self, kind: str, payload) -> None:
        if kind == "line":
            self._append(payload[0], payload[1])
        elif kind == "note":
            self._append(None, payload, note=True)
        elif kind == "status":
            self.status.configure(text=payload)
        elif kind == "badge":
            self.badge.configure(text=payload)
        elif kind == "path":
            self.path_label.configure(text=payload)
        elif kind == "running":
            self._set_running(payload)
        elif kind == "hotkey_toggle":
            self._on_toggle()
        elif kind == "hotkey_copy":
            self._on_copy()

    def _append(self, stamp: Optional[str], text: str, note: bool = False) -> None:
        self.view.configure(state="normal")
        if note:
            self.view.insert("end", f"-- {text} --\n", "note")
        else:
            self.view.insert("end", f"[{stamp}] ", "time")
            self.view.insert("end", f"{text}\n")
        self.view.see("end")
        self.view.configure(state="disabled")

    def _set_running(self, running: bool) -> None:
        if running:
            self.toggle.configure(text="STOP", bg=DANGER, fg="#2a1013",
                                  activebackground=DANGER)
        else:
            self.toggle.configure(text="START", bg=ACCENT, fg="#10221e",
                                  activebackground=ACCENT)

    def _tick_meter(self) -> None:
        self.meter.delete("all")
        width = max(1, self.meter.winfo_width())
        filled = int(width * max(0.0, min(1.0, self._level)))
        if filled > 0:
            colour = DANGER if self._level > 0.92 else ACCENT
            self.meter.create_rectangle(0, 0, filled, 10, fill=colour, width=0)
        self.root.after(33, self._tick_meter)

    # --------------------------------------------------------------- events

    def _on_toggle(self) -> None:
        self.controller.toggle()

    def _on_copy(self) -> None:
        # copy_all() posts its own status line, so the message is identical
        # whether the copy came from the button, the hotkey, or STOP.
        self.controller.copy_all()

    def _on_open(self) -> None:
        path = self.controller.transcript_path()
        if not os.path.exists(path):
            self.status.configure(text="no transcript yet")
            return
        try:
            subprocess.Popen(["notepad.exe", str(path)])
        except Exception:
            os.startfile(str(path))  # noqa: S606

    def _on_close(self) -> None:
        self.controller.shutdown()
        if self._hotkeys is not None:
            try:
                self._hotkeys.stop()
            except Exception:
                pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
