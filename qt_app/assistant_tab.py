"""
qt_app/assistant_tab.py
=======================
AI Virtual Assistant tab for the Human Airways Digital Twin desktop app.

Features
--------
- Chat interface backed by core.rag (same 31-doc KB as the Streamlit Ask AI page)
- Animated avatar with four states: Idle | Listening | Thinking | Speaking
- Voice input: sounddevice (record) + SpeechRecognition (Google free API, no key)
- Voice output: pyttsx3 (offline Windows SAPI TTS)
- Keyboard shortcut: Enter to send, Space to toggle voice recording (from text field)

STT note: uses Google's free public endpoint via SpeechRecognition.recognize_google().
No API key required — same service used by Chrome's voice input.
Requires internet connection for STT; TTS works fully offline.
"""
from __future__ import annotations

import io
import math
import wave

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QPainterPath, QLinearGradient,
)
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QSplitter, QScrollArea, QFrame,
    QSizePolicy, QApplication,
)

# ── State constants ───────────────────────────────────────────────────────────
IDLE      = 0
LISTENING = 1
THINKING  = 2
SPEAKING  = 3

BG     = '#0f1119'
PANEL  = '#1a1d26'
BORDER = '#2a2d3a'
TEAL   = '#4EB3D3'
GREEN  = '#4CAF50'
ORANGE = '#F4A261'
PURPLE = '#9C27B0'
FG     = '#e0e0e0'

STATE_COLORS = {IDLE: TEAL, LISTENING: GREEN, THINKING: ORANGE, SPEAKING: PURPLE}
STATE_LABELS = {IDLE: "Idle", LISTENING: "Listening…", THINKING: "Thinking…", SPEAKING: "Speaking…"}

SUGGESTED = [
    "What is POD and how is it used?",
    "Why does pressure need fewer modes than geometry?",
    "Which parameter affects pressure the most?",
    "How accurate is the RBF surrogate?",
    "What is airway resistance?",
    "How do Sobol indices differ from Pearson?",
    "What is a Digital Twin?",
    "How many nodes are in the mesh?",
]

SAMPLE_RATE = 16_000
REC_SECONDS = 6          # max recording duration per press


# ── Animated avatar widget ────────────────────────────────────────────────────

class AvatarWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(180, 180)
        self._state = IDLE
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(40)   # 25 fps

    def set_state(self, state: int):
        self._state = state

    def _tick(self):
        self._phase = (self._phase + 0.035) % 1.0
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx = self.width() // 2
        cy = self.height() // 2
        R  = min(cx, cy) - 8

        # Dark background disc
        p.setBrush(QBrush(QColor(PANEL)))
        p.setPen(Qt.NoPen)
        p.drawEllipse(cx - R, cy - R, 2 * R, 2 * R)

        col = QColor(STATE_COLORS[self._state])

        if self._state == IDLE:
            self._draw_idle(p, cx, cy, R, col)
        elif self._state == LISTENING:
            self._draw_listening(p, cx, cy, R, col)
        elif self._state == THINKING:
            self._draw_thinking(p, cx, cy, R, col)
        elif self._state == SPEAKING:
            self._draw_speaking(p, cx, cy, R, col)

        p.end()

    # Idle: two concentric rings + "AI" text
    def _draw_idle(self, p, cx, cy, R, col):
        pen = QPen(col, 2)
        p.setPen(pen); p.setBrush(Qt.NoBrush)
        p.drawEllipse(cx - R + 3,      cy - R + 3,      2*(R-3),    2*(R-3))
        p.drawEllipse(cx - R//2,       cy - R//2,       R,          R)
        # Inner filled circle
        p.setBrush(QBrush(col.darker(200))); p.setPen(Qt.NoPen)
        r2 = R // 3
        p.drawEllipse(cx - r2, cy - r2, 2*r2, 2*r2)
        # "AI" label
        f = QFont("Segoe UI", 16, QFont.Bold)
        p.setFont(f); p.setPen(col)
        p.drawText(cx - 14, cy + 7, "AI")

    # Listening: expanding green rings from center
    def _draw_listening(self, p, cx, cy, R, col):
        for offset in (0.0, 0.33, 0.66):
            t   = (self._phase + offset) % 1.0
            r   = int(R * 0.25 + R * 0.75 * t)
            alpha = int(220 * (1.0 - t))
            c2  = QColor(col); c2.setAlpha(alpha)
            p.setPen(QPen(c2, 2)); p.setBrush(Qt.NoBrush)
            p.drawEllipse(cx - r, cy - r, 2*r, 2*r)
        # Core solid disc
        r0 = int(R * 0.28)
        p.setBrush(QBrush(col)); p.setPen(Qt.NoPen)
        p.drawEllipse(cx - r0, cy - r0, 2*r0, 2*r0)
        # Mic text
        f = QFont("Segoe UI Emoji", 20)
        p.setFont(f); p.setPen(QColor('white'))
        p.drawText(cx - 12, cy + 8, "🎙")

    # Thinking: three orbiting dots
    def _draw_thinking(self, p, cx, cy, R, col):
        p.setPen(Qt.NoPen)
        orbit_r = int(R * 0.55)
        dot_r   = int(R * 0.13)
        for i in range(3):
            angle = self._phase * 2 * math.pi + i * (2 * math.pi / 3)
            dx = int(cx + orbit_r * math.cos(angle))
            dy = int(cy + orbit_r * math.sin(angle))
            brightness = int(100 + 155 * ((math.sin(angle + self._phase * math.pi) + 1) / 2))
            c2 = QColor(col)
            c2.setAlpha(brightness)
            p.setBrush(QBrush(c2))
            p.drawEllipse(dx - dot_r, dy - dot_r, 2*dot_r, 2*dot_r)
        # Center dot
        p.setBrush(QBrush(col.darker(150)))
        r0 = int(R * 0.22)
        p.drawEllipse(cx - r0, cy - r0, 2*r0, 2*r0)

    # Speaking: vertical waveform bars
    def _draw_speaking(self, p, cx, cy, R, col):
        n_bars = 7
        bar_w  = max(4, int(R * 1.1 / n_bars) - 2)
        total_w = n_bars * (bar_w + 2)
        x0 = cx - total_w // 2
        for i in range(n_bars):
            t     = self._phase * 2 * math.pi + i * 0.7
            h_frac = 0.25 + 0.65 * abs(math.sin(t))
            bar_h = int(R * h_frac)
            bx = x0 + i * (bar_w + 2)
            by = cy - bar_h // 2
            alpha = 180 + int(75 * abs(math.sin(t)))
            c2 = QColor(col); c2.setAlpha(alpha)
            p.setBrush(QBrush(c2)); p.setPen(Qt.NoPen)
            p.drawRoundedRect(bx, by, bar_w, bar_h, 2, 2)


# ── Background threads ────────────────────────────────────────────────────────

class RecordThread(QThread):
    """Records audio for REC_SECONDS then returns transcription via `done` signal."""
    done = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, duration: int = REC_SECONDS):
        super().__init__()
        self._dur = duration
        self._stopped = False

    def stop_early(self):
        self._stopped = True

    def run(self):
        try:
            import sounddevice as sd
            import speech_recognition as sr

            recording = sd.rec(
                int(self._dur * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='int16',
            )
            sd.wait()

            # Pack into WAV bytes
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(recording.tobytes())
            buf.seek(0)

            recognizer = sr.Recognizer()
            with sr.AudioFile(buf) as source:
                audio = recognizer.record(source)

            try:
                text = recognizer.recognize_google(audio)
                self.done.emit(text)
            except sr.UnknownValueError:
                self.done.emit("")
            except sr.RequestError as exc:
                self.error.emit(f"STT request error: {exc}")

        except ImportError as exc:
            self.error.emit(
                f"Missing package: {exc}\n"
                "Run: pip install sounddevice SpeechRecognition"
            )
        except Exception as exc:
            self.error.emit(str(exc))


class TTSThread(QThread):
    finished = pyqtSignal()

    def __init__(self, text: str):
        super().__init__()
        self._text = text

    def run(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 165)
            engine.setProperty('volume', 0.92)
            engine.say(self._text)
            engine.runAndWait()
        except Exception:
            pass
        self.finished.emit()


class RAGThread(QThread):
    result = pyqtSignal(str, list)   # answer, [(score, doc)]

    def __init__(self, query: str):
        super().__init__()
        self._query = query

    def run(self):
        from core.rag import get_knowledge_base, compose_local_answer, generate_answer, _get_api_key
        kb      = get_knowledge_base()
        results = kb.retrieve(self._query, top_k=3)
        if _get_api_key():
            ctx    = [f"### {d['title']}\n{d['content']}" for _, d in results]
            answer = generate_answer(self._query, ctx) or compose_local_answer(self._query, results, kb)
        else:
            answer = compose_local_answer(self._query, results, kb)
        self.result.emit(answer, results)


# ── Main assistant tab ────────────────────────────────────────────────────────

class AssistantTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state       = IDLE
        self._rec_thread  = None
        self._tts_thread  = None
        self._rag_thread  = None
        self._recording   = False
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        # ── Left panel: avatar + status + quick questions ─────────────────
        left = QWidget()
        left.setFixedWidth(220)
        left.setStyleSheet(f"background:{PANEL}; border-radius:8px;")
        lv = QVBoxLayout(left); lv.setContentsMargins(10, 12, 10, 12); lv.setSpacing(8)

        # Avatar
        self._avatar = AvatarWidget()
        lv.addWidget(self._avatar, alignment=Qt.AlignHCenter)

        # Status
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setAlignment(Qt.AlignCenter)
        self._status_lbl.setStyleSheet(f"color:{TEAL}; font-size:12px; font-weight:bold;")
        lv.addWidget(self._status_lbl)

        lv.addWidget(self._hsep())

        # Suggested questions
        sugg_lbl = QLabel("Suggested questions")
        sugg_lbl.setStyleSheet(f"color:#888; font-size:10px;")
        lv.addWidget(sugg_lbl)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background:transparent;")
        sugg_w = QWidget(); sugg_w.setStyleSheet("background:transparent;")
        sv = QVBoxLayout(sugg_w); sv.setSpacing(4)
        for q in SUGGESTED:
            btn = QPushButton(q)
            btn.setStyleSheet(
                "QPushButton { background:#252836; border:1px solid #2a2d3a; "
                "border-radius:4px; padding:4px 6px; font-size:9px; "
                "color:#ccc; text-align:left; }"
                "QPushButton:hover { border-color:#4EB3D3; color:#4EB3D3; }"
            )
            btn.clicked.connect(lambda _, q=q: self._send_question(q))
            sv.addWidget(btn)
        sv.addStretch()
        scroll.setWidget(sugg_w)
        lv.addWidget(scroll, stretch=1)

        root.addWidget(left)

        # ── Right panel: chat + input ─────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right); rv.setContentsMargins(0, 0, 0, 0); rv.setSpacing(6)

        # Header
        header = QLabel("🫁  Human Airways AI Assistant")
        header.setStyleSheet(
            f"color:{TEAL}; font-size:15px; font-weight:bold; "
            f"padding:6px 0; border-bottom:1px solid {BORDER};"
        )
        rv.addWidget(header)

        cap = QLabel(
            "Ask anything about the project. Offline answers always available. "
            "Voice uses Google's free STT (internet required)."
        )
        cap.setStyleSheet(f"color:#666; font-size:9.5px; padding:2px 0;")
        cap.setWordWrap(True)
        rv.addWidget(cap)

        # Chat history
        self._chat = QTextEdit()
        self._chat.setReadOnly(True)
        self._chat.setStyleSheet(
            f"background:{PANEL}; color:{FG}; border:1px solid {BORDER}; "
            "border-radius:6px; padding:8px; font-size:11px;"
        )
        rv.addWidget(self._chat, stretch=1)

        # Input row
        inp_row = QHBoxLayout(); inp_row.setSpacing(6)
        self._inp = QLineEdit()
        self._inp.setPlaceholderText("Ask a question about the Human Airways project…")
        self._inp.setStyleSheet(
            f"background:{PANEL}; color:{FG}; border:1px solid {BORDER}; "
            "border-radius:4px; padding:6px 10px; font-size:11px;"
        )
        self._inp.returnPressed.connect(self._on_send)
        inp_row.addWidget(self._inp, stretch=1)

        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedWidth(70)
        self._send_btn.setStyleSheet(
            "QPushButton { background:#4EB3D3; color:#000; border-radius:4px; "
            "font-weight:bold; padding:6px; }"
            "QPushButton:hover { background:#5ecde3; }"
            "QPushButton:disabled { background:#2a2d3a; color:#555; }"
        )
        self._send_btn.clicked.connect(self._on_send)
        inp_row.addWidget(self._send_btn)

        self._mic_btn = QPushButton("🎙 Voice")
        self._mic_btn.setFixedWidth(80)
        self._mic_btn.setCheckable(True)
        self._mic_btn.setStyleSheet(
            "QPushButton { background:#252836; border:1px solid #3a3d4e; "
            "border-radius:4px; padding:6px; font-size:11px; }"
            "QPushButton:checked { background:#4CAF50; color:#000; border-color:#4CAF50; }"
            "QPushButton:hover   { border-color:#4EB3D3; }"
            "QPushButton:disabled { background:#1a1d26; color:#444; }"
        )
        self._mic_btn.clicked.connect(self._on_mic_toggle)
        inp_row.addWidget(self._mic_btn)

        self._tts_btn = QPushButton("🔊")
        self._tts_btn.setFixedWidth(36)
        self._tts_btn.setCheckable(True)
        self._tts_btn.setChecked(True)
        self._tts_btn.setToolTip("Toggle text-to-speech")
        self._tts_btn.setStyleSheet(
            "QPushButton { background:#252836; border:1px solid #3a3d4e; "
            "border-radius:4px; padding:6px; }"
            "QPushButton:checked { border-color:#9C27B0; }"
        )
        inp_row.addWidget(self._tts_btn)
        rv.addLayout(inp_row)

        root.addWidget(right, stretch=1)

        # Welcome message
        self._add_msg("assistant",
                      "Hello! I'm your Human Airways Digital Twin assistant. "
                      "Ask me anything about the project — POD, RBF, CFD, anatomy, "
                      "validation, or how to use the dashboard.\n\n"
                      "You can also use the **🎙 Voice** button to speak your question "
                      "(requires internet for speech recognition).")

    # ── Message display ───────────────────────────────────────────────────────

    def _add_msg(self, role: str, text: str):
        if role == "user":
            bg    = "#1e2a3a"
            label = "You"
            color = "#6bc5e8"
        else:
            bg    = "#1a1d26"
            label = "🫁 AI Assistant"
            color = TEAL

        # Convert basic markdown bold (**text**) → <b>text</b>
        import re
        html_text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        html_text = html_text.replace('\n', '<br>')

        block = (
            f'<div style="background:{bg}; border-radius:8px; '
            f'padding:8px 10px; margin:6px 0;">'
            f'<span style="color:{color}; font-size:10px; '
            f'font-weight:bold;">{label}</span><br>'
            f'<span style="color:{FG}; font-size:11px;">{html_text}</span>'
            f'</div>'
        )
        self._chat.append(block)
        self._chat.verticalScrollBar().setValue(
            self._chat.verticalScrollBar().maximum())

    def _add_sources(self, results):
        if not results:
            return
        lines = ["<span style='color:#555; font-size:9px;'>Sources: "]
        for score, doc in results:
            lines.append(f"<b>{doc['title']}</b> ({score:.3f})")
        lines.append("</span>")
        block = f'<div style="padding:2px 10px;">{"  ·  ".join(lines)}</div>'
        self._chat.append(block)

    # ── State management ──────────────────────────────────────────────────────

    def _set_state(self, state: int):
        self._state = state
        self._avatar.set_state(state)
        self._status_lbl.setText(STATE_LABELS[state])
        self._status_lbl.setStyleSheet(
            f"color:{STATE_COLORS[state]}; font-size:12px; font-weight:bold;")

    # ── Send / receive ────────────────────────────────────────────────────────

    def _on_send(self):
        q = self._inp.text().strip()
        if not q:
            return
        self._send_question(q)

    def _send_question(self, question: str):
        if self._state in (THINKING, LISTENING):
            return
        self._inp.clear()
        self._add_msg("user", question)
        self._set_state(THINKING)
        self._send_btn.setEnabled(False)
        self._mic_btn.setEnabled(False)

        self._rag_thread = RAGThread(question)
        self._rag_thread.result.connect(self._on_answer)
        self._rag_thread.start()

    def _on_answer(self, answer: str, results: list):
        self._add_msg("assistant", answer)
        self._add_sources(results)
        self._set_state(IDLE)
        self._send_btn.setEnabled(True)
        self._mic_btn.setEnabled(True)

        if self._tts_btn.isChecked():
            self._speak(answer)

    # ── TTS ───────────────────────────────────────────────────────────────────

    def _speak(self, text: str):
        # Strip markdown and truncate to ~400 chars for reasonable TTS length
        import re
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        clean = re.sub(r'[#\*`]', '', clean)
        if len(clean) > 400:
            clean = clean[:400] + "…"
        self._set_state(SPEAKING)
        self._tts_thread = TTSThread(clean)
        self._tts_thread.finished.connect(lambda: self._set_state(IDLE))
        self._tts_thread.start()

    # ── Voice recording ───────────────────────────────────────────────────────

    def _on_mic_toggle(self, checked: bool):
        if checked:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self._set_state(LISTENING)
        self._rec_thread = RecordThread(REC_SECONDS)
        self._rec_thread.done.connect(self._on_stt_done)
        self._rec_thread.error.connect(self._on_stt_error)
        self._rec_thread.start()
        self._send_btn.setEnabled(False)

    def _stop_recording(self):
        if self._rec_thread and self._rec_thread.isRunning():
            self._rec_thread.stop_early()
        self._set_state(IDLE)
        self._mic_btn.setChecked(False)
        self._send_btn.setEnabled(True)

    def _on_stt_done(self, text: str):
        self._mic_btn.setChecked(False)
        self._send_btn.setEnabled(True)
        if text:
            self._inp.setText(text)
            self._send_question(text)
        else:
            self._set_state(IDLE)
            self._add_msg("assistant",
                          "_Couldn't understand the audio. Please try again or type your question._")

    def _on_stt_error(self, msg: str):
        self._mic_btn.setChecked(False)
        self._send_btn.setEnabled(True)
        self._set_state(IDLE)
        self._add_msg("assistant", f"_Voice error: {msg}_")

    # ── Helper ────────────────────────────────────────────────────────────────

    def _hsep(self):
        line = QFrame(); line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color:{BORDER};")
        return line
