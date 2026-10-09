"""Local text/voice entry point, sharing Mark-LV's HUD, memory and safety gates."""
import argparse
import os
import queue
import threading
from pathlib import Path

from core.local_backend import LocalChat
from core import confirm, undo

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", action="store_true", help="Text terminal without HUD or microphone")
    parser.add_argument("--voice", action="store_true", help="Offline Whisper + eSpeak voice with HUD")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--speech-model", type=Path, default=ROOT / ".local-data" / "whisper-base")
    args = parser.parse_args()
    if args.cli and args.voice:
        parser.error("--voice uses the HUD; omit --cli")
    chat = LocalChat(args.url, args.model)
    try:
        chat.check()
    except Exception as exc:
        parser.exit(1, f"Ollama unavailable: {exc}\nStart Ollama and pull {args.model}; see LOCAL_MODE.md\n")
    if args.cli:
        print("Local Mark-LV. /quit exits; /undo undoes. Irreversible actions require the HUD.")
        while True:
            try:
                text = input("You: ").strip()
                if text == "/quit":
                    return
                if text:
                    print("JARVIS:", undo.undo_last() if text == "/undo" else chat.reply(text))
            except (EOFError, KeyboardInterrupt):
                return
            except Exception as exc:
                print(f"ERR: {exc}")
    os.environ["MARK_LV_LOCAL"] = "1"
    from ui import JarvisUI
    ui = JarvisUI(str(ROOT / "face.png"))
    # The shared HUD includes cloud controls; mark unsupported controls explicitly.
    from PyQt6.QtWidgets import QPushButton
    unsupported = ("REMOTE CONTROL", "DESKTOP SHORTCUT", "AUTO-START", "CUSTOMISE",
                   "PLUGIN", "WAKE", "PUSH-TO-TALK", "BRIEF")
    for button in ui._win.findChildren(QPushButton):
        if any(label in button.text().upper() for label in unsupported):
            button.setEnabled(False)
            button.setToolTip("Unavailable in local mode; see LOCAL_MODE.md")
    ui._win._drop_zone.setEnabled(False)
    ui._win._file_hint.setText("Attachments unavailable in local mode")
    confirm.bind(ui.show_confirm, ui.hide_confirm, ui.write_log)
    pending = queue.Queue(maxsize=4)
    stop = threading.Event()
    cancelled = threading.Event()
    busy = threading.Event()
    speech = None
    if args.voice:
        try:
            from core.local_speech import OfflineSpeech
            speech = OfflineSpeech(args.speech_model)
        except Exception as exc:
            ui.write_log(f"ERR: Voice unavailable: {exc}. Text mode remains available.")
    if not speech:
        ui._win._mute_btn.setEnabled(False)
        ui._win._mute_btn.setText("TEXT MODE — MICROPHONE OFF")
    ui.on_audio_device_change = lambda: ui.write_log("SYS: Speaker selection applies next reply; restart to change microphone.")

    def submit(text):
        if text.strip():
            try:
                pending.put_nowait(text.strip())
            except queue.Full:
                ui.write_log("WRN: Request queue full; wait for the current reply.")

    def interrupt():
        cancelled.set()
        if speech:
            speech.interrupt()
        while True:
            try:
                pending.get_nowait()
            except queue.Empty:
                break

    ui.on_text_command = submit
    ui.on_interrupt = interrupt

    def worker():
        while not stop.is_set():
            try:
                text = pending.get(timeout=0.2)
            except queue.Empty:
                continue
            busy.set()
            cancelled.clear()
            ui.write_log(f"YOU: {text}")
            ui.set_state("THINKING")
            try:
                reply = undo.undo_last() if text == "/undo" else chat.reply(text, cancelled.is_set)
                if reply and not cancelled.is_set():
                    ui.write_log(f"JARVIS: {reply}")
                    if speech:
                        ui.set_state("SPEAKING")
                        speech.speak(reply, cancelled.is_set)
            except Exception as exc:
                ui.write_log(f"ERR: Local request failed: {exc}")
            finally:
                busy.clear()
                ui.set_state("LISTENING")

    def microphone():
        import numpy as np
        import sounddevice as sd
        from core.audio_devices import resolve
        from memory.config_manager import get_input_device
        # Half-duplex: audio is discarded during model replies and playback.
        frames, quiet, count = [], 0, 0
        try:
            with sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                                blocksize=1600, device=resolve(get_input_device(), "input")) as stream:
                while not stop.is_set():
                    audio, overflow = stream.read(1600)
                    if busy.is_set() or ui.muted:
                        frames, quiet, count = [], 0, 0
                        continue
                    block = audio[:, 0].copy()
                    level = float(np.sqrt(np.mean(block * block)))
                    ui.set_audio_level(min(1.0, level * 15))
                    if level > 0.012 or frames:
                        frames.append(block)
                        count += 1
                        quiet = quiet + 1 if level < 0.012 else 0
                        if quiet >= 8 or count >= 150:
                            if count - quiet >= 3:
                                text = speech.transcribe(np.concatenate(frames))
                                if text and not ui.muted and not stop.is_set():
                                    submit(text)
                            frames, quiet, count = [], 0, 0
        except Exception as exc:
            ui.write_log(f"ERR: Microphone unavailable: {exc}. Use text input.")

    threading.Thread(target=worker, daemon=True, name="local-chat").start()
    if speech:
        threading.Thread(target=microphone, daemon=True, name="local-microphone").start()
    ui.write_log(f"SYS: LOCAL Ollama ({args.model}). Tools: status, memory, notes, undo, web search/read.")
    ui.write_log("SYS: Web tools use internet search and public pages; AI responses stay local.")
    ui.write_log("SYS: Vision, browser, plugins, phone dashboard and Gemini voice settings unavailable locally.")
    ui.set_state("LISTENING")
    try:
        ui.root.mainloop()
    finally:
        stop.set()
        interrupt()
        confirm.resolve(False)


if __name__ == "__main__":
    main()
