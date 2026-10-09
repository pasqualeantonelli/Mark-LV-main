"""CPU speech recognition and eSpeak NG synthesis; no runtime downloads."""
import io
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd


class OfflineSpeech:
    def __init__(self, model_path):
        from faster_whisper import WhisperModel
        if not Path(model_path).is_dir():
            raise RuntimeError("Whisper model is missing; follow LOCAL_MODE.md to download it first")
        self.model = WhisperModel(str(model_path), device="cpu", compute_type="int8",
                                  cpu_threads=2, local_files_only=True)
        self.espeak = shutil.which("espeak-ng")
        if not self.espeak:
            raise RuntimeError("Local speech output requires espeak-ng")

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(audio, beam_size=1,
                                            condition_on_previous_text=False, vad_filter=True)
        return " ".join(segment.text for segment in segments).strip()

    def speak(self, text, cancelled=lambda: False):
        result = subprocess.run([self.espeak, "--stdout", "--stdin"], input=text.encode(),
                                capture_output=True, check=True, timeout=60)
        if cancelled():
            return
        with wave.open(io.BytesIO(result.stdout)) as wav:
            rate = wav.getframerate()
            # eSpeak's streamed WAV uses an oversized header; read available bytes.
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        from core.audio_devices import resolve
        from memory.config_manager import get_output_device
        sd.play(audio, rate, device=resolve(get_output_device(), "output"))
        sd.wait()

    @staticmethod
    def interrupt():
        sd.stop()
