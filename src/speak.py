"""Say a short reminder out loud without stalling the camera.

Nothing in this file is sent over the network. The order is:

1. The computer's own offline voice: macOS `say`, espeak, or Windows speech.
2. A soft two-note tone through sounddevice, when no voice is available.
3. A line on the console, when audio itself is unavailable.

`Speaker.say` returns immediately. A background thread does the speaking,
so the camera loop keeps drawing frames.
"""

import math
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable

import numpy as np

# A little slower than a typical assistant, so the reminder stays calm.
SPEAK_RATE = 150
SAMPLE_RATE = 22050

# Quiet on purpose. The tone should tap the shoulder, not startle anyone.
TONE_VOLUME = 0.2


def voice_command(message: str) -> list[str] | None:
    """A local command that speaks `message`, or None when this OS has no voice."""
    if sys.platform == "darwin" and shutil.which("say"):
        return ["say", "-r", str(SPEAK_RATE), message]

    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if espeak:
        return [espeak, "-s", str(SPEAK_RATE), message]

    if sys.platform == "win32" and shutil.which("powershell"):
        # Single-quoted PowerShell string. Double any quote so a sentence
        # cannot break out of the command.
        safe = message.replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$voice.Speak('{safe}')"
        )
        return ["powershell", "-NoProfile", "-Command", script]

    return None


def speak_with_voice(message: str) -> bool:
    """Try the operating system's voice. False if it is missing or it fails."""
    command = voice_command(message)
    if command is None:
        return False
    try:
        completed = subprocess.run(
            command,
            check=False,
            timeout=30,
            capture_output=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def chime_samples(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Two quiet notes a third apart, faded so they are not a harsh beep."""

    def tone(frequency: float, seconds: float) -> np.ndarray:
        count = int(sample_rate * seconds)
        times = np.arange(count, dtype=np.float32) / sample_rate
        wave = np.sin(2 * math.pi * frequency * times).astype(np.float32)
        fade = min(int(sample_rate * 0.02), max(count // 2, 1))
        envelope = np.ones(count, dtype=np.float32)
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        envelope[:fade] = ramp
        envelope[-fade:] = ramp[::-1]
        return wave * envelope

    gap = np.zeros(int(sample_rate * 0.05), dtype=np.float32)
    # G4, then B4. A small rising third sounds like a reminder, not an alarm.
    notes = (tone(392.0, 0.18), gap, tone(494.0, 0.28))
    chime = np.concatenate(notes) * TONE_VOLUME
    # float rounding can peek just past the volume we asked for.
    return np.clip(chime, -TONE_VOLUME, TONE_VOLUME)


def play_chime() -> bool:
    """Play the soft tone. False when there is no audio library or no speaker."""
    try:
        import sounddevice as sd
    except (ImportError, OSError):
        return False
    try:
        sd.play(chime_samples(), SAMPLE_RATE)
        sd.wait()
    except (OSError, RuntimeError, ValueError, sd.PortAudioError):
        # A missing speaker or a busy device should fall through to the console.
        return False
    return True


def deliver(message: str) -> str:
    """Present one reminder now. Return "voice", "tone", or "console".

    The camera loop should not call this directly. `Speaker.say` runs it
    on a background thread so the picture keeps moving.
    """
    if speak_with_voice(message):
        return "voice"
    if play_chime():
        return "tone"
    print(f"Nudge (no audio): {message}")
    return "console"


class Speaker:
    """Speak reminders one at a time, away from the camera thread.

    A new reminder that arrives while one is still playing waits, then
    speaks. Reminders are not played on top of each other.
    """

    def __init__(self, deliver_message: Callable[[str], str] | None = None) -> None:
        self._deliver = deliver if deliver_message is None else deliver_message
        self._lock = threading.Lock()
        self._pending: str | None = None
        self._running = False
        self._worker: threading.Thread | None = None

    def say(self, message: str) -> None:
        """Ask for `message` to be spoken. This returns immediately."""
        with self._lock:
            # Keep only the newest sentence. A repeat that arrives while the
            # previous one is still going replaces anything still waiting.
            self._pending = message
            if self._running:
                return
            self._running = True
            self._worker = threading.Thread(
                target=self._run,
                name="nudge-voice",
                daemon=True,
            )
            self._worker.start()

    def wait(self, timeout: float | None = 5.0) -> None:
        """Block until the phrase being spoken has finished. Tests use this."""
        with self._lock:
            worker = self._worker
        if worker is not None:
            worker.join(timeout)

    def _run(self) -> None:
        while True:
            with self._lock:
                message = self._pending
                self._pending = None
                if message is None:
                    self._running = False
                    return
            try:
                self._deliver(message)
            except Exception as error:  # noqa: BLE001  audio must not stop the camera
                print(f"Nudge (audio failed): {message} ({error})")
