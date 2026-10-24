"""The spoken reminder, with the voice and the speaker replaced by fakes."""

import io
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import speak


class VoiceCommandTest(unittest.TestCase):
    def test_macos_uses_the_built_in_say_command(self) -> None:
        def which(name: str) -> str | None:
            return "/usr/bin/say" if name == "say" else None

        with patch("speak.sys.platform", "darwin"), patch("speak.shutil.which", which):
            command = speak.voice_command("Something is still on the stove.")
        self.assertIsNotNone(command)
        assert command is not None
        self.assertEqual(command[:3], ["say", "-r", str(speak.SPEAK_RATE)])
        self.assertEqual(command[-1], "Something is still on the stove.")

    def test_linux_prefers_espeak_ng(self) -> None:
        def which(name: str) -> str | None:
            return {
                "espeak-ng": "/usr/bin/espeak-ng",
                "espeak": "/usr/bin/espeak",
            }.get(name)

        with patch("speak.sys.platform", "linux"), patch("speak.shutil.which", which):
            command = speak.voice_command("Hello")
        self.assertIsNotNone(command)
        assert command is not None
        self.assertEqual(command[0], "/usr/bin/espeak-ng")
        self.assertEqual(command[-1], "Hello")

    def test_windows_speech_escapes_quotes(self) -> None:
        def which(name: str) -> str | None:
            return "powershell" if name == "powershell" else None

        with patch("speak.sys.platform", "win32"), patch("speak.shutil.which", which):
            command = speak.voice_command("It's time to check the stove.")
        self.assertIsNotNone(command)
        assert command is not None
        self.assertEqual(command[0], "powershell")
        self.assertIn("It''s time to check the stove.", command[-1])

    def test_no_voice_tool_means_no_command(self) -> None:
        with (
            patch("speak.sys.platform", "linux"),
            patch("speak.shutil.which", lambda _name: None),
        ):
            self.assertIsNone(speak.voice_command("Hello"))


class SpeakWithVoiceTest(unittest.TestCase):
    def test_a_missing_voice_is_a_failure(self) -> None:
        with patch("speak.voice_command", return_value=None):
            self.assertFalse(speak.speak_with_voice("Hello"))

    def test_a_voice_error_is_a_failure(self) -> None:
        with (
            patch("speak.voice_command", return_value=["say", "Hello"]),
            patch("speak.subprocess.run", side_effect=speak.subprocess.TimeoutExpired("say", 30)),
        ):
            self.assertFalse(speak.speak_with_voice("Hello"))

    def test_a_nonzero_exit_is_a_failure(self) -> None:
        completed = speak.subprocess.CompletedProcess(["say"], returncode=1)
        with (
            patch("speak.voice_command", return_value=["say", "Hello"]),
            patch("speak.subprocess.run", return_value=completed),
        ):
            self.assertFalse(speak.speak_with_voice("Hello"))


class DeliverTest(unittest.TestCase):
    def test_a_voice_is_preferred_over_a_tone(self) -> None:
        with (
            patch("speak.speak_with_voice", return_value=True) as voice,
            patch("speak.play_chime") as chime,
        ):
            self.assertEqual(speak.deliver("Hello"), "voice")
        voice.assert_called_once_with("Hello")
        chime.assert_not_called()

    def test_a_tone_is_used_when_there_is_no_voice(self) -> None:
        with (
            patch("speak.speak_with_voice", return_value=False),
            patch("speak.play_chime", return_value=True) as chime,
        ):
            self.assertEqual(speak.deliver("Hello"), "tone")
        chime.assert_called_once_with()

    def test_the_console_is_used_when_nothing_can_play(self) -> None:
        with (
            patch("speak.speak_with_voice", return_value=False),
            patch("speak.play_chime", return_value=False),
            patch("sys.stdout", new_callable=io.StringIO) as out,
        ):
            method = speak.deliver("Something is still on the stove.")
        self.assertEqual(method, "console")
        self.assertIn("Something is still on the stove.", out.getvalue())


class ChimeTest(unittest.TestCase):
    def test_the_chime_is_short_and_quiet(self) -> None:
        samples = speak.chime_samples()
        loudest = float(np.max(np.abs(samples)))
        self.assertLess(samples.shape[0], speak.SAMPLE_RATE)
        self.assertGreater(loudest, 0.01)
        # 0.2 is not exact in float32, so allow a tiny rounding step.
        self.assertLessEqual(loudest, speak.TONE_VOLUME + 1e-6)


class SpeakerTest(unittest.TestCase):
    def test_say_returns_while_the_phrase_is_still_going(self) -> None:
        started = threading.Event()
        release = threading.Event()
        heard: list[str] = []

        def slow(message: str) -> str:
            started.set()
            release.wait(2)
            heard.append(message)
            return "voice"

        speaker = speak.Speaker(deliver_message=slow)
        began = time.monotonic()
        speaker.say("Hello from Nudge")
        self.assertTrue(started.wait(1))
        self.assertLess(time.monotonic() - began, 0.5)
        release.set()
        speaker.wait()
        self.assertEqual(heard, ["Hello from Nudge"])

    def test_a_second_phrase_waits_instead_of_overlapping(self) -> None:
        order: list[tuple[str, str]] = []
        first_started = threading.Event()
        release_first = threading.Event()

        def fake(message: str) -> str:
            order.append(("start", message))
            if message == "one":
                first_started.set()
                release_first.wait(2)
            order.append(("end", message))
            return "console"

        speaker = speak.Speaker(deliver_message=fake)
        speaker.say("one")
        self.assertTrue(first_started.wait(1))
        speaker.say("two")
        release_first.set()
        speaker.wait()
        self.assertLess(order.index(("end", "one")), order.index(("start", "two")))
        self.assertEqual(
            [item for item in order if item[0] == "start"],
            [("start", "one"), ("start", "two")],
        )


if __name__ == "__main__":
    unittest.main()
