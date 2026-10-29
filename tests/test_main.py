"""Checks for command-line defaults that do not need a camera."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import main


class NudgeArgumentTest(unittest.TestCase):
    def test_the_first_reminder_waits_a_few_minutes(self) -> None:
        sys.argv = ["main.py"]
        args = main.parse_arguments()
        self.assertEqual(args.nudge_after, 180)
        self.assertEqual(args.nudge_repeat, 60)
        self.assertEqual(args.max_reminders, 4)
        self.assertFalse(args.no_escalate)

    def test_demo_times_can_be_shortened(self) -> None:
        sys.argv = ["main.py", "--nudge-after", "15", "--nudge-repeat", "10"]
        args = main.parse_arguments()
        self.assertEqual(args.nudge_after, 15.0)
        self.assertEqual(args.nudge_repeat, 10.0)

    def test_a_zero_wait_is_rejected(self) -> None:
        sys.argv = ["main.py", "--nudge-after", "0"]
        with (
            self.assertRaises(SystemExit),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            main.parse_arguments()

    def test_escalation_can_be_turned_off(self) -> None:
        sys.argv = ["main.py", "--no-escalate", "--max-reminders", "2"]
        args = main.parse_arguments()
        self.assertTrue(args.no_escalate)
        self.assertEqual(args.max_reminders, 2)

    def test_a_zero_reminder_limit_is_rejected(self) -> None:
        sys.argv = ["main.py", "--max-reminders", "0"]
        with (
            self.assertRaises(SystemExit),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            main.parse_arguments()


class DecisionLogTest(unittest.TestCase):
    def test_a_decision_line_is_one_json_object(self) -> None:
        original_dir = main.DATA_DIR
        original_file = main.DECISIONS_FILE
        try:
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                main.DATA_DIR = folder
                main.DECISIONS_FILE = folder / "decisions.jsonl"
                main.log_decision(
                    "2026-10-25T12:00:00",
                    "12:00:00  WAIT       stove occupied, hands away, away for 4s, 0 reminders unanswered",
                )
                line = main.DECISIONS_FILE.read_text(encoding="utf-8").strip()
        finally:
            main.DATA_DIR = original_dir
            main.DECISIONS_FILE = original_file
        record = json.loads(line)
        self.assertEqual(record["time"], "2026-10-25T12:00:00")
        self.assertIn("stove occupied", record["trace"])
        self.assertNotIn("frame", line)

    def test_a_model_decision_is_stored_without_a_key(self) -> None:
        original_dir = main.DATA_DIR
        original_file = main.DECISIONS_FILE
        try:
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                main.DATA_DIR = folder
                main.DECISIONS_FILE = folder / "decisions.jsonl"
                step = main.nudge.Step(
                    nudge=main.nudge.Nudge(
                        level=1,
                        message="The pot is still on the stove.",
                        urgency="calm",
                    ),
                    source="nemotron",
                    model="nvidia/nemotron-3-super-120b-a12b",
                )
                main.log_decision(
                    "2026-10-27T12:00:00",
                    "12:00:00  REMIND     stove occupied, hands away, away for 180s, 1 reminder unanswered",
                    **main.decision_log_fields(step),
                )
                line = main.DECISIONS_FILE.read_text(encoding="utf-8").strip()
        finally:
            main.DATA_DIR = original_dir
            main.DECISIONS_FILE = original_file
        record = json.loads(line)
        self.assertEqual(record["source"], "nemotron")
        self.assertEqual(record["model"], "nvidia/nemotron-3-super-120b-a12b")
        self.assertEqual(record["action"], "remind")
        self.assertEqual(record["urgency"], "calm")
        self.assertEqual(record["message"], "The pot is still on the stove.")
        self.assertNotIn("NEBIUS_API_KEY", line)
        self.assertNotIn("frame", line)


class CameraArgumentTest(unittest.TestCase):
    def test_cam_defaults_to_the_built_in_camera(self) -> None:
        sys.argv = ["main.py"]
        args = main.parse_arguments()
        self.assertEqual(args.cam, 0)

    def test_cam_flag_overrides_the_default(self) -> None:
        sys.argv = ["main.py", "--cam", "2"]
        args = main.parse_arguments()
        self.assertEqual(args.cam, 2)


if __name__ == "__main__":
    unittest.main()
