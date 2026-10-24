"""Checks for command-line defaults that do not need a camera."""

import sys
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

    def test_demo_times_can_be_shortened(self) -> None:
        sys.argv = ["main.py", "--nudge-after", "15", "--nudge-repeat", "10"]
        args = main.parse_arguments()
        self.assertEqual(args.nudge_after, 15.0)
        self.assertEqual(args.nudge_repeat, 10.0)

    def test_a_zero_wait_is_rejected(self) -> None:
        sys.argv = ["main.py", "--nudge-after", "0"]
        with self.assertRaises(SystemExit):
            main.parse_arguments()


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
