"""Checks for command-line defaults that do not need a camera."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import main


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
