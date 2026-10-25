"""Stove checks that draw pictures in memory instead of opening a camera."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import stove


def empty_stove(width: int = 220, height: int = 160) -> np.ndarray:
    """A blurred grate, like the patch the camera loop actually compares."""
    image = np.full((height, width), 60, np.uint8)
    step = max(8, min(height, width) // 8)
    image[step // 2 : height : step, :] = 140
    image[:, step // 2 : width : step] = 140
    cv2.circle(image, (width // 3, height // 2), max(4, min(height, width) // 6), 35, 2)
    cv2.circle(
        image,
        (2 * width // 3, height // 2),
        max(4, min(height, width) // 6),
        35,
        2,
    )
    return cv2.GaussianBlur(image, (21, 21), 0)


def brighten(image: np.ndarray) -> np.ndarray:
    """A whole-frame exposure change, as when a light is switched on."""
    brighter = image.astype(np.float32) * 1.8 + 30
    return np.clip(brighter, 0, 255).astype(np.uint8)


def side_light(image: np.ndarray) -> np.ndarray:
    """Daylight falling more on one side of the stove than the other."""
    width = image.shape[1]
    ramp = np.linspace(0, 100, width, dtype=np.float32)
    lit = image.astype(np.float32) + ramp[np.newaxis, :]
    return np.clip(lit, 0, 255).astype(np.uint8)


def with_pot(image: np.ndarray) -> np.ndarray:
    """A dark round pan sitting on one burner."""
    pot = image.copy()
    height, width = pot.shape
    radius = int(min(height, width) * 0.22)
    cv2.circle(pot, (width // 3, height // 2), radius, 25, -1)
    return cv2.GaussianBlur(pot, (7, 7), 0)


class ChangePercentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.empty = empty_stove()

    def test_an_unchanged_stove_is_clear(self) -> None:
        self.assertEqual(stove.change_percent(self.empty, self.empty), 0.0)

    def test_global_brightness_alone_does_not_look_occupied(self) -> None:
        changed = stove.change_percent(self.empty, brighten(self.empty))
        self.assertLess(changed, 1.0)

    def test_without_normalization_a_light_looks_like_an_object(self) -> None:
        # The raw helper is the evaluation "off" column. The camera does not
        # use it. A brightness change should count there, and not above.
        changed = stove.change_percent_raw(self.empty, brighten(self.empty))
        self.assertGreaterEqual(changed, stove.OCCUPIED_PERCENT)

    def test_a_lighting_gradient_alone_does_not_look_occupied(self) -> None:
        changed = stove.change_percent(self.empty, side_light(self.empty))
        self.assertLess(changed, stove.OCCUPIED_PERCENT)

    def test_a_pot_still_looks_occupied(self) -> None:
        changed = stove.change_percent(self.empty, with_pot(self.empty))
        self.assertGreaterEqual(changed, stove.OCCUPIED_PERCENT)

    def test_a_pot_is_still_found_when_the_room_is_brighter(self) -> None:
        changed = stove.change_percent(self.empty, brighten(with_pot(self.empty)))
        self.assertGreaterEqual(changed, stove.OCCUPIED_PERCENT)

    def test_a_pot_changes_the_picture_much_more_than_the_lights_do(self) -> None:
        lights = stove.change_percent(self.empty, side_light(self.empty))
        pot = stove.change_percent(self.empty, with_pot(self.empty))
        self.assertGreater(pot, lights + stove.OCCUPIED_PERCENT)


class ReferenceAdaptTest(unittest.TestCase):
    def test_reference_learns_only_while_the_stove_is_clear(self) -> None:
        self.assertTrue(stove.should_adapt_reference(False, False))
        self.assertFalse(stove.should_adapt_reference(True, False))
        self.assertFalse(stove.should_adapt_reference(False, True))
        self.assertFalse(stove.should_adapt_reference(True, True))

    def test_one_second_moves_the_picture_only_a_little(self) -> None:
        reference = np.full((8, 8), 100, np.float32)
        current = np.full((8, 8), 200, np.uint8)
        adapted = stove.adapt_reference(reference, current, elapsed_seconds=1.0)
        alpha = 1.0 - math.exp(-1.0 / stove.ADAPT_SECONDS)
        expected = 100 + (200 - 100) * alpha
        self.assertAlmostEqual(float(adapted.mean()), expected, places=3)
        self.assertLess(float(adapted.mean()), 110)

    def test_a_long_pause_does_not_replace_the_picture(self) -> None:
        reference = np.full((8, 8), 0, np.float32)
        current = np.full((8, 8), 255, np.uint8)
        adapted = stove.adapt_reference(reference, current, elapsed_seconds=3600)
        alpha = 1.0 - math.exp(-stove.MAX_ADAPT_STEP_SECONDS / stove.ADAPT_SECONDS)
        self.assertAlmostEqual(float(adapted.mean()), 255 * alpha, places=2)
        self.assertLess(float(adapted.mean()), 30)

    def test_zero_elapsed_leaves_the_picture_unchanged(self) -> None:
        reference = np.full((4, 4), 40, np.float32)
        current = np.zeros((4, 4), np.uint8)
        adapted = stove.adapt_reference(reference, current, elapsed_seconds=0)
        np.testing.assert_allclose(adapted, reference)

    def test_working_copy_is_a_separate_float_picture(self) -> None:
        patch = np.full((4, 4), 7, np.uint8)
        copy = stove.working_copy(patch)
        copy[0, 0] = 0
        self.assertEqual(int(patch[0, 0]), 7)
        self.assertEqual(copy.dtype, np.float32)


class ReferenceFileTest(unittest.TestCase):
    def test_saved_reference_reloads_for_the_same_zone(self) -> None:
        patch = empty_stove(width=12, height=10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stove_reference.png"
            stove.save_reference(path, patch)
            loaded = stove.load_reference(path, (0, 0, 12, 10))
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.shape, (10, 12))
        self.assertEqual(loaded.dtype, np.float32)
        self.assertLess(stove.change_percent(loaded, patch), 1.0)

    def test_reference_is_ignored_when_the_zone_size_changed(self) -> None:
        patch = np.full((10, 12), 80, np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stove_reference.png"
            stove.save_reference(path, patch)
            loaded = stove.load_reference(path, (0, 0, 20, 10))
        self.assertIsNone(loaded)

    def test_prepare_patch_crops_to_the_zone_and_drops_color(self) -> None:
        frame = np.zeros((50, 60, 3), np.uint8)
        frame[10:30, 5:25] = (10, 20, 30)
        patch = stove.prepare_patch(frame, (5, 10, 20, 20))
        self.assertEqual(patch.shape, (20, 20))
        self.assertEqual(patch.ndim, 2)


if __name__ == "__main__":
    unittest.main()
