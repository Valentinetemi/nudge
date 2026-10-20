"""Detect whether something is sitting on the stove by comparing it to a
reference picture of the empty stove.

No object detector is needed: anything placed on the stove changes the
pixels in the stove zone, so measuring how much the zone differs from the
empty reference tells us whether it is occupied.
"""

from pathlib import Path

import cv2
import numpy as np

Zone = tuple[int, int, int, int]

# How different (0-255) a pixel must be from the reference to count as
# changed. Lower catches subtler objects but also more camera noise.
PIXEL_DIFF_THRESHOLD = 30

# Percentage of changed pixels above which the stove counts as occupied.
# Watch the on-screen percentage with the stove empty and with a pot on it,
# then pick a value comfortably between the two.
OCCUPIED_PERCENT = 5.0

# Removes isolated changed pixels (sensor noise) while keeping solid blobs.
NOISE_KERNEL = np.ones((5, 5), np.uint8)


def prepare_patch(frame: np.ndarray, zone: Zone) -> np.ndarray:
    """Crop the stove zone and simplify it so frames can be compared fairly."""
    x, y, width, height = zone
    patch = frame[y : y + height, x : x + width]
    # Color adds little here and doubles the noise, so compare brightness only.
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # Blurring hides tiny differences such as sensor grain and slight shake.
    return cv2.GaussianBlur(gray, (21, 21), 0)


def change_percent(reference: np.ndarray, current: np.ndarray) -> float:
    """Return the percentage of pixels that differ clearly from the reference."""
    difference = cv2.absdiff(reference, current)
    _, changed = cv2.threshold(
        difference, PIXEL_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY
    )
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, NOISE_KERNEL)
    return 100.0 * cv2.countNonZero(changed) / changed.size


def load_reference(path: Path, zone: Zone | None) -> np.ndarray | None:
    """Load the saved empty-stove picture if it still fits the stove zone."""
    if zone is None or not path.exists():
        return None
    reference = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    # A size mismatch means the stove zone was redrawn after the picture was
    # taken, so the old picture no longer lines up with the zone.
    if reference is None or reference.shape != (zone[3], zone[2]):
        return None
    return reference


def save_reference(path: Path, reference: np.ndarray) -> None:
    """Store the empty-stove picture so it survives restarts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), reference)


class Debounced:
    """A yes/no signal that only changes after holding steady for a while.

    Without this, one noisy frame would log "placed" and "clear" back to back.
    """

    def __init__(self, hold_seconds: float, initial: bool = False) -> None:
        self.hold_seconds = hold_seconds
        self.value = initial
        self._candidate = initial
        self._candidate_since = 0.0

    def update(self, raw: bool, now: float) -> bool:
        """Feed the latest reading; return True when the confirmed value flips."""
        if raw != self._candidate:
            self._candidate = raw
            self._candidate_since = now
            return False
        if raw != self.value and now - self._candidate_since >= self.hold_seconds:
            self.value = raw
            return True
        return False
