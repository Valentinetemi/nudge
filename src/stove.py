"""Detect whether something is sitting on the stove by comparing it to a
reference picture of the empty stove.

No object detector is needed: anything placed on the stove changes the
pixels in the stove zone, so measuring how much the zone differs from the
empty reference tells us whether it is occupied.

A lamp, a cloud, or the camera brightening the whole picture does that too.
Before comparing, broad lighting is cancelled. While the stove is clearly
empty, the reference also drifts slowly toward what the camera sees now, so
a long afternoon of changing daylight does not look like a new object.
"""

import math
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

# Width of the lighting blur, as a fraction of the longer side of the stove
# zone. A lamp lights the whole zone together. A pot is smaller than this
# blur, so dividing the blur out keeps the pot and drops the lamp.
LIGHTING_SIGMA_FRACTION = 0.12

# How long the empty-stove picture takes to catch up with a clear stove.
# Slow on purpose: daylight can drift in, but a real object would take many
# minutes to fade, and we never adapt while an object might be present.
ADAPT_SECONDS = 60.0

# A paused camera loop must not treat the whole pause as one huge blend.
MAX_ADAPT_STEP_SECONDS = 5.0


def prepare_patch(frame: np.ndarray, zone: Zone) -> np.ndarray:
    """Crop the stove zone and simplify it so frames can be compared fairly."""
    x, y, width, height = zone
    patch = frame[y : y + height, x : x + width]
    # Color adds little here and doubles the noise, so compare brightness only.
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # Blurring hides tiny differences such as sensor grain and slight shake.
    return cv2.GaussianBlur(gray, (21, 21), 0)


def normalize_lighting(patch: np.ndarray) -> np.ndarray:
    """Return `patch` with broad brightness removed.

    Dividing by a heavy blur cancels light that changes slowly across the
    stove, such as a lamp, a window, or the camera brightening every pixel.
    A pot covers less of the zone than that blur, so it stays visible.
    """
    sigma = max(3.0, LIGHTING_SIGMA_FRACTION * max(patch.shape))
    illumination = cv2.GaussianBlur(patch.astype(np.float32), (0, 0), sigmaX=sigma)
    # A black pixel would otherwise divide by zero and blow up.
    illumination = np.maximum(illumination, 1.0)
    flattened = patch.astype(np.float32) / illumination
    mean = float(flattened.mean())
    if mean > 1e-6:
        # Put every frame on the same mid-gray, so two pictures of an empty
        # stove still match after the room gets brighter or darker.
        flattened *= 128.0 / mean
    return np.clip(np.rint(flattened), 0, 255).astype(np.uint8)


def change_percent(reference: np.ndarray, current: np.ndarray) -> float:
    """Return the percentage of pixels that differ clearly from the reference.

    Lighting is normalized first, so a light switched on or off does not by
    itself look like an object was set down. This is what the camera uses.
    """
    return _changed_percent(
        normalize_lighting(reference),
        normalize_lighting(current),
    )


def change_percent_raw(reference: np.ndarray, current: np.ndarray) -> float:
    """Same comparison as `change_percent`, with lighting left in the picture.

    The live camera does not call this. The clip report uses it as the
    "normalization off" column, so a real recording can show the difference.
    """
    return _changed_percent(reference, current)


def _changed_percent(reference: np.ndarray, current: np.ndarray) -> float:
    """Percentage of pixels that differ, after the caller has prepared them."""
    difference = cv2.absdiff(reference, current)
    _, changed = cv2.threshold(
        difference, PIXEL_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY
    )
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, NOISE_KERNEL)
    return 100.0 * cv2.countNonZero(changed) / changed.size


def should_adapt_reference(raw_occupied: bool, confirmed_occupied: bool) -> bool:
    """The empty-stove picture may only learn while the stove is clear.

    `raw_occupied` is this single frame. `confirmed_occupied` has already
    waited out a short flicker. Either one is enough to hold the picture
    still: stop at the first glimpse of an object, and stay stopped until
    that object has been gone long enough to count as clear.
    """
    return not raw_occupied and not confirmed_occupied


def adapt_reference(
    reference: np.ndarray,
    current: np.ndarray,
    elapsed_seconds: float,
) -> np.ndarray:
    """Blend a little of a clear frame into the empty-stove picture.

    Only call this when `should_adapt_reference` is true. The blend is tiny,
    so afternoon daylight eases in, while a pot would take far longer to
    disappear than a reminder waits.
    """
    reference_float = reference.astype(np.float32)
    if elapsed_seconds <= 0:
        return reference_float

    # Cap the step so a frozen window does not leap to the newest frame.
    elapsed_seconds = min(elapsed_seconds, MAX_ADAPT_STEP_SECONDS)
    alpha = 1.0 - math.exp(-elapsed_seconds / ADAPT_SECONDS)
    current_float = current.astype(np.float32)
    return (1.0 - alpha) * reference_float + alpha * current_float


def working_copy(patch: np.ndarray) -> np.ndarray:
    """Float copy so a slow blend can move by less than one brightness level.

    Integer pixels would round that tiny step back to zero, and the empty
    picture would never actually update.
    """
    return np.array(patch, dtype=np.float32, copy=True)


def load_reference(path: Path, zone: Zone | None) -> np.ndarray | None:
    """Load the saved empty-stove picture if it still fits the stove zone."""
    if zone is None or not path.exists():
        return None
    reference = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    # A size mismatch means the stove zone was redrawn after the picture was
    # taken, so the old picture no longer lines up with the zone.
    if reference is None or reference.shape != (zone[3], zone[2]):
        return None
    return working_copy(reference)


def save_reference(path: Path, reference: np.ndarray) -> None:
    """Store the empty-stove picture so it survives restarts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = reference
    # The live picture is floating point so it can drift slowly. A PNG
    # stores whole brightness levels, which is plenty for the next launch.
    if image.dtype != np.uint8:
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), image)


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
