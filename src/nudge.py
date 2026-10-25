"""Decide when to gently remind someone about an unattended stove.

This file is only the rule. It does not open the camera, play a sound, or
write the event log, so it can be tested by handing it times and yes/no
readings.
"""

from dataclasses import dataclass

# A few minutes is long enough to step away for a spoon, and short enough
# that a forgotten burner is not left alone all afternoon.
DEFAULT_WAIT_SECONDS = 180.0

# Later reminders come sooner than the first wait, and they stay polite.
DEFAULT_REPEAT_SECONDS = 60.0

# Each step is a little more direct, then we stay on the last wording.
# These sentences may be read aloud to someone who is confused, so they
# never scold.
MESSAGES = (
    "Something is still on the stove.",
    "Reminder: the stove still needs a look.",
    "Please check the stove. It has been left for a while.",
    "If you can, have someone check the stove with you.",
)


def message_for_level(level: int) -> str:
    """Calm wording for reminder number `level`. The first reminder is 1."""
    if level < 1:
        raise ValueError("nudge level starts at 1")
    return MESSAGES[min(level, len(MESSAGES)) - 1]


def ended_message(reason: str) -> str:
    """Plain sentence for why a reminder stopped."""
    if reason == "returned":
        return "Someone came back, so the reminder stopped."
    if reason == "clear":
        return "The stove is clear, so the reminder stopped."
    raise ValueError(f"unknown reason {reason!r}")


@dataclass(frozen=True)
class Nudge:
    """One reminder the camera loop should say out loud and write to the log."""

    level: int
    message: str


@dataclass(frozen=True)
class Step:
    """What to do after one check of the stove and the hands.

    `nudge` is set when a reminder is due right now. `ended` is "returned"
    or "clear" when a reminder that was already showing has just stopped.
    Both are None when the screen should be left as it is.
    """

    nudge: Nudge | None = None
    ended: str | None = None


class NudgeRule:
    """Watch the stove and the hands, and decide when a reminder is due.

    Time passes only while something is on the stove and no hands are
    nearby. A person coming back, or the stove becoming clear, forgets the
    reminder. The next time someone walks away, the full wait starts over
    instead of jumping straight to a repeat.
    """

    def __init__(
        self,
        wait_seconds: float = DEFAULT_WAIT_SECONDS,
        repeat_seconds: float = DEFAULT_REPEAT_SECONDS,
    ) -> None:
        if wait_seconds <= 0 or repeat_seconds <= 0:
            raise ValueError("wait and repeat times must be greater than zero")
        self.wait_seconds = wait_seconds
        self.repeat_seconds = repeat_seconds
        self.level = 0
        self._unattended_since: float | None = None
        self._last_nudge_at: float | None = None
        self._reminder: str | None = None

    @property
    def reminder(self) -> str | None:
        """Words to keep on screen, or None when there is nothing to say."""
        return self._reminder

    def unattended_for(self, now: float) -> float | None:
        """How long the stove has sat occupied with nobody nearby.

        None when the stove is clear or someone is there. The count starts
        from the first unattended reading, not from when the object was put
        down, so standing at the stove does not use up the wait.
        """
        if self._unattended_since is None:
            return None
        return max(0.0, now - self._unattended_since)

    def update(self, stove_occupied: bool, hands_nearby: bool, now: float) -> Step:
        """Take the latest confirmed readings and say what should happen.

        Pass debounced readings, not a single camera frame. A one-frame miss
        from the hand tracker would otherwise restart the wait.
        """
        if not stove_occupied or hands_nearby:
            return self._stop(stove_occupied)

        if self._unattended_since is None:
            self._unattended_since = now
            return Step()

        if self.level == 0:
            due = now - self._unattended_since >= self.wait_seconds
        elif self._last_nudge_at is None:
            due = False
        else:
            due = now - self._last_nudge_at >= self.repeat_seconds

        if not due:
            return Step()
        return Step(nudge=self._fire(now))

    def _stop(self, stove_occupied: bool) -> Step:
        ended = None
        if self._reminder is not None:
            # A clear stove ends the task even if a person is standing there too.
            ended = "clear" if not stove_occupied else "returned"
        self.level = 0
        self._unattended_since = None
        self._last_nudge_at = None
        self._reminder = None
        return Step(ended=ended)

    def _fire(self, now: float) -> Nudge:
        self.level += 1
        self._last_nudge_at = now
        message = message_for_level(self.level)
        self._reminder = message
        return Nudge(level=self.level, message=message)
