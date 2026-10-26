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

# Spoken in the room when the reminders have gone unanswered. This asks a
# person for help. It does not claim the message has already been delivered.
ESCALATION_MESSAGE = (
    "I haven't seen you come back, so I'm letting your caregiver know."
)

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
class Escalation:
    """Ask an opted-in caregiver, once, after the spoken reminders run out.

    `message` is what to say in the room. Delivery to the caregiver is a
    separate step. This object only records that the ask is due.
    """

    reminders_unanswered: int
    away_seconds: float
    message: str


@dataclass(frozen=True)
class Step:
    """What to do after one check of the stove and the hands.

    `nudge` is set when a reminder is due right now. `escalate` is set once,
    after the configured reminders have gone unanswered. `ended` is
    "returned" or "clear" when a reminder that was already showing has just
    stopped. All three are None when the screen should be left as it is.
    """

    nudge: Nudge | None = None
    ended: str | None = None
    escalate: Escalation | None = None


class NudgeRule:
    """Watch the stove and the hands, and decide when a reminder is due.

    Time passes only while something is on the stove and no hands are
    nearby. A person coming back, or the stove becoming clear, forgets the
    reminder. The next time someone walks away, the full wait starts over
    instead of jumping straight to a repeat.

    Pass `max_reminders` to ask a caregiver once those reminders go
    unanswered. Leave it out and the last calm sentence keeps repeating.
    """

    def __init__(
        self,
        wait_seconds: float = DEFAULT_WAIT_SECONDS,
        repeat_seconds: float = DEFAULT_REPEAT_SECONDS,
        max_reminders: int | None = None,
    ) -> None:
        if wait_seconds <= 0 or repeat_seconds <= 0:
            raise ValueError("wait and repeat times must be greater than zero")
        # None keeps repeating the last calm sentence. A number asks a
        # caregiver once the spoken reminders have gone unanswered.
        if max_reminders is not None and max_reminders < 1:
            raise ValueError("max_reminders must be at least 1, or omitted")
        self.wait_seconds = wait_seconds
        self.repeat_seconds = repeat_seconds
        self.max_reminders = max_reminders
        self.level = 0
        self._unattended_since: float | None = None
        self._last_nudge_at: float | None = None
        self._reminder: str | None = None
        self._escalated = False

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

        if self._escalated:
            # One request per absence. Coming back or clearing the stove
            # is what starts a new episode.
            return Step()

        if self.level == 0:
            due = now - self._unattended_since >= self.wait_seconds
        elif self._last_nudge_at is None:
            due = False
        else:
            due = now - self._last_nudge_at >= self.repeat_seconds

        if not due:
            return Step()
        if self.max_reminders is not None and self.level >= self.max_reminders:
            return Step(escalate=self._escalate(now))
        return Step(nudge=self._fire(now))

    def describe(self, now: float, stove_occupied: bool, hands_nearby: bool) -> str:
        """One readable line for the decision log. No image data."""
        away = self.unattended_for(now)
        if away is None:
            away_text = "not unattended"
        else:
            away_text = f"away for {int(away)}s"
        if self._escalated:
            state = "ESCALATE"
        elif self._reminder is not None:
            state = "REMIND"
        else:
            state = "WAIT"
        reminders = (
            "1 reminder unanswered"
            if self.level == 1
            else f"{self.level} reminders unanswered"
        )
        stove = "stove occupied" if stove_occupied else "stove clear"
        hands = "hands nearby" if hands_nearby else "hands away"
        return f"{state:<10} {stove}, {hands}, {away_text}, {reminders}"

    def _stop(self, stove_occupied: bool) -> Step:
        ended = None
        if self._reminder is not None:
            # A clear stove ends the task even if a person is standing there too.
            ended = "clear" if not stove_occupied else "returned"
        self.level = 0
        self._unattended_since = None
        self._last_nudge_at = None
        self._reminder = None
        self._escalated = False
        return Step(ended=ended)

    def _escalate(self, now: float) -> Escalation:
        self._escalated = True
        away = self.unattended_for(now) or 0.0
        self._reminder = ESCALATION_MESSAGE
        return Escalation(
            reminders_unanswered=self.level,
            away_seconds=away,
            message=ESCALATION_MESSAGE,
        )

    def _fire(self, now: float) -> Nudge:
        self.level += 1
        self._last_nudge_at = now
        message = message_for_level(self.level)
        self._reminder = message
        return Nudge(level=self.level, message=message)
