"""Decide when to gently remind someone about an unattended stove.

This file is the reminder clock. It does not open the camera, play a sound,
or write the event log, so it can be tested by handing it times and yes/no
readings. An optional advisor may choose the words and whether to escalate
at a moment the clock already calls due. The advisor is not called on the
waits in between. If it is missing or fails, the built-in ladder speaks.
"""

import re
from collections.abc import Callable
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


# Whole words. A short reminder can still be firm; these are insults.
_INSULTS = frozenset(
    {
        "stupid",
        "idiot",
        "idiotic",
        "careless",
        "carelessly",
        "dumb",
        "foolish",
        "fool",
        "lazy",
        "lazily",
        "shame",
        "shameful",
        "irresponsible",
        "useless",
        "pathetic",
        "ridiculous",
        "incompetent",
        "moron",
        "jerk",
        "worthless",
        "selfish",
        "naughty",
        "clumsy",
        "hopeless",
    }
)

_SCOLD_PHRASES = (
    "what's wrong with you",
    "what is wrong with you",
    "pay attention",
    "your fault",
    "you always forget",
    "you never remember",
)

# A completed contact. The canned ask ("I'm letting your caregiver know")
# does not match: it says what is about to happen, not that someone was
# already reached. The verbs are matched even when "caregiver" is absent.
_DELIVERY = re.compile(
    r"("
    r"\balready\s+(?:been\s+)?(?:told|notified|contacted|called|emailed|texted|alerted|informed|messaged|reached|sent)\b"
    r"|"
    r"\b(?:have|has|had|was|were|i've|we've|they've)\s+(?:already\s+|just\s+)?(?:been\s+)?(?:told|notified|contacted|called|emailed|texted|alerted|informed|messaged|reached|let)\b"
    r"|"
    r"\bi\s+(?:just\s+|already\s+)?(?:told|notified|contacted|called|emailed|texted|alerted|informed|messaged|reached|let)\b"
    r"|"
    r"\b(?:message|text|email|alert|call)\s+(?:was|has been)\s+(?:sent|made|placed)\b"
    r"|"
    r"\bthey\s+(?:already\s+)?know\b"
    r"|"
    r"\b(?:someone|they|family|help)\s+(?:was|were|has been|have been)\s+(?:told|notified|contacted|called|alerted|informed|reached)\b"
    r"|"
    r"\b(?:was|were|been)\s+(?:just\s+|already\s+)?(?:told|notified|contacted|called|emailed|texted|alerted|informed|messaged|reached)\b"
    r"|"
    r"\b(?:told|notified|contacted|called|emailed|texted|alerted|informed)\s+(?:your|the|them|her|him|their)\b"
    r")",
    re.IGNORECASE,
)


def spoken_wording_problem(text: str) -> str | None:
    """Why this sentence must not be said, or None when it is safe.

    Used both when the model reply is parsed and again when an ask is
    turned back into a reminder. A delivery claim is refused even if it
    never says "caregiver". The canned ladder is safe.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    lowered = text.lower()
    tokens = set(re.findall(r"[a-z']+", lowered))
    if tokens & _INSULTS:
        return "model wording was not allowed: it scolds"
    for phrase in _SCOLD_PHRASES:
        if phrase in lowered:
            return "model wording was not allowed: it scolds"
    if _DELIVERY.search(lowered):
        return "model wording was not allowed: it claims someone was already contacted"
    return None


@dataclass(frozen=True)
class Nudge:
    """One reminder the camera loop should say out loud and write to the log."""

    level: int
    message: str
    # calm, direct, or firm. The canned ladder uses calm, then direct, then firm.
    urgency: str = "calm"


@dataclass(frozen=True)
class Escalation:
    """Ask an opted-in caregiver, once, after the spoken reminders run out.

    `message` is what to say in the room. `caregiver_summary` is the text
    for the caregiver. Delivery is a separate step. This object only
    records that the ask is due.
    """

    reminders_unanswered: int
    away_seconds: float
    message: str
    caregiver_summary: str = ""
    urgency: str = "firm"


@dataclass(frozen=True)
class Situation:
    """Facts a model may use at a moment when a reminder is already due.

    These are numbers and short sentences. There is no image in here.
    """

    stove_occupied: bool
    hands_nearby: bool
    away_seconds: float
    reminders_given: int
    max_reminders: int | None
    escalate_allowed: bool
    rule_would: str
    last_message: str | None
    wait_seconds: float
    repeat_seconds: float


@dataclass(frozen=True)
class Advice:
    """A model's choice for one due moment. The rule still applies the caps."""

    action: str
    urgency: str
    message: str
    caregiver_summary: str
    model: str


@dataclass(frozen=True)
class ReasoningResult:
    """Either a model decision or a reason the rules should speak instead."""

    advice: Advice | None = None
    fallback: str | None = None


@dataclass(frozen=True)
class Step:
    """What to do after one check of the stove and the hands.

    `nudge` is set when a reminder is due right now. `escalate` is set once,
    after the configured reminders have gone unanswered. `ended` is
    "returned" or "clear" when a reminder that was already showing has just
    stopped. All three are None when the screen should be left as it is.

    `source` is "rule" for the built-in ladder and "nemotron" when the model
    chose the step. `fallback` is set when the model was asked and could not
    answer, so the ladder spoke instead. `note` records a safety cap that
    changed the model's action.
    """

    nudge: Nudge | None = None
    ended: str | None = None
    escalate: Escalation | None = None
    source: str = "rule"
    model: str | None = None
    fallback: str | None = None
    note: str | None = None


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

    def update(
        self,
        stove_occupied: bool,
        hands_nearby: bool,
        now: float,
        advisor: Callable[[Situation], ReasoningResult] | None = None,
    ) -> Step:
        """Take the latest confirmed readings and say what should happen.

        Pass debounced readings, not a single camera frame. A one-frame miss
        from the hand tracker would otherwise restart the wait.

        `advisor`, when set, is asked only at a moment the clock already
        calls due. It may choose the spoken words and whether to escalate.
        If it returns no advice, or raises, the built-in ladder speaks.
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
        if advisor is not None:
            # The clock already says something should happen. The model may
            # choose the words and whether this is the caregiver ask. A miss
            # keeps the ladder below.
            try:
                result = advisor(self.situation(now, stove_occupied, hands_nearby))
            except Exception as error:
                result = ReasoningResult(
                    fallback=(
                        "Nemotron step failed "
                        f"({type(error).__name__}); using the reminder rules"
                    )
                )
            if not isinstance(result, ReasoningResult):
                result = ReasoningResult(
                    fallback="Nemotron step returned nothing; using the reminder rules"
                )
            if result.advice is not None:
                return self._apply_advice(result.advice, now)
            return self._due_from_rules(now, fallback=result.fallback)
        return self._due_from_rules(now)

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

    def situation(self, now: float, stove_occupied: bool, hands_nearby: bool) -> Situation:
        """The facts available at this moment, for a model or for the log."""
        away = self.unattended_for(now) or 0.0
        if self.max_reminders is not None and self.level >= self.max_reminders:
            rule_would = "escalate"
        else:
            rule_would = "remind"
        return Situation(
            stove_occupied=stove_occupied,
            hands_nearby=hands_nearby,
            away_seconds=away,
            reminders_given=self.level,
            max_reminders=self.max_reminders,
            escalate_allowed=self.max_reminders is not None,
            rule_would=rule_would,
            last_message=self._reminder,
            wait_seconds=self.wait_seconds,
            repeat_seconds=self.repeat_seconds,
        )

    def _due_from_rules(self, now: float, fallback: str | None = None) -> Step:
        if self.max_reminders is not None and self.level >= self.max_reminders:
            return Step(
                escalate=self._escalate(now),
                source="rule",
                fallback=fallback,
            )
        return Step(nudge=self._fire(now), source="rule", fallback=fallback)

    def _apply_advice(self, advice: Advice, now: float) -> Step:
        """Use the model's words, then apply the same safety caps as the rules.

        The room hears a reminder before any caregiver ask. `--no-escalate`
        cannot be overridden. Once the reminder cap is reached, a model that
        wants another reminder still produces the one caregiver ask, and the
        model's summary is kept when it wrote one.
        """
        action = advice.action
        message = advice.message
        summary = advice.caregiver_summary
        note = None
        coerced_to_remind = False
        if action == "escalate" and self.max_reminders is None:
            action = "remind"
            coerced_to_remind = True
            note = "Escalation is turned off, so this stays a reminder."
        elif action == "escalate" and self.level < 1:
            action = "remind"
            coerced_to_remind = True
            note = "The room has not heard a reminder yet, so this stays a reminder."
        if coerced_to_remind and "caregiver" in message.lower():
            # The model wrote an ask, but this moment is still a reminder.
            # Saying the caregiver is being told would be false.
            message = message_for_level(self.level + 1)
        elif (
            action != "escalate"
            and self.max_reminders is not None
            and self.level >= self.max_reminders
        ):
            action = "escalate"
            note = "The reminder cap is reached, so this asks the caregiver."
            if "caregiver" not in message.lower():
                message = ESCALATION_MESSAGE
            if not summary:
                summary = _rule_summary(self.unattended_for(now) or 0.0, self.level)

        # The same check as the parser. A delivery claim or an insult is
        # replaced even when this ask was just turned into a reminder, and
        # even when the sentence never says "caregiver".
        if spoken_wording_problem(message):
            message = (
                ESCALATION_MESSAGE
                if action == "escalate"
                else message_for_level(self.level + 1)
            )
            note = _join_note(note, "The model wording was not spoken.")
        if action == "escalate" and summary and spoken_wording_problem(summary):
            summary = _rule_summary(self.unattended_for(now) or 0.0, self.level)
            note = _join_note(note, "The model summary was not sent.")

        if action == "escalate":
            escalation = self._escalate(
                now,
                message=message,
                summary=summary,
                urgency=advice.urgency or "firm",
            )
            return Step(
                escalate=escalation,
                source="nemotron",
                model=advice.model,
                note=note,
            )
        nudge = self._fire(now, message=message, urgency=advice.urgency or "calm")
        return Step(nudge=nudge, source="nemotron", model=advice.model, note=note)

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

    def _escalate(
        self,
        now: float,
        message: str | None = None,
        summary: str = "",
        urgency: str = "firm",
    ) -> Escalation:
        self._escalated = True
        away = self.unattended_for(now) or 0.0
        spoken = message or ESCALATION_MESSAGE
        self._reminder = spoken
        return Escalation(
            reminders_unanswered=self.level,
            away_seconds=away,
            message=spoken,
            caregiver_summary=summary,
            urgency=urgency,
        )

    def _fire(self, now: float, message: str | None = None, urgency: str | None = None) -> Nudge:
        self.level += 1
        self._last_nudge_at = now
        spoken = message or message_for_level(self.level)
        self._reminder = spoken
        return Nudge(
            level=self.level,
            message=spoken,
            urgency=urgency or _urgency_for_level(self.level),
        )


def _join_note(note: str | None, extra: str) -> str:
    if not note:
        return extra
    if extra in note:
        return note
    return f"{note} {extra}"


def _urgency_for_level(level: int) -> str:
    """The canned ladder gets a little more direct, then stays there."""
    if level <= 1:
        return "calm"
    if level == 2:
        return "direct"
    return "firm"


def _rule_summary(away_seconds: float, reminders: int) -> str:
    """A factual caregiver line used when the model did not write one."""
    reminder_word = "reminder" if reminders == 1 else "reminders"
    return (
        f"The stove has been unattended for {int(away_seconds)} seconds "
        f"after {reminders} {reminder_word}. "
        "The reminder cap required this ask."
    )
