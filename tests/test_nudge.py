"""The nudge rule, fed by hand. No camera, clock, or speaker is used."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import nudge


def rule(wait: float = 10, repeat: float = 5) -> nudge.NudgeRule:
    """Short times so a test can walk the whole story in a few numbers."""
    return nudge.NudgeRule(wait_seconds=wait, repeat_seconds=repeat)


class NudgeRuleTest(unittest.TestCase):
    def test_a_clear_stove_never_nudges(self) -> None:
        checker = rule()
        for moment in range(0, 100, 10):
            step = checker.update(stove_occupied=False, hands_nearby=False, now=moment)
            self.assertIsNone(step.nudge)
            self.assertIsNone(checker.reminder)

    def test_hands_nearby_keep_an_occupied_stove_quiet(self) -> None:
        checker = rule()
        for moment in range(0, 100, 10):
            step = checker.update(stove_occupied=True, hands_nearby=True, now=moment)
            self.assertIsNone(step.nudge)
            self.assertIsNone(checker.unattended_for(moment))

    def test_nothing_is_said_before_the_wait_is_up(self) -> None:
        checker = rule(wait=10, repeat=5)
        self.assertIsNone(checker.update(True, False, 0).nudge)
        step = checker.update(True, False, 9.9)
        self.assertIsNone(step.nudge)
        self.assertIsNone(checker.reminder)
        self.assertAlmostEqual(checker.unattended_for(9.9) or 0, 9.9)

    def test_the_first_nudge_fires_when_the_wait_elapses(self) -> None:
        checker = rule(wait=10, repeat=5)
        checker.update(True, False, 0)
        step = checker.update(True, False, 10)
        self.assertIsNotNone(step.nudge)
        assert step.nudge is not None
        self.assertEqual(step.nudge.level, 1)
        self.assertEqual(step.nudge.message, nudge.MESSAGES[0])
        self.assertEqual(checker.reminder, nudge.MESSAGES[0])
        self.assertIsNone(step.ended)

    def test_the_wait_starts_when_hands_leave_not_when_the_pot_is_set_down(self) -> None:
        checker = rule(wait=10, repeat=5)
        # Someone is cooking, so the occupied stove is not unattended yet.
        checker.update(True, True, 0)
        checker.update(True, True, 100)
        checker.update(True, False, 100)
        self.assertIsNone(checker.update(True, False, 109).nudge)
        step = checker.update(True, False, 110)
        self.assertIsNotNone(step.nudge)
        assert step.nudge is not None
        self.assertEqual(step.nudge.level, 1)

    def test_a_repeat_waits_and_then_uses_a_more_direct_wording(self) -> None:
        checker = rule(wait=10, repeat=5)
        checker.update(True, False, 0)
        first = checker.update(True, False, 10).nudge
        self.assertIsNone(checker.update(True, False, 14).nudge)
        second = checker.update(True, False, 15).nudge
        assert first is not None and second is not None
        self.assertEqual(second.level, 2)
        self.assertNotEqual(second.message, first.message)
        self.assertEqual(checker.reminder, second.message)

    def test_wording_gets_more_direct_and_then_stays_kind(self) -> None:
        checker = rule(wait=10, repeat=5)
        checker.update(True, False, 0)
        seen = []
        moment = 10
        for _ in range(6):
            step = checker.update(True, False, moment)
            assert step.nudge is not None
            seen.append(step.nudge.message)
            moment += 5
        self.assertEqual(seen[0], nudge.MESSAGES[0])
        self.assertEqual(seen[1], nudge.MESSAGES[1])
        self.assertEqual(seen[2], nudge.MESSAGES[2])
        self.assertEqual(seen[3], nudge.MESSAGES[3])
        # Further repeats do not become harsher.
        self.assertEqual(seen[4], nudge.MESSAGES[3])
        self.assertEqual(seen[5], nudge.MESSAGES[3])

    def test_a_long_gap_fires_one_nudge_not_a_burst(self) -> None:
        checker = rule(wait=10, repeat=5)
        checker.update(True, False, 0)
        jumped = checker.update(True, False, 10_000)
        assert jumped.nudge is not None
        self.assertEqual(jumped.nudge.level, 1)
        self.assertIsNone(checker.update(True, False, 10_000).nudge)

    def test_coming_back_stops_the_reminder_and_restarts_the_full_wait(self) -> None:
        checker = rule(wait=10, repeat=5)
        checker.update(True, False, 0)
        checker.update(True, False, 10)
        back = checker.update(True, True, 12)
        self.assertEqual(back.ended, "returned")
        self.assertIsNone(back.nudge)
        self.assertIsNone(checker.reminder)
        self.assertEqual(checker.level, 0)
        # They leave again. A repeat must not fire early.
        checker.update(True, False, 12)
        self.assertIsNone(checker.update(True, False, 21).nudge)
        again = checker.update(True, False, 22).nudge
        assert again is not None
        self.assertEqual(again.level, 1)

    def test_clearing_the_stove_stops_the_reminder(self) -> None:
        checker = rule(wait=10, repeat=5)
        checker.update(True, False, 0)
        checker.update(True, False, 10)
        cleared = checker.update(False, False, 11)
        self.assertEqual(cleared.ended, "clear")
        self.assertIsNone(checker.reminder)
        self.assertIsNone(checker.unattended_for(11))

    def test_a_clear_stove_wins_when_hands_are_there_too(self) -> None:
        checker = rule(wait=10, repeat=5)
        checker.update(True, False, 0)
        checker.update(True, False, 10)
        step = checker.update(False, True, 11)
        self.assertEqual(step.ended, "clear")

    def test_leaving_before_a_nudge_does_not_log_an_ending(self) -> None:
        checker = rule(wait=10, repeat=5)
        checker.update(True, False, 0)
        step = checker.update(False, False, 4)
        self.assertIsNone(step.ended)
        self.assertIsNone(step.nudge)

    def test_the_reminder_stays_up_between_repeats(self) -> None:
        checker = rule(wait=10, repeat=5)
        checker.update(True, False, 0)
        checker.update(True, False, 10)
        checker.update(True, False, 12)
        self.assertEqual(checker.reminder, nudge.MESSAGES[0])

    def test_times_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            nudge.NudgeRule(wait_seconds=0, repeat_seconds=5)
        with self.assertRaises(ValueError):
            nudge.NudgeRule(wait_seconds=5, repeat_seconds=-1)

    def test_level_zero_is_not_a_message(self) -> None:
        with self.assertRaises(ValueError):
            nudge.message_for_level(0)

    def test_ended_messages_are_sentences(self) -> None:
        self.assertIn("came back", nudge.ended_message("returned"))
        self.assertIn("clear", nudge.ended_message("clear"))
        with self.assertRaises(ValueError):
            nudge.ended_message("nope")

    def test_unanswered_reminders_escalate_once(self) -> None:
        checker = nudge.NudgeRule(wait_seconds=10, repeat_seconds=5, max_reminders=2)
        checker.update(True, False, 0)
        first = checker.update(True, False, 10)
        self.assertIsNotNone(first.nudge)
        self.assertIsNone(first.escalate)
        second = checker.update(True, False, 15)
        assert second.nudge is not None
        self.assertEqual(second.nudge.level, 2)
        ask = checker.update(True, False, 20)
        self.assertIsNone(ask.nudge)
        assert ask.escalate is not None
        self.assertEqual(ask.escalate.reminders_unanswered, 2)
        self.assertIn("caregiver", ask.escalate.message)
        self.assertNotIn("already notified", ask.escalate.message)
        # Still unattended. Do not ask again on the next interval.
        later = checker.update(True, False, 40)
        self.assertIsNone(later.nudge)
        self.assertIsNone(later.escalate)
        self.assertEqual(checker.reminder, nudge.ESCALATION_MESSAGE)

    def test_coming_back_after_an_escalation_starts_a_new_wait(self) -> None:
        checker = nudge.NudgeRule(wait_seconds=10, repeat_seconds=5, max_reminders=1)
        checker.update(True, False, 0)
        checker.update(True, False, 10)
        ask = checker.update(True, False, 15)
        self.assertIsNotNone(ask.escalate)
        back = checker.update(True, True, 16)
        self.assertEqual(back.ended, "returned")
        self.assertIsNone(checker.reminder)
        checker.update(True, False, 16)
        self.assertIsNone(checker.update(True, False, 25).nudge)
        again = checker.update(True, False, 26)
        assert again.nudge is not None
        self.assertEqual(again.nudge.level, 1)
        self.assertIsNone(again.escalate)

    def test_the_decision_trace_names_the_camera_facts(self) -> None:
        checker = nudge.NudgeRule(wait_seconds=10, repeat_seconds=5, max_reminders=2)
        checker.update(True, False, 0)
        checker.update(True, False, 10)
        line = checker.describe(10, stove_occupied=True, hands_nearby=False)
        self.assertTrue(line.startswith("REMIND"))
        self.assertIn("stove occupied", line)
        self.assertIn("hands away", line)
        self.assertIn("away for 10s", line)
        self.assertIn("1 reminder unanswered", line)

    def test_without_a_limit_the_rule_does_not_escalate(self) -> None:
        checker = rule(wait=10, repeat=5)
        checker.update(True, False, 0)
        moment = 10
        for _ in range(6):
            step = checker.update(True, False, moment)
            self.assertIsNotNone(step.nudge)
            self.assertIsNone(step.escalate)
            moment += 5

    def test_max_reminders_must_be_at_least_one(self) -> None:
        with self.assertRaises(ValueError):
            nudge.NudgeRule(wait_seconds=10, repeat_seconds=5, max_reminders=0)

    def test_the_model_is_not_asked_before_the_wait_is_up(self) -> None:
        checker = rule(wait=10, repeat=5)
        asked: list[nudge.Situation] = []

        def advisor(facts: nudge.Situation) -> nudge.ReasoningResult:
            asked.append(facts)
            return nudge.ReasoningResult(fallback="should not be asked yet")

        checker.update(True, False, 0, advisor=advisor)
        checker.update(True, False, 9, advisor=advisor)
        self.assertEqual(asked, [])

    def test_the_model_can_choose_the_reminder_wording(self) -> None:
        checker = rule(wait=10, repeat=5)
        checker.update(True, False, 0)
        step = checker.update(True, False, 10, advisor=_remind("The pot is still on the stove."))
        assert step.nudge is not None
        self.assertEqual(step.nudge.message, "The pot is still on the stove.")
        self.assertEqual(step.nudge.urgency, "direct")
        self.assertEqual(step.source, "nemotron")
        self.assertEqual(step.model, "nvidia/nemotron-3-super-120b-a12b")
        self.assertIsNone(step.fallback)
        self.assertEqual(checker.reminder, step.nudge.message)

    def test_the_model_can_escalate_after_a_reminder(self) -> None:
        checker = nudge.NudgeRule(wait_seconds=10, repeat_seconds=5, max_reminders=4)
        checker.update(True, False, 0)
        first = checker.update(True, False, 10, advisor=_remind("Please glance at the stove."))
        self.assertIsNotNone(first.nudge)
        step = checker.update(True, False, 15, advisor=_escalate())
        self.assertIsNone(step.nudge)
        assert step.escalate is not None
        self.assertEqual(step.source, "nemotron")
        self.assertEqual(step.escalate.reminders_unanswered, 1)
        self.assertIn("240", step.escalate.caregiver_summary)
        self.assertIn("caregiver", step.escalate.message)
        self.assertNotIn("already notified", step.escalate.message)
        # One ask. The next interval does not ask again.
        later = checker.update(True, False, 30, advisor=_escalate())
        self.assertIsNone(later.escalate)
        self.assertIsNone(later.nudge)

    def test_the_first_moment_stays_a_reminder_even_if_the_model_escalates(self) -> None:
        checker = nudge.NudgeRule(wait_seconds=10, repeat_seconds=5, max_reminders=4)
        checker.update(True, False, 0)
        step = checker.update(True, False, 10, advisor=_escalate())
        assert step.nudge is not None
        self.assertIsNone(step.escalate)
        self.assertEqual(step.source, "nemotron")
        self.assertIn("not heard a reminder", step.note or "")
        self.assertEqual(step.nudge.level, 1)
        self.assertEqual(step.nudge.message, nudge.MESSAGES[0])
        self.assertNotIn("caregiver", step.nudge.message)

    def test_the_reminder_cap_still_asks_the_caregiver(self) -> None:
        checker = nudge.NudgeRule(wait_seconds=10, repeat_seconds=5, max_reminders=1)
        checker.update(True, False, 0)
        checker.update(True, False, 10, advisor=_remind("Something is still on the stove."))
        step = checker.update(True, False, 15, advisor=_remind("One more look at the stove."))
        self.assertIsNone(step.nudge)
        assert step.escalate is not None
        self.assertEqual(step.source, "nemotron")
        self.assertIn("cap", step.note or "")
        self.assertIn("caregiver", step.escalate.message)
        self.assertTrue(step.escalate.caregiver_summary)

    def test_escalation_stays_off_when_the_flag_says_so(self) -> None:
        checker = rule(wait=10, repeat=5)
        checker.update(True, False, 0)
        checker.update(True, False, 10, advisor=_remind("Something is still on the stove."))
        step = checker.update(True, False, 15, advisor=_escalate())
        assert step.nudge is not None
        self.assertIsNone(step.escalate)
        self.assertIn("turned off", step.note or "")
        self.assertNotIn("caregiver", step.nudge.message)

    def test_a_model_miss_uses_the_canned_reminder_and_records_why(self) -> None:
        checker = rule(wait=10, repeat=5)

        def advisor(facts: nudge.Situation) -> nudge.ReasoningResult:
            return nudge.ReasoningResult(fallback="Nebius call timed out; using the reminder rules")

        checker.update(True, False, 0)
        step = checker.update(True, False, 10, advisor=advisor)
        assert step.nudge is not None
        self.assertEqual(step.nudge.message, nudge.MESSAGES[0])
        self.assertEqual(step.source, "rule")
        self.assertIn("timed out", step.fallback or "")

    def test_an_advisor_that_raises_falls_back(self) -> None:
        checker = rule(wait=10, repeat=5)

        def advisor(facts: nudge.Situation) -> nudge.ReasoningResult:
            raise RuntimeError("connection reset")

        checker.update(True, False, 0)
        step = checker.update(True, False, 10, advisor=advisor)
        assert step.nudge is not None
        self.assertEqual(step.nudge.message, nudge.MESSAGES[0])
        self.assertIn("RuntimeError", step.fallback or "")
        self.assertNotIn("connection reset", step.fallback or "")


def _remind(message: str):
    def advisor(facts: nudge.Situation) -> nudge.ReasoningResult:
        return nudge.ReasoningResult(
            advice=nudge.Advice(
                action="remind",
                urgency="direct",
                message=message,
                caregiver_summary="",
                model="nvidia/nemotron-3-super-120b-a12b",
            )
        )

    return advisor


def _escalate():
    def advisor(facts: nudge.Situation) -> nudge.ReasoningResult:
        return nudge.ReasoningResult(
            advice=nudge.Advice(
                action="escalate",
                urgency="firm",
                message="I haven't seen you come back, so I'm letting your caregiver know.",
                caregiver_summary=(
                    "The stove has been unattended for 240 seconds after 2 reminders."
                ),
                model="nvidia/nemotron-3-super-120b-a12b",
            )
        )

    return advisor


if __name__ == "__main__":
    unittest.main()
