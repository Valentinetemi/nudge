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


if __name__ == "__main__":
    unittest.main()
