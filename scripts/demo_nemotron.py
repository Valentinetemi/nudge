"""Ask Nemotron for one decision about a sample unattended-stove event.

No camera. This is the call to show in a short demo:

    export NEBIUS_API_KEY=your-token-factory-key
    python scripts/demo_nemotron.py

The default sample is a stove that has been occupied with nobody nearby
for 240 seconds, after two spoken reminders. Pass --away and --reminders
to try a different moment. If the key is missing, or Token Factory does
not answer, the script prints the rule fallback and exits 1.

The key is read from the environment or from a gitignored .env file.
It is not printed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cloud  # noqa: E402  -- src/ is added to the path just above.
import nudge  # noqa: E402
import reason  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask Nemotron what Nudge should do about one sample event."
    )
    parser.add_argument(
        "--away",
        type=float,
        default=240,
        help="seconds the stove has been unattended (default: 240)",
    )
    parser.add_argument(
        "--reminders",
        type=int,
        default=2,
        help="reminders already spoken (default: 2)",
    )
    parser.add_argument(
        "--max-reminders",
        type=int,
        default=4,
        help="reminder cap before the rules require a caregiver ask (default: 4)",
    )
    return parser.parse_args()


def sample_situation(args: argparse.Namespace) -> nudge.Situation:
    """One text event. Nothing here is a frame."""
    if args.reminders < 0 or args.max_reminders < 1 or args.away < 0:
        raise SystemExit("--away must be >= 0, --reminders >= 0, and --max-reminders >= 1")
    last = nudge.message_for_level(args.reminders) if args.reminders else None
    rule_would = "escalate" if args.reminders >= args.max_reminders else "remind"
    return nudge.Situation(
        stove_occupied=True,
        hands_nearby=False,
        away_seconds=args.away,
        reminders_given=args.reminders,
        max_reminders=args.max_reminders,
        escalate_allowed=True,
        rule_would=rule_would,
        last_message=last,
        wait_seconds=nudge.DEFAULT_WAIT_SECONDS,
        repeat_seconds=nudge.DEFAULT_REPEAT_SECONDS,
    )


def main() -> int:
    args = parse_arguments()
    cloud.load_env_file(ROOT / ".env")
    client = reason.NebiusReasoner.from_env()
    facts = sample_situation(args)

    print("Nudge sample event")
    print("  stove occupied, hands away")
    print(f"  unattended for {int(facts.away_seconds)} seconds")
    print(f"  reminders already spoken: {facts.reminders_given}")
    print(f"  maximum reminders: {facts.max_reminders}")
    print(f"  the built-in rules would: {facts.rule_would}")
    print()
    print(client.describe())

    result = client.advise(facts)
    print()
    if result.advice is None:
        print("Fallback")
        print(f"  {result.fallback}")
        print("  The camera would speak the built-in reminder ladder.")
        return 1

    advice = result.advice
    print("Nemotron decision")
    print(f"  model: {advice.model}")
    print(f"  action: {advice.action}")
    print(f"  urgency: {advice.urgency}")
    print(f"  spoken: {advice.message}")
    if advice.caregiver_summary:
        print(f"  caregiver summary: {advice.caregiver_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
