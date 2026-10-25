"""The harness math is tested here. Kitchen rates are not invented."""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import evaluate_stove
import main
import nudge
import stove
from evaluate_stove import ClipResult, count_rates, format_rate


def test_debounce_constant_matches_the_camera_loop() -> None:
    assert evaluate_stove.DEBOUNCE_SEC == main.DEBOUNCE_SEC


def test_default_reminder_matches_the_camera(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate_stove.py"])
    args = evaluate_stove.parse_arguments()
    assert args.remind_after == nudge.DEFAULT_WAIT_SECONDS


def test_rates_are_n_a_when_a_side_has_no_clips() -> None:
    assert format_rate(0, 0, "no occupied clips") == "n/a (no occupied clips)"
    assert "0.0%" not in format_rate(0, 0, "no clear clips")
    counts = count_rates(
        [
            ClipResult("missing.mp4", True, False, False, None, False, "", "file not found"),
        ]
    )
    assert counts["occupied"] == 0
    assert counts["unreadable"] == 1


def test_rates_count_only_scored_clips() -> None:
    results = [
        ClipResult("hit.png", True, False, True, 60.0, True, ""),
        ClipResult("miss.png", True, False, False, None, False, ""),
        ClipResult("alarm.png", False, False, True, 60.0, True, ""),
        ClipResult("ok.png", False, False, False, None, False, ""),
    ]
    counts = count_rates(results)
    assert counts["detected"] == 1 and counts["occupied"] == 2
    assert counts["false_alarms"] == 1 and counts["clear"] == 2
    assert format_rate(counts["detected"], counts["occupied"], "no occupied clips") == "50.0% (1/2)"


def test_empty_labels_do_not_print_a_made_up_rate(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "file,label,hands\n# empty_daylight.mp4,clear,absent\n",
        encoding="utf-8",
    )
    report = evaluate_stove.evaluate(
        _args(tmp_path, labels, tmp_path / "reference.png")
    )
    assert "n/a (no occupied clips)" in report
    assert "n/a (no clear clips)" in report
    assert "No recorded clips were scored" in report
    assert "100.0%" not in report
    assert "0.0%" not in report


def test_synthetic_clips_show_the_lighting_difference(tmp_path: Path) -> None:
    """Drawn pictures, not a kitchen. The numbers belong only to these files."""
    clips = tmp_path / "clips"
    clips.mkdir()
    reference = np.full((160, 200, 3), 100, dtype=np.uint8)
    empty = reference.copy()
    bright = np.full((160, 200, 3), 160, dtype=np.uint8)
    pot = reference.copy()
    pot[50:110, 70:140] = 20
    cv2.imwrite(str(clips / "reference.png"), reference)
    cv2.imwrite(str(clips / "empty.png"), empty)
    cv2.imwrite(str(clips / "bright.png"), bright)
    cv2.imwrite(str(clips / "pot.png"), pot)
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "file,label,hands\n"
        "empty.png,clear,absent\n"
        "bright.png,clear,absent\n"
        "pot.png,occupied,absent\n",
        encoding="utf-8",
    )
    report = evaluate_stove.evaluate(
        _args(clips, labels, clips / "reference.png", remind_after=60)
    )

    off, on = report.split("Lighting normalization on", maxsplit=1)
    assert "detection rate: 100.0% (1/1)" in off
    assert "false-alarm rate: 50.0% (1/2)" in off
    assert "detection rate: 100.0% (1/1)" in on
    assert "false-alarm rate: 0.0% (0/2)" in on
    assert "off false alarms (clear clip called occupied): bright.png" in report
    assert "on false alarms (clear clip called occupied): none" in report
    assert "this is what the camera does" in report
    assert "pot.png:" in report
    # The still is shorter than the reminder delay, so the report says so.
    assert "extrapolated" in report
    assert stove.OCCUPIED_PERCENT == 5.0


def _args(clips: Path, labels: Path, reference: Path, remind_after: float = 60.0):
    return type(
        "Args",
        (),
        {
            "clips": clips,
            "labels": labels,
            "reference": reference,
            "zone": None,
            "remind_after": remind_after,
            "sample_fps": 5.0,
            "out": None,
        },
    )()
