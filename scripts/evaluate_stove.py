"""Score stove occupancy on recorded clips. It does not invent results.

Example:
    python scripts/evaluate_stove.py \
        --clips eval/clips \
        --labels eval/labels.csv \
        --reference eval/clips/reference.png

The labels file is CSV with a header and optional # comments:

    file,label,hands
    empty_daylight.mp4,clear,absent
    pot_on_burner.mp4,occupied,absent

label is occupied or clear. hands is present or absent, and is used only
for time-to-first-nudge. Film every clip, and the empty reference still,
so the stove fills the frame the same way. The picture saved with B in the
live app is already cropped and blurred; use a normal still here instead.

A clip is called occupied only after the same debounced comparison the live
app uses. "Normalization on" is that live comparison. "Normalization off"
is the same pixels with the lighting step skipped, so the two columns can
be compared. The slow empty-picture drift the camera does while a stove
stays clear is not part of these rates. It only happens in a live session.

Time-to-first-nudge uses the same reminder rule as the camera. Short clips
are held forward on that clock and marked extrapolated. That is the
reminder setting, not a measurement of a long video you do not have.

If the labels file has no clip rows, the report says n/a. Those are not zeros.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402  -- imported after the path insert, and only for clips.
import numpy as np  # noqa: E402

import nudge  # noqa: E402
import stove  # noqa: E402

# Same one-second confirmation the camera loop uses before it trusts a change.
DEBOUNCE_SEC = 1.0
# How often a video is sampled. Stills are repeated on this clock so the
# debounce has time to confirm a steady picture.
SAMPLE_FPS = 5.0
# A still has no duration. Repeat it this long so a steady pot can pass
# the one-second debounce, as it would in a short video of the same scene.
STILL_SECONDS = 2.0
# Past the end of a short clip, hold the last reading this long so the
# report can show when the first reminder would be spoken.
NUDGE_HORIZON_SEC = 15 * 60
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class ClipResult:
    """One clip, scored. An error means it was not counted as a hit or a miss."""

    filename: str
    label_occupied: bool | None
    hands_present: bool
    predicted_occupied: bool
    seconds_to_nudge: float | None
    extrapolated: bool
    note: str
    error: str | None = None


def load_labels(path: Path) -> list[tuple[str, str, str]]:
    """Read file,label,hands rows. Comments and a lone header are skipped."""
    if not path.exists():
        raise SystemExit(
            f"No labels file at {path}. Copy eval/labels.example.csv to "
            "eval/labels.csv and add one row per clip."
        )
    rows: list[tuple[str, str, str]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if parts[:2] == ["file", "label"] or parts == ["file", "label", "hands"]:
            continue
        if len(parts) < 2:
            raise SystemExit(f"{path}:{line_number} needs file,label,hands")
        filename, label = parts[0], parts[1].lower()
        hands = parts[2].lower() if len(parts) > 2 and parts[2] else "absent"
        if label not in {"occupied", "clear"}:
            raise SystemExit(f"{path}:{line_number} label must be occupied or clear")
        if hands not in {"present", "absent"}:
            raise SystemExit(f"{path}:{line_number} hands must be present or absent")
        rows.append((filename, label, hands))
    return rows


def load_media(path: Path, sample_fps: float) -> list[tuple[float, np.ndarray]]:
    """Return (time, frame) samples. A still is repeated so debounce can confirm."""
    if path.suffix.lower() in IMAGE_SUFFIXES:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"could not read image {path.name}")
        count = int(STILL_SECONDS * sample_fps) + 1
        return [(index / sample_fps, frame.copy()) for index in range(count)]

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video {path.name}")
    native = capture.get(cv2.CAP_PROP_FPS)
    if not native or native < 1:
        native = 30.0
    step = max(int(round(native / sample_fps)), 1)
    samples: list[tuple[float, np.ndarray]] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            samples.append((index / native, frame))
        index += 1
    capture.release()
    if not samples:
        raise ValueError(f"{path.name} has no readable frames")
    return samples


def prepare_reference(path: Path, zone: stove.Zone | None) -> np.ndarray:
    """Crop and blur the empty still the same way the camera prepares a frame.

    Lighting normalization is applied later, per column, so both columns
    start from this one picture.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read the reference image {path}")
    height, width = image.shape[:2]
    reference_zone = zone if zone is not None else (0, 0, width, height)
    reference = stove.prepare_patch(image, reference_zone)
    if reference.size == 0:
        raise SystemExit("The reference zone is empty. Check --zone.")
    return reference


def score_frames(
    filename: str,
    frames: list[tuple[float, np.ndarray]],
    reference: np.ndarray,
    *,
    label_occupied: bool,
    hands_present: bool,
    zone: stove.Zone | None,
    normalized: bool,
    remind_after_sec: float,
) -> ClipResult:
    """Run one occupancy comparison, then the reminder rule, on one clip."""
    compare = stove.change_percent if normalized else stove.change_percent_raw
    debounced = stove.Debounced(DEBOUNCE_SEC)
    timeline: list[tuple[float, bool]] = []
    for time_sec, frame in frames:
        frame_zone = zone
        if frame_zone is None:
            frame_zone = (0, 0, frame.shape[1], frame.shape[0])
        patch = stove.prepare_patch(frame, frame_zone)
        if patch.shape != reference.shape:
            return ClipResult(
                filename=filename,
                label_occupied=label_occupied,
                hands_present=hands_present,
                predicted_occupied=False,
                seconds_to_nudge=None,
                extrapolated=False,
                note="",
                error=(
                    f"frame size {patch.shape} does not match the reference "
                    f"{reference.shape}; film the stove the same way each time"
                ),
            )
        raw_occupied = compare(reference, patch) >= stove.OCCUPIED_PERCENT
        debounced.update(raw_occupied, time_sec)
        timeline.append((time_sec, debounced.value))

    predicted = bool(timeline and timeline[-1][1])
    seconds, extrapolated, note = _time_to_nudge(
        timeline, hands_present, remind_after_sec
    )
    return ClipResult(
        filename=filename,
        label_occupied=label_occupied,
        hands_present=hands_present,
        predicted_occupied=predicted,
        seconds_to_nudge=seconds,
        extrapolated=extrapolated,
        note=note,
    )


def _time_to_nudge(
    timeline: list[tuple[float, bool]],
    hands_present: bool,
    remind_after_sec: float,
) -> tuple[float | None, bool, str]:
    """Seconds from the first confirmed occupied sample until the first reminder.

    Hands marked present never produce a nudge. A clip that ends first is
    extended by holding the last occupied/clear reading, and marked as such.
    The clock is the same rule the camera uses. The first unattended sample
    starts the wait, and the reminder is spoken when that wait elapses.
    """
    if hands_present:
        return None, False, "hands marked present, so the reminder stays quiet"
    if not timeline:
        return None, False, "no frames"

    rule = nudge.NudgeRule(
        wait_seconds=remind_after_sec,
        repeat_seconds=remind_after_sec,
    )
    confirmed_at: float | None = None
    for time_sec, occupied in timeline:
        if occupied and confirmed_at is None:
            confirmed_at = time_sec
        step = rule.update(stove_occupied=occupied, hands_nearby=False, now=time_sec)
        if step.nudge is not None and confirmed_at is not None:
            return time_sec - confirmed_at, False, "nudge fell inside the clip"

    if confirmed_at is None:
        return None, False, "stove was not confirmed occupied, so there is no nudge"

    last_time, last_occupied = timeline[-1]
    if not last_occupied:
        return None, False, "stove was clear again before a nudge"
    clock = last_time
    limit = last_time + NUDGE_HORIZON_SEC
    while clock < limit:
        clock += 1.0
        step = rule.update(stove_occupied=True, hands_nearby=False, now=clock)
        if step.nudge is not None:
            return clock - confirmed_at, True, (
                "extrapolated by holding the last frame; "
                "the clip is shorter than the reminder delay"
            )
    return None, True, f"no nudge within {int(NUDGE_HORIZON_SEC)} seconds"


def score_folder(
    clips_dir: Path,
    labels: list[tuple[str, str, str]],
    reference: np.ndarray,
    *,
    zone: stove.Zone | None,
    normalized: bool,
    remind_after_sec: float,
    sample_fps: float,
) -> list[ClipResult]:
    results: list[ClipResult] = []
    for filename, label, hands in labels:
        path = clips_dir / filename
        if not path.exists():
            results.append(
                ClipResult(
                    filename=filename,
                    label_occupied=label == "occupied",
                    hands_present=hands == "present",
                    predicted_occupied=False,
                    seconds_to_nudge=None,
                    extrapolated=False,
                    note="",
                    error="file not found",
                )
            )
            continue
        try:
            frames = load_media(path, sample_fps)
        except ValueError as error:
            results.append(
                ClipResult(
                    filename=filename,
                    label_occupied=label == "occupied",
                    hands_present=hands == "present",
                    predicted_occupied=False,
                    seconds_to_nudge=None,
                    extrapolated=False,
                    note="",
                    error=str(error),
                )
            )
            continue
        results.append(
            score_frames(
                filename,
                frames,
                reference,
                label_occupied=label == "occupied",
                hands_present=hands == "present",
                zone=zone,
                normalized=normalized,
                remind_after_sec=remind_after_sec,
            )
        )
    return results


def count_rates(results: list[ClipResult]) -> dict[str, int]:
    """Count clips that were actually scored. Errors are left out of the rates."""
    scored = [item for item in results if item.error is None and item.label_occupied is not None]
    occupied = [item for item in scored if item.label_occupied]
    clear = [item for item in scored if not item.label_occupied]
    return {
        "occupied": len(occupied),
        "detected": sum(1 for item in occupied if item.predicted_occupied),
        "clear": len(clear),
        "false_alarms": sum(1 for item in clear if item.predicted_occupied),
        "unreadable": sum(1 for item in results if item.error is not None),
    }


def format_rate(hits: int, total: int, empty_reason: str) -> str:
    """Percent only when there are clips. An empty set is n/a, not zero."""
    if total == 0:
        return f"n/a ({empty_reason})"
    return f"{hits / total * 100:.1f}% ({hits}/{total})"


def render_report(
    off: list[ClipResult],
    on: list[ClipResult],
    *,
    clips_dir: Path,
    reference: Path,
    remind_after_sec: float,
) -> str:
    """The text the owner copies from. No row means no number."""
    lines = [
        "Stove evaluation",
        f"clips: {clips_dir}",
        f"reference: {reference}",
        f"debounce: {DEBOUNCE_SEC:.0f}s (same confirmation as the live camera)",
        (
            f"reminder delay used for time-to-first-nudge: {remind_after_sec:.0f}s "
            f"(the camera default is {nudge.DEFAULT_WAIT_SECONDS:.0f}s)"
        ),
        "",
        "Normalization on is the comparison the camera makes.",
        "Normalization off skips that lighting step. It is only a comparison.",
        "The live empty-picture drift is not included. It happens only while",
        "the camera is running and the stove looks clear.",
        "",
        "These rates are computed only from the labeled files above.",
        "An empty folder is n/a. It is not a result, and it is not zero.",
        "",
    ]
    lines.extend(
        _mode_section(
            "Lighting normalization off (comparison only; the live camera always normalizes)",
            off,
        )
    )
    lines.append("")
    lines.extend(
        _mode_section(
            "Lighting normalization on (this is what the camera does)",
            on,
        )
    )
    lines.append("")
    lines.append("Time to first nudge")
    if not off and not on:
        lines.append("  no clips")
    else:
        lines.extend(_nudge_lines("off", off))
        lines.extend(_nudge_lines("on", on))
    lines.append("")
    lines.append("Failure cases")
    lines.extend(_failure_lines("off", off))
    lines.extend(_failure_lines("on", on))
    if not any(item.error is None for item in off):
        lines.append("")
        lines.append(
            "No recorded clips were scored. Do not copy a number into the "
            "README table until this report prints one from your own files."
        )
    return "\n".join(lines) + "\n"


def _mode_section(title: str, results: list[ClipResult]) -> list[str]:
    counts = count_rates(results)
    return [
        title,
        f"  occupied clips: {counts['occupied']}",
        f"  clear clips: {counts['clear']}",
        f"  unreadable: {counts['unreadable']}",
        "  detection rate: "
        + format_rate(counts["detected"], counts["occupied"], "no occupied clips"),
        "  false-alarm rate: "
        + format_rate(counts["false_alarms"], counts["clear"], "no clear clips"),
    ]


def _nudge_lines(mode: str, results: list[ClipResult]) -> list[str]:
    if not results:
        return [f"  {mode}: no clips"]
    lines = [f"  {mode}:"]
    for item in results:
        if item.error:
            lines.append(f"    {item.filename}: not scored ({item.error})")
            continue
        if item.seconds_to_nudge is None:
            lines.append(f"    {item.filename}: no nudge ({item.note})")
            continue
        extra = " extrapolated" if item.extrapolated else ""
        lines.append(
            f"    {item.filename}: {item.seconds_to_nudge:.1f}s after the stove "
            f"was confirmed occupied ({item.note}){extra}"
        )
    return lines


def _failure_lines(mode: str, results: list[ClipResult]) -> list[str]:
    misses = [
        item.filename
        for item in results
        if item.error is None and item.label_occupied and not item.predicted_occupied
    ]
    alarms = [
        item.filename
        for item in results
        if item.error is None and item.label_occupied is False and item.predicted_occupied
    ]
    unreadable = [f"{item.filename} ({item.error})" for item in results if item.error]
    return [
        f"  {mode} misses (occupied clip called clear): {_join(misses)}",
        f"  {mode} false alarms (clear clip called occupied): {_join(alarms)}",
        f"  {mode} unreadable: {_join(unreadable)}",
    ]


def _join(names: list[str]) -> str:
    return ", ".join(names) if names else "none"


def parse_zone(value: str) -> stove.Zone:
    """Parse x,y,width,height. The default is the whole frame."""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("zone must be x,y,width,height")
    try:
        x, y, width, height = (int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("zone values must be integers") from error
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("zone must sit on the frame and have a size")
    return x, y, width, height


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score recorded stove clips. Prints n/a when there is nothing to score."
    )
    parser.add_argument("--clips", type=Path, default=ROOT / "eval" / "clips")
    parser.add_argument("--labels", type=Path, default=ROOT / "eval" / "labels.csv")
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "eval" / "clips" / "reference.png",
        help="still photo of the empty stove, same framing as the clips",
    )
    parser.add_argument("--zone", type=parse_zone, default=None)
    parser.add_argument(
        "--remind-after",
        type=float,
        default=nudge.DEFAULT_WAIT_SECONDS,
        help="seconds until the first reminder (default: the camera's 180)",
    )
    parser.add_argument("--sample-fps", type=float, default=SAMPLE_FPS)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="also write the report here (eval/last_results.md is gitignored)",
    )
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> str:
    labels = load_labels(args.labels)
    if not labels:
        return render_report(
            [],
            [],
            clips_dir=args.clips,
            reference=args.reference,
            remind_after_sec=args.remind_after,
        )
    if not args.reference.exists():
        raise SystemExit(
            f"No reference image at {args.reference}. Save a still of the empty "
            "stove in the same framing as the clips."
        )
    if args.remind_after <= 0:
        raise SystemExit("--remind-after must be greater than zero")
    reference = prepare_reference(args.reference, args.zone)
    shared = dict(
        zone=args.zone,
        remind_after_sec=args.remind_after,
        sample_fps=args.sample_fps,
    )
    off = score_folder(
        args.clips, labels, reference, normalized=False, **shared
    )
    on = score_folder(
        args.clips, labels, reference, normalized=True, **shared
    )
    return render_report(
        off,
        on,
        clips_dir=args.clips,
        reference=args.reference,
        remind_after_sec=args.remind_after,
    )


def main() -> None:
    args = parse_arguments()
    report = evaluate(args)
    print(report, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
