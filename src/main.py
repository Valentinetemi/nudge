"""Watch a stove and gently remind someone if they walk away.

Example:
    python src/main.py
    python src/main.py --nudge-after 15

Press Z to draw or replace the hand zone, S to draw or replace the stove
zone, B to photograph the empty stove (the reference used to notice items
placed on it), and Q to quit. Named zones are stored in data/zones.json so
they can be reused later.

A reminder is spoken when the stove still looks occupied and no hands have
been in the hand zone or on the stove for a few minutes. It repeats a little
more directly. After those reminders go unanswered, Nudge asks a caregiver
once. Coming back or clearing the stove stops the episode. Every check is
also written to data/decisions.jsonl so the reason can be read later.
"""

import argparse
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cloud
import nudge
import reason

# Resolve files from the repository root instead of the terminal's current
# directory. This lets the program work when launched from somewhere else.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
ZONES_FILE = DATA_DIR / "zones.json"
LOG_FILE = DATA_DIR / "events.jsonl"
# One JSON line per check of the rule, including the waits. Separate from
# events.jsonl so the activity log stays short and the decision trace stays
# complete.
DECISIONS_FILE = DATA_DIR / "decisions.jsonl"
STOVE_REFERENCE_FILE = DATA_DIR / "stove_reference.png"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
DEBOUNCE_SEC = 1.0

# Set by main() once the cloud mode is known. None only before startup,
# and in tests that never open a camera.
active_cloud: cloud.CloudClient | None = None

# The stove always has this fixed name so later steps can find it in zones.json.
STOVE_ZONE = "stove"
ORANGE = (0, 165, 255)  # OpenCV colors are BGR, not RGB.

# An OpenCV rectangle contains x, y, width, and height.
Zone = tuple[int, int, int, int]


def parse_zone_name(value: str) -> str:
    """Validate and normalize a zone name supplied on the command line."""
    name = value.strip()
    if not name:
        raise argparse.ArgumentTypeError("zone name cannot be empty")
    if len(name) > 64:
        raise argparse.ArgumentTypeError("zone name must be 64 characters or fewer")
    return name


def parse_max_reminders(value: str) -> int:
    """How many spoken reminders happen before one caregiver request."""
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "max reminders must be a whole number"
        ) from error
    if count < 1:
        raise argparse.ArgumentTypeError("max reminders must be at least 1")
    return count


def parse_positive_seconds(value: str) -> float:
    """Parse a number of seconds that must be greater than zero."""
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("time must be a number of seconds") from error
    if seconds <= 0:
        raise argparse.ArgumentTypeError("time must be greater than zero")
    return seconds


def parse_arguments() -> argparse.Namespace:
    """Read camera, zone, and reminder timing from command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Watch a stove and gently remind someone who walks away."
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=0,
        help="camera index to open (default: 0)",
    )
    parser.add_argument(
        "--zone",
        type=parse_zone_name,
        default="counter",
        help="zone to load or create (default: counter)",
    )
    parser.add_argument(
        "--nudge-after",
        type=parse_positive_seconds,
        default=nudge.DEFAULT_WAIT_SECONDS,
        help=(
            "seconds with nobody nearby before the first reminder "
            "(default: 180). Use 15 for a short demo."
        ),
    )
    parser.add_argument(
        "--nudge-repeat",
        type=parse_positive_seconds,
        default=nudge.DEFAULT_REPEAT_SECONDS,
        help="seconds between later reminders if nobody returns (default: 60)",
    )
    parser.add_argument(
        "--max-reminders",
        type=parse_max_reminders,
        default=len(nudge.MESSAGES),
        help=(
            "spoken reminders before one caregiver request "
            f"(default: {len(nudge.MESSAGES)})"
        ),
    )
    parser.add_argument(
        "--no-escalate",
        action="store_true",
        help="keep repeating the last reminder and do not ask a caregiver",
    )
    parser.add_argument(
        "--list-zones",
        action="store_true",
        help="print saved zone names and exit",
    )
    return parser.parse_args()


def ensure_model() -> None:
    """Download MediaPipe's hand-landmarker model when it is not available."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading hand model (about 8 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def validate_zone(name: str, value: object) -> Zone:
    """Convert one JSON value into a valid OpenCV zone rectangle."""
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f'zone "{name}" must contain [x, y, width, height]')
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f'zone "{name}" coordinates must all be integers')

    x, y, width, height = value
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(
            f'zone "{name}" must have non-negative coordinates and a positive size'
        )
    return x, y, width, height


def load_zones() -> dict[str, Zone]:
    """Load every saved zone, keyed by its human-readable name."""
    if not ZONES_FILE.exists():
        return {}

    try:
        raw_zones = json.loads(ZONES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read {ZONES_FILE}: {error}") from error

    if not isinstance(raw_zones, dict):
        raise SystemExit(f"{ZONES_FILE} must contain a JSON object")

    try:
        return {
            parse_zone_name(name): validate_zone(name, value)
            for name, value in raw_zones.items()
            if isinstance(name, str)
        }
    except (argparse.ArgumentTypeError, ValueError) as error:
        raise SystemExit(f"Invalid zone data in {ZONES_FILE}: {error}") from error


def save_zones(zones: dict[str, Zone]) -> None:
    """Persist all named zones with an atomic file replacement."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {name: list(zone) for name, zone in sorted(zones.items())}
    temporary_file = ZONES_FILE.with_suffix(".json.tmp")
    temporary_file.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_file.replace(ZONES_FILE)


def log_event(
    event: str,
    observed: str,
    recent: list[str],
    details: dict | None = None,
) -> None:
    """Append one machine-readable event and update the on-screen history.

    The local file always gets the line. When a cloud client is active it
    receives the same words, plus any caregiver numbers in `details`. Frames
    are not a field this function accepts.
    """
    entry = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "observed": observed,
    }
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as log:
        log.write(json.dumps(entry) + "\n")

    line = f'{entry["time"][11:]}  {observed}'
    print(line)
    recent.append(line)
    del recent[:-6]  # Keep the display readable by showing only six events.

    if active_cloud is not None:
        payload = dict(entry)
        if details:
            payload.update(details)
        active_cloud.send(payload)


def log_decision(when: str, trace: str, **fields: object) -> None:
    """Append one readable decision. `trace` is already a sentence of facts.

    Extra fields are for a model decision or a fallback. Empty values are
    left out so a plain wait line stays a time and a trace. The API key is
    not a field this function accepts.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {"time": when, "trace": trace}
    for key, value in fields.items():
        if value is None or value == "":
            continue
        record[key] = value
    with DECISIONS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def model_event_fields(step: nudge.Step) -> dict[str, str]:
    """Text fields a caregiver event may carry about who chose the step.

    Empty values are omitted. The API key is not one of these fields.
    """
    fields: dict[str, str] = {}
    if step.model:
        fields["model"] = step.model
    if step.fallback:
        fields["source"] = "rule"
        fields["fallback"] = step.fallback
    elif step.source == "nemotron":
        fields["source"] = "nemotron"
    if step.note:
        fields["note"] = step.note
    if step.escalate is not None and step.escalate.caregiver_summary:
        fields["caregiver_summary"] = step.escalate.caregiver_summary
    return fields


def decision_log_fields(step: nudge.Step) -> dict[str, str]:
    """Fields to store beside the trace when a model was involved.

    A plain rule step with no fallback adds nothing, so older log lines
    keep their shape. A Nemotron choice records the action, the urgency,
    the spoken line, and the caregiver summary. A fallback records why
    the rules spoke instead.
    """
    fields: dict[str, str] = {}
    if step.fallback:
        fields["source"] = "rule"
        fields["fallback"] = step.fallback
        return fields
    if step.source != "nemotron":
        return fields
    fields["source"] = "nemotron"
    if step.model:
        fields["model"] = step.model
    if step.note:
        fields["note"] = step.note
    if step.nudge is not None:
        fields["action"] = "remind"
        fields["urgency"] = step.nudge.urgency
        fields["message"] = step.nudge.message
    elif step.escalate is not None:
        fields["action"] = "escalate"
        fields["urgency"] = step.escalate.urgency
        fields["message"] = step.escalate.message
        if step.escalate.caregiver_summary:
            fields["caregiver_summary"] = step.escalate.caregiver_summary
    return fields


def count_hands_in_zone(
    result: object,
    zone: Zone | None,
    frame_width: int,
    frame_height: int,
) -> int:
    """Count hands that have at least one landmark inside the selected zone."""
    if zone is None:
        return 0

    x, y, zone_width, zone_height = zone
    count = 0
    for hand in result.hand_landmarks:
        for landmark in hand:
            # MediaPipe coordinates range from 0 to 1, so convert them to pixels.
            pixel_x = landmark.x * frame_width
            pixel_y = landmark.y * frame_height
            if x <= pixel_x <= x + zone_width and y <= pixel_y <= y + zone_height:
                count += 1
                break  # Count each hand once, even if several landmarks are inside.
    return count


def select_zone(frame: object) -> Zone | None:
    """Let the user drag a rectangle on the frame; None if they cancel."""
    import cv2

    x, y, width, height = (
        int(value) for value in cv2.selectROI("Nudge", frame, showCrosshair=False)
    )
    # selectROI returns a zero-sized rectangle when the user presses Esc or C.
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def describe_duration(seconds: float) -> str:
    """Format elapsed seconds as a short, human-readable duration."""
    whole_seconds = max(0, int(seconds))
    if whole_seconds < 60:
        return f"{whole_seconds} seconds"
    minutes, remaining_seconds = divmod(whole_seconds, 60)
    if remaining_seconds == 0:
        return f"{minutes} min"
    return f"{minutes} min {remaining_seconds} sec"


def draw_reminder(frame: object, message: str) -> None:
    """Paint the current reminder in a warm bar across the top of the picture.

    The bar is clay colored, not red. This is a tap on the shoulder, not an alarm.
    """
    import cv2

    _height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 56), (45, 85, 130), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    # Shrink the sentence until it fits a narrow camera picture.
    while scale > 0.45:
        (text_width, _text_height), _baseline = cv2.getTextSize(
            message, font, scale, thickness
        )
        if text_width <= width - 32:
            break
        scale -= 0.05
    cv2.putText(
        frame,
        message,
        (16, 36),
        font,
        scale,
        (255, 255, 255),
        thickness,
    )


def print_saved_zones(zones: dict[str, Zone]) -> None:
    """Print saved names without requiring a camera or MediaPipe model."""
    if not zones:
        print("No zones have been saved yet.")
        return

    print("Saved zones:")
    for name in sorted(zones):
        print(f"- {name}: {zones[name]}")


def main() -> None:
    """Run the camera loop for the selected named zone."""
    args = parse_arguments()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zones = load_zones()

    if args.list_zones:
        print_saved_zones(zones)
        return

    if args.zone == STOVE_ZONE:
        # The stove is drawn with S and watched for objects, not hands.
        raise SystemExit(f'"{STOVE_ZONE}" is reserved for the stove; pick another --zone')

    # Load the large camera libraries only when tracking is about to start.
    # Commands such as --list-zones can then run without camera dependencies.
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    import speak
    import stove

    zone_name = args.zone
    zone = zones.get(zone_name)
    ensure_model()

    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    camera = cv2.VideoCapture(args.cam)

    if not camera.isOpened():
        landmarker.close()
        raise SystemExit(
            f"Camera {args.cam} didn't open. Try a different --cam number."
        )

    # Debouncing prevents a brief missed detection from creating false events.
    confirmed_inside = False
    candidate_inside = False
    candidate_since = time.monotonic()
    arrived_at: float | None = None
    hands_seen = 0
    recent_events: list[str] = []
    started_at = time.monotonic()

    # The empty-stove picture that every frame is compared against. While
    # the stove stays clear it drifts slowly, so daylight does not pile up
    # into a false "something was placed" event.
    stove_reference = stove.load_reference(
        STOVE_REFERENCE_FILE, zones.get(STOVE_ZONE)
    )
    stove_occupied = stove.Debounced(DEBOUNCE_SEC)
    last_reference_adapt = time.monotonic()
    # Hands count as nearby in the named zone or over the stove. The same
    # one-second debounce as the event log, so a missed frame does not
    # restart the reminder clock.
    hands_nearby = stove.Debounced(DEBOUNCE_SEC)
    # Omitting the limit keeps the original behavior: the last reminder
    # repeats for as long as the stove stays unattended.
    max_reminders = None if args.no_escalate else args.max_reminders
    reminders = nudge.NudgeRule(
        args.nudge_after,
        args.nudge_repeat,
        max_reminders=max_reminders,
    )
    speaker = speak.Speaker()
    last_decision_log = 0.0
    last_cloud_retry = 0.0
    # The camera does not hide a bad NUDGE_CLOUD setting. The default is off,
    # which keeps every event on this machine.
    global active_cloud
    active_cloud = cloud.CloudClient.from_env(
        env_file=PROJECT_ROOT / ".env",
        queue_path=DATA_DIR / "outbox.jsonl",
    )
    print(active_cloud.describe())
    # .env is already loaded by the cloud client. A missing key leaves the
    # reminder ladder in charge and does not open a connection.
    nemotron = reason.NebiusReasoner.from_env()
    print(nemotron.describe())

    if max_reminders is None:
        follow_up = "and keeps repeating the last reminder"
    else:
        follow_up = f"then asks a caregiver after {max_reminders} reminders"
    log_event(
        "nudge_started",
        (
            f'Nudge started watching zone "{zone_name}" '
            f"(reminder after {describe_duration(args.nudge_after)} unattended, "
            f"{follow_up})"
        ),
        recent_events,
    )

    try:
        while True:
            frame_available, frame = camera.read()
            if not frame_available:
                break

            frame_height, frame_width = frame.shape[:2]

            # Measure the stove before anything is drawn on the frame, so the
            # overlays themselves never count as "something on the stove".
            stove_zone = zones.get(STOVE_ZONE)
            stove_patch = (
                stove.prepare_patch(frame, stove_zone) if stove_zone is not None else None
            )
            stove_change: float | None = None
            # Shapes differ if the camera resolution changed since the
            # reference was taken; comparing then would be meaningless.
            if (
                stove_patch is not None
                and stove_reference is not None
                and stove_patch.shape == stove_reference.shape
            ):
                stove_change = stove.change_percent(stove_reference, stove_patch)
                raw_occupied = stove_change >= stove.OCCUPIED_PERCENT
                now = time.monotonic()
                # Learn only from frames that look empty. Adapting while a
                # pot is in view would teach the picture to ignore the pot.
                if stove.should_adapt_reference(raw_occupied, stove_occupied.value):
                    stove_reference = stove.adapt_reference(
                        stove_reference,
                        stove_patch,
                        now - last_reference_adapt,
                    )
                last_reference_adapt = now
                if stove_occupied.update(raw_occupied, now):
                    if stove_occupied.value:
                        log_event(
                            "stove_item_placed",
                            "Something was placed on the stove",
                            recent_events,
                        )
                    else:
                        log_event(
                            "stove_clear",
                            "The stove is clear again",
                            recent_events,
                        )

            # Convert the camera frame into the image format MediaPipe expects.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.monotonic() - started_at) * 1000)
            result = landmarker.detect_for_video(media_image, timestamp_ms)

            # Draw detected landmarks so the user can see what the model found.
            for hand in result.hand_landmarks:
                for landmark in hand:
                    point = (
                        int(landmark.x * frame_width),
                        int(landmark.y * frame_height),
                    )
                    cv2.circle(frame, point, 3, (0, 255, 0), -1)

            detected_hands = count_hands_in_zone(
                result,
                zone,
                frame_width,
                frame_height,
            )
            currently_inside = detected_hands > 0

            # Confirm a state change only after it lasts for DEBOUNCE_SEC.
            if currently_inside != candidate_inside:
                candidate_inside = currently_inside
                candidate_since = time.monotonic()
                hands_seen = detected_hands
            else:
                hands_seen = max(hands_seen, detected_hands)
                if (
                    candidate_inside != confirmed_inside
                    and time.monotonic() - candidate_since >= DEBOUNCE_SEC
                ):
                    confirmed_inside = candidate_inside
                    if confirmed_inside:
                        arrived_at = candidate_since
                        subject = "Both hands are" if hands_seen >= 2 else "A hand is"
                        log_event(
                            "hands_in_zone",
                            f'{subject} working in zone "{zone_name}"',
                            recent_events,
                        )
                    else:
                        duration = (
                            describe_duration(candidate_since - arrived_at)
                            if arrived_at is not None
                            else "a while"
                        )
                        log_event(
                            "hands_left_zone",
                            f'Hands left zone "{zone_name}" after {duration}',
                            recent_events,
                        )

            hands_over_stove = count_hands_in_zone(
                result,
                stove_zone,
                frame_width,
                frame_height,
            )
            now = time.monotonic()
            hands_nearby.update(currently_inside or hands_over_stove > 0, now)
            # Confirmed readings only. A single noisy frame must not nudge,
            # and must not cancel a nudge either.
            step = reminders.update(
                stove_occupied=stove_occupied.value,
                hands_nearby=hands_nearby.value,
                now=now,
                advisor=nemotron.advise,
            )
            trace = reminders.describe(
                now,
                stove_occupied=stove_occupied.value,
                hands_nearby=hands_nearby.value,
            )
            away = reminders.unattended_for(now)
            if step.fallback:
                print(f"Nemotron fallback: {step.fallback}")
            elif step.source == "nemotron":
                chosen = "escalate" if step.escalate is not None else "remind"
                print(f"Nemotron ({step.model}) chose {chosen}.")
                if step.note:
                    print(step.note)
            if step.nudge is not None:
                log_event(
                    "nudge",
                    step.nudge.message,
                    recent_events,
                    details={
                        "action": "remind",
                        "prompt": step.nudge.message,
                        "reason": step.nudge.message,
                        "trace": trace,
                        "away_seconds": away if away is not None else 0,
                        "reminders_unanswered": step.nudge.level,
                        "stove_occupied": stove_occupied.value,
                        "hands_present": hands_nearby.value,
                        # Newer fields. An older API that rejects them gets
                        # a second post without them, so the event is kept.
                        "urgency": step.nudge.urgency,
                        **model_event_fields(step),
                    },
                )
                speaker.say(step.nudge.message)
            elif step.escalate is not None:
                log_event(
                    "escalate",
                    step.escalate.message,
                    recent_events,
                    details={
                        "action": "escalate",
                        "prompt": step.escalate.message,
                        "reason": step.escalate.message,
                        "trace": trace,
                        "away_seconds": step.escalate.away_seconds,
                        "reminders_unanswered": step.escalate.reminders_unanswered,
                        "stove_occupied": stove_occupied.value,
                        "hands_present": hands_nearby.value,
                        "urgency": step.escalate.urgency,
                        **model_event_fields(step),
                    },
                )
                if step.escalate.caregiver_summary:
                    print(f"Caregiver summary: {step.escalate.caregiver_summary}")
                speaker.say(step.escalate.message)
            elif step.ended is not None:
                ended_text = nudge.ended_message(step.ended)
                log_event(
                    "nudge_ended",
                    ended_text,
                    recent_events,
                    details={
                        "action": "stand_down",
                        "reason": ended_text,
                        "trace": trace,
                        "away_seconds": 0,
                        "reminders_unanswered": 0,
                        "stove_occupied": stove_occupied.value,
                        "hands_present": hands_nearby.value,
                    },
                )

            # The file gets a line about once a second, and again whenever
            # the rule speaks, asks a caregiver, or stops. The terminal
            # already prints those three. Waits stay in the file.
            acted = (
                step.nudge is not None
                or step.escalate is not None
                or step.ended is not None
            )
            if acted or now - last_decision_log >= 1.0:
                when = datetime.now().isoformat(timespec="seconds")
                # Model fields ride on the line for the moment that spoke
                # or asked. A once-a-second wait stays a trace only.
                extra = decision_log_fields(step) if acted else {}
                log_decision(when, f"{when[11:]}  {trace}", **extra)
                last_decision_log = now
            # A dead API must not be contacted on every frame. The client
            # also waits out its own cooldown after a failure.
            if active_cloud is not None and now - last_cloud_retry >= 1.0:
                active_cloud.retry_queued()
                last_cloud_retry = now

            # A reminder banner covers the top of the picture, so other labels
            # start lower while one is showing.
            top = 78 if reminders.reminder else 0

            # Draw the active zone, or explain how to create it when missing.
            if zone is not None:
                x, y, zone_width, zone_height = zone
                color = (0, 200, 0) if confirmed_inside else (0, 0, 255)
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + zone_width, y + zone_height),
                    color,
                    2,
                )
                cv2.putText(
                    frame,
                    zone_name.upper(),
                    (x, max(y - 8, 20 + top)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    f'Press Z to draw zone "{zone_name}"',
                    (20, 40 + top),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

            # Draw the stove in orange so it is easy to tell apart from the hand
            # zone. The live change percentage is shown to help tune
            # stove.OCCUPIED_PERCENT.
            if stove_zone is not None:
                x, y, zone_width, zone_height = stove_zone
                if stove_change is None:
                    stove_label = "STOVE - press B while empty"
                else:
                    state = "occupied" if stove_occupied.value else "clear"
                    away = reminders.unattended_for(time.monotonic())
                    # Show the wait counting up so a demo is obvious before
                    # the first sentence is spoken. The banner takes over after that.
                    if away is not None and reminders.reminder is None:
                        state = f"{state}, away {describe_duration(away)}"
                    stove_label = f"STOVE {stove_change:.1f}% ({state})"
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + zone_width, y + zone_height),
                    ORANGE,
                    2,
                )
                cv2.putText(
                    frame,
                    stove_label,
                    (x, max(y - 8, 20 + top)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    ORANGE,
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Press S to draw the stove",
                    (20, 75 + top),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    ORANGE,
                    2,
                )

            if reminders.reminder is not None:
                draw_reminder(frame, reminders.reminder)

            # Show the six most recent events at the bottom of the camera view.
            for index, line in enumerate(recent_events):
                y_position = frame_height - 20 - 25 * (
                    len(recent_events) - 1 - index
                )
                cv2.putText(
                    frame,
                    line,
                    (20, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Nudge", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("z"):
                selected_zone = select_zone(frame)
                if selected_zone is not None:
                    zone = selected_zone
                    zones[zone_name] = zone
                    save_zones(zones)
                    log_event(
                        "zone_saved",
                        f'Zone "{zone_name}" was saved',
                        recent_events,
                    )

            if key == ord("s"):
                selected_zone = select_zone(frame)
                if selected_zone is not None:
                    zones[STOVE_ZONE] = selected_zone
                    save_zones(zones)
                    # The old empty-stove picture no longer lines up with the
                    # new box, so a fresh one must be taken with B.
                    stove_reference = None
                    stove_occupied = stove.Debounced(DEBOUNCE_SEC)
                    last_reference_adapt = time.monotonic()
                    log_event("stove_zone_set", "Stove zone set", recent_events)

            if key == ord("b"):
                if stove_patch is None:
                    print("Draw the stove zone with S before pressing B.")
                else:
                    # stove_patch was taken from this frame before any drawing.
                    stove_reference = stove.working_copy(stove_patch)
                    stove.save_reference(STOVE_REFERENCE_FILE, stove_reference)
                    # The stove is empty by definition when the picture is taken.
                    stove_occupied = stove.Debounced(DEBOUNCE_SEC)
                    last_reference_adapt = time.monotonic()
                    log_event(
                        "stove_reference_saved",
                        "Saved a picture of the empty stove",
                        recent_events,
                    )
    finally:
        # Always release native camera and MediaPipe resources, even after errors.
        log_event(
            "nudge_stopped",
            f'Nudge stopped watching zone "{zone_name}"',
            recent_events,
        )
        camera.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
