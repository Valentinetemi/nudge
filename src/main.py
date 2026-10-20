"""Track hands inside a named camera zone and record activity events.

Example:
    python src/main.py --cam 0 --zone counter

Press Z to draw or replace the selected zone, and press Q to quit. Named
zones are stored in data/zones.json so they can be reused later.
"""

import argparse
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path


# Resolve files from the repository root instead of the terminal's current
# directory. This lets the program work when launched from somewhere else.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
ZONES_FILE = DATA_DIR / "zones.json"
LOG_FILE = DATA_DIR / "events.jsonl"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
DEBOUNCE_SEC = 1.0

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


def parse_arguments() -> argparse.Namespace:
    """Read camera and zone choices from command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Track hands inside a named area of the camera frame."
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=1,
        help="camera index to open (default: 1)",
    )
    parser.add_argument(
        "--zone",
        type=parse_zone_name,
        default="counter",
        help="zone to load or create (default: counter)",
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


def log_event(event: str, observed: str, recent: list[str]) -> None:
    """Append one machine-readable event and update the on-screen history."""
    entry = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "observed": observed,
    }
    with LOG_FILE.open("a", encoding="utf-8") as log:
        log.write(json.dumps(entry) + "\n")

    line = f'{entry["time"][11:]}  {observed}'
    print(line)
    recent.append(line)
    del recent[:-6]  # Keep the display readable by showing only six events.


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


def describe_duration(seconds: float) -> str:
    """Format elapsed seconds as a short, human-readable duration."""
    whole_seconds = int(seconds)
    if whole_seconds < 60:
        return f"{whole_seconds} seconds"
    minutes, remaining_seconds = divmod(whole_seconds, 60)
    return f"{minutes} min {remaining_seconds} sec"


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

    # Load the large camera libraries only when tracking is about to start.
    # Commands such as --list-zones can then run without camera dependencies.
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

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
        raise SystemExit(f"Camera {args.cam} didn't open. Try --cam 0")

    # Debouncing prevents a brief missed detection from creating false events.
    confirmed_inside = False
    candidate_inside = False
    candidate_since = time.time()
    arrived_at: float | None = None
    hands_seen = 0
    recent_events: list[str] = []
    started_at = time.time()

    log_event(
        "nudge_started",
        f'Nudge started watching zone "{zone_name}"',
        recent_events,
    )

    try:
        while True:
            frame_available, frame = camera.read()
            if not frame_available:
                break

            frame_height, frame_width = frame.shape[:2]

            # Convert the camera frame into the image format MediaPipe expects.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - started_at) * 1000)
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
                candidate_since = time.time()
                hands_seen = detected_hands
            else:
                hands_seen = max(hands_seen, detected_hands)
                if (
                    candidate_inside != confirmed_inside
                    and time.time() - candidate_since >= DEBOUNCE_SEC
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
                    (x, max(y - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    f'Press Z to draw zone "{zone_name}"',
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

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
                selected_rectangle = cv2.selectROI(
                    "Nudge",
                    frame,
                    showCrosshair=False,
                )
                if selected_rectangle[2] > 0 and selected_rectangle[3] > 0:
                    zone = tuple(int(value) for value in selected_rectangle)
                    zones[zone_name] = zone
                    save_zones(zones)
                    log_event(
                        "zone_saved",
                        f'Zone "{zone_name}" was saved',
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
