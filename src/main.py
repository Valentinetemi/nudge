import argparse #this is to pass options from my script to terminal
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_PATH = Path("models/hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
ZONE_FILE = Path("data/zone.json")  #rememeber where the counter is
LOG_FILE = Path("data/events.jsonl") #this is nudge memory: one events per line
DEBOUNCE_SEC = 1.0 # this is how long a change must hold before we count it as a new event


 
def ensure_model():
    """Download MediaPipe's hand model once."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(exist_ok=True)
        print("Downloading hand model (about 8 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

 
def load_zone():
    if ZONE_FILE.exists():
        return tuple(json.loads(ZONE_FILE.read_text()))
    return None


def save_zone(zone):
    ZONE_FILE.write_text(json.dumps(list(zone)))

def log_event(event, observed, recent):
    """
    this is is to save one event.
    """
    entry = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "observed": observed,
    }
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    line = f'{entry["time"][11:]}  {observed}'
    print(line)
    recent.append(line)
    del recent[:-6]  # keep only the last 6 lines on screen

def count_hands_in_zone(result, zone, w, h):
    """How many hands have at least one point inside the counter zone."""
    if not zone:
        return 0
    x, y, zw, zh = zone
    count = 0
    for hand in result.hand_landmarks:
        for lm in hand:
            px, py = lm.x * w, lm.y * h  # landmarks are 0-1, convert to pixels
            if x <= px <= x + zw and y <= py <= y + zh:
                count += 1
                break  # this hand counts once, move on to the next hand
    return count

def describe_duration(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} seconds"
    minutes, secs = divmod(seconds, 60)
    return f"{minutes} min {secs} sec"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=1)
    args = ap.parse_args()

    Path("data").mkdir(exist_ok=True)
    ensure_model()
    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"Camera {args.cam} didn't open. Try --cam 0")

    zone = load_zone()
    state = False
    candidate, since = False, time.time()
    arrived_at = None
    hands_seen = 0
    recent = []
    start = time.time()
    log_event("nudge_started", "Nudge started watching the counter", recent)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        # 1. Perceive
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - start) * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        for hand in result.hand_landmarks:
            for lm in hand:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)

        # 2. Decide
        n_hands = count_hands_in_zone(result, zone, w, h)
        now_in = n_hands > 0
        if now_in != candidate:
            candidate, since = now_in, time.time()
            hands_seen = n_hands
        else:
            hands_seen = max(hands_seen, n_hands)
            if candidate != state and time.time() - since >= DEBOUNCE_SEC:
                state = candidate
                if state:
                    arrived_at = since
                    who = "Both hands are" if hands_seen >= 2 else "A hand is"
                    log_event("hands_at_counter", f"{who} working at the counter", recent)
                else:
                    stayed = describe_duration(since - arrived_at) if arrived_at else "a while"
                    log_event("hands_left_counter",
                              f"Hands left the counter after {stayed} there", recent)

        # 3. Show
        if zone:
            x, y, zw, zh = zone
            color = (0, 200, 0) if state else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + zw, y + zh), color, 2)
            cv2.putText(frame, "COUNTER", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(frame, "Press Z to draw the counter zone", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        for i, line in enumerate(recent):
            y_pos = h - 20 - 25 * (len(recent) - 1 - i)
            cv2.putText(frame, line, (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Nudge", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("z"):
            r = cv2.selectROI("Nudge", frame, showCrosshair=False)
            if r[2] > 0 and r[3] > 0:
                zone = tuple(int(v) for v in r)
                save_zone(zone)
                log_event("counter_zone_set", "Counter zone set", recent)

    log_event("nudge_stopped", "Nudge stopped watching", recent)
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()