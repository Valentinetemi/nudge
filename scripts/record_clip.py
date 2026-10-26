"""Record a short stove clip for the evaluation harness.

Example:
    python scripts/record_clip.py --cam 0 --seconds 12 --out eval/clips/empty_daylight.mp4

Point the camera so the stove fills the frame, and keep that framing for
every clip and for the empty reference still. Then add a row to
eval/labels.csv:

    empty_daylight.mp4,clear,absent

A reference still can be any frame of an empty clip, or a photo saved as
eval/clips/reference.png. Do not use data/stove_reference.png here. That
file is already cropped and blurred for the live app.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a clip of the stove.")
    parser.add_argument("--cam", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--seconds", type=float, default=12.0, help="how long to record")
    parser.add_argument("--out", type=Path, required=True, help="mp4 path to write")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.seconds <= 0:
        raise SystemExit("--seconds must be positive")

    camera = cv2.VideoCapture(args.cam)
    if not camera.isOpened():
        raise SystemExit(f"Camera {args.cam} didn't open. Try --cam 0 or --cam 1")

    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = camera.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 20.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        camera.release()
        raise SystemExit(
            f"Could not open {args.out} for writing. "
            "The mp4v codec may be missing; try a different --out folder or install ffmpeg."
        )

    frame_count = int(args.seconds * fps)
    print(f"Recording {args.seconds:.0f}s from camera {args.cam} to {args.out}")
    print("Keep the stove framing still. Press Q to stop early.")
    written = 0
    try:
        for _ in range(frame_count):
            ok, frame = camera.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
            cv2.imshow("Record stove clip", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        writer.release()
        cv2.destroyAllWindows()
    if written == 0:
        raise SystemExit("No frames were written.")
    print(f"Wrote {written} frames. Add a row to eval/labels.csv for this file.")


if __name__ == "__main__":
    main()
