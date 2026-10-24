# Nudge

Nudge is a real-time camera helper for dementia care. People living with dementia sometimes start a task, walk away, and do not come back. A pot left on the stove is the case this prototype watches. Nudge does not scold and it does not beep like an alarm. It waits, and if the stove still looks occupied and nobody is nearby, it speaks a short, calm reminder. If nothing changes, it says so again a little more directly. When the person returns, or the stove is clear, it stops.

This is a prototype for an OpenCV project. It is not a medical device, and it is not a substitute for someone checking in or for a real stove shutoff.

## How it works

Three small pieces, all on the laptop:

1. **Hands.** MediaPipe's hand landmarker looks for hands in a zone you draw (the counter, by default). A hand counts only after it has been seen for about a second, so a flicker does not spam the log. Hands over the stove count too, so stirring a pot counts as "someone is here" even if the hand zone is the counter beside it.
2. **Stove.** You draw a box around the stove and photograph it while it is empty (`B`). Each new frame is compared with that picture. The percentage of pixels that changed is shown on screen. Above 5%, the stove counts as occupied, again only after the change holds for about a second.
3. **Nudge.** If the stove stays occupied and no hands have been nearby, a clock starts. The default wait is **3 minutes** (`--nudge-after 180`). The first reminder is "Something is still on the stove." If nobody comes back, Nudge speaks again after **1 minute** (`--nudge-repeat 60`), then in slightly more direct words, and then stops getting sterner. Coming back or clearing the stove resets the clock. The next absence waits the full 3 minutes again.

Lighting is handled in two ways, because a lamp or a cloud used to look exactly like a pot:

- Before comparing, each stove picture is divided by a wide blur. Light that falls across the whole zone drops out. A pot is smaller than that blur, so it stays.
- While the stove is confirmed clear, the empty picture drifts slowly toward the live view (on the order of a minute). Afternoon daylight can ease in. The drift stops at the first frame that might contain an object, so a pot is not learned away.

## Privacy

Everything stays on the machine. The camera frames are not uploaded, and there is no account and no cloud model. The only network request in the program is an optional download of the hand model, and only when `models/hand_landmarker.task` is missing. That file is already in the repository so a demo can run offline.

What is written to disk, under `data/` (gitignored):

- `zones.json`, the boxes you draw
- `stove_reference.png`, the empty-stove photo
- `events.jsonl`, a plain-language log

Delete `data/` to forget a session.

## Setup

Python 3.10 or newer. On macOS, allow the terminal (or the IDE) to use the camera under System Settings, Privacy & Security, Camera.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The hand model is MediaPipe's hand landmarker, about 8 MB, stored at `models/hand_landmarker.task`. It is committed on purpose. `ensure_model()` can download the same file from Google's model storage, but a judging laptop may be offline, and without the file the camera loop cannot start. The model is separate from this project's MIT license. Use it under MediaPipe's own terms.

`sounddevice` is used only for the fallback chime. On macOS the spoken reminder uses the built-in `say` command, which is offline. Linux uses `espeak` or `espeak-ng` if one of those is installed. Windows uses the built-in speech synthesizer. If none of that can play audio, Nudge prints the sentence instead. The camera loop does not wait on the voice.

## Run

From the repository root:

```bash
python src/main.py
```

A short demo, so you do not have to wait three minutes:

```bash
python src/main.py --nudge-after 15 --nudge-repeat 15
```

Other flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--cam` | `0` | Camera index. Try `--cam 1` if the wrong camera opens. |
| `--zone` | `counter` | Name of the hand zone to load or create. `stove` is reserved. |
| `--nudge-after` | `180` | Seconds with nobody nearby before the first reminder. |
| `--nudge-repeat` | `60` | Seconds between later reminders. |
| `--list-zones` | | Print saved zones and exit. No camera needed. |

### Keys

| Key | Action |
| --- | --- |
| `Z` | Draw or replace the hand zone. Drag a rectangle, then Enter. Esc cancels. |
| `S` | Draw or replace the stove zone. Take a new empty photo afterward. |
| `B` | Photograph the empty stove. Do this with the burners clear. |
| `Q` | Quit. |

Suggested first run: press `Z` and box in the counter where hands rest while cooking, press `S` and box in the burners, press `B` while the stove is empty, then set a pot down and walk out of view.

The stove label shows the live change, for example `STOVE 1.2% (clear)` or `STOVE 12.0% (occupied, away 20 seconds)`. When a reminder is due, a warm banner across the top shows the same words that are spoken.

## Event log

Each line in `data/events.jsonl` is one JSON object with `time`, `event`, and `observed`:

```json
{"time": "2026-10-24T18:05:01", "event": "stove_item_placed", "observed": "Something was placed on the stove"}
{"time": "2026-10-24T18:08:12", "event": "nudge", "observed": "Something is still on the stove."}
{"time": "2026-10-24T18:08:40", "event": "nudge_ended", "observed": "Someone came back, so the reminder stopped."}
```

`nudge` is the reminder itself. Later reminders use a more direct sentence. `nudge_ended` is written when a reminder stops, and the sentence says whether someone came back or the stove is clear.

## Tuning

Watch the on-screen percentage with the stove empty, with a light switched on, and with a pot in place. Empty, including a brightness change, should sit well under 5%. A pot should sit comfortably above it. If your kitchen falls in between, edit `OCCUPIED_PERCENT` or `PIXEL_DIFF_THRESHOLD` in `src/stove.py`.

`DEBOUNCE_SEC` in `src/main.py` is the one-second pause before a hand or a stove change is believed. Reminder times shorter than a couple of seconds fight that pause. Use 15 seconds or more for a demo.

The hand zone should cover where hands actually are while someone cooks. Hands inside the stove box count as nearby even if the hand zone does not overlap it.

## Tests

No camera is required:

```bash
python -m unittest discover -s tests -v
```

The tests cover the reminder clock, the lighting comparison (drawn pictures, not a webcam), the voice/tone/console fallback, and the command-line defaults.

## Limitations

- Nudge cannot see heat. A cold pot and a hot burner look the same if they change the picture. A cloth left on the stove can cause a reminder too.
- A small item, such as a teaspoon, may stay under the 5% line after the stove image is blurred to hide camera noise.
- A hard shadow that cuts across the stove, or a camera that gets bumped, can still look like an object. Press `B` again after you move the camera.
- Smooth lighting changes (a lamp, a window, auto-exposure) are what the lighting step is built for. It is not a promise about every shadow.
- The reminder only knows that hands were in view. It does not know who the person is, and it misses hands that are out of frame or hidden.
- There is one stove zone. The reference photo is the one you last saved with `B`. The slow daylight drift is kept in memory and is not written back over that photo.
- If the laptop has no working voice and no speaker, the reminder is only on screen and in the terminal.

## Next steps

- Tell a pot from a cloth, a spoon, or a hand passing through the zone.
- Add a heat signal (a burner sensor, or a thermal camera) so "occupied" means hot, not merely visually different.
- A daily summary from `events.jsonl` for a caregiver, still stored locally.
- More than one unfinished task: a running tap, an open door, a kettle.
- A later, separate step that can text a chosen person, only with consent, after several unanswered reminders.
- A shutoff or smart plug as its own careful decision, never as a surprise from this prototype.

## License

The code is MIT. See `LICENSE`.
