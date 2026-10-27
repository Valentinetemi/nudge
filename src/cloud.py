"""Send small text events to a caregiver API. Frames are never accepted.

NUDGE_CLOUD selects where events go:

    off     default. The camera logs stay on this machine. Nothing is sent.
    mock    a local server (scripts/mock_cloud.py) with the same JSON shape
            the AWS API will use. No AWS account is required.
    aws     the deployed API Gateway URL.

A failed send is appended to data/outbox.jsonl and tried again later. The
camera loop is never raised into. The API key is read from the environment
or from a gitignored .env file, and it is never written into the event body.

The sentence below is the privacy rule this module enforces. Video frames
are not a field we accept, and a blocked field rejects the whole event.
"""

from __future__ import annotations

import hmac
import json
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# Shown on the dashboard and used by tests so the wording cannot drift quietly.
PRIVACY_SENTENCE = (
    "Frames never leave the device; only text events go to AWS, with caregiver consent."
)

# The mock server accepts this so a demo can start with no setup. The AWS
# path refuses it. It is not a secret, and it must not be deployed.
LOCAL_DEV_KEY = "local-dev-key"

# Text fields a caregiver can read. Anything else is rejected.
TEXT_FIELDS = {
    "time",
    "event",
    "observed",
    "action",
    "reason",
    "device",
    "prompt",
    "phase",
    "trace",
    # Written by the Nemotron step. Still text. Still no picture.
    "caregiver_summary",
    "urgency",
    "model",
    "source",
    "fallback",
    "note",
}
NUMBER_FIELDS = {"away_seconds", "reminders_unanswered"}
BOOL_FIELDS = {"stove_occupied", "hands_present"}

# A key containing one of these words is an attempt to attach a picture.
BLOCKED_PARTS = (
    "frame",
    "image",
    "jpeg",
    "jpg",
    "png",
    "pixel",
    "video",
    "screenshot",
    "thumbnail",
    "ndarray",
)

MAX_STRING = 2000
MAX_BYTES = 4096
DEFAULT_COOLDOWN_SEC = 15.0
DEFAULT_TIMEOUT_SEC = 1.5


class EventRejected(ValueError):
    """The payload is not a small text event. It was not sent."""


def load_env_file(path: Path, environ: dict[str, str] | None = None) -> None:
    """Fill missing keys from a KEY=VALUE file. Real environment wins.

    Blank lines and # comments are ignored. Values may be wrapped in quotes.
    This is intentionally small so the app does not gain a dependency.
    """
    target = os.environ if environ is None else environ
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            target.setdefault(key, value)


def keys_match(provided: str, expected: str) -> bool:
    """Compare two API keys without leaking which characters matched."""
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided, expected)


def sanitize_event(event: object) -> dict[str, object]:
    """Return a small text event, or raise EventRejected.

    The returned dict is safe to hand to the network: known fields only,
    short strings, no nested objects, and no picture-shaped keys.
    """
    if not isinstance(event, dict):
        raise EventRejected("event must be a JSON object")

    clean: dict[str, object] = {}
    for key, value in event.items():
        if not isinstance(key, str):
            raise EventRejected("event fields must be named with strings")
        name = key.strip().lower()
        if any(part in name for part in BLOCKED_PARTS):
            raise EventRejected(
                f"field {key!r} is not allowed; frames never leave the device"
            )
        if name in TEXT_FIELDS:
            if not isinstance(value, str):
                raise EventRejected(f"{key} must be text")
            if len(value) > MAX_STRING:
                raise EventRejected(f"{key} is too long to be a text event")
            if value.startswith(("data:image", "data:video")):
                raise EventRejected(f"{key} looks like an image, which is not allowed")
            clean[name] = value
        elif name in NUMBER_FIELDS:
            # bool is a subclass of int, and it is not a duration or a count.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EventRejected(f"{key} must be a number")
            clean[name] = value
        elif name in BOOL_FIELDS:
            if not isinstance(value, bool):
                raise EventRejected(f"{key} must be true or false")
            clean[name] = value
        else:
            raise EventRejected(f"field {key!r} is not part of a text event")

    if "event" not in clean and "action" not in clean:
        raise EventRejected("event needs an event name or an action")
    if "time" not in clean:
        clean["time"] = datetime.now().isoformat(timespec="seconds")

    encoded = json.dumps(clean).encode("utf-8")
    if len(encoded) > MAX_BYTES:
        raise EventRejected("event is larger than 4096 bytes")
    return clean


def text_event(
    event: str,
    observed: str,
    when: str,
    device: str,
    *,
    action: str | None = None,
    reason: str = "",
    trace: str = "",
    phase: str = "",
    away_seconds: float | None = None,
    reminders_unanswered: int | None = None,
    stove_occupied: bool | None = None,
    hands_present: bool | None = None,
    prompt: str | None = None,
) -> dict[str, object]:
    """Build one caregiver event from plain facts. No pixels are copied.

    The camera loop already decided. This only copies the words and numbers
    a caregiver can read.
    """
    payload: dict[str, object] = {
        "time": when,
        "event": event,
        "observed": observed,
        "reason": reason or observed,
        "device": device,
        "trace": trace,
    }
    if action:
        payload["action"] = action
    if phase:
        payload["phase"] = phase
    if prompt:
        payload["prompt"] = prompt
    if away_seconds is not None:
        payload["away_seconds"] = round(float(away_seconds), 3)
    if reminders_unanswered is not None:
        payload["reminders_unanswered"] = int(reminders_unanswered)
    if stove_occupied is not None:
        payload["stove_occupied"] = bool(stove_occupied)
    if hands_present is not None:
        payload["hands_present"] = bool(hands_present)
    return payload


def build_request(url: str, api_key: str, event: dict[str, object]) -> urllib.request.Request:
    """POST /events with the key in a header. The body stays free of the key."""
    payload = json.dumps(event).encode("utf-8")
    return urllib.request.Request(
        url.rstrip("/") + "/events",
        data=payload,
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
        },
        method="POST",
    )


def urlopen_post(
    url: str,
    api_key: str,
    event: dict[str, object],
    timeout: float,
) -> str:
    """Send one event. Return 'ok', 'drop' (do not retry), or 'retry'."""
    request = build_request(url, api_key, event)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
        return "ok"
    except urllib.error.HTTPError as error:
        error.read()
        # 408/429 and server errors are temporary. Other 4xx responses mean
        # this payload will not start working if we keep sending it.
        if error.code in (408, 429) or 500 <= error.code <= 599:
            print(
                f"Caregiver API returned HTTP {error.code}. "
                "The event stays in the local outbox."
            )
            return "retry"
        print(
            f"Caregiver API rejected an event with HTTP {error.code}. "
            "It was not queued."
        )
        return "drop"
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        print(f"Caregiver API unreachable ({error}). The event stays in the local outbox.")
        return "retry"


class CloudClient:
    """Optional sender. `off` does nothing. Failures are queued, not raised."""

    def __init__(
        self,
        mode: str,
        api_url: str,
        api_key: str,
        device: str,
        queue_path: Path,
        *,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        cooldown_sec: float = DEFAULT_COOLDOWN_SEC,
        transport=None,
        now=None,
    ) -> None:
        self.mode = mode
        self.api_url = api_url
        self.api_key = api_key
        self.device = device[:64] if device else "kitchen"
        self.queue_path = queue_path
        self.timeout_sec = timeout_sec
        self.cooldown_sec = cooldown_sec
        self._transport = transport
        self._now = now or time.monotonic
        self._lock = threading.Lock()
        self._last_failure_at: float | None = None
        self._warned = False

    @classmethod
    def from_env(
        cls,
        env_file: Path | None = None,
        queue_path: Path | None = None,
        environ: dict[str, str] | None = None,
    ) -> CloudClient:
        """Read NUDGE_* settings. Unknown modes fall back to off."""
        if env_file is not None:
            load_env_file(env_file, environ)
        env = os.environ if environ is None else environ
        mode = env.get("NUDGE_CLOUD", "off").strip().lower() or "off"
        if mode not in {"off", "mock", "aws"}:
            print(f"NUDGE_CLOUD={mode!r} is not off, mock, or aws. Staying offline.")
            mode = "off"
        root = Path(__file__).resolve().parents[1]
        device = env.get("NUDGE_DEVICE_NAME", "kitchen").strip() or "kitchen"
        return cls(
            mode=mode,
            api_url=env.get("NUDGE_API_URL", "").strip(),
            api_key=env.get("NUDGE_API_KEY", "").strip(),
            device=device,
            queue_path=queue_path or (root / "data" / "outbox.jsonl"),
        )

    def describe(self) -> str:
        """One startup line. Does not include the API key."""
        if self.mode == "off":
            return (
                "Caregiver events: off. Local logs only. "
                "Set NUDGE_CLOUD=mock to use the local dashboard."
            )
        target = self.api_url or "(set NUDGE_API_URL)"
        return f"Caregiver events: {self.mode} at {target}, device {self.device}."

    def send(self, event: dict) -> bool:
        """Deliver one event, or queue it. Return True when it was accepted.

        `off` returns True because there is nothing to deliver. A rejected
        payload returns False and is not queued. Never raises.
        """
        try:
            clean = sanitize_event(event)
        except EventRejected as error:
            print(f"Not sending event ({error}).")
            return False
        if "device" not in clean:
            clean["device"] = self.device
        if self.mode == "off":
            return True
        with self._lock:
            return self._deliver_locked(clean)

    def retry_queued(self) -> None:
        """Try the outbox again. Called about once a second from the camera loop.

        Network attempts are skipped until the cooldown after a failure has
        passed, so a dead API does not stall the camera on every frame.
        """
        if self.mode == "off":
            return
        with self._lock:
            self._flush_locked()

    def queued_events(self) -> list[dict]:
        """Read the outbox. Used by tests and by the retry path."""
        with self._lock:
            return self._read_queue()

    def _deliver_locked(self, event: dict[str, object]) -> bool:
        self._flush_locked()
        if self._read_queue():
            # Something older is still waiting. Keep the order.
            self._enqueue_locked(event)
            return False
        result = self._post(event)
        if result == "ok":
            self._last_failure_at = None
            return True
        if result == "retry":
            self._last_failure_at = self._now()
            self._enqueue_locked(event)
        return False

    def _flush_locked(self) -> None:
        pending = self._read_queue()
        if not pending or not self._cooldown_elapsed():
            return
        kept: list[dict] = []
        for index, event in enumerate(pending):
            result = self._post(event)
            if result == "ok":
                self._last_failure_at = None
                continue
            if result == "drop":
                continue
            self._last_failure_at = self._now()
            kept = pending[index:]
            break
        self._write_queue(kept)

    def _cooldown_elapsed(self) -> bool:
        if self._last_failure_at is None:
            return True
        return (self._now() - self._last_failure_at) >= self.cooldown_sec

    def _post(self, event: dict[str, object]) -> str:
        if self.mode == "aws" and self.api_key == LOCAL_DEV_KEY:
            self._warn(
                "Refusing to send: local-dev-key is only for the mock server, not AWS."
            )
            return "drop"
        if not self.api_url or not self.api_key:
            self._warn(
                "Set NUDGE_API_URL and NUDGE_API_KEY, or use NUDGE_CLOUD=off. "
                "Events are queued on this machine."
            )
            return "retry"
        try:
            if self._transport is not None:
                return self._transport(event)
            return urlopen_post(self.api_url, self.api_key, event, self.timeout_sec)
        except Exception as error:
            # A bug in the transport must not take down the camera loop.
            print(f"Caregiver send failed ({error}). The event stays in the local outbox.")
            return "retry"

    def _warn(self, message: str) -> None:
        if not self._warned:
            print(message)
            self._warned = True

    def _enqueue_locked(self, event: dict[str, object]) -> None:
        pending = self._read_queue()
        pending.append(event)
        self._write_queue(pending)

    def _read_queue(self) -> list[dict]:
        if not self.queue_path.exists():
            return []
        events: list[dict] = []
        for line in self.queue_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                print("Skipped one unreadable outbox line.")
                continue
            if isinstance(value, dict):
                events.append(value)
        return events

    def _write_queue(self, events: list[dict]) -> None:
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.queue_path.with_suffix(".jsonl.tmp")
        payload = "".join(json.dumps(event) + "\n" for event in events)
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.queue_path)
