"""Caregiver API for text events. This file runs on AWS Lambda.

The laptop is supposed to send JSON only. This function checks that again
before writing to DynamoDB or asking SNS to message a caregiver. A picture
field is rejected. The public health page contains no personal events.

Routes:
    GET  / and /health   explain the service; no token and no events
    GET  /dashboard      the caregiver page (the page fetches /events)
    GET  /events         recent text events; requires the dashboard token
    POST /events         store one text event; requires the ingest key

boto3 is provided by the Lambda runtime. Tests replace put_event,
query_events, and publish_alert so they never call AWS.

The validation rules are a copy of src/cloud.py. tests/test_lambda.py
fails if the two copies disagree.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

# Same wording as src/cloud.py. Tests compare the two strings.
PRIVACY_SENTENCE = (
    "Frames never leave the device; only text events go to AWS, with caregiver consent."
)
LOCAL_DEV_KEY = "local-dev-key"

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
    "caregiver_summary",
    "urgency",
    "model",
    "source",
    "fallback",
    "note",
}
NUMBER_FIELDS = {"away_seconds", "reminders_unanswered"}
BOOL_FIELDS = {"stove_occupied", "hands_present"}
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
# One month. DynamoDB deletes the item after this clock time.
TTL_SECONDS = 30 * 24 * 60 * 60
PARTITION = "nudge"

DASHBOARD_FILE = Path(__file__).with_name("dashboard.html")


class EventRejected(ValueError):
    """The body is not a small text event."""


def keys_match(provided: str, expected: str) -> bool:
    """Compare two API keys without leaking which characters matched."""
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided, expected)


def validate_event(event: object) -> dict:
    """Return a small text event, or raise EventRejected.

    Keep this in step with sanitize_event in src/cloud.py.
    """
    if not isinstance(event, dict):
        raise EventRejected("event must be a JSON object")

    clean: dict = {}
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
        clean["time"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    encoded = json.dumps(clean).encode("utf-8")
    if len(encoded) > MAX_BYTES:
        raise EventRejected("event is larger than 4096 bytes")
    return clean


def escalation_message(event: dict) -> str:
    """Plain-text SMS or email. There is nowhere to put a picture."""
    device = event.get("device") or "the kitchen"
    observed = event.get("observed") or event.get("reason") or "The stove may be unattended."
    reminders = event.get("reminders_unanswered", "unknown")
    away = event.get("away_seconds", "unknown")
    summary = event.get("caregiver_summary") or ""
    lines = [
        f"Nudge escalation from {device}.",
        str(observed),
    ]
    if summary:
        lines.append(str(summary))
    lines.append(f"Reminders unanswered: {reminders}. Away for about {away} seconds.")
    lines.append("No video was included.")
    lines.append(PRIVACY_SENTENCE)
    return "\n".join(lines)


def to_item(event: dict) -> dict:
    """DynamoDB item. Floats become Decimal because DynamoDB requires that."""
    item = {
        "pk": PARTITION,
        "sk": f"{event.get('time', '')}#{uuid.uuid4().hex[:12]}",
        "ttl": int(time.time()) + TTL_SECONDS,
    }
    for key, value in event.items():
        if isinstance(value, float):
            item[key] = Decimal(str(value))
        else:
            item[key] = value
    return item


def from_item(item: dict) -> dict:
    """Drop the table keys and turn Decimal back into int or float."""
    public = {}
    for key, value in item.items():
        if key in {"pk", "sk", "ttl"}:
            continue
        if isinstance(value, Decimal):
            public[key] = int(value) if value == value.to_integral_value() else float(value)
        else:
            public[key] = value
    return public


def health_page() -> str:
    """Public HTML. It must not contain caregiver events."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Nudge</title>
</head>
<body>
  <h1>Nudge caregiver endpoint</h1>
  <p>{PRIVACY_SENTENCE}</p>
  <p>Open /dashboard and add the token you were given to see text events.
  This page itself shows none.</p>
</body>
</html>
"""


def dashboard_page() -> str:
    if DASHBOARD_FILE.exists():
        return DASHBOARD_FILE.read_text(encoding="utf-8")
    return health_page()


def handler(event: dict, context: object) -> dict:
    """API Gateway HTTP API entry point (payload format 2.0)."""
    method = (
        event.get("requestContext", {}).get("http", {}).get("method", "GET").upper()
    )
    path = _path(event)
    if method == "OPTIONS":
        return _response(204, "", "text/plain; charset=utf-8")
    if method == "GET" and path in {"/", "/health"}:
        return _response(200, health_page(), "text/html; charset=utf-8")
    if method == "GET" and path == "/dashboard":
        return _response(200, dashboard_page(), "text/html; charset=utf-8")
    if method == "GET" and path == "/events":
        return _read(event)
    if method == "POST" and path == "/events":
        return _write(event)
    return _json(404, {"error": "not found"})


def _read(event: dict) -> dict:
    if not _authorized(event, write=False):
        return _json(401, {"error": "missing or wrong API key"})
    limit = _limit(_query(event, "limit"))
    try:
        events = [from_item(item) for item in query_events(limit)]
    except Exception as error:
        print(f"Could not read events ({error}).")
        return _json(500, {"error": "could not read events"})
    return _json(200, {"events": events})


def _write(event: dict) -> dict:
    if _key_rejected():
        return _json(500, {"error": "server API key is not configured"})
    if not _authorized(event, write=True):
        return _json(401, {"error": "missing or wrong API key"})
    try:
        payload = _parse_body(event)
        clean = validate_event(payload)
    except EventRejected as error:
        return _json(400, {"error": str(error)})
    except (UnicodeError, json.JSONDecodeError, ValueError):
        return _json(400, {"error": "body must be a small JSON object"})

    try:
        put_event(to_item(clean))
    except Exception as error:
        print(f"Could not store event ({error}).")
        return _json(500, {"error": "could not store event"})

    notified = False
    if clean.get("event") == "escalate" or clean.get("action") == "escalate":
        try:
            publish_alert(escalation_message(clean))
            notified = True
        except Exception as error:
            # The event is already stored, so the dashboard can still show it
            # when SMS or email is not confirmed yet.
            print(f"Stored the escalation but could not notify ({error}).")
    return _json(201, {"ok": True, "notified": notified})


def put_event(item: dict) -> None:
    """Write one item. Tests replace this."""
    table().put_item(Item=item)


def query_events(limit: int) -> list[dict]:
    """Newest text events first. Tests replace this."""
    result = table().query(
        KeyConditionExpression="pk = :pk",
        ExpressionAttributeValues={":pk": PARTITION},
        ScanIndexForward=False,
        Limit=limit,
    )
    return result.get("Items", [])


def publish_alert(message: str) -> None:
    """Publish one plain-text alert. Tests replace this."""
    topic = os.environ.get("CAREGIVER_TOPIC_ARN", "").strip()
    if not topic:
        raise RuntimeError("CAREGIVER_TOPIC_ARN is not set")
    sns().publish(TopicArn=topic, Subject="Nudge escalation", Message=message)


def table():
    import boto3

    return boto3.resource("dynamodb").Table(os.environ["EVENTS_TABLE"])


def sns():
    import boto3

    return boto3.client("sns")


def _key_rejected() -> bool:
    """True when the ingest key is missing or is the localhost demo key."""
    ingest = os.environ.get("INGEST_API_KEY", "").strip()
    if not ingest or ingest == LOCAL_DEV_KEY or len(ingest) < 20:
        return True
    dashboard = os.environ.get("DASHBOARD_TOKEN", "").strip()
    if dashboard and (dashboard == LOCAL_DEV_KEY or len(dashboard) < 20):
        return True
    return False


def _authorized(event: dict, write: bool) -> bool:
    if _key_rejected():
        return False
    provided = _header(event, "x-api-key")
    if not write and not provided:
        provided = _query(event, "token")
    ingest = os.environ.get("INGEST_API_KEY", "").strip()
    if write:
        # The header is the only write credential. A token in the URL
        # must not be enough to create an escalation.
        return keys_match(_header(event, "x-api-key"), ingest)
    dashboard = os.environ.get("DASHBOARD_TOKEN", "").strip()
    if dashboard and keys_match(provided, dashboard):
        return True
    return keys_match(provided, ingest)


def _parse_body(event: dict) -> object:
    raw = event.get("body") or ""
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError("body must be text")
    if len(raw.encode("utf-8")) > MAX_BYTES:
        raise EventRejected("event is too large")
    if not raw.strip():
        raise ValueError("empty body")
    return json.loads(raw)


def _path(event: dict) -> str:
    raw = event.get("rawPath") or "/"
    stage = event.get("requestContext", {}).get("stage")
    prefix = f"/{stage}"
    if stage and stage != "$default" and (raw == prefix or raw.startswith(prefix + "/")):
        raw = raw[len(prefix):] or "/"
    if len(raw) > 1:
        raw = raw.rstrip("/")
    return raw


def _header(event: dict, name: str) -> str:
    headers = event.get("headers") or {}
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == name:
            return "" if value is None else str(value)
    return ""


def _query(event: dict, name: str) -> str:
    query = event.get("queryStringParameters") or {}
    value = query.get(name)
    return "" if value is None else str(value)


def _limit(raw: str) -> int:
    try:
        value = int(raw) if raw else 50
    except ValueError:
        return 50
    return min(max(value, 1), 200)


def _json(status: int, payload: dict) -> dict:
    return _response(status, json.dumps(payload), "application/json")


def _response(status: int, body: str, content_type: str) -> dict:
    return {
        "statusCode": status,
        "headers": {
            "content-type": content_type,
            "access-control-allow-origin": "*",
            "access-control-allow-headers": "content-type, x-api-key",
            "access-control-allow-methods": "GET, POST, OPTIONS",
        },
        "body": body,
    }
