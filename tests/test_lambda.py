"""Lambda tests do not call AWS. They check the same text-only rules as the laptop."""

import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "aws" / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import app
import cloud

# Payloads the laptop and the Lambda must both accept or both reject.
CASES = [
    {
        "time": "2026-10-25T12:00:00",
        "event": "remind",
        "observed": "Come back when you can.",
        "away_seconds": 60,
        "reminders_unanswered": 1,
        "stove_occupied": True,
        "hands_present": False,
    },
    {"event": "remind", "observed": "no clock in the payload"},
    {"event": "remind", "observed": "hi", "frame": "abc"},
    {"event": "remind", "observed": "hi", "thumbnail": "abc"},
    {"event": "remind", "observed": "data:image/png;base64,aaaa"},
    {"event": "remind", "away_seconds": True, "observed": "hi", "time": "t"},
    {"event": "remind", "observed": "hi", "mystery": 1, "time": "t"},
    {
        "time": "2026-10-27T12:00:00",
        "event": "escalate",
        "observed": "I haven't seen you come back, so I'm letting your caregiver know.",
        "caregiver_summary": "Unattended for 240 seconds after 2 reminders.",
        "urgency": "firm",
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "source": "nemotron",
    },
    ["not", "a", "dict"],
    {"observed": "missing the event name", "time": "t"},
]


def _outcome(function, payload):
    try:
        result = dict(function(payload))
    except (cloud.EventRejected, app.EventRejected) as error:
        return ("error", str(error))
    if isinstance(payload, dict) and "time" not in payload:
        result.pop("time", None)
    return ("ok", result)


def test_lambda_validation_matches_the_laptop() -> None:
    for payload in CASES:
        assert _outcome(app.validate_event, payload) == _outcome(cloud.sanitize_event, payload)


def test_model_fields_are_accepted_and_preserved() -> None:
    """Both validators must keep the Nemotron fields, not merely agree.

    A parity check alone still passes when both sides reject the event.
    """
    payload = {
        "time": "2026-10-27T12:00:00",
        "event": "escalate",
        "observed": "I haven't seen you come back, so I'm letting your caregiver know.",
        "caregiver_summary": "Unattended for 240 seconds after 2 reminders.",
        "urgency": "firm",
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "source": "nemotron",
        "fallback": "not used",
        "note": "The reminder cap is reached, so this asks the caregiver.",
    }
    for validate in (cloud.sanitize_event, app.validate_event):
        clean = validate(payload)
        assert clean["event"] == "escalate"
        assert clean["observed"] == payload["observed"]
        assert clean["caregiver_summary"] == payload["caregiver_summary"]
        assert clean["urgency"] == "firm"
        assert clean["model"] == payload["model"]
        assert clean["source"] == "nemotron"
        assert clean["fallback"] == "not used"
        assert clean["note"] == payload["note"]


def test_privacy_sentence_matches_the_laptop() -> None:
    assert app.PRIVACY_SENTENCE == cloud.PRIVACY_SENTENCE


def test_importing_the_handler_does_not_import_boto3() -> None:
    assert "boto3" not in __import__("sys").modules


def test_item_round_trip_drops_table_keys_and_keeps_text() -> None:
    item = app.to_item(
        {
            "time": "2026-10-25T12:00:00",
            "event": "escalate",
            "observed": "Please check the stove.",
            "away_seconds": 180.5,
            "reminders_unanswered": 2,
            "stove_occupied": True,
        }
    )
    assert item["pk"] == "nudge"
    assert isinstance(item["away_seconds"], Decimal)
    assert item["ttl"] > 0
    assert "frame" not in item
    public = app.from_item(item)
    assert "pk" not in public
    assert public["away_seconds"] == 180.5
    assert public["reminders_unanswered"] == 2
    assert public["observed"] == "Please check the stove."


def test_escalation_message_has_no_video() -> None:
    message = app.escalation_message(
        {
            "device": "kitchen",
            "observed": "Please check the stove.",
            "reminders_unanswered": 2,
            "away_seconds": 180,
        }
    )
    assert "kitchen" in message
    assert "Please check the stove." in message
    assert "No video was included." in message
    assert cloud.PRIVACY_SENTENCE in message
    summary = app.escalation_message(
        {
            "device": "kitchen",
            "observed": "Please check the stove.",
            "reminders_unanswered": 2,
            "away_seconds": 180,
            "caregiver_summary": "Unattended for 180 seconds after 2 reminders.",
        }
    )
    assert "Unattended for 180 seconds after 2 reminders." in summary
    assert "frame" not in summary


def _request(method: str, path: str, body: dict | None = None, key: str | None = None):
    event = {
        "rawPath": path,
        "requestContext": {"http": {"method": method}, "stage": "$default"},
        "headers": {},
        "isBase64Encoded": False,
    }
    if key is not None:
        event["headers"]["x-api-key"] = key
    if body is not None:
        event["body"] = json.dumps(body)
    return event


def _install_fakes(monkeypatch, stored: list, published: list) -> None:
    monkeypatch.setenv("INGEST_API_KEY", "ingest-key-0123456789")
    monkeypatch.setenv("DASHBOARD_TOKEN", "dashboard-token-012345")
    monkeypatch.setenv("CAREGIVER_TOPIC_ARN", "arn:aws:sns:us-east-1:123:nudge")
    monkeypatch.setattr(app, "put_event", stored.append)
    monkeypatch.setattr(app, "publish_alert", published.append)
    monkeypatch.setattr(app, "query_events", lambda _limit: list(reversed(stored)))


def test_post_stores_text_and_escalation_notifies(monkeypatch) -> None:
    stored: list = []
    published: list = []
    _install_fakes(monkeypatch, stored, published)
    body = {
        "time": "2026-10-25T12:00:00",
        "event": "escalate",
        "observed": "Please check the stove.",
        "device": "kitchen",
        "reminders_unanswered": 2,
        "away_seconds": 180,
    }
    response = app.handler(_request("POST", "/events", body, key="ingest-key-0123456789"), None)
    assert response["statusCode"] == 201
    assert json.loads(response["body"])["notified"] is True
    assert stored[0]["event"] == "escalate"
    assert "frame" not in stored[0]
    assert "Please check the stove." in published[0]

    published.clear()
    remind = dict(body, event="remind", observed="Come back when you can.")
    reminder = app.handler(
        _request("POST", "/events", remind, key="ingest-key-0123456789"),
        None,
    )
    assert reminder["statusCode"] == 201
    assert published == []


def test_dashboard_token_can_read_but_not_write(monkeypatch) -> None:
    stored: list = []
    published: list = []
    _install_fakes(monkeypatch, stored, published)
    stored.append(
        {
            "pk": "nudge",
            "sk": "1",
            "ttl": 1,
            "event": "escalate",
            "observed": "Please check the stove.",
            "time": "2026-10-25T12:00:00",
        }
    )
    denied = app.handler(
        _request(
            "POST",
            "/events",
            {"event": "remind", "observed": "hi", "time": "t"},
            key="dashboard-token-012345",
        ),
        None,
    )
    assert denied["statusCode"] == 401

    allowed = app.handler(_request("GET", "/events", key="dashboard-token-012345"), None)
    assert allowed["statusCode"] == 200
    body = json.loads(allowed["body"])
    assert body["events"][0]["observed"] == "Please check the stove."
    assert "pk" not in body["events"][0]

    missing = app.handler(_request("GET", "/events"), None)
    assert missing["statusCode"] == 401


def test_health_and_dashboard_are_public_and_event_free(monkeypatch) -> None:
    monkeypatch.setenv("INGEST_API_KEY", "ingest-key-0123456789")
    health = app.handler(_request("GET", "/health"), None)
    assert health["statusCode"] == 200
    assert cloud.PRIVACY_SENTENCE in health["body"]
    assert "Please check" not in health["body"]

    dashboard = app.handler(_request("GET", "/dashboard"), None)
    assert dashboard["statusCode"] == 200
    assert cloud.PRIVACY_SENTENCE in dashboard["body"]
    assert "What happened" in dashboard["body"]


def test_frames_are_refused_and_not_stored(monkeypatch) -> None:
    stored: list = []
    published: list = []
    _install_fakes(monkeypatch, stored, published)
    response = app.handler(
        _request(
            "POST",
            "/events",
            {"event": "remind", "observed": "hi", "time": "t", "frame": "pixels"},
            key="ingest-key-0123456789",
        ),
        None,
    )
    assert response["statusCode"] == 400
    assert stored == []
    assert published == []


def test_localhost_demo_key_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("INGEST_API_KEY", cloud.LOCAL_DEV_KEY)
    monkeypatch.delenv("DASHBOARD_TOKEN", raising=False)
    response = app.handler(
        _request(
            "POST",
            "/events",
            {"event": "escalate", "observed": "hi", "time": "t"},
            key=cloud.LOCAL_DEV_KEY,
        ),
        None,
    )
    assert response["statusCode"] == 500


def test_template_declares_the_text_path_and_no_bucket_for_video() -> None:
    template = (
        __import__("pathlib").Path(__file__).resolve().parents[1] / "aws" / "template.yaml"
    ).read_text(encoding="utf-8")
    for name in ("EventsTable", "CaregiverTopic", "NudgeApi", "EventsFunction"):
        assert name in template
    assert "AWS::S3::Bucket" not in template
    assert "dynamodb" in template.lower() or "DynamoDB" in template
    assert "sns:Publish" in template
