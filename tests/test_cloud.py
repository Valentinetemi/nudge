"""Cloud tests stay on this machine. They check that pictures cannot be sent."""

import io
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "scripts", ROOT / "aws" / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import cloud
import main
import mock_cloud


def _event(name: str = "remind", **extra: object) -> dict:
    payload: dict = {
        "time": "2026-10-25T12:00:00",
        "event": name,
        "observed": "The stove still has something on it.",
    }
    payload.update(extra)
    return payload


def _client(tmp_path: Path, transport, **overrides) -> cloud.CloudClient:
    settings = dict(
        mode="mock",
        api_url="http://127.0.0.1:8787",
        api_key="test-key",
        device="kitchen",
        queue_path=tmp_path / "outbox.jsonl",
        cooldown_sec=15,
        transport=transport,
        now=lambda: 0.0,
    )
    settings.update(overrides)
    return cloud.CloudClient(**settings)


def test_sanitize_rejects_pictures_and_unknown_fields() -> None:
    with_frame = _event(frame="not-really-bytes")
    try:
        cloud.sanitize_event(with_frame)
    except cloud.EventRejected as error:
        assert "frames never leave" in str(error)
    else:
        raise AssertionError("a frame field must be rejected")

    try:
        cloud.sanitize_event(_event(thumbnail="abc"))
    except cloud.EventRejected:
        pass
    else:
        raise AssertionError("a thumbnail field must be rejected")

    try:
        cloud.sanitize_event({"event": "remind", "observed": "data:image/png;base64,aaaa"})
    except cloud.EventRejected as error:
        assert "image" in str(error)
    else:
        raise AssertionError("a data URL must be rejected")


def test_sanitize_keeps_a_normal_text_event() -> None:
    clean = cloud.sanitize_event(
        _event(
            action="remind",
            away_seconds=60,
            reminders_unanswered=1,
            stove_occupied=True,
            hands_present=False,
        )
    )
    assert clean["event"] == "remind"
    assert clean["away_seconds"] == 60
    assert "frame" not in clean


def test_bool_is_not_accepted_as_a_number() -> None:
    try:
        cloud.sanitize_event(_event(away_seconds=True))
    except cloud.EventRejected:
        return
    raise AssertionError("true is not a duration")


def test_off_mode_does_not_touch_the_network(tmp_path: Path) -> None:
    def transport(_event: dict) -> str:
        raise AssertionError("off mode must not send")

    client = _client(tmp_path, transport, mode="off")
    assert client.send(_event()) is True
    assert client.queued_events() == []


def test_failure_is_queued_and_retried_in_order(tmp_path: Path) -> None:
    calls: list[str] = []
    clock = {"now": 0.0}

    def transport(event: dict) -> str:
        calls.append(str(event["observed"]))
        if event["observed"] == "first" and calls.count("first") == 1:
            return "retry"
        return "ok"

    client = _client(tmp_path, transport, now=lambda: clock["now"])
    assert client.send(_event(observed="first")) is False
    assert [item["observed"] for item in client.queued_events()] == ["first"]

    # Inside the cooldown the new event waits behind the old one. No network.
    assert client.send(_event(observed="second")) is False
    assert calls == ["first"]
    assert [item["observed"] for item in client.queued_events()] == ["first", "second"]

    clock["now"] = 15.0
    assert client.send(_event(observed="third")) is True
    assert calls == ["first", "first", "second", "third"]
    assert client.queued_events() == []


def test_rejected_payload_is_not_queued(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda _event: "drop")
    assert client.send(_event()) is False
    assert client.queued_events() == []


def test_aws_refuses_the_localhost_demo_key(tmp_path: Path) -> None:
    def transport(_event: dict) -> str:
        raise AssertionError("the demo key must not be sent to AWS")

    client = _client(
        tmp_path,
        transport,
        mode="aws",
        api_key=cloud.LOCAL_DEV_KEY,
        api_url="https://example.amazonaws.com",
    )
    assert client.send(_event()) is False
    assert client.queued_events() == []


class _QuietResponse:
    def read(self) -> bytes:
        return b""

    def __enter__(self):
        return self

    def __exit__(self, *args) -> bool:
        return False


def test_a_rejected_new_field_is_sent_again_without_it(monkeypatch) -> None:
    bodies: list[dict] = []

    def fake_open(request, timeout=0):
        body = json.loads(request.data.decode())
        bodies.append(body)
        if "urgency" in body or "caregiver_summary" in body:
            raise urllib.error.HTTPError(
                request.full_url,
                400,
                "bad",
                None,
                io.BytesIO(b"unknown field"),
            )
        return _QuietResponse()

    monkeypatch.setattr(cloud.urllib.request, "urlopen", fake_open)
    event = _event(
        name="escalate",
        reason="I haven't seen you come back, so I'm letting your caregiver know.",
        urgency="firm",
        model="nvidia/nemotron-3-super-120b-a12b",
        source="nemotron",
        caregiver_summary="Unattended for 240 seconds after 2 reminders.",
    )
    assert cloud.urlopen_post("https://example.amazonaws.com", "secret", event, 1.5) == "ok"
    assert len(bodies) == 2
    assert "urgency" not in bodies[1]
    assert "model" not in bodies[1]
    assert "source" not in bodies[1]
    assert "caregiver_summary" not in bodies[1]
    assert "240" in bodies[1]["reason"]
    assert "secret" not in json.dumps(bodies[1])
    assert bodies[1]["event"] == "escalate"


def test_an_auth_failure_is_not_resent_without_fields(monkeypatch) -> None:
    calls = {"n": 0}

    def fake_open(request, timeout=0):
        calls["n"] += 1
        raise urllib.error.HTTPError(
            request.full_url, 401, "no", None, io.BytesIO(b"no")
        )

    monkeypatch.setattr(cloud.urllib.request, "urlopen", fake_open)
    event = _event(urgency="firm", caregiver_summary="Unattended for 240 seconds.")
    assert cloud.urlopen_post("https://example.amazonaws.com", "secret", event, 1.5) == "drop"
    assert calls["n"] == 1


def test_a_legacy_rejection_is_not_retried(monkeypatch) -> None:
    calls = {"n": 0}

    def fake_open(request, timeout=0):
        calls["n"] += 1
        raise urllib.error.HTTPError(
            request.full_url, 400, "bad", None, io.BytesIO(b"no")
        )

    monkeypatch.setattr(cloud.urllib.request, "urlopen", fake_open)
    assert cloud.urlopen_post("https://example.amazonaws.com", "secret", _event(), 1.5) == "drop"
    assert calls["n"] == 1


def test_request_keeps_the_key_out_of_the_body() -> None:
    request = cloud.build_request(
        "http://127.0.0.1:8787",
        "secret-key",
        _event(),
    )
    assert request.get_header("X-api-key") == "secret-key"
    assert request.data is not None
    assert b"secret-key" not in request.data


def test_env_file_does_not_override_real_variables(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "NUDGE_CLOUD=aws\nNUDGE_API_KEY=from-file\nNUDGE_DEVICE_NAME=stove\n",
        encoding="utf-8",
    )
    environ = {"NUDGE_CLOUD": "off", "NUDGE_API_KEY": "from-env"}
    cloud.load_env_file(env_file, environ)
    assert environ["NUDGE_CLOUD"] == "off"
    assert environ["NUDGE_API_KEY"] == "from-env"
    assert environ["NUDGE_DEVICE_NAME"] == "stove"


def test_unknown_cloud_mode_stays_offline(tmp_path: Path) -> None:
    client = cloud.CloudClient.from_env(
        queue_path=tmp_path / "outbox.jsonl",
        environ={"NUDGE_CLOUD": "maybe"},
    )
    assert client.mode == "off"


def test_decision_event_is_text_only() -> None:
    payload = cloud.text_event(
        "escalate",
        "I haven't seen you come back, so I'm letting your caregiver know.",
        "2026-10-25T12:00:00",
        "kitchen",
        action="escalate",
        reason="Four reminders went unanswered.",
        trace="ESCALATE   stove occupied, hands away, away for 420s, 4 reminders unanswered",
        away_seconds=420,
        reminders_unanswered=4,
        stove_occupied=True,
        hands_present=False,
        prompt="I haven't seen you come back, so I'm letting your caregiver know.",
    )
    clean = cloud.sanitize_event(payload)
    assert clean["event"] == "escalate"
    assert clean["device"] == "kitchen"
    assert clean["reminders_unanswered"] == 4
    encoded = json.dumps(clean)
    assert "frame" not in encoded
    assert "already notified" not in encoded
    assert len(encoded.encode("utf-8")) < cloud.MAX_BYTES


def test_log_event_can_carry_caregiver_numbers(tmp_path: Path, monkeypatch) -> None:
    sent: list[dict] = []
    monkeypatch.setattr(main, "LOG_FILE", tmp_path / "events.jsonl")
    client = _client(tmp_path, lambda event: sent.append(event) or "ok")
    monkeypatch.setattr(main, "active_cloud", client)
    main.log_event(
        "escalate",
        "I haven't seen you come back, so I'm letting your caregiver know.",
        [],
        details={
            "action": "escalate",
            "reminders_unanswered": 4,
            "away_seconds": 420.0,
            "stove_occupied": True,
            "hands_present": False,
        },
    )
    assert sent[0]["event"] == "escalate"
    assert sent[0]["reminders_unanswered"] == 4
    assert sent[0]["device"] == "kitchen"
    assert "frame" not in sent[0]


def test_log_event_forwards_text_only(tmp_path: Path, monkeypatch) -> None:
    sent: list[dict] = []
    monkeypatch.setattr(main, "LOG_FILE", tmp_path / "events.jsonl")
    client = _client(tmp_path, lambda event: sent.append(event) or "ok")
    monkeypatch.setattr(main, "active_cloud", client)
    main.log_event("hands_left_zone", "Hands left the counter", [])
    assert sent[0]["event"] == "hands_left_zone"
    assert sent[0]["device"] == "kitchen"
    assert "image" not in sent[0]


def test_privacy_sentence_is_on_the_dashboard() -> None:
    html = (Path(__file__).resolve().parents[1] / "aws" / "src" / "dashboard.html").read_text()
    assert cloud.PRIVACY_SENTENCE in html


def test_mock_server_stores_text_and_refuses_frames(tmp_path: Path) -> None:
    server = mock_cloud.make_server(
        "127.0.0.1",
        0,
        tmp_path / "events.jsonl",
        cloud.LOCAL_DEV_KEY,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        with urllib.request.urlopen(base + "/health", timeout=2) as response:
            page = response.read().decode("utf-8")
        assert cloud.PRIVACY_SENTENCE in page

        with urllib.request.urlopen(base + "/dashboard", timeout=2) as response:
            dashboard = response.read().decode("utf-8")
        assert cloud.PRIVACY_SENTENCE in dashboard
        assert "What happened" in dashboard

        client = cloud.CloudClient(
            mode="mock",
            api_url=base,
            api_key=cloud.LOCAL_DEV_KEY,
            device="kitchen",
            queue_path=tmp_path / "outbox.jsonl",
            timeout_sec=2,
        )
        assert client.send(_event("escalate", observed="Please check the stove."))

        request = urllib.request.Request(
            base + "/events",
            headers={"x-api-key": cloud.LOCAL_DEV_KEY},
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            body = json.loads(response.read().decode("utf-8"))
        assert body["events"][0]["event"] == "escalate"
        assert body["events"][0]["device"] == "kitchen"

        denied = urllib.request.Request(base + "/events")
        try:
            urllib.request.urlopen(denied, timeout=2)
        except urllib.error.HTTPError as error:
            assert error.code == 401
        else:
            raise AssertionError("events must require a key")

        bad = json.dumps(_event(frame="pixels")).encode("utf-8")
        rejected = urllib.request.Request(
            base + "/events",
            data=bad,
            headers={
                "content-type": "application/json",
                "x-api-key": cloud.LOCAL_DEV_KEY,
            },
        )
        try:
            urllib.request.urlopen(rejected, timeout=2)
        except urllib.error.HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("frames must be refused")

        with urllib.request.urlopen(request, timeout=2) as response:
            after = json.loads(response.read().decode("utf-8"))
        assert len(after["events"]) == 1
    finally:
        server.shutdown()
        server.server_close()
