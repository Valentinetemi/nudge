"""Nemotron decisions with the Nebius HTTP call replaced by a fake."""

import io
import json
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import nudge
import reason

MODEL = "nvidia/nemotron-3-super-120b-a12b"
KEY = "test-key-not-real"


def situation(**overrides) -> nudge.Situation:
    facts = {
        "stove_occupied": True,
        "hands_nearby": False,
        "away_seconds": 240,
        "reminders_given": 2,
        "max_reminders": 4,
        "escalate_allowed": True,
        "rule_would": "remind",
        "last_message": "Something is still on the stove.",
        "wait_seconds": 180,
        "repeat_seconds": 60,
    }
    facts.update(overrides)
    return nudge.Situation(**facts)


def completion(content) -> bytes:
    body = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    return json.dumps(body).encode("utf-8")


ESCALATE = json.dumps(
    {
        "action": "escalate",
        "urgency": "firm",
        "message": "I haven't seen you come back, so I'm letting your caregiver know.",
        "caregiver_summary": (
            "The stove has been unattended for 240 seconds after 2 reminders."
        ),
    }
)


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args) -> bool:
        return False


def _http_error(status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://api.tokenfactory.us-central1.nebius.com/v1/chat/completions",
        status,
        "error",
        hdrs=None,
        fp=io.BytesIO(b"no"),
    )


def test_a_missing_key_does_not_call_the_network() -> None:
    def opener(request, timeout=0):
        raise AssertionError("the network must stay unused without a key")

    client = reason.NebiusReasoner(api_key="", opener=opener)
    result = client.advise(situation())
    assert result.advice is None
    assert result.fallback is not None
    assert "NEBIUS_API_KEY" in result.fallback
    assert KEY not in result.fallback


def test_the_default_model_and_host_match_the_nemotron_quickstart() -> None:
    client = reason.NebiusReasoner.from_env(environ={})
    assert client.model == "nvidia/nemotron-3-super-120b-a12b"
    assert client.base_url == "https://api.tokenfactory.us-central1.nebius.com/v1"
    assert client.endpoint == (
        "https://api.tokenfactory.us-central1.nebius.com/v1/chat/completions"
    )
    described = client.describe()
    assert "NEBIUS_API_KEY is not set" in described
    assert "sk-" not in described


def test_the_model_id_can_be_overridden() -> None:
    client = reason.NebiusReasoner.from_env(
        environ={
            "NEBIUS_API_KEY": KEY,
            "NEBIUS_MODEL": "nvidia/Nemotron-3-Ultra-550b-a55b",
            "NEBIUS_BASE_URL": "https://api.tokenfactory.us-central1.nebius.com/v1/",
        }
    )
    assert client.model == "nvidia/Nemotron-3-Ultra-550b-a55b"
    assert client.base_url == "https://api.tokenfactory.us-central1.nebius.com/v1"
    assert KEY not in client.describe()


def test_a_successful_call_posts_text_facts_and_returns_the_decision() -> None:
    seen = []

    def opener(request, timeout=0):
        seen.append(request)
        return _Response(completion(ESCALATE))

    client = reason.NebiusReasoner(api_key=KEY, opener=opener, timeout_sec=5)
    result = client.advise(situation())
    assert len(seen) == 1
    request = seen[0]
    assert request.full_url == client.endpoint
    assert request.get_header("Authorization") == f"Bearer {KEY}"
    assert request.data is not None
    body = json.loads(request.data.decode("utf-8"))
    assert body["model"] == MODEL
    assert body["stream"] is False
    assert body["reasoning_effort"] == "low"
    assert body["response_format"] == {"type": "json_object"}
    encoded = request.data.decode("utf-8")
    assert KEY not in encoded
    assert "240" in encoded
    assert "reminders already spoken" in encoded
    # The prompt may say that no camera image was sent. It must not send one.
    assert "data:image" not in encoded
    assert "image_url" not in encoded
    assert "frame" not in encoded
    assert result.advice is not None
    assert result.fallback is None
    assert result.advice.action == "escalate"
    assert result.advice.urgency == "firm"
    assert "caregiver" in result.advice.message
    assert "240" in result.advice.caregiver_summary
    assert result.advice.model == MODEL
    assert "already notified" not in result.advice.message


def test_a_timeout_falls_back_to_the_rules() -> None:
    def opener(request, timeout=0):
        raise TimeoutError("timed out")

    result = reason.NebiusReasoner(api_key=KEY, opener=opener).advise(situation())
    assert result.advice is None
    assert result.fallback is not None
    assert "timed out" in result.fallback
    assert KEY not in result.fallback


def test_an_http_error_falls_back_without_the_key() -> None:
    def opener(request, timeout=0):
        raise _http_error(500)

    result = reason.NebiusReasoner(api_key=KEY, opener=opener).advise(situation())
    assert result.advice is None
    assert result.fallback is not None
    assert "HTTP 500" in result.fallback
    assert KEY not in result.fallback


def test_a_rejected_optional_field_is_retried_once() -> None:
    calls = {"n": 0}

    def opener(request, timeout=0):
        calls["n"] += 1
        body = json.loads(request.data.decode("utf-8"))
        if "reasoning_effort" in body:
            raise _http_error(400)
        return _Response(completion(ESCALATE))

    result = reason.NebiusReasoner(api_key=KEY, opener=opener).advise(situation())
    assert calls["n"] == 2
    assert result.advice is not None
    assert result.advice.action == "escalate"


def test_thinking_tags_and_a_fence_still_parse() -> None:
    text = (
        "<think>The person has already heard two reminders.</think>\n"
        "```json\n"
        + ESCALATE
        + "\n```"
    )
    advice = reason.parse_decision(text, model=MODEL)
    assert advice.action == "escalate"
    assert "240" in advice.caregiver_summary


def test_a_list_of_content_parts_is_read() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": ESCALATE},
                    ]
                }
            }
        ]
    }
    assert "escalate" in reason.message_text(payload)


def test_bad_json_and_a_scolding_line_fall_back() -> None:
    def bad_json(request, timeout=0):
        return _Response(completion("not json at all"))

    result = reason.NebiusReasoner(api_key=KEY, opener=bad_json).advise(situation())
    assert result.advice is None
    assert result.fallback is not None
    assert "JSON" in result.fallback

    def scold(request, timeout=0):
        payload = json.dumps(
            {
                "action": "remind",
                "urgency": "firm",
                "message": "That was careless. Check the stove.",
                "caregiver_summary": "",
            }
        )
        return _Response(completion(payload))

    result = reason.NebiusReasoner(api_key=KEY, opener=scold).advise(situation())
    assert result.advice is None
    assert result.fallback is not None
    assert "not allowed" in result.fallback


def test_a_delivery_claim_that_omits_caregiver_is_rejected() -> None:
    text = json.dumps(
        {
            "action": "remind",
            "urgency": "firm",
            "message": "I already called them about the stove.",
            "caregiver_summary": "",
        }
    )
    try:
        reason.parse_decision(text, model=MODEL)
    except reason.ReasoningError as error:
        assert "not allowed" in str(error)
        assert "contacted" in str(error)
    else:
        raise AssertionError("a delivery claim must be rejected")


def test_scolding_outside_the_old_phrase_list_is_rejected() -> None:
    text = json.dumps(
        {
            "action": "remind",
            "urgency": "firm",
            "message": "That was foolish. Look at the stove.",
            "caregiver_summary": "",
        }
    )
    try:
        reason.parse_decision(text, model=MODEL)
    except reason.ReasoningError as error:
        assert "scolds" in str(error)
    else:
        raise AssertionError("an insult must be rejected")


def test_the_canned_ask_is_still_allowed() -> None:
    text = json.dumps(
        {
            "action": "escalate",
            "urgency": "firm",
            "message": "I haven't seen you come back, so I'm letting your caregiver know.",
            "caregiver_summary": "The stove has been unattended for 240 seconds after 2 reminders.",
        }
    )
    advice = reason.parse_decision(text, model=MODEL)
    assert advice.action == "escalate"
    assert "caregiver" in advice.message


def test_a_claim_that_the_caregiver_was_already_told_is_rejected() -> None:
    text = json.dumps(
        {
            "action": "escalate",
            "urgency": "firm",
            "message": "Your caregiver was already notified about the stove.",
            "caregiver_summary": "Unattended for 240 seconds after 2 reminders.",
        }
    )
    try:
        reason.parse_decision(text, model=MODEL)
    except reason.ReasoningError as error:
        assert "not allowed" in str(error)
    else:
        raise AssertionError("a claim of delivery must be rejected")


def test_an_untrusted_base_url_does_not_receive_the_key() -> None:
    refused = [
        "http://api.tokenfactory.us-central1.nebius.com/v1",
        "https://evil.example/v1",
        "https://api.tokenfactory.us-central1.nebius.com.evil.example/v1",
        "https://user:secret@api.tokenfactory.us-central1.nebius.com/v1",
        "https://api.tokenfactory.us-central1.nebius.com/v1?token=1",
        "https://127.0.0.1/v1",
        "https://api.tokenfactory.us-central1.nebius.com:8443/v1",
        "https://api.tokenfactory.nebius.com/v1/chat/completions",
    ]

    def opener(request, timeout=0):
        raise AssertionError("the key must not be sent")

    for base in refused:
        client = reason.NebiusReasoner(api_key=KEY, base_url=base, opener=opener)
        result = client.advise(situation())
        assert result.advice is None
        assert result.fallback is not None
        assert "API key" in result.fallback
        assert KEY not in result.fallback
        assert "evil.example" not in result.fallback
        assert "secret" not in result.fallback
        assert client.endpoint == ""
        assert "evil.example" not in client.describe()
        try:
            client._post({"model": MODEL})
        except reason.ReasoningError as error:
            assert KEY not in str(error)
            assert "evil.example" not in str(error)
        else:
            raise AssertionError("post must fail closed before opening a connection")


def test_a_documented_token_factory_host_is_allowed() -> None:
    seen = []

    def opener(request, timeout=0):
        seen.append(request.full_url)
        return _Response(completion(ESCALATE))

    client = reason.NebiusReasoner(
        api_key=KEY,
        base_url="https://api.tokenfactory.nebius.com/v1/",
        opener=opener,
    )
    result = client.advise(situation())
    assert result.advice is not None
    assert seen == ["https://api.tokenfactory.nebius.com/v1/chat/completions"]


def test_the_prompt_names_the_event_and_not_a_picture() -> None:
    messages = reason.build_messages(situation(away_seconds=95, reminders_given=1))
    user = messages[1]["content"]
    assert "seconds unattended: 95" in user
    assert "reminders already spoken: 1" in user
    assert "frame" not in user
    assert "jpeg" not in user
    assert "png" not in user
