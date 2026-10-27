"""Ask NVIDIA Nemotron, on Nebius Token Factory, what Nudge should do next.

The camera still decides when a reminder is due. This module runs only at
that moment. It sends text facts (how long the stove has been unattended,
how many reminders were already spoken) to the OpenAI-compatible chat
completions API and reads back a wording, an urgency, and whether to ask a
caregiver. No frame is included.

If the API key is missing, or the call fails or times out, the caller keeps
the built-in reminder rules. The failure text is safe to write to the
decision log: it never contains the key.

The default model id and base URL are the pair Nebius documents for
Nemotron 3 Super on Token Factory (us-central1):

    model    nvidia/nemotron-3-super-120b-a12b
    base URL https://api.tokenfactory.us-central1.nebius.com/v1
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

import nudge

# Catalog entry checked 27 October 2026: text-to-text, region us-central1,
# status active. The same id is in the Token Factory Nemotron quickstart.
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"

# Regional OpenAI-compatible host from that quickstart and from the
# Nemotron 3 Super cookbook page. The generic api.tokenfactory.nebius.com
# host is for other models; this one is served from us-central1.
DEFAULT_BASE_URL = "https://api.tokenfactory.us-central1.nebius.com/v1"

# Long enough for a short reasoning trace plus the JSON object, short
# enough that a stuck call gives the camera loop back to the rules.
DEFAULT_TIMEOUT_SEC = 20.0

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

# Spoken lines are said to someone who may be confused. These phrases
# either scold or claim the caregiver has already been told.
_FORBIDDEN = (
    "already notified",
    "already told",
    "have notified",
    "has been notified",
    "i notified",
    "stupid",
    "idiot",
    "careless",
    "dumb",
)

_URGENCY = {
    "calm": "calm",
    "gentle": "calm",
    "low": "calm",
    "direct": "direct",
    "medium": "direct",
    "firm": "firm",
    "urgent": "firm",
    "high": "firm",
}


class ReasoningError(Exception):
    """The model call did not produce a decision. The message has no API key."""


def build_messages(situation: nudge.Situation) -> list[dict[str, str]]:
    """The text prompt for one due moment. Facts only, no picture."""
    if situation.max_reminders is None:
        limit = "no fixed limit, and escalation is not allowed"
    else:
        limit = str(situation.max_reminders)
    last = situation.last_message or "(nothing has been said yet)"
    user = (
        "Facts:\n"
        f"- stove occupied: {'yes' if situation.stove_occupied else 'no'}\n"
        f"- hands nearby: {'yes' if situation.hands_nearby else 'no'}\n"
        f"- seconds unattended: {int(situation.away_seconds)}\n"
        f"- reminders already spoken: {situation.reminders_given}\n"
        f"- maximum reminders before the rules require a caregiver ask: {limit}\n"
        f"- escalation allowed: {'yes' if situation.escalate_allowed else 'no'}\n"
        f"- first wait seconds: {int(situation.wait_seconds)}\n"
        f"- repeat wait seconds: {int(situation.repeat_seconds)}\n"
        f"- last sentence spoken in the room: {last}\n"
        f"- the built-in rules would: {situation.rule_would}\n"
        "\n"
        "Choose the next action from these facts."
    )
    system = (
        "You are the decision step for Nudge, a calm stove reminder for "
        "someone who may be confused. You choose the next spoken sentence "
        "and whether a caregiver should be asked. You never scold, never "
        "use sarcasm, and never say a caregiver has already been told. "
        "You know only the facts in the user message. No camera image is "
        "available. Reply with one JSON object and nothing else.\n"
        "Fields:\n"
        '- action: "remind" or "escalate"\n'
        '- urgency: "calm", "direct", or "firm"\n'
        "- message: one short sentence to say aloud\n"
        '- caregiver_summary: one or two sentences for the caregiver when '
        'action is escalate, otherwise ""\n'
        "Use escalate only after at least one reminder has already been "
        "spoken, the person is still away, and another reminder is unlikely "
        "to be enough. If escalation is not allowed, action must be remind. "
        "A firm sentence is still kind. The summary must mention how long "
        "the stove has been unattended and how many reminders were spoken."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def message_text(payload: object) -> str:
    """Pull the assistant text out of a chat-completions JSON body."""
    if not isinstance(payload, dict):
        raise ReasoningError("response was not a JSON object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ReasoningError("response had no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ReasoningError("response choice was not an object")
    message = first.get("message") or {}
    if not isinstance(message, dict):
        raise ReasoningError("response message was not an object")
    content = _coerce_content(message.get("content"))
    if content:
        return content
    # Some reasoning models leave the answer in a side field.
    for key in ("reasoning_content", "reasoning"):
        content = _coerce_content(message.get(key))
        if content:
            return content
    raise ReasoningError("response had no text")


def parse_decision(text: str, *, model: str) -> nudge.Advice:
    """Turn model text into a decision, or raise ReasoningError.

    Nemotron may wrap the object in a thinking block or a markdown fence.
    Either is stripped before the JSON parse. The spoken line is checked
    so a scolding sentence, or a claim that the caregiver was already
    told, never reaches the room.
    """
    data = _load_json_object(text)
    action = str(data.get("action", "")).strip().lower()
    if action not in {"remind", "escalate"}:
        raise ReasoningError("model action was not remind or escalate")

    urgency = _URGENCY.get(str(data.get("urgency", "calm")).strip().lower(), "calm")
    spoken = _one_line(data.get("message", ""), limit=280)
    if len(spoken) < 8 or not any(character.isalpha() for character in spoken):
        raise ReasoningError("model message was empty")
    _reject_forbidden(spoken)

    summary_raw = data.get("caregiver_summary") or ""
    summary = _one_line(summary_raw, limit=600)
    if action == "remind":
        summary = ""
    else:
        if len(summary) < 8:
            raise ReasoningError("escalate decision had no caregiver summary")
        _reject_forbidden(summary)

    return nudge.Advice(
        action=action,
        urgency=urgency,
        message=spoken,
        caregiver_summary=summary,
        model=model,
    )


class NebiusReasoner:
    """One OpenAI-compatible client for Nebius Token Factory."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        opener=None,
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model.strip() or DEFAULT_MODEL
        self.base_url = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
        self.timeout_sec = timeout_sec
        self._opener = opener or urllib.request.urlopen

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None, opener=None) -> NebiusReasoner:
        """Read NEBIUS_API_KEY and the optional model and base URL.

        A missing key still returns a reasoner. `advise` then reports a
        fallback and does not open a connection. Real environment variables
        are the caller's job to load; this does not read `.env` itself.
        """
        env = os.environ if environ is None else environ
        return cls(
            api_key=env.get("NEBIUS_API_KEY", ""),
            model=env.get("NEBIUS_MODEL", ""),
            base_url=env.get("NEBIUS_BASE_URL", ""),
            opener=opener,
        )

    @property
    def endpoint(self) -> str:
        """Chat completions URL. The key is not part of the URL."""
        return self.base_url + "/chat/completions"

    def describe(self) -> str:
        """One startup line. Does not include the API key."""
        if not self.api_key:
            return (
                "Nemotron: off. NEBIUS_API_KEY is not set, "
                "so reminder wording stays on the built-in rules."
            )
        return f"Nemotron: {self.model} via Nebius Token Factory at {self.base_url}."

    def advise(self, situation: nudge.Situation) -> nudge.ReasoningResult:
        """Ask the model, or return a fallback the rules can act on."""
        if not self.api_key:
            return nudge.ReasoningResult(
                fallback="NEBIUS_API_KEY is not set; using the reminder rules"
            )
        print(
            f"Asking {self.model} about {int(situation.away_seconds)}s unattended, "
            f"{situation.reminders_given} reminders already spoken..."
        )
        try:
            text = self._complete(build_messages(situation))
            return nudge.ReasoningResult(advice=parse_decision(text, model=self.model))
        except TimeoutError:
            return nudge.ReasoningResult(
                fallback="Nebius call timed out; using the reminder rules"
            )
        except ReasoningError as error:
            return nudge.ReasoningResult(
                fallback=f"Nebius call failed ({error}); using the reminder rules"
            )
        except Exception:
            # A bug in the parser must not take down the camera loop.
            return nudge.ReasoningResult(
                fallback="Nebius call failed (unexpected error); using the reminder rules"
            )

    def _complete(self, messages: list[dict[str, str]]) -> str:
        """POST once, and once more without optional fields if those are rejected.

        `reasoning_effort` keeps Nemotron 3 Super on a short trace so the
        JSON fits in the timeout. `response_format` asks for a JSON object.
        A 400 or 422 means this host does not accept one of those fields,
        so the second try is a plain chat completion. Timeouts are not
        retried.
        """
        for include_extras in (True, False):
            try:
                payload = self._post(self._body(messages, include_extras=include_extras))
            except urllib.error.HTTPError as error:
                error.read()
                if error.code in (400, 422) and include_extras:
                    continue
                raise ReasoningError(f"HTTP {error.code}") from None
            except TimeoutError:
                raise
            except urllib.error.URLError as error:
                reason = getattr(error, "reason", None)
                if isinstance(reason, TimeoutError):
                    raise TimeoutError("timed out") from None
                raise ReasoningError("network") from None
            except OSError as error:
                raise ReasoningError("network") from error
            return message_text(payload)
        raise ReasoningError("HTTP 400")

    def _body(self, messages: list[dict[str, str]], *, include_extras: bool) -> dict:
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1024 if include_extras else 2048,
            "stream": False,
        }
        if include_extras:
            body["reasoning_effort"] = "low"
            body["response_format"] = {"type": "json_object"}
        return body

    def _post(self, body: dict) -> dict:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with self._opener(request, timeout=self.timeout_sec) as response:
            raw = response.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReasoningError("response was not JSON") from error
        if not isinstance(payload, dict):
            raise ReasoningError("response was not a JSON object")
        return payload


def _coerce_content(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    return ""


def _load_json_object(text: str) -> dict:
    cleaned = _THINK_BLOCK.sub("", text).strip()
    cleaned = _FENCE.sub("", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ReasoningError("model did not return a JSON object")
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as error:
        raise ReasoningError("model did not return a JSON object") from error
    if not isinstance(data, dict):
        raise ReasoningError("model did not return a JSON object")
    return data


def _one_line(value: object, *, limit: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ReasoningError("model field was not text")
    text = " ".join(value.split())
    if len(text) > limit:
        raise ReasoningError("model field was too long")
    if "data:image" in text or "data:video" in text:
        raise ReasoningError("model field looked like an image")
    return text


def _reject_forbidden(text: str) -> None:
    lowered = text.lower()
    for phrase in _FORBIDDEN:
        if phrase in lowered:
            raise ReasoningError("model wording was not allowed")
