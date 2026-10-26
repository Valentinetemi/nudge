"""Local stand-in for the AWS caregiver API. No credentials and no video.

Example:
    python scripts/mock_cloud.py

Then open the printed dashboard URL, and in another terminal:

    NUDGE_CLOUD=mock NUDGE_API_URL=http://127.0.0.1:8787 NUDGE_API_KEY=local-dev-key \
        python src/main.py --nudge-after 15

The default key is for this laptop only. The AWS client refuses to send it.
Events are stored as JSON lines under data/mock_events.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cloud  # noqa: E402  -- src/ is added to the path just above.

DASHBOARD_FILE = ROOT / "aws" / "src" / "dashboard.html"
DEFAULT_STORE = ROOT / "data" / "mock_events.jsonl"


def health_page() -> bytes:
    """A public page with no personal events, so a browser can see the service."""
    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Nudge</title>
</head>
<body>
  <h1>Nudge caregiver endpoint</h1>
  <p>{cloud.PRIVACY_SENTENCE}</p>
  <p>Open <a href="/dashboard">/dashboard</a> and add the token you were given
  to see text events. This page itself stores nothing.</p>
</body>
</html>
"""
    return body.encode("utf-8")


class CaregiverHandler(BaseHTTPRequestHandler):
    """The four routes the real API will have: health, dashboard, read, write."""

    server_version = "NudgeMock/1.0"

    def log_message(self, fmt: str, *args) -> None:
        # Keep request lines, but never print headers (they hold the API key).
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _server_state(self):
        return self.server

    def do_OPTIONS(self) -> None:  # noqa: N802  -- stdlib handler name
        self._send(204, b"", "text/plain")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/health"):
            self._send(200, health_page(), "text/html; charset=utf-8")
            return
        if parsed.path == "/dashboard":
            if DASHBOARD_FILE.exists():
                body = DASHBOARD_FILE.read_bytes()
            else:
                body = health_page()
            self._send(200, body, "text/html; charset=utf-8")
            return
        if parsed.path == "/events":
            if not self._allowed(write=False):
                self._send_json(401, {"error": "missing or wrong API key"})
                return
            limit = _limit(parse_qs(parsed.query).get("limit", ["50"])[0])
            events = _read_events(self.server.store_path)[-limit:]
            events.reverse()
            self._send_json(200, {"events": events})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/events":
            self._send_json(404, {"error": "not found"})
            return
        if not self._allowed(write=True):
            self._send_json(401, {"error": "missing or wrong API key"})
            return
        length = int(self.headers.get("content-length", "0") or "0")
        if length > cloud.MAX_BYTES:
            self._send_json(413, {"error": "event is too large"})
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            event = cloud.sanitize_event(payload)
        except (UnicodeError, json.JSONDecodeError, cloud.EventRejected) as error:
            self._send_json(400, {"error": str(error)})
            return
        _append_event(self.server.store_path, self.server.store_lock, event)
        if event.get("event") == "escalate" or event.get("action") == "escalate":
            print("Local caregiver alert (no SMS from the mock):")
            print(f"  {event.get('observed', '')}")
        self._send_json(201, {"ok": True})

    def _allowed(self, write: bool) -> bool:
        provided = self.headers.get("x-api-key", "")
        if not write and not provided:
            # The dashboard page puts the token in the header itself. A query
            # token is accepted on reads so a caregiver link still works if a
            # client cannot set headers.
            query = parse_qs(urlparse(self.path).query)
            provided = (query.get("token") or [""])[0]
        expected = self.server.ingest_key if write else self.server.read_key
        return cloud.keys_match(provided, expected)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-headers", "content-type, x-api-key")
        self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")


def make_server(
    host: str,
    port: int,
    store_path: Path,
    ingest_key: str,
    read_key: str | None = None,
) -> ThreadingHTTPServer:
    """Build the local API. Port 0 asks the OS for a free port (used by tests)."""
    httpd = ThreadingHTTPServer((host, port), CaregiverHandler)
    httpd.store_path = store_path
    httpd.store_lock = threading.Lock()
    httpd.ingest_key = ingest_key
    # One key is enough for a demo. A separate read key matches the AWS option.
    httpd.read_key = read_key or ingest_key
    return httpd


def _limit(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        return 50
    return min(max(value, 1), 200)


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _append_event(path: Path, lock: threading.Lock, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Nudge caregiver API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    # load a local .env if present, without overriding exported variables.
    cloud.load_env_file(ROOT / ".env")
    ingest_key = os.environ.get("NUDGE_API_KEY", "").strip() or cloud.LOCAL_DEV_KEY
    read_key = os.environ.get("NUDGE_DASHBOARD_TOKEN", "").strip() or ingest_key
    if ingest_key == cloud.LOCAL_DEV_KEY:
        print("Using local-dev-key. This key is for localhost only. Do not deploy it.")
    httpd = make_server(args.host, args.port, args.store, ingest_key, read_key)
    host, port = httpd.server_address[:2]
    print(f"Mock caregiver API at http://{host}:{port}")
    print(f"Dashboard: http://{host}:{port}/dashboard?token={read_key}")
    print(cloud.PRIVACY_SENTENCE)
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
