"""Tests for scripts/calendar-delete-event.sh.

The script is the last hop before a human believes a calendar event is gone,
so it is exercised end to end: a real bash process, a real curl, against a
throwaway HTTP server standing in for calendar-agent. Nothing here touches a
live calendar or proxy.

Issue #4: the wrapper must distinguish three outcomes — the deletion happened,
it definitely did not happen, or nobody knows yet — and must decide by
re-reading the event rather than by trusting the delete's own answer.
"""

import json
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "calendar-delete-event.sh"

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_UNKNOWN = 2
EXIT_NOT_FOUND = 3
EXIT_USAGE = 4

EVENT_PATH = "/calendars/primary/events/event_1"
ROUTER_404 = {"status": 404, "body": {"detail": "Not Found"}}  # FastAPI, no envelope
# calendar-agent's own refusal of a missing or wrong token (issue #15).
UNAUTHORIZED = {
    "status": 401,
    "body": {"success": False, "error": "Missing or invalid bearer token"},
}
API_TOKEN = "synthetic-script-token-0123456789abcdef"  # not a real credential


def token_fragments(token: str) -> list[str]:
    """The whole token and three 16-character pieces of it (start, middle,
    end), so output that carried most of a token is caught too."""
    middle = len(token) // 2 - 8
    return [token, token[:16], token[middle : middle + 16], token[-16:]]


class _FakeAgentHandler(BaseHTTPRequestHandler):
    """Serves ``server.script`` for the one real event route, and records what
    was asked. Any other path gets FastAPI's bare router 404, so a script that
    drops a path segment or hits a misconfigured base URL is caught (F10)."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # noqa: A003 - silence stderr noise
        pass

    def _respond(self, spec: dict) -> None:
        self.server.authorization.append(self.headers.get_all("Authorization") or [])
        self.server.injected.extend(self.headers.get_all("X-Injected") or [])
        required = self.server.required_token
        if self.path != EVENT_PATH:
            spec = ROUTER_404
        elif required and self.headers.get("Authorization") != f"Bearer {required}":
            spec = UNAUTHORIZED
        if spec.get("delay"):
            time.sleep(spec["delay"])
        payload = json.dumps(spec.get("body", {})).encode()
        try:
            self.send_response(spec["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            # curl hit --max-time and closed the socket: exactly what the
            # timeout scenarios are simulating.
            self.close_connection = True

    def do_DELETE(self):  # noqa: N802 - BaseHTTPRequestHandler naming
        self.server.calls.append(("DELETE", self.path))
        self._respond(self.server.script["delete"])

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler naming
        self.server.calls.append(("GET", self.path))
        self._respond(self.server.script["get"])


@pytest.fixture
def fake_agent():
    """Start a stub calendar-agent on loopback; yield a configurable handle."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeAgentHandler)
    server.daemon_threads = True
    server.calls = []
    server.authorization = []  # the Authorization header values of each request
    server.injected = []  # any X-Injected header values (see the whitespace test)
    server.required_token = None  # set to make the stub demand a token
    server.script = {
        "delete": {"status": 200, "body": {"success": True, "message": "ok"}},
        "get": {"status": 404, "body": {"success": False, "event": None}},
    }
    server.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def run_script(
    fake_agent,
    *,
    event_id="event_1",
    max_time="5",
    url=None,
    extra_env=None,
    path="/usr/bin:/bin:/usr/local/bin",
):
    # The stub answers in seconds, so its "mutation budget" is declared as 0:
    # the script's deadline floor is relative to it (F1).
    env = {
        "PATH": path,
        "CALENDAR_AGENT_URL": url or fake_agent.url,
        "CALENDAR_AGENT_CONFIRM_TIMEOUT": "0",
    }
    if max_time is not None:
        env["CALENDAR_DELETE_MAX_TIME"] = max_time
    for key, value in (extra_env or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return subprocess.run(
        [str(SCRIPT), event_id, "primary"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def confirmed_event(event_id="event_1"):
    return {
        "status": 200,
        "body": {"success": True, "event": {"id": event_id, "status": "confirmed"}},
    }


class TestVerifiedSuccess:
    """Exit 0 requires evidence the event is gone, never the delete's claim."""

    def test_deleted_event_gone_on_re_read_is_success(self, fake_agent):
        result = run_script(fake_agent)
        assert result.returncode == EXIT_SUCCESS
        assert "RESULT: SUCCESS" in result.stdout

    def test_cancelled_status_on_re_read_is_success(self, fake_agent):
        fake_agent.script["get"] = {
            "status": 200,
            "body": {"success": True, "event": {"id": "event_1", "status": "cancelled"}},
        }
        result = run_script(fake_agent)
        assert result.returncode == EXIT_SUCCESS
        assert "RESULT: SUCCESS" in result.stdout

    def test_exactly_one_delete_then_one_get_on_the_event_route(self, fake_agent):
        """Both requests must hit the real route with both ids in place (F10:
        a stub that answered any path let a dropped calendar segment pass)."""
        run_script(fake_agent)
        assert fake_agent.calls == [("DELETE", EVENT_PATH), ("GET", EVENT_PATH)]


class TestDefiniteFailure:
    """Exit 1 means the event demonstrably survived and nothing is pending."""

    def test_success_claim_contradicted_by_re_read_is_failure(self, fake_agent):
        """The 2026-08-07 incident: HTTP 200 for an event still confirmed."""
        fake_agent.script["get"] = confirmed_event()
        result = run_script(fake_agent)
        assert result.returncode == EXIT_FAILURE
        assert "RESULT: FAILURE" in result.stdout

    def test_success_false_in_body_is_not_success(self, fake_agent):
        """A 200 whose body says success:false must not be read as deleted."""
        fake_agent.script["delete"] = {
            "status": 200,
            "body": {"success": False, "message": "Failed to delete event"},
        }
        fake_agent.script["get"] = confirmed_event()
        result = run_script(fake_agent)
        assert result.returncode == EXIT_FAILURE

    def test_operator_rejection_is_failure(self, fake_agent):
        fake_agent.script["delete"] = {
            "status": 403,
            "body": {"success": False, "message": "rejected by operator"},
        }
        fake_agent.script["get"] = confirmed_event()
        result = run_script(fake_agent)
        assert result.returncode == EXIT_FAILURE
        assert "RESULT: FAILURE" in result.stdout


class TestUnknownOutcome:
    """Exit 2 means the deletion may still land; do not act on it either way."""

    def test_server_reported_unknown_outcome_is_unknown(self, fake_agent):
        """504 from calendar-agent: it gave up waiting for the operator."""
        fake_agent.script["delete"] = {
            "status": 504,
            "body": {"success": False, "message": "Deletion outcome unknown"},
        }
        fake_agent.script["get"] = confirmed_event()
        result = run_script(fake_agent)
        assert result.returncode == EXIT_UNKNOWN
        assert "RESULT: UNKNOWN" in result.stdout

    def test_client_timeout_with_event_still_present_is_unknown(self, fake_agent):
        """curl's own deadline expiring is not evidence the delete failed."""
        fake_agent.script["delete"] = {"status": 200, "body": {"success": True}, "delay": 3}
        fake_agent.script["get"] = confirmed_event()
        result = run_script(fake_agent, max_time="1")
        assert result.returncode == EXIT_UNKNOWN
        assert "RESULT: UNKNOWN" in result.stdout

    def test_client_timeout_still_verifies(self, fake_agent):
        """The verify GET must run even when the delete never answered."""
        fake_agent.script["delete"] = {"status": 200, "body": {"success": True}, "delay": 3}
        run_script(fake_agent, max_time="1")
        assert ("GET", "/calendars/primary/events/event_1") in fake_agent.calls

    def test_client_timeout_but_event_gone_is_success(self, fake_agent):
        """Approved late but before the re-read: the deletion did happen."""
        fake_agent.script["delete"] = {"status": 200, "body": {"success": True}, "delay": 3}
        result = run_script(fake_agent, max_time="1")
        assert result.returncode == EXIT_SUCCESS

    def test_unreadable_verification_is_unknown(self, fake_agent):
        """If the re-read itself fails, nothing has been established."""
        fake_agent.script["get"] = {"status": 502, "body": {"success": False}}
        result = run_script(fake_agent)
        assert result.returncode == EXIT_UNKNOWN
        assert "RESULT: UNKNOWN" in result.stdout

    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_server_error_with_event_still_present_is_unknown(self, fake_agent, status):
        """F2: calendar-agent answers 502 for a transport fault *after* the
        request reached the proxy, where it stays queued for approval. A 5xx
        is therefore not "nothing outstanding" — reporting FAILURE invites the
        compensating create the whole script exists to prevent."""
        fake_agent.script["delete"] = {
            "status": status,
            "body": {"success": False, "error": "Proxy error: connection reset"},
        }
        fake_agent.script["get"] = confirmed_event()
        result = run_script(fake_agent)
        assert result.returncode == EXIT_UNKNOWN
        assert "RESULT: UNKNOWN" in result.stdout

    def test_verify_has_its_own_shorter_deadline(self, fake_agent):
        """F1: the DELETE deadline must outlive the operator window, but the
        re-read is an ordinary 30s-bounded GET and must not inherit a 340s
        wait. A slow re-read is inconclusive, and the DELETE is untouched."""
        fake_agent.script["get"] = {**confirmed_event(), "delay": 3}
        result = run_script(
            fake_agent, max_time="5", extra_env={"CALENDAR_DELETE_VERIFY_MAX_TIME": "1"}
        )
        assert result.returncode == EXIT_UNKNOWN
        assert "VERIFY event_1 (verify deadline 1s) -> no response" in result.stdout
        assert "DELETE primary event event_1 (deadline 5s) -> HTTP 200" in result.stdout


class TestNotFound:
    """Exit 3 means the id never existed; a 404 delete is not evidence of work.

    A DELETE that itself 404s is indistinguishable, by exit-0 SUCCESS alone,
    from a completed deletion of a real event: both re-reads come back 404.
    A mistyped or already-gone event id must not be reported as success.
    """

    def test_delete_404_with_get_404_is_not_found(self, fake_agent):
        fake_agent.script["delete"] = {
            "status": 404,
            "body": {"success": False, "message": "Event not found"},
        }
        result = run_script(fake_agent)
        assert result.returncode == EXIT_NOT_FOUND
        assert "RESULT: NOT FOUND" in result.stdout
        assert "RESULT: SUCCESS" not in result.stdout

    def test_delete_410_with_get_404_is_not_found(self, fake_agent):
        fake_agent.script["delete"] = {
            "status": 410,
            "body": {"success": False, "message": "Event gone"},
        }
        result = run_script(fake_agent)
        assert result.returncode == EXIT_NOT_FOUND
        assert "RESULT: NOT FOUND" in result.stdout

    def test_router_404_is_not_read_as_absent(self, fake_agent):
        """F10: a 404 without calendar-agent's envelope is the *router* saying
        "no such route" — a wrong CALENDAR_AGENT_URL, not a missing event.
        Reporting NOT FOUND ("check the id") sends the operator hunting the
        wrong thing; nothing has been established."""
        result = run_script(fake_agent, url=fake_agent.url + "/wrong-prefix")
        assert result.returncode == EXIT_UNKNOWN
        assert "RESULT: NOT FOUND" not in result.stdout
        assert "envelope" in result.stdout

    def test_delete_404_but_get_still_present_is_failure(self, fake_agent):
        """A 404 delete followed by a present re-read is the FAILURE path,
        not NOT FOUND — the claim only matters when verify says gone."""
        fake_agent.script["delete"] = {
            "status": 404,
            "body": {"success": False, "message": "Event not found"},
        }
        fake_agent.script["get"] = confirmed_event()
        result = run_script(fake_agent)
        assert result.returncode == EXIT_FAILURE
        assert "RESULT: FAILURE" in result.stdout


class TestNoRetryOnItsOwn:
    """Enqueuing a second approval is a side effect; the script must not."""

    def test_exactly_one_delete_is_issued(self, fake_agent):
        fake_agent.script["delete"] = {"status": 504, "body": {"success": False}}
        fake_agent.script["get"] = confirmed_event()
        run_script(fake_agent)
        assert [m for m, _ in fake_agent.calls].count("DELETE") == 1


class TestDeadlines:
    """F1: the DELETE deadline must outlive calendar-agent's mutation budget,
    or a late approval lands after curl has given up — the same client/server
    mismatch the server guards against, re-created one hop out."""

    def test_default_deadline_outlives_the_server_budget(self, fake_agent):
        """Unset, the deadline is 340s against the 330s default budget, and
        the re-read gets its own 35s (get_event is READ_TIMEOUT-bounded)."""
        result = run_script(
            fake_agent, max_time=None, extra_env={"CALENDAR_AGENT_CONFIRM_TIMEOUT": None}
        )
        assert result.returncode == EXIT_SUCCESS
        assert "deadline 340s" in result.stdout
        assert "verify deadline 35s" in result.stdout

    def test_deadline_inside_the_server_budget_is_refused(self, fake_agent):
        """The PR's original default (90s) is exactly this misconfiguration."""
        result = run_script(
            fake_agent, max_time="90", extra_env={"CALENDAR_AGENT_CONFIRM_TIMEOUT": "330"}
        )
        assert result.returncode == EXIT_USAGE
        assert "CALENDAR_DELETE_MAX_TIME" in result.stderr
        assert "330" in result.stderr
        assert fake_agent.calls == []  # refused before any request


class TestUsage:
    """Guard rails that keep the whitelisted script narrowly scoped. Usage and
    configuration failures exit 4: nothing was attempted, so none of the
    outcome codes (0-3) would be true."""

    def test_missing_event_id_exits_usage(self, fake_agent):
        result = subprocess.run(
            [str(SCRIPT)],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "CALENDAR_AGENT_URL": fake_agent.url},
            timeout=30,
        )
        assert result.returncode == EXIT_USAGE
        assert "usage" in result.stderr.lower()

    def test_missing_agent_url_exits_usage(self):
        result = subprocess.run(
            [str(SCRIPT), "event_1"],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin"},
            timeout=30,
        )
        assert result.returncode == EXIT_USAGE
        assert "CALENDAR_AGENT_URL" in result.stderr

    def test_missing_python3_is_fatal_before_any_request(self, fake_agent, tmp_path):
        """F10: ``enc()`` failing used to be non-fatal under ``set -uo pipefail``
        — the URL silently lost both ids and the DELETE went to ``/calendars//events/``."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for tool in ("bash", "curl"):
            (bin_dir / tool).symlink_to(shutil.which(tool))
        result = run_script(fake_agent, path=str(bin_dir))
        assert result.returncode == EXIT_USAGE
        assert "python3" in result.stderr
        assert fake_agent.calls == []


class TestBearerToken:
    """Issue #15: calendar-agent requires ``Authorization: Bearer <token>``.
    The script reads the token from CALENDAR_AGENT_API_KEY."""

    def test_the_token_is_sent_on_the_delete_and_on_the_verifying_get(self, fake_agent):
        fake_agent.required_token = API_TOKEN
        result = run_script(fake_agent, extra_env={"CALENDAR_AGENT_API_KEY": API_TOKEN})
        assert result.returncode == EXIT_SUCCESS
        assert fake_agent.calls == [("DELETE", EVENT_PATH), ("GET", EVENT_PATH)]
        assert fake_agent.authorization == [[f"Bearer {API_TOKEN}"], [f"Bearer {API_TOKEN}"]]

    def test_no_authorization_header_is_sent_when_the_variable_is_unset(self, fake_agent):
        """A server running without the requirement keeps working unchanged."""
        result = run_script(fake_agent)
        assert result.returncode == EXIT_SUCCESS
        assert fake_agent.authorization == [[], []]

    @pytest.mark.parametrize(
        "value",
        [
            f"{API_TOKEN}\r",
            f"{API_TOKEN}\n",
            f"{API_TOKEN}\r\n",
            f"\t{API_TOKEN}\t",
            f"\n{API_TOKEN}",
            f"{API_TOKEN[:20]}\r\n{API_TOKEN[20:]}",
        ],
        ids=["trailing CR", "trailing LF", "trailing CRLF", "tabs", "leading LF", "CRLF inside"],
    )
    def test_whitespace_in_the_variable_is_removed_before_the_header_is_sent(
        self, fake_agent, value
    ):
        """A carriage return or newline left in the variable (a token file
        saved with Windows line endings, a ``$(cat file)``) must not reach
        curl's header input, where it would end the header early or start
        another one. The stub demands the exact token."""
        fake_agent.required_token = API_TOKEN
        result = run_script(fake_agent, extra_env={"CALENDAR_AGENT_API_KEY": value})
        assert result.returncode == EXIT_SUCCESS, result.stderr
        assert fake_agent.authorization == [[f"Bearer {API_TOKEN}"], [f"Bearer {API_TOKEN}"]]
        assert fake_agent.injected == []

    def test_a_newline_in_the_variable_cannot_add_a_header(self, fake_agent):
        """What follows a newline is not sent as a header line of its own."""
        result = run_script(
            fake_agent, extra_env={"CALENDAR_AGENT_API_KEY": f"{API_TOKEN}\r\nX-Injected:1"}
        )
        assert result.returncode == EXIT_SUCCESS, result.stderr
        assert fake_agent.injected == []
        assert fake_agent.authorization == [[f"Bearer {API_TOKEN}X-Injected:1"]] * 2

    @pytest.mark.parametrize("value", ["", "   ", "\t", "\r\n", " \n\t\r "])
    def test_an_empty_variable_sends_no_header(self, fake_agent, value):
        result = run_script(fake_agent, extra_env={"CALENDAR_AGENT_API_KEY": value})
        assert result.returncode == EXIT_SUCCESS
        assert fake_agent.authorization == [[], []]

    @pytest.mark.parametrize(
        "token", [None, "synthetic-script-wrong-0123456789abcdef"], ids=["no token", "wrong token"]
    )
    def test_a_refused_token_is_a_configuration_error_and_nothing_else_is_sent(
        self, fake_agent, token
    ):
        """The server refuses before the request reaches a route, so nothing
        was attempted: exit 4, not "unknown", and no verifying GET."""
        fake_agent.required_token = API_TOKEN
        fake_agent.script["get"] = confirmed_event()
        result = run_script(fake_agent, extra_env={"CALENDAR_AGENT_API_KEY": token})
        assert result.returncode == EXIT_USAGE
        assert fake_agent.calls == [("DELETE", EVENT_PATH)]
        assert "CALENDAR_AGENT_API_KEY" in result.stderr
        assert "nothing was attempted" in result.stderr.lower()
        assert "RESULT:" not in result.stdout

    def test_a_401_without_the_envelope_is_not_read_as_a_token_refusal(self, fake_agent):
        """Some other server answering 401 says nothing about calendar-agent;
        that stays "unknown", decided by the re-read like any other answer
        without the envelope."""
        fake_agent.script["delete"] = {"status": 401, "body": {"detail": "Unauthorized"}}
        fake_agent.script["get"] = confirmed_event()
        result = run_script(fake_agent, extra_env={"CALENDAR_AGENT_API_KEY": API_TOKEN})
        assert result.returncode == EXIT_UNKNOWN
        assert fake_agent.calls == [("DELETE", EVENT_PATH), ("GET", EVENT_PATH)]

    def test_the_token_is_never_printed(self, fake_agent):
        """Neither the token nor a 16-character piece of it (start, middle,
        end) is in the script's output, accepted or refused."""
        for required in (API_TOKEN, "synthetic-script-other-0123456789abcdef"):
            fake_agent.required_token = required
            result = run_script(fake_agent, extra_env={"CALENDAR_AGENT_API_KEY": API_TOKEN})
            for fragment in token_fragments(API_TOKEN):
                assert fragment not in result.stdout
                assert fragment not in result.stderr

    def test_the_token_is_not_passed_on_curls_command_line(self, fake_agent, tmp_path):
        """A command line is readable by every process on the machine (``ps``).
        A stand-in ``curl`` records its arguments, then runs the real one."""
        real_curl = shutil.which("curl")
        log = tmp_path / "curl-args.log"
        shim = tmp_path / "bin" / "curl"
        shim.parent.mkdir()
        shim.write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" >> "{log}"\nexec "{real_curl}" "$@"\n')
        shim.chmod(0o755)
        fake_agent.required_token = API_TOKEN
        result = run_script(
            fake_agent,
            extra_env={"CALENDAR_AGENT_API_KEY": API_TOKEN},
            path=f"{shim.parent}:/usr/bin:/bin:/usr/local/bin",
        )
        assert result.returncode == EXIT_SUCCESS
        recorded = log.read_text()
        assert "DELETE" in recorded, "the stand-in curl was not the one that ran"
        for fragment in token_fragments(API_TOKEN):
            assert fragment not in recorded
