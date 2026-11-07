"""Start-up behaviour of the token settings (issue #15).

Two layers. In-process: the app's start-up hook, run by entering a
``TestClient`` context, with the environment patched. Out-of-process: the
real server (``uvicorn calendar_agent.calendar_server:app``, the command the
Dockerfile runs) started on a loopback port, to pin what an operator sees --
the exit status and what reaches stderr. Nothing here calls an upstream
service: the only routes requested are ``/health`` and ``/openapi.json``.
Every token is synthetic.
"""

import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from calendar_agent import auth
from calendar_agent.calendar_server import app

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKEN = "synthetic-startup-token-0123456789abcdef"
OTHER_TOKEN = "synthetic-startup-other-0123456789abcdef"
SHORT_TOKEN = "synthetic-short-token"
RIGHT = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def environ(monkeypatch):
    """Set (or clear) the two settings in this process's environment."""

    def apply(mode: str | None, keys: str | None):
        for name, value in ((auth.MODE_ENV, mode), (auth.KEYS_ENV, keys)):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)

    return apply


class TestStartUpHook:
    def test_the_settings_are_read_from_the_environment_at_start_up(self, environ):
        """Not at import: the module was imported long before this test set
        the variables."""
        environ("required", f"laptop:{TOKEN}")
        with TestClient(app) as client:
            assert [key.name for key in app.state.auth_settings.keys] == ["laptop"]
            assert client.get("/openapi.json", headers=RIGHT).status_code == 200
            assert client.get("/openapi.json").status_code == 401

    @pytest.mark.parametrize(
        "mode, keys", [(None, None), ("required", None), (None, ""), ("required", " , ")]
    )
    def test_required_with_no_tokens_refuses_to_start_naming_both_settings(
        self, environ, mode, keys
    ):
        environ(mode, keys)
        with pytest.raises(auth.AuthConfigError) as excinfo, TestClient(app):
            pass
        assert "CALENDAR_AGENT_API_KEYS" in str(excinfo.value)
        assert "CALENDAR_AGENT_AUTH" in str(excinfo.value)

    @pytest.mark.parametrize("mode", ["required", "optional"])
    def test_a_malformed_keys_value_refuses_to_start_without_echoing_it(self, environ, mode):
        environ(mode, f"laptop:{TOKEN},scheduler:{SHORT_TOKEN}")
        with pytest.raises(auth.AuthConfigError) as excinfo, TestClient(app):
            pass
        assert TOKEN not in str(excinfo.value)
        assert SHORT_TOKEN not in str(excinfo.value)

    def test_an_unknown_mode_refuses_to_start(self, environ):
        environ("disabled", f"laptop:{TOKEN}")
        with pytest.raises(auth.AuthConfigError), TestClient(app):
            pass

    def test_a_refusal_to_start_is_logged_as_one_error_line(self, environ, caplog):
        """The exception reaches the operator inside a traceback; the same
        message is also logged on a line of its own, where it can be found."""
        environ("required", f"laptop:{TOKEN},scheduler:{SHORT_TOKEN}")
        with pytest.raises(auth.AuthConfigError) as excinfo, TestClient(app):
            pass
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert errors == [f"refusing to start: {excinfo.value}"]
        assert TOKEN not in caplog.text and SHORT_TOKEN not in caplog.text

    def test_off_starts_and_warns(self, environ, caplog):
        environ("off", None)
        caplog.set_level(logging.INFO, logger="calendar_agent.auth")
        with TestClient(app) as client:
            assert client.get("/openapi.json").status_code == 200
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "CALENDAR_AGENT_AUTH=off" in warnings[0]
        assert "no token is required" in warnings[0]

    def test_optional_starts_and_warns_that_tokenless_requests_are_served(self, environ, caplog):
        environ("optional", f"laptop:{TOKEN}")
        caplog.set_level(logging.INFO, logger="calendar_agent.auth")
        with TestClient(app):
            start_up = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(start_up) == 1
        assert "CALENDAR_AGENT_AUTH=optional" in start_up[0].getMessage()
        assert "laptop" in start_up[0].getMessage()

    def test_required_starts_and_says_which_callers_are_configured(self, environ, caplog):
        environ(None, f"laptop:{TOKEN},scheduler:{OTHER_TOKEN}")
        caplog.set_level(logging.INFO, logger="calendar_agent.auth")
        with TestClient(app):
            lines = [r.getMessage() for r in caplog.records if r.name == "calendar_agent.auth"]
        assert len(lines) == 1
        assert "CALENDAR_AGENT_AUTH=required" in lines[0]
        assert "laptop" in lines[0] and "scheduler" in lines[0]
        assert TOKEN not in caplog.text and OTHER_TOKEN not in caplog.text

    def test_logging_already_configured_by_the_host_is_left_alone(self, environ):
        """Under pytest the root logger has handlers, so the start-up hook
        must not add its own (lines would be printed twice)."""
        environ("off", None)
        with TestClient(app):
            pass
        assert logging.getLogger("calendar_agent").handlers == []


# ----------------------------------------------------------------------------
# The real server on a loopback port
# ----------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _server_env(mode: str | None, keys: str | None) -> dict[str, str]:
    """A minimal environment: nothing inherited but what Python needs. Both
    settings are always present (empty when "unset") so a stray .env file
    cannot supply them."""
    env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG") if key in os.environ}
    env[auth.MODE_ENV] = mode or ""
    env[auth.KEYS_ENV] = keys or ""
    return env


class _Server:
    def __init__(self, mode: str | None, keys: str | None, *extra_args: str):
        self.port = _free_port()
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "calendar_agent.calendar_server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                *extra_args,
            ],  # fmt: skip
            cwd=REPO_ROOT,
            env=_server_env(mode, keys),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.url = f"http://127.0.0.1:{self.port}"

    def wait_until_serving(
        self, timeout: float = 30.0, path: str = "/health", headers: dict | None = None
    ) -> None:
        deadline = time.monotonic() + timeout
        with httpx.Client(trust_env=False) as client:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise AssertionError(f"server exited early:\n{self.process.stderr.read()}")
                try:
                    response = client.get(f"{self.url}{path}", headers=headers, timeout=1.0)
                    if response.status_code == 200:
                        return
                except httpx.TransportError:
                    time.sleep(0.1)
        raise AssertionError(f"server did not start serving {path} in time")

    def stop(self) -> tuple[int, str]:
        """Stop the server; return its exit status and everything it wrote."""
        if self.process.poll() is None:
            self.process.terminate()
        try:
            stdout, stderr = self.process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            stdout, stderr = self.process.communicate()
        return self.process.returncode, stdout + stderr


@pytest.fixture
def server():
    started: list[_Server] = []

    def start(mode: str | None, keys: str | None, *extra_args: str) -> _Server:
        instance = _Server(mode, keys, *extra_args)
        started.append(instance)
        return instance

    yield start
    for instance in started:
        instance.stop()


class TestTheRealServer:
    def test_required_with_no_tokens_exits_at_start_up_naming_both_settings(self, server):
        instance = server(None, None)
        try:
            instance.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pytest.fail("the server kept running with no tokens in required mode")
        status, output = instance.stop()
        assert status != 0
        assert "CALENDAR_AGENT_API_KEYS" in output
        assert "CALENDAR_AGENT_AUTH" in output

    def test_a_malformed_keys_value_exits_at_start_up_without_printing_a_token(self, server):
        instance = server("required", f"laptop:{TOKEN},scheduler:{SHORT_TOKEN}")
        try:
            instance.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pytest.fail("the server kept running with a malformed keys value")
        status, output = instance.stop()
        assert status != 0
        assert "CALENDAR_AGENT_API_KEYS" in output and "entry 2" in output
        assert TOKEN not in output and SHORT_TOKEN not in output

    def test_off_serves_without_a_token_and_warns_on_stderr(self, server):
        instance = server("off", None)
        instance.wait_until_serving()
        with httpx.Client(trust_env=False) as client:
            assert client.get(f"{instance.url}/openapi.json").status_code == 200
        _, output = instance.stop()
        assert "CALENDAR_AGENT_AUTH=off" in output

    def test_optional_serves_and_logs_and_the_log_reaches_stderr(self, server):
        instance = server("optional", f"laptop:{TOKEN}")
        instance.wait_until_serving()
        with httpx.Client(trust_env=False) as client:
            assert client.get(f"{instance.url}/openapi.json").status_code == 200
            assert client.get(f"{instance.url}/openapi.json", headers=RIGHT).status_code == 200
            wrong = client.get(
                f"{instance.url}/openapi.json", headers={"Authorization": f"Bearer {OTHER_TOKEN}"}
            )
            assert wrong.status_code == 401
            assert wrong.headers["www-authenticate"] == "Bearer"
        _, output = instance.stop()
        assert "unauthenticated request served" in output
        assert "route=/openapi.json" in output
        assert "client=127.0.0.1" in output
        assert "authenticated request: caller=laptop" in output
        assert "refused request" in output
        assert TOKEN not in output and OTHER_TOKEN not in output

    def test_required_serves_health_without_a_token_and_nothing_else(self, server):
        instance = server("required", f"laptop:{TOKEN}")
        instance.wait_until_serving()
        with httpx.Client(trust_env=False) as client:
            assert client.get(f"{instance.url}/health").status_code == 200
            for path in ("/openapi.json", "/docs", "/redoc", "/calendars", "/no-such-route"):
                assert client.get(f"{instance.url}{path}").status_code == 401, path
            assert client.get(f"{instance.url}/openapi.json", headers=RIGHT).status_code == 200
        _, output = instance.stop()
        assert TOKEN not in output

    def test_under_a_root_path_health_needs_the_token(self, server):
        """Pins current behaviour. The exemption compares the path exactly as
        the server reports it, and ``uvicorn --root-path /cal`` reports
        ``/cal/health``, so behind a root path ``/health`` is refused without
        a token like any other route (it fails closed). The Dockerfile serves
        at the root of the host, where ``/health`` is open. The comparison is
        exact on purpose: a suffix match would also open
        ``/calendars/health/events/health`` (see test_auth_routes.py)."""
        instance = server("required", f"laptop:{TOKEN}", "--root-path", "/cal")
        instance.wait_until_serving(headers=RIGHT)
        with httpx.Client(trust_env=False) as client:
            refused = client.get(f"{instance.url}/health")
            assert refused.status_code == 401
            assert refused.headers["www-authenticate"] == "Bearer"
            assert client.get(f"{instance.url}/health", headers=RIGHT).status_code == 200
        _, output = instance.stop()
        assert TOKEN not in output
