"""The middleware on its own, around a small stand-in app (issue #15).

The real app has only plain HTTP routes today. Default deny has to hold for
what it does not have yet, so these tests wrap an app that also has a
WebSocket route, and one that is handed a connection type nobody has
invented. Every token is synthetic.
"""

import logging

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from calendar_agent import auth

TOKEN = "synthetic-middleware-token-0123456789abcdef"
WRONG_TOKEN = "synthetic-middleware-wrong-0123456789abcdef"
RIGHT = {"Authorization": f"Bearer {TOKEN}"}
WRONG = {"Authorization": f"Bearer {WRONG_TOKEN}"}


def wrapped_app(mode: str, keys: str | None = f"laptop:{TOKEN}"):
    async def page(request):
        return PlainTextResponse("served")

    async def socket(websocket):
        await websocket.accept()
        await websocket.send_text("served")
        await websocket.close()

    inner = Starlette(
        routes=[
            Route("/page", page),
            Route("/health", page),
            WebSocketRoute("/socket", socket),
            WebSocketRoute("/health", socket),
        ]
    )
    environ = {auth.MODE_ENV: mode}
    if keys is not None:
        environ[auth.KEYS_ENV] = keys
    settings = auth.load_auth_settings(environ)
    return auth.BearerAuthMiddleware(inner, get_settings=lambda: settings)


def connects(client: TestClient, path: str, headers: dict[str, str]) -> bool:
    try:
        with client.websocket_connect(path, headers=headers) as websocket:
            return websocket.receive_text() == "served"
    except WebSocketDisconnect:
        return False


class TestWebSocketConnections:
    def test_required_refuses_a_connection_without_the_token(self):
        client = TestClient(wrapped_app("required"))
        assert connects(client, "/socket", {}) is False
        assert connects(client, "/socket", WRONG) is False

    def test_required_accepts_a_connection_with_the_token(self):
        client = TestClient(wrapped_app("required"))
        assert connects(client, "/socket", RIGHT) is True

    def test_a_refused_connection_is_closed_with_policy_violation(self):
        client = TestClient(wrapped_app("required"))
        with pytest.raises(WebSocketDisconnect) as excinfo, client.websocket_connect("/socket"):
            pass
        assert excinfo.value.code == 1008

    def test_the_health_exemption_is_for_http_only(self):
        client = TestClient(wrapped_app("required"))
        assert client.get("/health").status_code == 200
        assert connects(client, "/health", {}) is False

    def test_optional_serves_a_tokenless_connection_and_logs_it(self, caplog):
        caplog.set_level(logging.INFO, logger="calendar_agent.auth")
        client = TestClient(wrapped_app("optional"))
        assert connects(client, "/socket", {}) is True
        assert connects(client, "/socket", WRONG) is False
        served = [r.getMessage() for r in caplog.records if "unauthenticated" in r.getMessage()]
        assert len(served) == 1
        assert "method=WEBSOCKET" in served[0]

    def test_a_connection_is_refused_when_no_settings_are_loaded(self):
        """The fail-closed path (the start-up hook never ran) refuses a
        WebSocket handshake too, with the right token or without it."""

        async def socket(websocket):
            await websocket.accept()
            await websocket.send_text("served")
            await websocket.close()

        inner = Starlette(routes=[WebSocketRoute("/socket", socket)])
        client = TestClient(auth.BearerAuthMiddleware(inner, get_settings=lambda: None))
        assert connects(client, "/socket", RIGHT) is False
        with pytest.raises(WebSocketDisconnect) as excinfo, client.websocket_connect("/socket"):
            pass
        assert excinfo.value.code == 1008

    def test_off_serves_any_connection(self):
        client = TestClient(wrapped_app("off"))
        assert connects(client, "/socket", {}) is True
        assert connects(client, "/socket", WRONG) is True


class TestOtherConnectionTypes:
    async def test_lifespan_messages_pass_through(self):
        seen = []

        async def inner(scope, receive, send):
            seen.append(scope["type"])

        middleware = auth.BearerAuthMiddleware(inner, get_settings=lambda: None)
        await middleware({"type": "lifespan"}, None, None)
        assert seen == ["lifespan"]

    @pytest.mark.parametrize("mode", ["required", "optional"])
    async def test_a_connection_type_the_check_does_not_know_is_not_served(self, mode):
        """Nothing sends one today. If something does, it is dropped rather
        than handed to the application unchecked."""
        seen = []

        async def inner(scope, receive, send):
            seen.append(scope["type"])

        settings = auth.load_auth_settings({auth.MODE_ENV: mode, auth.KEYS_ENV: f"laptop:{TOKEN}"})
        middleware = auth.BearerAuthMiddleware(inner, get_settings=lambda: settings)
        await middleware({"type": "future-transport", "path": "/page", "headers": []}, None, None)
        assert seen == []


class TestLogLinesAreNotForgeable:
    """The method and the client address come from the connection. They go
    into a log line through a small safe alphabet, so text carrying a newline
    does not start a second, forged line."""

    @pytest.mark.parametrize("mode", ["optional", "required"])
    async def test_a_newline_in_the_method_or_client_does_not_reach_the_log(self, caplog, mode):
        caplog.set_level(logging.INFO, logger="calendar_agent.auth")
        sent = []

        async def inner(scope, receive, send):
            pass

        async def send(message):
            sent.append(message)

        settings = auth.load_auth_settings({auth.MODE_ENV: mode, auth.KEYS_ENV: f"laptop:{TOKEN}"})
        middleware = auth.BearerAuthMiddleware(inner, get_settings=lambda: settings)
        scope = {
            "type": "http",
            "method": "GET\nWARNING forged-method-line",
            "path": "/page",
            "headers": [],
            "client": ("192.0.2.7\nWARNING forged-client-line", 4321),
        }
        await middleware(scope, None, send)

        lines = [r.getMessage() for r in caplog.records if r.name == "calendar_agent.auth"]
        assert len(lines) == 1
        assert "\n" not in lines[0]
        assert "method=GET?WARNING?forged-method-line " in lines[0]
        assert "client=192.0.2.7?WARNING?forged-client-line" in lines[0]
