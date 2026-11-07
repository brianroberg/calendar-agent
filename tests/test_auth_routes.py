"""The token check in front of the real app (issue #15).

These tests drive ``calendar_agent.calendar_server.app`` through a test
client, with the proxy and the LLM mocked. Every token here is synthetic.
"""

import logging
import re

import pytest
from fastapi.testclient import TestClient
from starlette.routing import Route

from calendar_agent import auth
from calendar_agent.calendar_server import app
from tests.conftest import TEST_API_TOKEN

WRONG_TOKEN = "synthetic-token-wrong-0123456789abcdef"
RIGHT = {"Authorization": f"Bearer {TEST_API_TOKEN}"}
WRONG = {"Authorization": f"Bearer {WRONG_TOKEN}"}


def route_requests(fill: str = "placeholder") -> list[tuple[str, str, str]]:
    """(method, concrete path, route template) for every route the app has
    registered, derived from the app itself, with every path parameter set
    to ``fill``. A route type this walk does not understand fails it, rather
    than being skipped."""
    requests = []
    for route in app.routes:
        assert isinstance(route, Route), (
            f"{route!r} is not a plain HTTP route: teach this walk how to call it, "
            "so it cannot go unchecked"
        )
        path = re.sub(r"\{[^}]+\}", fill, route.path)
        for method in sorted(route.methods):
            requests.append((method, path, route.path))
    assert len(requests) > 18, "the walk found suspiciously few routes"
    return requests


def parameterised_route_requests(fill: str) -> list[tuple[str, str, str]]:
    """The walk restricted to routes with a path parameter."""
    requests = [request for request in route_requests(fill) if "{" in request[2]]
    assert len(requests) >= 8, "the walk found suspiciously few parameterised routes"
    return requests


# What the walk compares: (method, route template) pairs, so one method
# answering wrongly is not hidden by another method on the same template.
HEALTH = {("GET", "/health")}


def every_route() -> set[tuple[str, str]]:
    return {(method, template) for method, _, template in route_requests()}


def routes_not_answering_401(client: TestClient, headers: dict[str, str]) -> set[tuple[str, str]]:
    return {
        (method, template)
        for method, path, template in route_requests()
        if client.request(method, path, headers=headers).status_code != 401
    }


@pytest.fixture
def anonymous_client(mock_proxy_client, mock_llm_service):
    """A client that sends no Authorization header unless a test adds one."""
    return TestClient(app)


@pytest.fixture
def use_settings():
    """Swap the app's settings for the test: ``use_settings(mode, keys)``
    goes through the real loader. The conftest fixture cleans up."""

    def install(mode: str, keys: str | None = f"test-suite:{TEST_API_TOKEN}"):
        environ = {auth.MODE_ENV: mode}
        if keys is not None:
            environ[auth.KEYS_ENV] = keys
        app.state.auth_settings = auth.load_auth_settings(environ)

    return install


@pytest.fixture
def auth_log(caplog):
    """Everything the auth module logs, INFO and up."""
    caplog.set_level(logging.INFO, logger="calendar_agent.auth")
    return caplog


def auth_records(caplog, level: int) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "calendar_agent.auth" and record.levelno == level
    ]


class TestRequiredMode:
    """The conftest default: ``required`` with one configured token."""

    def test_every_route_but_health_refuses_a_request_with_no_token(
        self, anonymous_client, mock_proxy_client, mock_llm_service
    ):
        """Walks the app's own route list. The only route left open is
        ``/health``: a new route is covered without being listed anywhere,
        and adding a path to the exempt list fails this test."""
        assert routes_not_answering_401(anonymous_client, {}) == HEALTH
        assert mock_proxy_client.mock_calls == []
        assert mock_llm_service.mock_calls == []

    def test_every_route_but_health_refuses_a_wrong_token(
        self, anonymous_client, mock_proxy_client, mock_llm_service
    ):
        assert routes_not_answering_401(anonymous_client, WRONG) == HEALTH
        assert mock_proxy_client.mock_calls == []
        assert mock_llm_service.mock_calls == []

    def test_a_path_parameter_spelled_health_does_not_open_a_route(
        self, anonymous_client, mock_proxy_client, mock_llm_service
    ):
        """Every parameterised route again, with each parameter set to
        ``health``: ``DELETE /calendars/health/events/health`` must be refused
        like any other path. Catches an exemption that matches the end of a
        path rather than the whole path."""
        not_refused = {
            (method, path)
            for method, path, _ in parameterised_route_requests("health")
            if anonymous_client.request(method, path).status_code != 401
        }
        assert not_refused == set()
        assert mock_proxy_client.mock_calls == []
        assert mock_llm_service.mock_calls == []

    def test_the_token_check_is_the_outermost_middleware(self):
        """``add_middleware`` puts each new middleware in front of the ones
        added before it, and a middleware in front of the check would see,
        and could answer, a request before its token is checked."""
        assert app.user_middleware[0].cls is auth.BearerAuthMiddleware

    def test_the_exempt_list_is_health_alone(self):
        assert set(auth.EXEMPT_PATHS) == {"/health"}

    def test_no_route_refuses_the_right_token(self, anonymous_client):
        assert routes_not_answering_401(anonymous_client, RIGHT) == every_route()

    def test_the_right_token_reaches_the_route(self, anonymous_client, mock_proxy_client):
        response = anonymous_client.get("/calendars", headers=RIGHT)
        assert response.status_code == 200
        assert response.json()["success"] is True
        mock_proxy_client.list_calendars.assert_awaited_once()

    def test_a_path_or_method_the_app_does_not_have_is_refused_too(self, anonymous_client):
        """Default deny: without a token the answer is 401 whether or not the
        route exists, so the 401 says nothing about which paths are real."""
        assert anonymous_client.get("/no-such-route").status_code == 401
        assert anonymous_client.options("/calendars").status_code == 401
        assert anonymous_client.post("/health/extra").status_code == 401
        assert anonymous_client.get("/health/").status_code == 401
        for path in (
            "/x/health",
            "/no-such-route/health",
            "/calendars/primary/events/event_1/health",
            "//health",
        ):
            assert anonymous_client.get(path).status_code == 401, path
            assert anonymous_client.delete(path).status_code == 401, path

    @pytest.mark.parametrize("headers", [{}, WRONG], ids=["no token", "wrong token"])
    def test_a_refusal_is_401_with_the_envelope_and_a_bearer_challenge(
        self, anonymous_client, headers
    ):
        response = anonymous_client.get("/calendars", headers=headers)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.headers["content-type"] == "application/json"
        assert response.json() == {"success": False, "error": "Missing or invalid bearer token"}

    def test_the_body_does_not_say_whether_the_token_was_missing_or_wrong(self, anonymous_client):
        missing = anonymous_client.get("/calendars")
        wrong = anonymous_client.get("/calendars", headers=WRONG)
        assert missing.content == wrong.content
        assert missing.headers["www-authenticate"] == wrong.headers["www-authenticate"]

    @pytest.mark.parametrize(
        "value",
        [
            TEST_API_TOKEN,  # no scheme
            f"Basic {TEST_API_TOKEN}",
            f"Token {TEST_API_TOKEN}",
            "Bearer",
            "Bearer ",
            f"Bearer {TEST_API_TOKEN} extra",
            f"Bearer {TEST_API_TOKEN},{TEST_API_TOKEN}",
            f"Bearer{TEST_API_TOKEN}",
            f"Bearer {TEST_API_TOKEN[:-1]}",
            "",
        ],
    )
    def test_an_authorization_header_that_is_not_bearer_plus_the_token_is_refused(
        self, anonymous_client, value
    ):
        response = anonymous_client.get("/calendars", headers={"Authorization": value})
        assert response.status_code == 401

    @pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER"])
    def test_the_scheme_name_is_case_insensitive(self, anonymous_client, scheme):
        response = anonymous_client.get(
            "/calendars", headers={"Authorization": f"{scheme} {TEST_API_TOKEN}"}
        )
        assert response.status_code == 200

    def test_two_authorization_headers_are_refused_even_if_one_is_right(self, anonymous_client):
        response = anonymous_client.get(
            "/calendars",
            headers=[
                ("Authorization", f"Bearer {WRONG_TOKEN}"),
                ("Authorization", f"Bearer {TEST_API_TOKEN}"),
            ],
        )
        assert response.status_code == 401

    def test_the_token_in_the_query_string_or_another_header_is_not_accepted(
        self, anonymous_client
    ):
        assert anonymous_client.get(f"/calendars?token={TEST_API_TOKEN}").status_code == 401
        assert anonymous_client.get(f"/calendars?access_token={TEST_API_TOKEN}").status_code == 401
        assert (
            anonymous_client.get("/calendars", headers={"X-API-Key": TEST_API_TOKEN}).status_code
            == 401
        )

    def test_the_token_is_checked_before_the_request_is_validated(self, anonymous_client):
        """A request that would be a 422 with a token is a 401 without one:
        an unauthenticated caller learns nothing about field names."""
        bad_body = {"no_such_field": 1}
        assert anonymous_client.post("/search", json=bad_body, headers=RIGHT).status_code == 422
        assert anonymous_client.post("/search", json=bad_body).status_code == 401
        assert anonymous_client.get("/calendars?bogus=1", headers=RIGHT).status_code == 422
        assert anonymous_client.get("/calendars?bogus=1").status_code == 401

    def test_health_needs_no_token_and_ignores_a_wrong_one(self, anonymous_client):
        for headers in ({}, WRONG, RIGHT, {"Authorization": "garbage"}):
            response = anonymous_client.get("/health", headers=headers)
            assert response.status_code == 200
            assert response.json()["status"] == "ok"


class TestSettingsNeverLoaded:
    """If the start-up hook has not run (so nothing read the settings), the
    check fails closed rather than open or with a 500."""

    def test_everything_but_health_is_refused_even_with_a_plausible_token(self, anonymous_client):
        del app.state.auth_settings
        try:
            assert routes_not_answering_401(anonymous_client, {}) == HEALTH
            assert routes_not_answering_401(anonymous_client, RIGHT) == HEALTH
        finally:
            app.state.auth_settings = None  # the conftest fixture deletes it

    @pytest.mark.parametrize("headers", [{}, RIGHT, WRONG], ids=["no token", "right", "wrong"])
    def test_the_refusal_is_logged_as_an_error_naming_the_missing_settings(
        self, anonymous_client, auth_log, headers
    ):
        """The reason given is the one the operator has to act on: the
        settings were never loaded. Not "token not recognised", which would
        send them looking for a token mismatch. The caller's answer is the
        ordinary refusal."""
        del app.state.auth_settings
        try:
            response = anonymous_client.delete("/calendars/primary/events/event_1", headers=headers)
        finally:
            app.state.auth_settings = None  # the conftest fixture deletes it
        assert response.status_code == 401
        assert response.json() == {"success": False, "error": "Missing or invalid bearer token"}
        assert response.headers["www-authenticate"] == "Bearer"
        (line,) = auth_records(auth_log, logging.ERROR)
        assert line.startswith("refused request: method=DELETE")
        assert "route=/calendars/{calendar_id}/events/{event_id}" in line
        assert "settings are not loaded" in line
        assert "start-up hook did not run" in line
        assert "token not recognised" not in line
        assert "no Authorization header" not in line
        assert auth_records(auth_log, logging.WARNING) == []


# Requests that, sent with a token, go on to a call the proxy holds open while
# a human operator approves it (for minutes). Bodies are valid, so the only
# thing standing between the request and that wait is the token check.
GATED_REQUESTS = [
    ("POST", "/calendars/primary/events", {"summary": "Synthetic event"}, "create_event"),
    ("PUT", "/calendars/primary/events/event_1", {"summary": "Synthetic event"}, "update_event"),
    ("PATCH", "/calendars/primary/events/event_1", {"summary": "Synthetic event"}, "patch_event"),
    ("DELETE", "/calendars/primary/events/event_1", None, "delete_event"),
    (
        "POST",
        "/calendars/primary/events/event_1/respond",
        {"response_status": "accepted"},
        "respond_to_event",
    ),
    (
        "POST",
        "/bulk-actions",
        {"operations": [{"operation": "delete", "event_id": "event_1", "calendar_id": "primary"}]},
        "delete_event",
    ),
]


class TestRefusalComesBeforeTheOperatorApprovalWait:
    @pytest.mark.parametrize(
        "method, path, body, gated_call",
        GATED_REQUESTS,
        ids=[f"{m} {p}" for m, p, _, _ in GATED_REQUESTS],
    )
    @pytest.mark.parametrize("headers", [{}, WRONG], ids=["no token", "wrong token"])
    def test_an_unauthenticated_mutation_is_refused_without_reaching_the_proxy(
        self, anonymous_client, mock_proxy_client, method, path, body, gated_call, headers
    ):
        response = anonymous_client.request(method, path, json=body, headers=headers)
        assert response.status_code == 401
        assert mock_proxy_client.mock_calls == [], "the request got as far as the proxy client"

    @pytest.mark.parametrize(
        "method, path, body, gated_call",
        GATED_REQUESTS,
        ids=[f"{m} {p}" for m, p, _, _ in GATED_REQUESTS],
    )
    def test_control_the_same_request_with_the_token_does_reach_the_gated_call(
        self, anonymous_client, mock_proxy_client, method, path, body, gated_call
    ):
        """Without this the test above could pass for a request that would
        never have reached the proxy anyway (a 422, a wrong path)."""
        anonymous_client.request(method, path, json=body, headers=RIGHT)
        assert getattr(mock_proxy_client, gated_call).await_count == 1

    def test_a_401_on_a_mutation_route_carries_no_outcome(self, anonymous_client):
        """Like a request-validation 422: nothing was attempted, so there is
        no outcome to report."""
        response = anonymous_client.delete("/calendars/primary/events/event_1")
        assert response.status_code == 401
        assert "outcome" not in response.json()


class TestHealthInEveryMode:
    @pytest.mark.parametrize("mode", ["required", "optional", "off"])
    @pytest.mark.parametrize(
        "headers",
        [{}, WRONG, RIGHT, {"Authorization": "garbage"}],
        ids=["no token", "wrong token", "right token", "garbage header"],
    )
    def test_health_answers_200_whatever_the_request_carries(
        self, use_settings, anonymous_client, mode, headers
    ):
        use_settings(mode)
        response = anonymous_client.get("/health", headers=headers)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    @pytest.mark.parametrize("mode", ["required", "optional"])
    def test_health_is_open_even_with_no_tokens_loaded(self, anonymous_client, mode):
        app.state.auth_settings = auth.AuthSettings(mode=auth.AuthMode(mode))
        assert anonymous_client.get("/health").status_code == 200


class TestOptionalMode:
    def test_a_request_with_no_token_is_served(
        self, use_settings, anonymous_client, mock_proxy_client
    ):
        use_settings("optional")
        response = anonymous_client.get("/calendars")
        assert response.status_code == 200
        mock_proxy_client.list_calendars.assert_awaited_once()

    def test_a_request_with_no_token_is_logged_once_with_method_route_and_client(
        self, use_settings, anonymous_client, auth_log
    ):
        use_settings("optional")
        anonymous_client.get("/calendars/someone@example.com/events")
        warnings = auth_records(auth_log, logging.WARNING)
        assert len(warnings) == 1
        (line,) = warnings
        assert "unauthenticated" in line
        assert "method=GET" in line
        assert "route=/calendars/{calendar_id}/events" in line
        assert "client=testclient" in line

    def test_the_log_names_the_route_template_not_the_ids_in_the_path(
        self, use_settings, anonymous_client, auth_log
    ):
        use_settings("optional")
        anonymous_client.get("/calendars/someone@example.com/events/synthetic-event-id")
        assert "someone@example.com" not in auth_log.text
        assert "synthetic-event-id" not in auth_log.text

    def test_a_path_with_no_route_is_logged_without_repeating_the_path(
        self, use_settings, anonymous_client, auth_log
    ):
        """The raw path is caller-controlled text (it can carry an encoded
        newline), so it is not copied into a log line."""
        use_settings("optional")
        response = anonymous_client.get("/no-such-route/%0Aforged-line")
        assert response.status_code == 404
        (line,) = auth_records(auth_log, logging.WARNING)
        assert "route=(no matching route)" in line
        assert "forged-line" not in auth_log.text
        assert "no-such-route" not in auth_log.text

    def test_every_route_but_health_still_refuses_a_wrong_token(
        self, use_settings, anonymous_client, mock_proxy_client, mock_llm_service
    ):
        use_settings("optional")
        assert routes_not_answering_401(anonymous_client, WRONG) == HEALTH
        assert mock_proxy_client.mock_calls == []
        assert mock_llm_service.mock_calls == []

    @pytest.mark.parametrize("value", ["", "Bearer", f"Basic {TEST_API_TOKEN}", TEST_API_TOKEN])
    def test_an_authorization_header_that_is_present_but_unusable_is_refused(
        self, use_settings, anonymous_client, value
    ):
        """Only the absence of the header counts as "no token"."""
        use_settings("optional")
        response = anonymous_client.get("/calendars", headers={"Authorization": value})
        assert response.status_code == 401

    def test_the_right_token_is_served_and_not_logged_as_unauthenticated(
        self, use_settings, anonymous_client, auth_log
    ):
        use_settings("optional")
        assert anonymous_client.get("/calendars", headers=RIGHT).status_code == 200
        assert auth_records(auth_log, logging.WARNING) == []

    def test_health_is_served_and_not_logged(self, use_settings, anonymous_client, auth_log):
        use_settings("optional")
        assert anonymous_client.get("/health").status_code == 200
        assert auth_records(auth_log, logging.WARNING) == []
        assert auth_records(auth_log, logging.INFO) == []

    def test_with_no_tokens_configured_tokenless_is_served_and_any_token_refused(
        self, use_settings, anonymous_client
    ):
        use_settings("optional", keys=None)
        assert anonymous_client.get("/calendars").status_code == 200
        assert anonymous_client.get("/calendars", headers=RIGHT).status_code == 401


class TestOffMode:
    @pytest.mark.parametrize("headers", [{}, WRONG, RIGHT, {"Authorization": "garbage"}], ids=str)
    def test_no_route_answers_401_whatever_the_request_carries(
        self, use_settings, anonymous_client, headers
    ):
        use_settings("off")
        assert routes_not_answering_401(anonymous_client, headers) == every_route()

    def test_requests_are_not_logged(self, use_settings, anonymous_client, auth_log):
        use_settings("off")
        anonymous_client.get("/calendars")
        anonymous_client.get("/calendars", headers=WRONG)
        assert auth_records(auth_log, logging.WARNING) == []
        assert auth_records(auth_log, logging.INFO) == []


class TestRequestLogging:
    def test_an_authenticated_request_is_logged_with_the_callers_name(
        self, anonymous_client, auth_log
    ):
        anonymous_client.get("/calendars/primary/events", headers=RIGHT)
        (line,) = auth_records(auth_log, logging.INFO)
        assert "caller=test-suite" in line
        assert "method=GET" in line
        assert "route=/calendars/{calendar_id}/events" in line

    def test_the_log_names_which_of_several_callers_it_was(
        self, use_settings, anonymous_client, auth_log
    ):
        other = "synthetic-token-other-0123456789abcdef"
        use_settings("required", keys=f"test-suite:{TEST_API_TOKEN},scheduler:{other}")
        anonymous_client.get("/calendars", headers={"Authorization": f"Bearer {other}"})
        (line,) = auth_records(auth_log, logging.INFO)
        assert "caller=scheduler" in line

    @pytest.mark.parametrize(
        "headers, reason",
        [({}, "no Authorization header"), (WRONG, "token not recognised")],
        ids=["no token", "wrong token"],
    )
    def test_a_refusal_is_logged_with_method_route_client_and_reason(
        self, anonymous_client, auth_log, headers, reason
    ):
        anonymous_client.delete("/calendars/primary/events/event_1", headers=headers)
        (line,) = auth_records(auth_log, logging.WARNING)
        assert "refused" in line
        assert "method=DELETE" in line
        assert "route=/calendars/{calendar_id}/events/{event_id}" in line
        assert "client=testclient" in line
        assert reason in line


def token_fragments(token: str) -> list[str]:
    """The whole token and three 16-character pieces of it (start, middle,
    end), so a log line that printed most of a token is caught too."""
    middle = len(token) // 2 - 8
    return [token, token[:16], token[middle : middle + 16], token[-16:]]


class TestNoTokenTextAnywhere:
    @pytest.mark.parametrize("mode", ["required", "optional", "off"])
    def test_no_token_appears_in_a_log_line_or_a_response(
        self, use_settings, anonymous_client, auth_log, mode
    ):
        """Right token, wrong token, no token and malformed headers, in each
        mode, against a read, a mutation, an unknown path and /health:
        neither the configured token nor the presented one, nor a 16-character
        piece of either, is in any captured log record (all loggers, DEBUG
        up) or any response body or header."""
        auth_log.set_level(logging.DEBUG)
        use_settings(mode)
        presented = [
            {},
            RIGHT,
            WRONG,
            {"Authorization": f"Basic {WRONG_TOKEN}"},
            {"Authorization": f"Bearer {TEST_API_TOKEN} {WRONG_TOKEN}"},
            {"Authorization": WRONG_TOKEN},
        ]
        requests = [
            ("GET", "/calendars"),
            ("DELETE", "/calendars/primary/events/event_1"),
            ("GET", "/no-such-route"),
            ("GET", "/health"),
            ("GET", "/openapi.json"),
        ]
        responses = [
            anonymous_client.request(method, path, headers=headers)
            for headers in presented
            for method, path in requests
        ]
        if mode != "off":  # `off` logs nothing per request
            assert any(record.name == "calendar_agent.auth" for record in auth_log.records), (
                "the auth module logged nothing, so this would prove nothing about it"
            )
        for token in (TEST_API_TOKEN, WRONG_TOKEN):
            for secret in token_fragments(token):
                assert secret not in auth_log.text
                for record in auth_log.records:
                    assert secret not in record.getMessage()
                    assert secret not in repr(record.args)
                for response in responses:
                    assert secret not in response.text
                    assert secret not in repr(response.headers)


def _operations(document: dict):
    for path, item in document["paths"].items():
        for method, operation in item.items():
            yield path, method, operation


class TestOpenApiDocument:
    def test_it_declares_one_http_bearer_scheme(self):
        schemes = app.openapi()["components"]["securitySchemes"]
        assert schemes == {"bearerAuth": {"type": "http", "scheme": "bearer"}}

    def test_the_scheme_is_required_for_the_whole_api(self):
        assert app.openapi()["security"] == [{"bearerAuth": []}]

    def test_health_is_marked_as_needing_no_token_and_nothing_else_is(self):
        """An empty ``security`` list on an operation overrides the
        document-wide requirement: that is how OpenAPI says "open"."""
        opted_out = {
            path for path, _, operation in _operations(app.openapi()) if "security" in operation
        }
        assert opted_out == {"/health"}
        assert app.openapi()["paths"]["/health"]["get"]["security"] == []

    def test_every_operation_but_health_documents_the_401(self):
        document = app.openapi()
        operations = list(_operations(document))
        assert len(operations) == 18
        without_401 = {
            path for path, _, operation in operations if "401" not in operation["responses"]
        }
        assert without_401 == {"/health"}

    def test_the_rest_of_the_document_is_untouched(self):
        document = app.openapi()
        assert "EventSummary" in document["components"]["schemas"]
        assert document["info"]["title"] == "Calendar Agent"

    def test_the_served_document_is_the_same_one(self, anonymous_client):
        served = anonymous_client.get("/openapi.json", headers=RIGHT)
        assert served.status_code == 200
        assert served.json() == app.openapi()
        assert served.json()["components"]["securitySchemes"]["bearerAuth"]["scheme"] == "bearer"
