"""Tests for Calendar Agent server endpoints."""


import re
import typing
from collections.abc import Mapping
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from calendar_agent.calendar_server import EventSummary, SearchFilters, app, event_to_summary
from calendar_agent.exceptions import (
    ProxyAuthError,
    ProxyConfigError,
    ProxyError,
    ProxyForbiddenError,
    ProxyNotFoundError,
    ProxyRequestError,
    ProxyTimeoutError,
)
from calendar_agent.proxy_client import (
    CONFIRM_TIMEOUT,
    CONFIRM_TIMEOUT_MARGIN,
    PROXY_CONFIRMATION_WINDOW,
    READ_TIMEOUT,
    CalendarProxyClient,
    resolve_confirm_timeout,
    resolve_confirmation_window,
)
from tests.factories import (
    AUTH_USER_EMAIL,
    CANCELLED_STUB,
    COLLEAGUE_EMAIL,
    GROUP_CALENDAR_ID,
    SAMPLE_EVENTS,
    colleague_copy,
    get_sample_event,
)

# ============================================================================
# Minimal valid request bodies, one per request-body route. Shared by the
# per-route endpoint tests and the unknown-field tables below so there is one
# copy of "what a valid body looks like".
# ============================================================================

CREATE_EVENT_BODY = {
    "summary": "New Meeting",
    "start": {"dateTime": "2024-01-15T10:00:00Z"},
    "end": {"dateTime": "2024-01-15T11:00:00Z"},
}
UPDATE_EVENT_BODY = {
    "summary": "Updated Meeting",
    "start": {"dateTime": "2024-01-15T14:00:00Z"},
    "end": {"dateTime": "2024-01-15T15:00:00Z"},
}
PATCH_EVENT_BODY = {"summary": "Renamed Meeting"}
RESPOND_BODY = {"response_status": "accepted"}
SUMMARIZE_BODY = {"calendar_id": "primary", "event_id": "event_123", "format": "brief"}
ASK_ABOUT_BODY = {
    "calendar_id": "primary",
    "event_id": "event_123",
    "question": "What time is the meeting?",
}
BATCH_SUMMARIZE_BODY = {
    "calendar_id": "primary",
    "event_ids": ["event_1", "event_2", "event_3"],
    "triage": False,
}
FIND_FREE_TIME_BODY = {
    "calendar_id": "primary",
    "time_min": "2024-01-15T09:00:00Z",
    "time_max": "2024-01-15T17:00:00Z",
    "duration_minutes": 30,
}
ANALYZE_SCHEDULE_BODY = {
    "calendar_id": "primary",
    "time_min": "2024-01-15T00:00:00Z",
    "time_max": "2024-01-22T00:00:00Z",
    "analysis_type": "overview",
}
PREPARE_BRIEFING_BODY = {"calendar_id": "primary", "briefing_type": "daily"}
SEARCH_BODY = {
    "calendar_id": "primary",
    "filters": {
        "query": "meeting",
        "time_min": "2024-01-01T00:00:00Z",
        "time_max": "2024-01-31T23:59:59Z",
    },
}
BULK_DELETE_BODY = {
    "operations": [
        {"operation": "delete", "event_id": "event_1", "calendar_id": "primary"},
        {"operation": "delete", "event_id": "event_2", "calendar_id": "primary"},
    ]
}


# ============================================================================
# Health Endpoint Tests
# ============================================================================


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self, client):
        """Health endpoint returns status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_returns_version(self, client):
        """Health endpoint returns correct version."""
        response = client.get("/health")
        data = response.json()
        assert data["version"] == "1.0.0"


# ============================================================================
# Calendar Endpoint Tests
# ============================================================================


class TestCalendarsEndpoint:
    """Tests for the /calendars endpoint."""

    def test_list_calendars_success(self, client, mock_proxy_client):
        """List calendars returns all calendars."""
        response = client.get("/calendars")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["calendars"]) == 3
        assert data["calendars"][0]["id"] == AUTH_USER_EMAIL

    def test_list_calendars_empty(self, client, mock_proxy_client):
        """List calendars handles empty list."""
        mock_proxy_client.list_calendars.return_value = {"items": []}
        response = client.get("/calendars")
        data = response.json()
        assert data["success"] is True
        assert len(data["calendars"]) == 0

    def test_list_calendars_with_pagination(self, client, mock_proxy_client):
        """List calendars supports pagination parameters."""
        response = client.get("/calendars?max_results=10&page_token=abc123")
        assert response.status_code == 200
        mock_proxy_client.list_calendars.assert_called_with(
            max_results=10,
            page_token="abc123",
        )

    def test_list_calendars_error(self, client, mock_proxy_client):
        """List calendars handles proxy errors."""
        mock_proxy_client.list_calendars.side_effect = ProxyError("Connection failed")
        response = client.get("/calendars")
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        assert "Proxy error" in data["error"]

    def test_get_calendar_success(self, client, mock_proxy_client):
        """Get specific calendar returns calendar details."""
        response = client.get("/calendars/primary")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["calendar"]["id"] == AUTH_USER_EMAIL

    def test_get_calendar_not_found(self, client, mock_proxy_client):
        """Get calendar handles not found."""
        mock_proxy_client.get_calendar.side_effect = ProxyError("Calendar not found")
        response = client.get("/calendars/nonexistent")
        data = response.json()
        assert data["success"] is False


# ============================================================================
# Event CRUD Endpoint Tests
# ============================================================================


class TestEventToSummary:
    """event_to_summary(): organizer exposure and RSVP state (issue #9).

    Google defines ``attendees[].self`` / ``organizer.self`` relative to the
    calendar the event copy sits on, so every ``calendar_*`` field describes
    the calendar named by ``calendar_id`` -- which is the authenticated user
    only when that calendar is the user's own.
    """

    def test_self_flags_describe_the_calendar_being_read(self):
        summary = event_to_summary(colleague_copy(), COLLEAGUE_EMAIL)
        assert summary.calendar_id == COLLEAGUE_EMAIL
        assert summary.calendar_is_organizer is False
        assert summary.calendar_rsvp_state == "accepted"  # Carol's RSVP, not john.doe@'s
        assert summary.organizer_email == "dave@example.com"
        assert summary.creator_email == "dave@example.com"

    def test_summary_field_set_is_exactly_the_documented_one(self):
        """Pins the README's Key Privacy Features claim for list/search rows:
        these fields and no others -- in particular no description and no
        attendee list. Adding a field here is a documentation change too."""
        expected = {
            "id",
            "calendar_id",
            "summary",
            "start",
            "end",
            "location",
            "attendee_count",
            "is_all_day",
            "status",
            "html_link",
            "organizer_email",
            "creator_email",
            "calendar_is_organizer",
            "calendar_rsvp_state",
        }
        assert set(EventSummary.model_fields) == expected

    def test_no_field_claims_to_be_the_authenticated_users_rsvp(self):
        """The old contract (is_organizer / response_status "of the
        authenticated user") is gone: nothing on the wire is named as if it
        described the caller rather than the calendar."""
        dumped = event_to_summary(colleague_copy(), COLLEAGUE_EMAIL).model_dump()
        assert "is_organizer" not in dumped
        assert "response_status" not in dumped
        assert "calendar_response_status" not in dumped  # dropped in round 3 (redundant)
        assert not any(k.startswith("user_") for k in dumped)

    def test_perspective_fields_are_required_in_the_openapi_schema(self):
        """Both perspective fields are always emitted, so the schema must not
        mark either optional (finding 15, round 3)."""
        schema = app.openapi()["components"]["schemas"]["EventSummary"]
        assert {"calendar_is_organizer", "calendar_rsvp_state"} <= set(schema["required"])
        assert "calendar_response_status" not in schema["properties"]

    def test_malformed_organizer_or_attendees_degrade_instead_of_raising(self):
        """Finding 8 (round 3): a non-dict organizer/creator or attendee row
        reads as absent instead of raising. Only these fields are guarded;
        a malformed start/end or summary is not."""
        event = get_sample_event()
        event["organizer"] = "dave@example.com"
        event["creator"] = ["dave@example.com"]
        event["attendees"] = [None, "bob@example.com"]
        summary = event_to_summary(event, "primary")
        assert summary.organizer_email is None
        assert summary.creator_email is None
        assert summary.calendar_is_organizer is False
        assert summary.calendar_rsvp_state == "unknown"
        # attendee_count agrees with the perspective: the non-dict rows are
        # not attendees, so the count is 0, not 2 (finding 7, round 4).
        assert summary.attendee_count == 0
        event["attendees"] = "not-a-list"
        assert event_to_summary(event, "primary").attendee_count == 0

    def test_pending_invitation_on_own_calendar(self):
        event = get_sample_event(
            organizer={"email": "dave@example.com", "self": False},
            attendees=[{"email": AUTH_USER_EMAIL, "self": True, "responseStatus": "needsAction"}],
        )
        summary = event_to_summary(event, "primary")
        assert summary.calendar_is_organizer is False
        assert summary.calendar_rsvp_state == "needsAction"

    def test_own_event_with_attendees_but_no_own_entry(self):
        """The dangerous null from issue #9: the calendar organizes, others
        are invited, the calendar has no attendee entry of its own. Reads as
        the calendar's own event, not an unanswered invitation."""
        event = get_sample_event(
            organizer={"email": AUTH_USER_EMAIL, "self": True},
            creator={"email": AUTH_USER_EMAIL, "self": True},
            attendees=[
                {"email": "alice@example.com", "responseStatus": "accepted"},
                {"email": "bob@example.com", "responseStatus": "needsAction"},
            ],
        )
        summary = event_to_summary(event, "primary")
        assert summary.attendee_count == 2
        assert summary.calendar_is_organizer is True
        assert summary.calendar_rsvp_state == "organizer_no_rsvp"

    def test_group_calendar_native_event_names_the_calendar_and_the_creator(self):
        """An event created directly on a group calendar: Google makes the
        calendar itself the organizer (email == calendar id, self:true); the
        person who created it is only in ``creator``."""
        group = "abc123@group.calendar.google.com"
        event = get_sample_event(
            organizer={"email": group, "displayName": "Team Calendar", "self": True},
            creator={"email": AUTH_USER_EMAIL},
            attendees=None,
        )
        summary = event_to_summary(event, group)
        assert summary.attendee_count == 0
        assert summary.calendar_is_organizer is True
        assert summary.calendar_rsvp_state == "organizer_no_rsvp"
        assert summary.organizer_email == group
        assert summary.creator_email == AUTH_USER_EMAIL

    def test_calendar_neither_organizes_nor_attends(self):
        event = get_sample_event(
            organizer={"email": "dave@example.com", "self": False},
            attendees=[{"email": "alice@example.com", "responseStatus": "accepted"}],
        )
        summary = event_to_summary(event, "primary")
        assert summary.calendar_is_organizer is False
        assert summary.calendar_rsvp_state == "not_attendee"

    def test_cancelled_recurring_stub_is_unknown_not_own_event(self):
        event = get_sample_event(status="cancelled", with_times=False)
        summary = event_to_summary(event, "primary")
        assert summary.status == "cancelled"
        assert summary.attendee_count == 0
        assert summary.calendar_is_organizer is False
        assert summary.calendar_rsvp_state == "unknown"
        assert summary.organizer_email is None
        assert summary.creator_email is None

    def test_bare_cancelled_stub_with_no_start_or_end(self):
        """The real shape of a deleted recurring instance: no start, no end,
        no summary (finding 5, round 4). Empty times, not all-day, not a
        crash."""
        summary = event_to_summary(CANCELLED_STUB, "primary")
        assert summary.id == CANCELLED_STUB["id"]
        assert summary.status == "cancelled"
        assert summary.start == ""
        assert summary.end == ""
        assert summary.is_all_day is False
        assert summary.summary == "Untitled Event"
        assert summary.calendar_rsvp_state == "unknown"

    def test_attendee_addresses_are_not_on_the_wire(self):
        """Brian's 2026-09-03 decision: organizer and creator addresses are
        exposed; the attendee list is still only a count."""
        event = colleague_copy()
        # Guard the assertion below against a fixture drift that would make
        # it vacuous: the address must really be in the attendee list.
        assert any(a["email"] == AUTH_USER_EMAIL for a in event["attendees"])
        dumped = event_to_summary(event, COLLEAGUE_EMAIL).model_dump_json()
        assert COLLEAGUE_EMAIL in dumped  # it is the calendar_id
        assert AUTH_USER_EMAIL not in dumped
        assert "attendees" not in dumped


class TestEventsListEndpoint:
    """Tests for GET /calendars/{calendar_id}/events."""

    def test_list_events_success(self, client, mock_proxy_client):
        """List events returns event summaries."""
        response = client.get("/calendars/primary/events")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["events"]) == 1
        assert data["events"][0]["summary"] == "Team Standup"

    def test_list_events_with_filters(self, client, mock_proxy_client):
        """List events accepts filter parameters."""
        response = client.get(
            "/calendars/primary/events",
            params={
                "time_min": "2024-01-01T00:00:00Z",
                "time_max": "2024-01-31T23:59:59Z",
                "q": "meeting",
                "max_results": 50,
            }
        )
        assert response.status_code == 200
        mock_proxy_client.list_events.assert_called_once()
        call_kwargs = mock_proxy_client.list_events.call_args.kwargs
        assert call_kwargs["time_min"] == "2024-01-01T00:00:00Z"
        assert call_kwargs["q"] == "meeting"

    def test_list_events_single_events_default_true(self, client, mock_proxy_client):
        """List events defaults to singleEvents=true for recurring expansion."""
        client.get("/calendars/primary/events")
        call_kwargs = mock_proxy_client.list_events.call_args.kwargs
        assert call_kwargs["single_events"] is True

    def test_list_event_without_attendees_counts_zero(self, client, mock_proxy_client):
        """An event with no attendees key is a count of 0 on the wire, not an
        error and not a phantom count (finding 9, round 3)."""
        mock_proxy_client.list_events.return_value = {"items": [SAMPLE_EVENTS["all_day_event"]]}
        response = client.get("/calendars/primary/events")
        assert response.status_code == 200
        assert response.json()["events"][0]["attendee_count"] == 0

    def test_list_survives_a_malformed_event_row(self, client, mock_proxy_client):
        """A row with a non-dict organizer and a non-dict attendee entry is
        served as null/unknown rather than 500ing the page (finding 8,
        round 3). Only organizer, creator and attendee rows are guarded."""
        bad = {
            **SAMPLE_EVENTS["basic_meeting"],
            "organizer": "dave@example.com",
            "attendees": [None],
        }
        mock_proxy_client.list_events.return_value = {"items": [bad]}
        response = client.get("/calendars/primary/events")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["events"]) == 1
        assert data["events"][0]["calendar_rsvp_state"] == "unknown"
        assert data["events"][0]["attendee_count"] == 0

    def test_list_serves_a_bare_cancelled_stub(self, client, mock_proxy_client):
        """A plain GET with single_events=false returns stubs without
        start/end; the page must still be 200 (finding 5, round 4)."""
        mock_proxy_client.list_events.return_value = {"items": [CANCELLED_STUB]}
        response = client.get("/calendars/primary/events", params={"single_events": "false"})
        assert response.status_code == 200
        row = response.json()["events"][0]
        assert row["status"] == "cancelled"
        assert row["start"] == "" and row["end"] == ""
        assert row["is_all_day"] is False

    def test_list_events_empty(self, client, mock_proxy_client):
        """List events handles empty results."""
        mock_proxy_client.list_events.return_value = {"items": []}
        response = client.get("/calendars/primary/events")
        data = response.json()
        assert data["success"] is True
        assert len(data["events"]) == 0

    def test_list_events_with_pagination_token(self, client, mock_proxy_client):
        """List events returns pagination token when available."""
        mock_proxy_client.list_events.return_value = {
            "items": [],
            "nextPageToken": "next_page_123",
        }
        response = client.get("/calendars/primary/events")
        data = response.json()
        assert data["next_page_token"] == "next_page_123"


class TestRsvpFieldsOnTheWire:
    """The organizer/RSVP fields as list and search actually emit them."""

    def test_list_on_primary(self, client, mock_proxy_client):
        mock_proxy_client.list_events.return_value = {"items": [SAMPLE_EVENTS["invitation"]]}
        response = client.get("/calendars/primary/events")
        assert response.status_code == 200
        event = response.json()["events"][0]
        assert event["organizer_email"] == "dave@example.com"
        assert event["creator_email"] == "dave@example.com"
        assert event["calendar_is_organizer"] is False
        assert event["calendar_rsvp_state"] == "needsAction"
        assert event["status"] == "confirmed"
        assert "is_organizer" not in event
        assert "response_status" not in event
        assert "calendar_response_status" not in event

    def test_list_on_colleague_calendar_reports_the_colleagues_rsvp(
        self, client, mock_proxy_client
    ):
        mock_proxy_client.list_events.return_value = {"items": [SAMPLE_EVENTS["colleague_copy"]]}
        response = client.get("/calendars/carol@example.com/events")
        assert response.status_code == 200
        event = response.json()["events"][0]
        assert event["calendar_id"] == "carol@example.com"
        assert event["calendar_rsvp_state"] == "accepted"  # Carol's, not john.doe's "declined"
        assert not any(k.startswith("user_") for k in event)

    def test_search_on_colleague_calendar_reports_the_colleagues_rsvp(
        self, client, mock_proxy_client
    ):
        mock_proxy_client.list_events.return_value = {"items": [SAMPLE_EVENTS["colleague_copy"]]}
        response = client.post(
            "/search", json={"calendar_id": "carol@example.com", "filters": {"query": "Budget"}}
        )
        assert response.status_code == 200
        event = response.json()["events"][0]
        assert event["calendar_is_organizer"] is False
        assert event["calendar_rsvp_state"] == "accepted"

    def test_read_needs_no_extra_proxy_call(self, client, mock_proxy_client):
        """Deriving the calendar's perspective is local: one proxy call per
        list, nothing to resolve the caller's identity."""
        mock_proxy_client.list_events.return_value = {"items": [SAMPLE_EVENTS["colleague_copy"]]}
        assert client.get("/calendars/carol@example.com/events").status_code == 200
        assert mock_proxy_client.list_events.await_count == 1
        assert mock_proxy_client.get_calendar.await_count == 0


class TestEventCreateEndpoint:
    """Tests for POST /calendars/{calendar_id}/events."""

    def test_create_event_success(self, client, mock_proxy_client):
        """Create event returns created event."""
        response = client.post("/calendars/primary/events", json=CREATE_EVENT_BODY)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["event"] is not None

    def test_create_event_minimal(self, client, mock_proxy_client):
        """Create event with minimal data."""
        event_data = {
            "summary": "Quick Note",
        }
        response = client.post("/calendars/primary/events", json=event_data)
        assert response.status_code == 200

    def test_create_event_with_attendees(self, client, mock_proxy_client):
        """Create event with attendees."""
        event_data = {
            "summary": "Team Meeting",
            "start": {"dateTime": "2024-01-15T10:00:00Z"},
            "end": {"dateTime": "2024-01-15T11:00:00Z"},
            "attendees": [
                {"email": "alice@example.com"},
                {"email": "bob@example.com", "optional": True},
            ],
        }
        response = client.post("/calendars/primary/events", json=event_data)
        assert response.status_code == 200

    def test_create_event_with_send_updates(self, client, mock_proxy_client):
        """Create event with sendUpdates parameter."""
        event_data = {"summary": "Meeting"}
        client.post(
            "/calendars/primary/events?send_updates=all",
            json=event_data,
        )
        mock_proxy_client.create_event.assert_called_once()


class TestEventGetEndpoint:
    """Tests for GET /calendars/{calendar_id}/events/{event_id}."""

    def test_get_event_success(self, client, mock_proxy_client):
        """Get event returns full event details."""
        response = client.get("/calendars/primary/events/event_123")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["event"]["id"] == "meeting_001"

    def test_get_event_returns_the_full_google_event(self, client, mock_proxy_client):
        """The detail route is not a summary: description and the attendee
        list with addresses come back as the proxy sent them (the README's
        privacy section says exactly this)."""
        mock_proxy_client.get_event.return_value = SAMPLE_EVENTS["invitation"]
        event = client.get("/calendars/primary/events/invite_001").json()["event"]
        assert event["description"] == "Quarterly budget walkthrough"
        assert any(a["email"] == AUTH_USER_EMAIL for a in event["attendees"])

    def test_get_event_detail_has_empty_warnings(self, client, mock_proxy_client):
        """`warnings` is on every EventDetailResponse (added for /respond);
        the other detail routes emit an empty list."""
        response = client.get("/calendars/primary/events/event_123")
        assert response.status_code == 200
        assert response.json()["warnings"] == []

    def test_get_event_with_timezone(self, client, mock_proxy_client):
        """Get event with timezone parameter."""
        client.get("/calendars/primary/events/event_123?time_zone=America/New_York")
        call_kwargs = mock_proxy_client.get_event.call_args.kwargs
        assert call_kwargs["time_zone"] == "America/New_York"

    def test_get_event_not_found(self, client, mock_proxy_client):
        """Get event handles not found."""
        mock_proxy_client.get_event.side_effect = ProxyError("Event not found")
        response = client.get("/calendars/primary/events/nonexistent")
        data = response.json()
        assert data["success"] is False


class TestEventUpdateEndpoint:
    """Tests for PUT /calendars/{calendar_id}/events/{event_id}."""

    def test_update_event_success(self, client, mock_proxy_client):
        """Update event returns updated event."""
        response = client.put("/calendars/primary/events/event_123", json=UPDATE_EVENT_BODY)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestEventPatchEndpoint:
    """Tests for PATCH /calendars/{calendar_id}/events/{event_id}."""

    def test_patch_event_success(self, client, mock_proxy_client):
        """Patch event with partial update."""
        response = client.patch("/calendars/primary/events/event_123", json=PATCH_EVENT_BODY)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_patch_event_only_changed_fields(self, client, mock_proxy_client):
        """Patch sends only the changed fields."""
        patch_data = {"location": "New Room"}
        client.patch("/calendars/primary/events/event_123", json=patch_data)
        call_kwargs = mock_proxy_client.patch_event.call_args.kwargs
        assert "location" in call_kwargs["event_data"]


class TestEventDeleteEndpoint:
    """Tests for DELETE /calendars/{calendar_id}/events/{event_id}."""

    def test_delete_event_success(self, client, mock_proxy_client):
        """Delete event returns success once the re-read shows it gone."""
        mock_proxy_client.get_event.side_effect = ProxyNotFoundError("Not Found")
        response = client.delete("/calendars/primary/events/event_123")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["outcome"] == "succeeded"
        assert "deleted" in data["message"].lower()

    def test_delete_is_verified_by_re_reading_the_event(self, client, mock_proxy_client):
        """The proxy's answer is a claim; the re-read is the evidence (F4)."""
        mock_proxy_client.get_event.side_effect = ProxyNotFoundError("Not Found")
        client.delete("/calendars/primary/events/event_123")
        assert [c[0] for c in mock_proxy_client.mock_calls] == ["delete_event", "get_event"]
        assert mock_proxy_client.get_event.call_args.args[:2] == ("primary", "event_123")

    def test_cancelled_status_on_re_read_is_success(self, client, mock_proxy_client):
        mock_proxy_client.get_event.return_value = {"id": "event_123", "status": "cancelled"}
        response = client.delete("/calendars/primary/events/event_123")
        assert response.status_code == 200
        assert response.json()["outcome"] == "succeeded"

    def test_success_claim_contradicted_by_re_read_is_failure(self, client, mock_proxy_client):
        """The 2026-08-07 incident: the proxy said 200 for an event still
        confirmed. Fixture default: get_event returns a confirmed event."""
        response = client.delete("/calendars/primary/events/event_123")
        assert response.status_code == 502
        data = response.json()
        assert data["success"] is False
        assert data["outcome"] == "failed"
        assert "still present" in data["error"]

    def test_delete_event_rejected_by_operator(self, client, mock_proxy_client):
        """An operator rejection surfaces as 403 with the rejection message."""
        mock_proxy_client.delete_event.side_effect = ProxyForbiddenError(
            "Request rejected by operator"
        )
        response = client.delete("/calendars/primary/events/event_123")
        assert response.status_code == 403
        data = response.json()
        assert data["success"] is False
        assert "rejected" in data["message"].lower()
        assert "rejected by operator" in data["error"]

    def test_delete_event_timeout_outcome_unknown(self, client, mock_proxy_client):
        """A timed-out delete with the event still present is 504/unknown."""
        mock_proxy_client.delete_event.side_effect = ProxyTimeoutError(
            "No response from proxy after 330s"
        )
        response = client.delete("/calendars/primary/events/event_123")
        assert response.status_code == 504
        data = response.json()
        assert data["success"] is False
        assert data["outcome"] == "unknown"
        assert "unknown" in data["message"].lower()

    def test_delete_timeout_but_event_gone_is_success(self, client, mock_proxy_client):
        """Approved after this server gave up, but before the re-read: the
        deletion did happen, and saying 504 would invite a compensating
        mutation against an event that no longer exists."""
        mock_proxy_client.delete_event.side_effect = ProxyTimeoutError("no response")
        mock_proxy_client.get_event.side_effect = ProxyNotFoundError("Not Found")
        response = client.delete("/calendars/primary/events/event_123")
        assert response.status_code == 200
        assert response.json()["outcome"] == "succeeded"

    def test_delete_timeout_and_unreadable_re_read_is_unknown(
        self, client, mock_proxy_client
    ):
        mock_proxy_client.delete_event.side_effect = ProxyTimeoutError("no response")
        mock_proxy_client.get_event.side_effect = ProxyError("proxy down")
        response = client.delete("/calendars/primary/events/event_123")
        assert response.status_code == 504
        assert response.json()["outcome"] == "unknown"

    def test_rejection_is_failed_without_a_re_read(self, client, mock_proxy_client):
        """A 403 is the proxy saying it dropped the request: definitive."""
        mock_proxy_client.delete_event.side_effect = ProxyForbiddenError("rejected")
        response = client.delete("/calendars/primary/events/event_123")
        assert response.status_code == 403
        assert response.json()["outcome"] == "failed"
        mock_proxy_client.get_event.assert_not_called()

    def test_nonexistent_event_is_404_failed(self, client, mock_proxy_client):
        mock_proxy_client.delete_event.side_effect = ProxyNotFoundError("Not Found")
        response = client.delete("/calendars/primary/events/event_123")
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert data["outcome"] == "failed"
        mock_proxy_client.get_event.assert_not_called()


class TestEventRespondEndpoint:
    """Tests for POST /calendars/{calendar_id}/events/{event_id}/respond."""

    def test_respond_success(self, client, mock_proxy_client):
        """RSVP forwards to the proxy client and returns the updated event."""
        mock_proxy_client.respond_to_event.return_value = {"id": "e1", "summary": "GMDM"}
        resp = client.post("/calendars/primary/events/e1/respond", json=RESPOND_BODY)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["event"]["id"] == "e1"
        mock_proxy_client.respond_to_event.assert_called_once_with("primary", "e1", "accepted")

    def test_respond_on_primary_carries_no_warning(self, client, mock_proxy_client):
        resp = client.post(
            "/calendars/primary/events/e1/respond",
            json={"response_status": "accepted"},
        )
        assert resp.status_code == 200
        assert resp.json()["warnings"] == []

    def test_respond_error_envelope_has_no_warnings(self, client, mock_proxy_client):
        mock_proxy_client.respond_to_event.side_effect = ProxyForbiddenError("blocked")
        resp = client.post(
            "/calendars/primary/events/e1/respond",
            json={"response_status": "accepted"},
        )
        assert resp.status_code == 403
        assert resp.json()["warnings"] == []


class TestEventRespondRefusesForeignCalendars:
    """POST .../respond refuses any calendar_id that is not the authenticated
    user's own (Brian's decision, 2026-09-04).

    The read fields describe the calendar being read; /respond always writes
    the authenticated user's entry. Refusing keeps the two on the same
    calendar, where they agree.
    """

    def test_refuses_a_group_calendar_id(self, client, mock_proxy_client):
        """A group calendar is never the authenticated account's own."""
        mock_proxy_client.get_calendar.return_value = {"id": AUTH_USER_EMAIL}
        resp = client.post(
            f"/calendars/{GROUP_CALENDAR_ID}/events/invite_001/respond",
            json={"response_status": "accepted"},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["success"] is False
        assert GROUP_CALENDAR_ID in data["error"]
        assert "primary" in data["error"]
        mock_proxy_client.respond_to_event.assert_not_called()

    def test_refuses_a_colleagues_address(self, client, mock_proxy_client):
        mock_proxy_client.get_calendar.return_value = {"id": AUTH_USER_EMAIL}
        resp = client.post(
            f"/calendars/{COLLEAGUE_EMAIL}/events/invite_001/respond",
            json={"response_status": "declined"},
        )
        assert resp.status_code == 400
        assert COLLEAGUE_EMAIL in resp.json()["error"]
        mock_proxy_client.respond_to_event.assert_not_called()

    def test_refusal_envelope_has_no_event_and_no_warnings(self, client, mock_proxy_client):
        mock_proxy_client.get_calendar.return_value = {"id": AUTH_USER_EMAIL}
        resp = client.post(
            f"/calendars/{COLLEAGUE_EMAIL}/events/invite_001/respond",
            json={"response_status": "declined"},
        )
        assert resp.json()["event"] is None
        assert resp.json()["warnings"] == []

    def test_allows_the_literal_primary_without_resolving_an_address(
        self, client, mock_proxy_client
    ):
        """'primary' needs no identity lookup, so the common path costs no
        extra proxy call."""
        resp = client.post(
            "/calendars/primary/events/e1/respond",
            json={"response_status": "accepted"},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        assert mock_proxy_client.get_calendar.await_count == 0

    def test_literal_primary_is_matched_case_insensitively(self, client, mock_proxy_client):
        """'Primary' takes the same no-lookup fast path as 'primary': the
        address comparison already casefolds, so the literal must too
        (Opus delta review, finding 2)."""
        mock_proxy_client.get_calendar.return_value = {"id": AUTH_USER_EMAIL}
        resp = client.post(
            "/calendars/Primary/events/e1/respond",
            json={"response_status": "accepted"},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        assert mock_proxy_client.get_calendar.await_count == 0

    def test_allows_the_authenticated_users_own_address(self, client, mock_proxy_client):
        mock_proxy_client.get_calendar.return_value = {"id": AUTH_USER_EMAIL}
        resp = client.post(
            f"/calendars/{AUTH_USER_EMAIL}/events/invite_001/respond",
            json={"response_status": "accepted"},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        mock_proxy_client.respond_to_event.assert_called_once_with(
            AUTH_USER_EMAIL, "invite_001", "accepted"
        )

    def test_own_address_matches_case_insensitively(self, client, mock_proxy_client):
        """Google lowercases calendar ids; a caller's capitalisation must not
        turn an allowed RSVP into a refusal."""
        mock_proxy_client.get_calendar.return_value = {"id": AUTH_USER_EMAIL}
        resp = client.post(
            f"/calendars/{AUTH_USER_EMAIL.upper()}/events/invite_001/respond",
            json={"response_status": "accepted"},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_resolves_the_authenticated_address_once_per_process(
        self, client, mock_proxy_client
    ):
        """The identity lookup is cached: two RSVPs, one GET /calendars/primary."""
        mock_proxy_client.get_calendar.return_value = {"id": AUTH_USER_EMAIL}
        for _ in range(2):
            client.post(
                f"/calendars/{AUTH_USER_EMAIL}/events/invite_001/respond",
                json={"response_status": "accepted"},
            )
        assert mock_proxy_client.get_calendar.await_count == 1

    def test_unresolvable_identity_is_an_upstream_failure_not_a_refusal(
        self, client, mock_proxy_client
    ):
        """If the proxy's primary calendar carries no id there is nothing to
        compare against: 502, and the RSVP is not forwarded."""
        mock_proxy_client.get_calendar.return_value = {"summary": "no id here"}
        resp = client.post(
            f"/calendars/{AUTH_USER_EMAIL}/events/invite_001/respond",
            json={"response_status": "accepted"},
        )
        assert resp.status_code == 502
        assert resp.json()["success"] is False
        mock_proxy_client.respond_to_event.assert_not_called()

    def test_identity_lookup_timeout_is_an_upstream_failure_not_an_unknown_outcome(
        self, client, mock_proxy_client
    ):
        """A read timeout on GET /calendars/primary happens before anything is
        sent, so it must not be reported as a mutation whose outcome is
        unknown (Opus delta review, finding 1): 502, the message says the
        ownership check could not be performed, and no RSVP is forwarded."""
        mock_proxy_client.get_calendar.side_effect = ProxyTimeoutError(
            "No response from proxy after 30s"
        )
        resp = client.post(
            f"/calendars/{AUTH_USER_EMAIL}/events/invite_001/respond",
            json={"response_status": "accepted"},
        )
        assert resp.status_code == 502
        data = resp.json()
        assert data["success"] is False
        assert "Outcome unknown" not in data["error"]
        assert "ownership check" in data["error"]
        assert "nothing was sent" in data["error"]
        mock_proxy_client.respond_to_event.assert_not_awaited()

    def test_identity_lookup_not_found_is_an_upstream_failure_not_a_404(
        self, client, mock_proxy_client
    ):
        """A 404 on GET /calendars/primary is not "no such event": the URL
        names an event this route never looked at. 502, nothing sent."""
        mock_proxy_client.get_calendar.side_effect = ProxyNotFoundError(
            "Calendar not found"
        )
        resp = client.post(
            f"/calendars/{AUTH_USER_EMAIL}/events/invite_001/respond",
            json={"response_status": "accepted"},
        )
        assert resp.status_code == 502
        assert resp.json()["success"] is False
        assert "ownership check" in resp.json()["error"]
        mock_proxy_client.respond_to_event.assert_not_awaited()

    @pytest.mark.parametrize(
        "lookup_failure",
        [
            ProxyForbiddenError("blocked"),
            ProxyRequestError(429, "slow down"),
            ProxyAuthError("bad key"),
            ProxyError("connection refused"),
        ],
        ids=["forbidden", "other-4xx", "auth", "generic"],
    )
    def test_any_identity_lookup_failure_is_502_with_nothing_sent(
        self, client, mock_proxy_client, lookup_failure
    ):
        """Pins the documented guarantee: whatever the proxy does to the
        identity read, the caller sees 502 and no RSVP was attempted. A
        forbidden here would otherwise read as "the operator rejected your
        RSVP" for a read no operator ever saw."""
        mock_proxy_client.get_calendar.side_effect = lookup_failure
        resp = client.post(
            f"/calendars/{AUTH_USER_EMAIL}/events/invite_001/respond",
            json={"response_status": "accepted"},
        )
        assert resp.status_code == 502
        assert resp.json()["success"] is False
        assert "nothing was sent" in resp.json()["error"]
        mock_proxy_client.respond_to_event.assert_not_awaited()

    def test_respond_invalid_status_rejected(self, client, mock_proxy_client):
        """Values outside accepted/declined/tentative are rejected with 422."""
        resp = client.post(
            "/calendars/primary/events/e1/respond",
            json={"response_status": "maybe"},
        )
        assert resp.status_code == 422
        mock_proxy_client.respond_to_event.assert_not_called()

    def test_respond_forbidden_returns_error(self, client, mock_proxy_client):
        """Proxy 403 surfaces as success=false with an error message."""
        mock_proxy_client.respond_to_event.side_effect = ProxyForbiddenError("blocked")
        resp = client.post(
            "/calendars/primary/events/e1/respond",
            json={"response_status": "accepted"},
        )
        assert resp.status_code == 403
        data = resp.json()
        assert data["success"] is False
        assert data["error"]

    def test_respond_timeout_returns_504(self, client, mock_proxy_client):
        """A timed-out RSVP surfaces as 504 with outcome-unknown error text."""
        mock_proxy_client.respond_to_event.side_effect = ProxyTimeoutError(
            "No response from proxy after 330s"
        )
        resp = client.post(
            "/calendars/primary/events/e1/respond",
            json={"response_status": "accepted"},
        )
        assert resp.status_code == 504
        data = resp.json()
        assert data["success"] is False
        assert "Outcome unknown" in data["error"]

    def test_respond_not_an_attendee_passes_the_proxys_400_through(self, client, mock_proxy_client):
        """The proxy's 400 ("not an attendee") reaches the caller as 400 with
        the proxy's message in ``error`` (finding 3, round 4) -- not 502."""
        mock_proxy_client.respond_to_event.side_effect = ProxyRequestError(
            400, "You are not an attendee of this event; cannot RSVP."
        )
        resp = client.post(
            "/calendars/primary/events/e1/respond",
            json={"response_status": "accepted"},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["success"] is False
        assert "You are not an attendee of this event; cannot RSVP." in data["error"]


# ============================================================================
# Proxy Client Tests
# ============================================================================


class TestProxyClientRespond:
    """CalendarProxyClient.respond_to_event forwards to the proxy /respond route."""

    @pytest.fixture
    def mock_response(self):
        r = AsyncMock(spec=httpx.Response)
        r.status_code = 200
        r.json.return_value = {
            "id": "e1",
            "attendees": [{"email": "me@x", "responseStatus": "accepted", "self": True}],
        }
        return r

    async def _call(self, mock_response, calendar_id, event_id, status):
        client = CalendarProxyClient(proxy_url="http://proxy", api_key="k")
        with patch("calendar_agent.proxy_client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.respond_to_event(calendar_id, event_id, status)
        return result, mock_client

    async def test_posts_to_respond_path_with_status(self, mock_response):
        result, mock_client = await self._call(mock_response, "robergb@dm.org", "e1", "accepted")

        mock_client.post.assert_called_once()
        call = mock_client.post.call_args
        assert call.args[0] == "http://proxy/calendar/v3/calendars/robergb@dm.org/events/e1/respond"
        assert call.kwargs["json"] == {"responseStatus": "accepted"}
        assert result["id"] == "e1"

    async def test_encodes_special_characters_in_path(self, mock_response):
        """A '#' in the calendar ID must be percent-encoded, not left as a fragment."""
        _, mock_client = await self._call(
            mock_response, "en.usa#holiday@group.v.calendar.google.com", "e1", "declined"
        )

        url = mock_client.post.call_args.args[0]
        assert "%23" in url
        assert "#" not in url
        assert url == (
            "http://proxy/calendar/v3/calendars/"
            "en.usa%23holiday@group.v.calendar.google.com/events/e1/respond"
        )


class TestProxyClientPathQuoting:
    """Every event route must percent-encode its path segments (F9).

    Google's built-in calendar ids contain ``#`` (``#contacts@group.v...``,
    ``en.usa#holiday@...``). Unquoted, httpx reads the ``#`` as a fragment and
    the request goes to ``/calendars/`` — a different route, whose 404 now maps
    to "the event does not exist" instead of a recognisable upstream error.
    """

    CAL = "#contacts@group.v.calendar.google.com"
    EXPECTED = (
        "http://proxy/calendar/v3/calendars/"
        "%23contacts@group.v.calendar.google.com/events/e%2F1"
    )

    @pytest.fixture
    def ok_response(self):
        r = AsyncMock(spec=httpx.Response)
        r.status_code = 200
        r.json.return_value = {"id": "e/1"}
        return r

    @pytest.mark.parametrize(
        ("method", "call"),
        [
            ("get", lambda c, cal: c.get_event(cal, "e/1")),
            ("put", lambda c, cal: c.update_event(cal, "e/1", {"summary": "x"})),
            ("patch", lambda c, cal: c.patch_event(cal, "e/1", {"summary": "x"})),
            ("delete", lambda c, cal: c.delete_event(cal, "e/1")),
        ],
    )
    async def test_event_routes_quote_both_segments(self, ok_response, method, call):
        client = CalendarProxyClient(proxy_url="http://proxy", api_key="k")
        with patch("calendar_agent.proxy_client.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            getattr(mock_http, method).return_value = ok_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await call(client, self.CAL)
        url = getattr(mock_http, method).call_args.args[0]
        assert url == self.EXPECTED
        assert httpx.URL(url).raw_path.decode().endswith("/events/e%2F1")

    async def test_proxy_410_is_not_found(self):
        """Google answers 410 Gone for a deleted event; that is "absent", the
        expected answer when verifying a delete, not a generic proxy error."""
        client = CalendarProxyClient(proxy_url="http://proxy", api_key="k")
        response = AsyncMock(spec=httpx.Response)
        response.status_code = 410
        response.json.return_value = {"detail": "Resource has been deleted"}
        with pytest.raises(ProxyNotFoundError):
            client._handle_response(response)


class TestProxyClientTimeouts:
    """Mutations must outlive the proxy's 300s confirmation window (issue #4)."""

    def _mock_http(self, mock_cls, method: str, response=None, side_effect=None):
        mock_http = AsyncMock()
        if response is not None:
            getattr(mock_http, method).return_value = response
        if side_effect is not None:
            getattr(mock_http, method).side_effect = side_effect
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        return mock_http

    @pytest.fixture
    def ok_response(self):
        r = AsyncMock(spec=httpx.Response)
        r.status_code = 200
        r.json.return_value = {"id": "e1"}
        return r

    def test_confirm_timeout_outlives_confirmation_window(self):
        """The mutation timeout must exceed the proxy's 300s approval window."""
        assert CONFIRM_TIMEOUT > 300

    async def test_mutating_call_uses_confirm_timeout(self, ok_response):
        client = CalendarProxyClient(proxy_url="http://proxy", api_key="k")
        with patch("calendar_agent.proxy_client.httpx.AsyncClient") as mock_cls:
            self._mock_http(mock_cls, "post", response=ok_response)
            await client.respond_to_event("primary", "e1", "accepted")
        assert mock_cls.call_args.kwargs["timeout"] == httpx.Timeout(
            CONFIRM_TIMEOUT, connect=READ_TIMEOUT
        )

    async def test_read_call_uses_read_timeout(self, ok_response):
        client = CalendarProxyClient(proxy_url="http://proxy", api_key="k")
        with patch("calendar_agent.proxy_client.httpx.AsyncClient") as mock_cls:
            self._mock_http(mock_cls, "get", response=ok_response)
            await client.get_event("primary", "e1")
        assert mock_cls.call_args.kwargs["timeout"] == httpx.Timeout(
            READ_TIMEOUT, connect=READ_TIMEOUT
        )

    async def test_timeout_raises_proxy_timeout_error(self):
        """An httpx timeout surfaces as ProxyTimeoutError with unknown-outcome text."""
        client = CalendarProxyClient(proxy_url="http://proxy", api_key="k")
        with patch("calendar_agent.proxy_client.httpx.AsyncClient") as mock_cls:
            self._mock_http(mock_cls, "delete", side_effect=httpx.ReadTimeout("timed out"))
            with pytest.raises(ProxyTimeoutError) as exc_info:
                await client.delete_event("primary", "e1")
        assert "outcome is unknown" in str(exc_info.value)

    async def test_connect_timeout_is_definitive_failure(self):
        """A connect-phase timeout never reached the proxy: ProxyError, not
        outcome-unknown ProxyTimeoutError."""
        client = CalendarProxyClient(proxy_url="http://proxy", api_key="k")
        with patch("calendar_agent.proxy_client.httpx.AsyncClient") as mock_cls:
            self._mock_http(mock_cls, "post", side_effect=httpx.ConnectTimeout("connect"))
            with pytest.raises(ProxyError) as exc_info:
                await client.create_event("primary", {"summary": "s"})
        assert not isinstance(exc_info.value, ProxyTimeoutError)
        assert "Could not connect" in str(exc_info.value)

    async def test_connection_error_is_proxy_error(self):
        """A down/unreachable proxy surfaces as ProxyError (502), not a raw
        httpx exception (500)."""
        client = CalendarProxyClient(proxy_url="http://proxy", api_key="k")
        with patch("calendar_agent.proxy_client.httpx.AsyncClient") as mock_cls:
            self._mock_http(
                mock_cls, "get", side_effect=httpx.ConnectError("All connection attempts failed")
            )
            with pytest.raises(ProxyError) as exc_info:
                await client.get_event("primary", "e1")
        assert not isinstance(exc_info.value, ProxyTimeoutError)
        assert "Proxy connection failed" in str(exc_info.value)


class TestProxyClientRequestErrors:
    """A proxy 4xx other than 401/403/404/410 raises ProxyRequestError carrying
    the upstream status and message (finding 3, round 4). 404 and 410 stay
    ProxyNotFoundError, which delete verification depends on (issue #4)."""

    def _response(self, status: int, detail: str):
        r = AsyncMock(spec=httpx.Response)
        r.status_code = status
        r.json.return_value = {"detail": detail}
        return r

    def test_400_carries_status_and_message(self):
        client = CalendarProxyClient(proxy_url="http://proxy", api_key="k")
        with pytest.raises(ProxyRequestError) as exc_info:
            client._handle_response(
                self._response(400, "You are not an attendee of this event; cannot RSVP.")
            )
        assert exc_info.value.status_code == 400
        assert str(exc_info.value) == "You are not an attendee of this event; cannot RSVP."

    def test_404_is_not_found_error_not_request_error(self):
        client = CalendarProxyClient(proxy_url="http://proxy", api_key="k")
        with pytest.raises(ProxyNotFoundError) as exc_info:
            client._handle_response(self._response(404, "Not Found"))
        assert not isinstance(exc_info.value, ProxyRequestError)
        assert str(exc_info.value) == "Not Found"

    def test_other_4xx_is_still_a_request_error(self):
        client = CalendarProxyClient(proxy_url="http://proxy", api_key="k")
        with pytest.raises(ProxyRequestError) as exc_info:
            client._handle_response(self._response(409, "conflict"))
        assert exc_info.value.status_code == 409
        assert isinstance(exc_info.value, ProxyError)

    def test_401_and_403_keep_their_own_types(self):
        client = CalendarProxyClient(proxy_url="http://proxy", api_key="k")
        with pytest.raises(ProxyAuthError):
            client._handle_response(self._response(401, "bad key"))
        with pytest.raises(ProxyForbiddenError):
            client._handle_response(self._response(403, "rejected"))


# ============================================================================
# LLM Endpoint Tests - Basic
# ============================================================================


class TestSummarizeEndpoint:
    """Tests for POST /summarize."""

    def test_summarize_success(self, client, mock_proxy_client, mock_llm_service):
        """Summarize event returns AI summary."""
        response = client.post("/summarize", json=SUMMARIZE_BODY)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "summary" in data["data"]

    def test_summarize_detailed_format(self, client, mock_proxy_client, mock_llm_service):
        """Summarize with detailed format."""
        request_data = {
            "calendar_id": "primary",
            "event_id": "event_123",
            "format": "detailed",
        }
        client.post("/summarize", json=request_data)
        mock_llm_service.summarize_event.assert_called_once()
        call_kwargs = mock_llm_service.summarize_event.call_args.kwargs
        assert call_kwargs["format"] == "detailed"

    def test_summarize_event_not_found(self, client, mock_proxy_client, mock_llm_service):
        """Summarize handles event not found."""
        mock_proxy_client.get_event.side_effect = ProxyError("Event not found")
        request_data = {
            "calendar_id": "primary",
            "event_id": "nonexistent",
        }
        response = client.post("/summarize", json=request_data)
        data = response.json()
        assert data["success"] is False


class TestAskAboutEndpoint:
    """Tests for POST /ask-about."""

    def test_ask_about_success(self, client, mock_proxy_client, mock_llm_service):
        """Ask about event returns AI answer."""
        response = client.post("/ask-about", json=ASK_ABOUT_BODY)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "answer" in data["data"]

    def test_ask_about_requires_question(self, client):
        """Ask about requires question field."""
        request_data = {
            "calendar_id": "primary",
            "event_id": "event_123",
        }
        response = client.post("/ask-about", json=request_data)
        assert response.status_code == 422  # Validation error


class TestBatchSummarizeEndpoint:
    """Tests for POST /batch-summarize."""

    def test_batch_summarize_success(self, client, mock_proxy_client, mock_llm_service):
        """Batch summarize returns summaries for multiple events."""
        response = client.post("/batch-summarize", json=BATCH_SUMMARIZE_BODY)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_batch_summarize_with_triage(self, client, mock_proxy_client, mock_llm_service):
        """Batch summarize with triage classification."""
        request_data = {
            "calendar_id": "primary",
            "event_ids": ["event_1"],
            "triage": True,
        }
        client.post("/batch-summarize", json=request_data)
        call_kwargs = mock_llm_service.batch_summarize.call_args.kwargs
        assert call_kwargs["triage"] is True

    def test_batch_summarize_continues_on_fetch_error(
        self, client, mock_proxy_client, mock_llm_service
    ):
        """Batch summarize continues if individual event fetch fails."""
        # First call succeeds, second fails
        mock_proxy_client.get_event.side_effect = [
            {"id": "event_1", "summary": "Event 1"},
            ProxyError("Event not found"),
        ]
        request_data = {
            "calendar_id": "primary",
            "event_ids": ["event_1", "event_2"],
        }
        response = client.post("/batch-summarize", json=request_data)
        data = response.json()
        assert data["success"] is True


# ============================================================================
# LLM Endpoint Tests - Calendar-Specific
# ============================================================================


class TestFindFreeTimeEndpoint:
    """Tests for POST /find-free-time."""

    def test_find_free_time_success(self, client, mock_proxy_client, mock_llm_service):
        """Find free time returns available slots and suggestions."""
        response = client.post("/find-free-time", json=FIND_FREE_TIME_BODY)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "suggestions" in data["data"]

    def test_find_free_time_with_preferences(
        self, client, mock_proxy_client, mock_llm_service
    ):
        """Find free time with scheduling preferences."""
        request_data = {
            "calendar_id": "primary",
            "time_min": "2024-01-15T09:00:00Z",
            "time_max": "2024-01-15T17:00:00Z",
            "duration_minutes": 60,
            "working_hours_only": True,
            "prefer_morning": True,
            "buffer_minutes": 15,
        }
        response = client.post("/find-free-time", json=request_data)
        assert response.status_code == 200

    def test_find_free_time_validation(self, client):
        """Find free time validates duration."""
        request_data = {
            "calendar_id": "primary",
            "time_min": "2024-01-15T09:00:00Z",
            "time_max": "2024-01-15T17:00:00Z",
            "duration_minutes": 0,  # Invalid
        }
        response = client.post("/find-free-time", json=request_data)
        assert response.status_code == 422


class TestAnalyzeScheduleEndpoint:
    """Tests for POST /analyze-schedule."""

    def test_analyze_schedule_success(self, client, mock_proxy_client, mock_llm_service):
        """Analyze schedule returns insights and metrics."""
        response = client.post("/analyze-schedule", json=ANALYZE_SCHEDULE_BODY)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "insights" in data["data"]

    @pytest.mark.parametrize("analysis_type", ["overview", "workload", "patterns", "conflicts"])
    def test_analyze_schedule_types(
        self, client, mock_proxy_client, mock_llm_service, analysis_type
    ):
        """Analyze schedule supports different analysis types."""
        request_data = {
            "calendar_id": "primary",
            "time_min": "2024-01-15T00:00:00Z",
            "time_max": "2024-01-22T00:00:00Z",
            "analysis_type": analysis_type,
        }
        response = client.post("/analyze-schedule", json=request_data)
        assert response.status_code == 200


class TestPrepareBriefingEndpoint:
    """Tests for POST /prepare-briefing."""

    def test_prepare_briefing_daily(self, client, mock_proxy_client, mock_llm_service):
        """Prepare daily briefing returns schedule overview."""
        response = client.post("/prepare-briefing", json=PREPARE_BRIEFING_BODY)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "briefing" in data["data"]

    def test_prepare_briefing_weekly(self, client, mock_proxy_client, mock_llm_service):
        """Prepare weekly briefing."""
        request_data = {
            "calendar_id": "primary",
            "briefing_type": "weekly",
        }
        response = client.post("/prepare-briefing", json=request_data)
        assert response.status_code == 200

    def test_prepare_briefing_custom_range(
        self, client, mock_proxy_client, mock_llm_service
    ):
        """Prepare briefing with custom time range."""
        request_data = {
            "calendar_id": "primary",
            "briefing_type": "daily",
            "time_min": "2024-01-15T00:00:00Z",
            "time_max": "2024-01-15T23:59:59Z",
        }
        response = client.post("/prepare-briefing", json=request_data)
        assert response.status_code == 200


# ============================================================================
# Operations Endpoint Tests
# ============================================================================


class TestSearchEndpoint:
    """Tests for POST /search."""

    def test_search_success(self, client, mock_proxy_client):
        """Search events with filters."""
        response = client.post("/search", json=SEARCH_BODY)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "events" in data

    def test_search_with_all_filters(self, client, mock_proxy_client):
        """Search with all available filters."""
        request_data = {
            "calendar_id": "primary",
            "filters": {
                "query": "project",
                "time_min": "2024-01-01T00:00:00Z",
                "time_max": "2024-12-31T23:59:59Z",
                "max_results": 50,
                "order_by": "startTime",
                "show_deleted": False,
            },
        }
        response = client.post("/search", json=request_data)
        assert response.status_code == 200

    def test_search_show_deleted_forwards_and_classifies_cancelled_rows(
        self, client, mock_proxy_client
    ):
        """show_deleted is forwarded with single_events fixed to True, and a
        cancelled row that carries organizer/attendees is classified like
        any other row -- not blanked to 'unknown' (finding 2, round 4)."""
        cancelled = colleague_copy()
        cancelled["status"] = "cancelled"
        mock_proxy_client.list_events.return_value = {"items": [cancelled]}
        response = client.post(
            "/search",
            json={"calendar_id": COLLEAGUE_EMAIL, "filters": {"show_deleted": True}},
        )
        assert response.status_code == 200
        kwargs = mock_proxy_client.list_events.call_args.kwargs
        assert kwargs["show_deleted"] is True
        assert kwargs["single_events"] is True
        row = response.json()["events"][0]
        assert row["status"] == "cancelled"
        assert row["organizer_email"] == "dave@example.com"
        assert row["calendar_rsvp_state"] == "accepted"
        assert row["start"] != ""

    def test_search_default_filters(self, client, mock_proxy_client):
        """Search with default filters."""
        request_data = {
            "calendar_id": "primary",
            "filters": {},
        }
        response = client.post("/search", json=request_data)
        assert response.status_code == 200


class TestBulkActionsEndpoint:
    """Tests for POST /bulk-actions."""

    def test_bulk_delete_success(self, client, mock_proxy_client):
        """Bulk delete events."""
        mock_proxy_client.get_event.side_effect = ProxyNotFoundError("Not Found")
        response = client.post("/bulk-actions", json=BULK_DELETE_BODY)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["success_count"] == 2
        assert data["error_count"] == 0

    def test_bulk_patch_success(self, client, mock_proxy_client):
        """Bulk patch events."""
        request_data = {
            "operations": [
                {
                    "operation": "patch",
                    "event_id": "event_1",
                    "calendar_id": "primary",
                    "updates": {"summary": "Updated Title"},
                },
            ]
        }
        response = client.post("/bulk-actions", json=request_data)
        data = response.json()
        assert data["success"] is True

    def test_bulk_mixed_operations(self, client, mock_proxy_client):
        """Bulk operations with mixed types."""
        mock_proxy_client.get_event.side_effect = ProxyNotFoundError("Not Found")
        request_data = {
            "operations": [
                {"operation": "delete", "event_id": "event_1", "calendar_id": "primary"},
                {
                    "operation": "patch",
                    "event_id": "event_2",
                    "calendar_id": "primary",
                    "updates": {"location": "New Room"},
                },
            ]
        }
        response = client.post("/bulk-actions", json=request_data)
        assert response.status_code == 200

    def test_bulk_partial_failure(self, client, mock_proxy_client):
        """Bulk operations continue on individual failures."""
        # First succeeds, second fails
        mock_proxy_client.delete_event.side_effect = [
            {"success": True},
            ProxyError("Event not found"),
        ]
        mock_proxy_client.get_event.side_effect = [
            ProxyNotFoundError("Not Found"),
            {"id": "event_2", "status": "confirmed"},
        ]
        request_data = {
            "operations": [
                {"operation": "delete", "event_id": "event_1", "calendar_id": "primary"},
                {"operation": "delete", "event_id": "event_2", "calendar_id": "primary"},
            ]
        }
        response = client.post("/bulk-actions", json=request_data)
        data = response.json()
        assert data["success"] is False  # one operation did not happen
        assert data["success_count"] == 1
        assert data["error_count"] == 1

    def test_bulk_update_without_data(self, client, mock_proxy_client):
        """Bulk update without update data is a validation error (422)."""
        request_data = {
            "operations": [
                {"operation": "update", "event_id": "event_1", "calendar_id": "primary"},
            ]
        }
        response = client.post("/bulk-actions", json=request_data)
        assert response.status_code == 422
        assert "updates" in response.text
        mock_proxy_client.update_event.assert_not_called()


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestProxyErrorHandling:
    """Tests for proxy error handling across endpoints."""

    def test_auth_error_handling(self, client, mock_proxy_client):
        """Authentication errors are properly formatted."""
        mock_proxy_client.list_calendars.side_effect = ProxyAuthError("Invalid API key")
        response = client.get("/calendars")
        assert response.status_code == 502
        data = response.json()
        assert data["success"] is False
        assert "Authentication error" in data["error"]

    def test_forbidden_error_handling(self, client, mock_proxy_client):
        """Forbidden errors are properly formatted."""
        mock_proxy_client.delete_event.side_effect = ProxyForbiddenError(
            "Request rejected by operator"
        )
        response = client.delete("/calendars/primary/events/event_123")
        assert response.status_code == 403
        data = response.json()
        assert data["success"] is False
        assert "blocked" in data["error"].lower()

    def test_generic_proxy_error_handling(self, client, mock_proxy_client):
        """Generic proxy errors are properly formatted."""
        mock_proxy_client.list_events.side_effect = ProxyError("Connection timeout")
        response = client.get("/calendars/primary/events")
        assert response.status_code == 502
        data = response.json()
        assert data["success"] is False
        assert "Proxy error" in data["error"]

    def test_proxy_404_passes_message_through_as_404(self, client, mock_proxy_client):
        mock_proxy_client.get_event.side_effect = ProxyNotFoundError("Not Found")
        response = client.get("/calendars/primary/events/nope")
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert "Not Found" in data["error"]

    def test_proxy_400_passes_through_as_400(self, client, mock_proxy_client):
        mock_proxy_client.list_events.side_effect = ProxyRequestError(400, "Bad Request")
        response = client.get("/calendars/primary/events")
        assert response.status_code == 400
        assert response.json()["success"] is False

    def test_other_proxy_4xx_still_maps_to_502(self, client, mock_proxy_client):
        """Only 400 passes through (404 is ProxyNotFoundError); any other proxy
        4xx is still 502."""
        mock_proxy_client.get_event.side_effect = ProxyRequestError(409, "conflict")
        response = client.get("/calendars/primary/events/e1")
        assert response.status_code == 502
        assert "conflict" in response.json()["error"]

    def test_proxy_401_maps_to_502(self, client, mock_proxy_client):
        """A proxy 401 (this service's own key rejected) is an upstream
        failure from the caller's point of view: 502, not 401."""
        mock_proxy_client.get_event.side_effect = ProxyAuthError("Invalid API key")
        response = client.get("/calendars/primary/events/e1")
        assert response.status_code == 502


# ============================================================================
# Validation Tests
# ============================================================================


class TestRequestValidation:
    """Tests for request validation."""

    def test_create_event_validates_attendee_email(self, client):
        """Event creation validates attendee email format."""
        event_data = {
            "summary": "Meeting",
            "attendees": [{"email": "not-an-email"}],
        }
        # Note: Pydantic doesn't validate email format by default
        response = client.post("/calendars/primary/events", json=event_data)
        # Should succeed (email format not strictly validated)
        assert response.status_code == 200

    def test_bulk_actions_requires_operations(self, client):
        """Bulk actions requires non-empty operations list."""
        request_data = {"operations": []}
        response = client.post("/bulk-actions", json=request_data)
        assert response.status_code == 422

    def test_find_free_time_requires_positive_duration(self, client):
        """Find free time requires positive duration."""
        request_data = {
            "calendar_id": "primary",
            "time_min": "2024-01-15T00:00:00Z",
            "time_max": "2024-01-16T00:00:00Z",
            "duration_minutes": -30,
        }
        response = client.post("/find-free-time", json=request_data)
        assert response.status_code == 422


class TestConfirmTimeoutResolution:
    """The confirm timeout is env-overridable; it must never silently drop
    back below the proxy's approval window (issue #4, item 1)."""

    def test_default_outlives_the_proxy_confirmation_window(self):
        """With nothing configured, the mutation budget clears the window."""
        assert resolve_confirm_timeout(None) > PROXY_CONFIRMATION_WINDOW

    def test_override_above_the_window_is_honoured(self):
        assert resolve_confirm_timeout("400") == 400.0

    def test_override_below_the_window_is_rejected(self):
        """A 30s override is the original incident's configuration; accepting
        it silently makes every approval outcome undeliverable again."""
        with pytest.raises(ProxyConfigError) as exc_info:
            resolve_confirm_timeout("30")
        assert "PROXY_CONFIRM_TIMEOUT" in str(exc_info.value)
        assert "300" in str(exc_info.value)

    def test_non_numeric_override_is_rejected(self):
        """A typo must fail loudly rather than fall back to a short default."""
        with pytest.raises(ProxyConfigError):
            resolve_confirm_timeout("5 minutes")

    @pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "NaN", "Infinity"])
    def test_non_finite_override_is_rejected(self, raw):
        """float('nan') parses and compares False against everything, so the
        old ``<= window`` guard let it through; ``inf`` disables the timeout.
        Neither is a budget that outlives anything."""
        with pytest.raises(ProxyConfigError):
            resolve_confirm_timeout(raw)

    @pytest.mark.parametrize("raw", ["301", "329.9"])
    def test_override_inside_the_margin_is_rejected(self, raw):
        """Clearing the window by a second is not clearing it: the proxy's own
        timeout, its response, and the network all sit inside that gap."""
        with pytest.raises(ProxyConfigError) as exc_info:
            resolve_confirm_timeout(raw)
        assert str(int(PROXY_CONFIRMATION_WINDOW + CONFIRM_TIMEOUT_MARGIN)) in str(
            exc_info.value
        )

    def test_override_at_window_plus_margin_is_accepted(self):
        assert resolve_confirm_timeout("330") == 330.0

    def test_guard_follows_an_overridden_confirmation_window(self):
        """The window is api-proxy's ``--confirmation-timeout``, not a law of
        nature; an operator who raises it there must be able to keep this
        guard honest, or it passes while the invariant is violated (F5)."""
        with pytest.raises(ProxyConfigError):
            resolve_confirm_timeout("330", window=600.0)
        assert resolve_confirm_timeout("630", window=600.0) == 630.0


class TestConfirmationWindowResolution:
    """``PROXY_CONFIRMATION_WINDOW`` mirrors api-proxy's confirmation timeout."""

    def test_default_matches_api_proxy_default(self):
        assert resolve_confirmation_window(None) == 300.0

    def test_override_is_honoured(self):
        assert resolve_confirmation_window("600") == 600.0

    @pytest.mark.parametrize("raw", ["0", "-1", "nan", "inf", "five"])
    def test_unbounded_or_invalid_window_is_rejected(self, raw):
        """api-proxy treats ``<= 0`` as "wait forever" — a window no client
        timeout can outlive, so the guard cannot be honest and must say so."""
        with pytest.raises(ProxyConfigError) as exc_info:
            resolve_confirmation_window(raw)
        assert "PROXY_CONFIRMATION_WINDOW" in str(exc_info.value)


class TestMissingEventStatus:
    """A resource the proxy says is gone must surface as 404, not 502.

    Item 3's verify-by-re-read (issue #4) is specified as "expect 404 or
    status: cancelled"; collapsing 404 into the generic upstream-error bucket
    leaves a caller unable to tell "deleted" from "proxy is broken"."""

    async def test_proxy_404_raises_not_found(self):
        client = CalendarProxyClient(proxy_url="http://proxy", api_key="k")
        response = AsyncMock(spec=httpx.Response)
        response.status_code = 404
        response.json.return_value = {"detail": "Not Found"}
        with pytest.raises(ProxyNotFoundError):
            client._handle_response(response)

    def test_get_deleted_event_returns_404(self, client, mock_proxy_client):
        mock_proxy_client.get_event.side_effect = ProxyNotFoundError("Event not found")
        response = client.get("/calendars/primary/events/gone_123")
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert data["event"] is None


class TestBulkActionsOutcomeHonesty:
    """/bulk-actions must not answer 200/success:true for work that failed or
    whose outcome is unknown (issue #4, item 4)."""

    def _delete_ops(self, *event_ids: str) -> dict:
        return {
            "operations": [
                {"operation": "delete", "event_id": e, "calendar_id": "primary"}
                for e in event_ids
            ]
        }

    def test_all_operations_failing_is_not_reported_as_success(
        self, client, mock_proxy_client
    ):
        mock_proxy_client.delete_event.side_effect = ProxyError("proxy exploded")
        response = client.post("/bulk-actions", json=self._delete_ops("e1", "e2"))
        assert response.status_code == 502
        data = response.json()
        assert data["success"] is False
        assert data["error_count"] == 2
        assert data["error"]  # F7: the envelope must say so, not just the items

    def test_partial_failure_is_not_reported_as_success(self, client, mock_proxy_client):
        mock_proxy_client.delete_event.side_effect = [
            {"success": True},
            ProxyError("proxy exploded"),
        ]
        mock_proxy_client.get_event.side_effect = [
            ProxyNotFoundError("Not Found"),
            {"id": "e2", "status": "confirmed"},
        ]
        response = client.post("/bulk-actions", json=self._delete_ops("e1", "e2"))
        data = response.json()
        assert data["success"] is False
        assert response.status_code == 502
        assert "1 of 2" in data["error"]
        assert "1 failed" in data["error"]

    def test_timed_out_operation_is_unknown_not_failed(self, client, mock_proxy_client):
        """A mutation that timed out may still be applied after later approval;
        recording it as a plain failure is the incident's error inverted."""
        mock_proxy_client.delete_event.side_effect = ProxyTimeoutError("no response")
        response = client.post("/bulk-actions", json=self._delete_ops("e1"))
        assert response.status_code == 504
        data = response.json()
        assert data["success"] is False
        assert data["unknown_count"] == 1
        assert data["error_count"] == 0
        assert data["results"][0]["outcome"] == "unknown"
        assert "unknown" in data["error"]  # F7

    def test_unknown_outcome_outranks_a_definite_failure(self, client, mock_proxy_client):
        """504 (verify before acting) must win over 502, because the unknown
        operation is the one that can still change the calendar."""
        mock_proxy_client.delete_event.side_effect = [
            ProxyError("proxy exploded"),
            ProxyTimeoutError("no response"),
        ]
        response = client.post("/bulk-actions", json=self._delete_ops("e1", "e2"))
        assert response.status_code == 504

    def test_rejected_operation_propagates_403(self, client, mock_proxy_client):
        mock_proxy_client.delete_event.side_effect = ProxyForbiddenError("rejected")
        response = client.post("/bulk-actions", json=self._delete_ops("e1"))
        assert response.status_code == 403
        assert response.json()["error"]  # F7

    def test_nonexistent_event_propagates_404(self, client, mock_proxy_client):
        """The README's 404 bulk contract (F10): a proxy 404 is "absent",
        not a 502 upstream failure."""
        mock_proxy_client.delete_event.side_effect = ProxyNotFoundError("Not Found")
        response = client.post("/bulk-actions", json=self._delete_ops("e1"))
        assert response.status_code == 404
        data = response.json()
        assert data["results"][0]["outcome"] == "failed"
        assert data["error"]

    def test_all_succeeding_stays_200_and_true(self, client, mock_proxy_client):
        mock_proxy_client.get_event.side_effect = ProxyNotFoundError("Not Found")
        response = client.post("/bulk-actions", json=self._delete_ops("e1", "e2"))
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["unknown_count"] == 0
        assert [r["outcome"] for r in data["results"]] == ["succeeded", "succeeded"]

    @pytest.mark.parametrize("operation", ["update", "patch"])
    def test_missing_update_payload_rejects_the_whole_batch(
        self, client, mock_proxy_client, operation
    ):
        """F3: a malformed operation anywhere in the batch is a 422 before any
        operation runs — never a 400 that claims "nothing happened" after
        earlier operations already did."""
        request_data = {
            "operations": [
                {"operation": "delete", "event_id": "e1", "calendar_id": "primary"},
                {"operation": operation, "event_id": "e2", "calendar_id": "primary"},
            ]
        }
        response = client.post("/bulk-actions", json=request_data)
        assert response.status_code == 422
        mock_proxy_client.delete_event.assert_not_called()

    @pytest.mark.parametrize("order", ["not_found_first", "rejected_first"])
    def test_status_code_is_ranked_by_severity_not_position(
        self, client, mock_proxy_client, order
    ):
        """F3: the envelope's status must not depend on operation order."""
        errors = [ProxyNotFoundError("Not Found"), ProxyForbiddenError("rejected")]
        if order == "rejected_first":
            errors.reverse()
        mock_proxy_client.delete_event.side_effect = errors
        response = client.post("/bulk-actions", json=self._delete_ops("e1", "e2"))
        assert response.status_code == 403

    def test_bulk_delete_is_verified_by_re_read(self, client, mock_proxy_client):
        """F4: bulk deletes get the same after-write check as single deletes."""
        # Fixture default: get_event returns a confirmed event.
        response = client.post("/bulk-actions", json=self._delete_ops("e1"))
        assert response.status_code == 502
        data = response.json()
        assert data["results"][0]["outcome"] == "failed"
        assert "still present" in data["results"][0]["error"]

    def test_stops_after_the_first_unknown_outcome(self, client, mock_proxy_client):
        """F8: once one gated mutation has timed out, the operator is not
        answering; each further operation would hold the connection another
        full CONFIRM_TIMEOUT and enqueue another approval. Stop, and say so."""
        mock_proxy_client.delete_event.side_effect = [
            {"success": True},
            ProxyTimeoutError("no response"),
            {"success": True},
        ]
        mock_proxy_client.get_event.side_effect = [
            ProxyNotFoundError("Not Found"),
            {"id": "e2", "status": "confirmed"},
        ]
        response = client.post("/bulk-actions", json=self._delete_ops("e1", "e2", "e3"))
        assert response.status_code == 504
        data = response.json()
        assert [r["outcome"] for r in data["results"]] == [
            "succeeded", "unknown", "not_attempted"
        ]
        assert data["results"][2]["success"] is False
        assert "not attempted" in data["results"][2]["error"].lower()
        assert data["success_count"] == 1
        assert data["unknown_count"] == 1
        assert data["error_count"] == 0
        assert data["not_attempted_count"] == 1
        assert mock_proxy_client.delete_event.call_count == 2
        assert "1 not attempted" in data["error"]


class TestBulkStatusCode:
    """Pure ranking of per-operation status codes into one envelope status."""

    @pytest.mark.parametrize(
        ("codes", "expected"),
        [
            ([200, 200], 200),
            ([404, 403], 403),
            ([403, 404], 403),
            ([502, 403], 403),
            ([404, 502], 502),
            ([403, 504], 504),
            ([200, 404], 404),
            ([500, 404], 500),
        ],
    )
    def test_precedence(self, codes, expected):
        from calendar_agent.calendar_server import bulk_status_code

        assert bulk_status_code(codes) == expected
# ============================================================================
# Unknown-Field Rejection Tests (issue #8)
#
# Every request model used to inherit Pydantic's default extra="ignore": an
# unrecognized key was silently dropped instead of rejected. That produced
# HTTP 200 / success:true responses that were confidently wrong -- e.g. a
# caller sending Google's own "timeMin"/"timeMax" got an unbounded search
# with no error at all. These tests pin extra="forbid" on every request
# model: an unknown or misspelled field must be a 422, not a silent no-op.
# ============================================================================


def assert_rejected_for_unknown_key(response, bogus_key, mock_proxy_client):
    """The response is a 422 whose ONLY error is extra_forbidden on `bogus_key`.

    Asserting the cause (not just the status) proves the base payload itself
    was valid and the rejection is the unknown key, nothing else -- and that
    nothing reached the proxy.
    """
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert [(err["type"], err["loc"][-1]) for err in detail] == [
        ("extra_forbidden", bogus_key)
    ], detail
    assert mock_proxy_client.method_calls == [], "a rejected request reached the proxy"


# One row per request-body route: (method, path, minimal valid body).
route_cases = pytest.mark.parametrize(
    "method,path,base_payload",
    [
        pytest.param(method, path, payload, id=f"{method.upper()} {path}")
        for method, path, payload in [
            ("post", "/calendars/primary/events", CREATE_EVENT_BODY),
            ("put", "/calendars/primary/events/event_123", UPDATE_EVENT_BODY),
            ("patch", "/calendars/primary/events/event_123", PATCH_EVENT_BODY),
            ("post", "/calendars/primary/events/event_123/respond", RESPOND_BODY),
            ("post", "/summarize", SUMMARIZE_BODY),
            ("post", "/ask-about", ASK_ABOUT_BODY),
            ("post", "/batch-summarize", BATCH_SUMMARIZE_BODY),
            ("post", "/find-free-time", FIND_FREE_TIME_BODY),
            ("post", "/analyze-schedule", ANALYZE_SCHEDULE_BODY),
            ("post", "/prepare-briefing", PREPARE_BRIEFING_BODY),
            ("post", "/search", SEARCH_BODY),
            ("post", "/bulk-actions", BULK_DELETE_BODY),
        ]
    ],
)


# Unknown keys below the top level, plus the issue's own three examples:
# (method, path, payload containing the bogus key, the bogus key).
nested_cases = pytest.mark.parametrize(
    "method,path,payload,bogus_key",
    [
        pytest.param(
            "post", "/search",
            {"calendar_id": "primary", "timeMin": "2026-01-01T00:00:00Z"},
            "timeMin", id="issue example: timeMin on /search",
        ),
        pytest.param(
            "post", "/calendars/primary/events",
            {"title": "Standup"},
            "title", id="issue example: title on create",
        ),
        pytest.param(
            "post", "/find-free-time",
            {**FIND_FREE_TIME_BODY, "preferMorning": True},
            "preferMorning", id="issue example: preferMorning on /find-free-time",
        ),
        pytest.param(
            "post", "/search",
            {"calendar_id": "primary", "filters": {"query": "meeting", "bogus": "x"}},
            "bogus", id="inside filters",
        ),
        pytest.param(
            "post", "/bulk-actions",
            {"operations": [{**BULK_DELETE_BODY["operations"][0], "bogus": "x"}]},
            "bogus", id="inside a bulk operation",
        ),
        pytest.param(
            "post", "/bulk-actions",
            {"operations": [{"operation": "update", "event_id": "e1", "calendar_id": "primary",
                             "updates": {"summary": "s", "bogus": "x"}}]},
            "bogus", id="inside bulk update `updates`",
        ),
        pytest.param(
            "post", "/bulk-actions",
            {"operations": [{"operation": "patch", "event_id": "e1", "calendar_id": "primary",
                             "updates": {"summary": "s", "bogus": "x"}}]},
            "bogus", id="inside bulk patch `updates`",
        ),
        pytest.param(
            "post", "/calendars/primary/events",
            {"summary": "s", "start": {"dateTime": "2024-01-15T09:00:00Z", "bogus": "x"}},
            "bogus", id="inside start",
        ),
        pytest.param(
            "post", "/calendars/primary/events",
            {"summary": "s", "attendees": [{"email": "alice@example.com", "bogus": "x"}]},
            "bogus", id="inside an attendee",
        ),
        pytest.param(
            "post", "/calendars/primary/events",
            {"summary": "s", "reminders": {"useDefault": False, "bogus": "x"}},
            "bogus", id="inside reminders",
        ),
        pytest.param(
            "post", "/calendars/primary/events",
            {"summary": "s", "reminders": {"useDefault": False,
                                           "overrides": [{"method": "popup", "minutes": 5, "bogus": "x"}]}},
            "bogus", id="inside a reminder override",
        ),
    ],
)


class TestUnknownFieldsRejected:
    """An unrecognized field on any request body is a 422, not a silent drop."""

    @route_cases
    def test_unknown_top_level_field_rejected(
        self, client, mock_proxy_client, method, path, base_payload
    ):
        payload = {**base_payload, "bogus_field_xyz": "should not be accepted"}
        response = getattr(client, method)(path, json=payload)
        assert_rejected_for_unknown_key(response, "bogus_field_xyz", mock_proxy_client)

    @nested_cases
    def test_unknown_nested_field_rejected(
        self, client, mock_proxy_client, method, path, payload, bogus_key
    ):
        response = getattr(client, method)(path, json=payload)
        assert_rejected_for_unknown_key(response, bogus_key, mock_proxy_client)


# ---------------------------------------------------------------------------
# Structural guard: the tables above are hand-maintained. This walks the
# app's own routes so the next request model -- or a field typed as a plain
# dict / Mapping / Any / object, through which unknown keys would pass
# unvalidated -- cannot silently revert to extra="ignore" without a test
# failing. Body models declared on `Depends(...)` functions are walked too;
# `TestRouteWalkGuards` pins each of these against a throwaway model.
# ---------------------------------------------------------------------------


# Annotations through which an unknown key would pass unvalidated. The walker
# reports each as the normalised type here (any mapping -> dict).
OPEN_TYPES = frozenset({dict, typing.Any, object})


def _iter_model_types(annotation):
    """Yield every BaseModel subclass reachable from a type annotation,
    unwrapping Optional/list/Union/Annotated, and yield an OPEN_TYPES member
    for any annotation that would accept unknown keys unvalidated: a mapping
    (bare or parametrised, reported as `dict`), `typing.Any`, or `object`."""
    from pydantic import BaseModel

    if annotation is typing.Any or annotation is object:
        yield annotation
        return
    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            yield annotation
            return
        if issubclass(annotation, Mapping):
            yield dict
            return
    origin = typing.get_origin(annotation)
    if isinstance(origin, type) and issubclass(origin, Mapping):
        yield dict
        return
    for arg in typing.get_args(annotation):
        yield from _iter_model_types(arg)


def _iter_body_params(dependant):
    """Body params of a route, including those of its `Depends` sub-dependencies
    at any depth -- the same set FastAPI resolves for the request body."""
    yield from dependant.body_params
    for sub in dependant.dependencies:
        yield from _iter_body_params(sub)


def iter_request_models(app=None):
    """Every request-body model the app accepts, including nested ones and
    those declared on `Depends(...)` functions.

    Returns {model: "<where it was reached from>"}; an OPEN_TYPES member
    appears as a key if any reachable field is typed that loosely. `app`
    defaults to the real server; the guard's own tests pass a throwaway one.
    """
    from pydantic import BaseModel

    if app is None:
        from calendar_agent.calendar_server import app

    found: dict[object, str] = {}
    pending = []
    for route in app.routes:
        dependant = getattr(route, "dependant", None)
        if dependant is None:
            continue
        for param in _iter_body_params(dependant):
            for model in _iter_model_types(param.type_):
                pending.append((model, f"{sorted(route.methods)[0]} {route.path} body"))
    while pending:
        model, origin = pending.pop()
        if model in found:
            continue
        found[model] = origin
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            continue  # an open type: nothing beneath it to walk
        for name, field in model.model_fields.items():
            for nested in _iter_model_types(field.annotation):
                pending.append((nested, f"{model.__name__}.{name}"))
    return found


class TestRequestModelsAreStrict:
    """Every request model, at every depth, forbids unknown fields (T1)."""

    def test_walk_finds_the_models(self):
        from calendar_agent.calendar_server import (
            EventAttendee,
            EventPatchRequest,
            EventReminder,
            EventUpdateRequest,
            SearchFilters,
        )

        found = iter_request_models()
        # Sanity: the walk reaches top-level, nested and union-member models.
        for model in (EventUpdateRequest, EventPatchRequest, EventAttendee,
                      EventReminder, SearchFilters):
            assert model in found, f"{model.__name__} not reached by the route walk"
        assert len(found) >= 15

    def test_every_request_model_forbids_extra(self, subtests):
        for model, origin in iter_request_models().items():
            if model in OPEN_TYPES:
                continue
            with subtests.test(model=model.__name__, reached_from=origin):
                assert model.model_config.get("extra") == "forbid", (
                    f"{model.__name__} (reached from {origin}) does not forbid "
                    "unknown fields -- an unrecognized key would be silently dropped"
                )

    def test_no_request_field_is_open(self):
        """No reachable field is typed so loosely that unknown keys inside it
        would be forwarded verbatim (see #8, bulk `updates`)."""
        found = iter_request_models()
        open_fields = {found[t] for t in OPEN_TYPES if t in found}
        assert not open_fields, (
            f"request field(s) reached from {sorted(open_fields)} are typed as a "
            "plain dict / Mapping / Any / object"
        )


class TestRouteWalkGuards:
    """The structural guard is itself guarded (fix round, item 5): a walker
    that cannot see a loosely typed field, or a model reached only through a
    `Depends(...)` sub-dependency, is a guard that passes vacuously. Each
    case below is a throwaway model the pre-fix walker missed."""

    @pytest.mark.parametrize(
        "annotation",
        [
            pytest.param(dict, id="bare-dict"),
            pytest.param(Mapping, id="bare-Mapping"),
            pytest.param(typing.Any, id="Any"),
            pytest.param(object, id="object"),
            pytest.param(dict[str, typing.Any], id="dict[str, Any]"),
            pytest.param(dict | None, id="dict | None"),
            pytest.param(list[typing.Any], id="list[Any]"),
            pytest.param(typing.Mapping[str, int], id="typing.Mapping[str, int]"),
        ],
    )
    def test_open_annotation_is_flagged(self, annotation):
        assert set(_iter_model_types(annotation)) & OPEN_TYPES, (
            f"{annotation!r} lets unknown keys through but the walker does not flag it"
        )

    @pytest.mark.parametrize(
        "annotation", [str, int, str | None, list[str], typing.Literal["a"]]
    )
    def test_closed_annotation_is_not_flagged(self, annotation):
        assert not set(_iter_model_types(annotation)) & OPEN_TYPES

    def test_walk_flags_open_fields_on_a_body_model(self):
        from fastapi import FastAPI
        from pydantic import BaseModel, ConfigDict

        class Blob(BaseModel):
            model_config = ConfigDict(extra="forbid")
            payload: typing.Any
            tags: object
            meta: Mapping

        mini = FastAPI()

        @mini.post("/blob")
        def route(body: Blob):  # pragma: no cover - never called
            return body

        found = iter_request_models(mini)
        assert Blob in found
        assert {t for t in OPEN_TYPES if t in found} == OPEN_TYPES
        assert found[typing.Any] == "Blob.payload"
        assert found[object] == "Blob.tags"
        assert found[dict] == "Blob.meta"

    def test_walk_reaches_models_behind_depends(self):
        """A body model declared on a `Depends` function -- one or two levels
        down -- is request surface just like a route parameter."""
        from fastapi import Depends, FastAPI
        from pydantic import BaseModel, ConfigDict

        class ViaDepends(BaseModel):  # deliberately NOT extra="forbid"
            x: int

        class ViaNestedDepends(BaseModel):
            y: int

        class Direct(BaseModel):
            model_config = ConfigDict(extra="forbid")
            z: int

        def inner(body: ViaNestedDepends):  # pragma: no cover
            return body

        def outer(
            body: ViaDepends, nested: typing.Annotated[object, Depends(inner)]
        ):  # pragma: no cover
            return body

        mini = FastAPI()

        @mini.post("/deep")
        def route(
            direct: Direct, dep: typing.Annotated[object, Depends(outer)]
        ):  # pragma: no cover
            return direct

        found = iter_request_models(mini)
        assert Direct in found
        assert ViaDepends in found, "model behind one Depends not reached"
        assert ViaNestedDepends in found, "model behind two Depends not reached"

    def test_walk_matches_fastapi_flat_dependant_on_the_real_app(self):
        """Cross-check: per route, the top-level body models the walk starts
        from equal what FastAPI itself resolves for the request body."""
        from fastapi.dependencies.utils import get_flat_dependant

        from calendar_agent.calendar_server import app

        for route in app.routes:
            dependant = getattr(route, "dependant", None)
            if dependant is None:
                continue
            via_fastapi = {
                m for p in get_flat_dependant(dependant).body_params
                for m in _iter_model_types(p.type_)
            }
            via_walk = {m for p in _iter_body_params(dependant) for m in _iter_model_types(p.type_)}
            assert via_walk == via_fastapi, route.path


# ============================================================================
# Validation-Error Envelope Tests (issue #8, second review round)
#
# With extra="forbid", a 422 is now the main way a misnamed field fails. The
# briefing skills pipe responses through
#   jq 'if .success then .briefing else "Error: " + .error end'
# so a 422 that carries only FastAPI's default {"detail": [...]} renders as a
# blank "Error: ". Every 422 must carry the same success/error envelope as
# every other failure; `detail` is kept for clients that read it.
# ============================================================================


class TestValidationErrorEnvelope:
    """A 422 carries {"success": false, "error": "..."} plus FastAPI's `detail`."""

    def test_unknown_field_422_carries_envelope(self, client):
        response = client.post("/calendars/primary/events", json={"title": "Standup"})
        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert "title" in data["error"]
        assert "Extra inputs are not permitted" in data["error"]
        assert data["detail"][0]["type"] == "extra_forbidden"
        assert data["detail"][0]["loc"] == ["body", "title"]

    def test_missing_field_422_carries_envelope(self, client):
        response = client.post(
            "/ask-about", json={"calendar_id": "primary", "event_id": "event_123"}
        )
        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert "question" in data["error"]
        assert "Field required" in data["error"]

    def test_multiple_errors_are_all_named(self, client):
        response = client.post(
            "/find-free-time",
            json={
                "calendar_id": "primary",
                "time_min": "2024-01-15T09:00:00Z",
                "time_max": "2024-01-15T17:00:00Z",
                "duration_minutes": 0,
                "preferMorning": True,
            },
        )
        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert "duration_minutes" in data["error"]
        assert "preferMorning" in data["error"]
        assert len(data["detail"]) == 2


class TestValidation422CarriesNoOutcome:
    """A request-validation 422 on a mutation route carries the validation
    envelope, not an `outcome` (fix round, item 9): nothing was attempted,
    so there is no outcome to report. Pins the exception the docs state to
    "mutation envelopes carry an `outcome`"."""

    @pytest.mark.parametrize(
        "method,path,kwargs",
        [
            pytest.param(
                "delete", "/calendars/primary/events/event_123?sendUpdates=all", {},
                id="delete-undeclared-query-key",
            ),
            pytest.param(
                "post", "/bulk-actions",
                {"json": {"operations": [{"operation": "bogus", "event_id": "e", "calendar_id": "p"}]}},
                id="bulk-bad-operation",
            ),
            pytest.param(
                "put", "/calendars/primary/events/event_123", {"json": {}},
                id="put-empty-body",
            ),
        ],
    )
    def test_422_on_a_mutation_route_is_the_validation_envelope(
        self, client, mock_proxy_client, method, path, kwargs
    ):
        response = getattr(client, method)(path, **kwargs)
        assert response.status_code == 422, response.text
        body = response.json()
        assert body["success"] is False
        assert body["error"]
        assert isinstance(body["detail"], list) and body["detail"]
        assert "outcome" not in body
        assert "results" not in body
        for name in ("delete_event", "update_event", "patch_event", "get_event"):
            getattr(mock_proxy_client, name).assert_not_called()


# ============================================================================
# Writable-field coverage and fetch-modify-write round trips (issue #8,
# second review round)
#
# extra="forbid" only holds up if the models declare every field a caller
# legitimately sends. Two gaps were found: EventPatchRequest lacked fields the
# create model accepts (transparency, visibility, guestsCan*) plus `status`
# (Google's documented cancel path), and a fetch -> modify -> PUT of a real
# Google event 422'd on Google's own server-populated fields (id, etag,
# htmlLink, ...) and on attendee fields Google returns (id, resource, ...).
# ============================================================================


# Every read-only, server-populated key Google puts on an event it returns.
# A caller doing fetch -> modify -> write sends these back verbatim; the
# server strips exactly this set (and nothing else) before forwarding.
# `outOfOfficeProperties` / `workingLocationProperties` only appear on events
# of those types; they are folded into this one fixture so the strip set is
# exercised in full, at the cost of the fixture being a composite.
GOOGLE_READ_ONLY_EVENT_FIELDS = {
    "kind": "calendar#event",
    "etag": '"3181161784712000"',
    "id": "meeting_001",
    "htmlLink": "https://www.google.com/calendar/event?eid=bWVldGluZ18wMDE",
    "created": "2024-01-08T10:00:00.000Z",
    "updated": "2024-01-15T10:00:00.712Z",
    "creator": {"email": "owner@example.com", "self": True},
    "organizer": {"email": "owner@example.com", "self": True},
    "iCalUID": "recurring_001@google.com",
    "sequence": 3,
    "eventType": "default",
    "hangoutLink": "https://meet.google.com/abc-defg-hij",
    "recurringEventId": "recurring_001",
    "originalStartTime": {"dateTime": "2024-01-15T10:00:00-05:00", "timeZone": "America/New_York"},
    "privateCopy": False,
    "locked": False,
    "attendeesOmitted": False,
    "endTimeUnspecified": False,
    "outOfOfficeProperties": {
        "autoDeclineMode": "declineAllConflictingInvitations",
        "declineMessage": "Out until Monday",
    },
    "workingLocationProperties": {"type": "homeOffice", "homeOffice": {}},
}

# Keys Google returns that ARE writable but this server does not declare:
# accepting them would be new write surface, and `conferenceData` is ignored
# by Google on a write unless `conferenceDataVersion` is sent (this server
# never sends it). A caller round-tripping a fetched event removes these
# before writing; Meet and attachment events are the common case. README
# "Round-tripping a fetched event" names the same five.
GOOGLE_WRITABLE_UNDECLARED_FIELDS = {
    "conferenceData": {
        "entryPoints": [
            {
                "entryPointType": "video",
                "uri": "https://meet.google.com/abc-defg-hij",
                "label": "meet.google.com/abc-defg-hij",
            }
        ],
        "conferenceSolution": {
            "key": {"type": "hangoutsMeet"},
            "name": "Google Meet",
            "iconUri": "https://fonts.gstatic.com/s/i/productlogos/meet_2020q4/v6/web-512dp/logo_meet_2020q4_color_2x_web_512dp.png",
        },
        "conferenceId": "abc-defg-hij",
    },
    "attachments": [
        {
            "fileUrl": "https://drive.google.com/open?id=1AbCdEfG",
            "title": "Agenda",
            "mimeType": "application/vnd.google-apps.document",
            "iconLink": "https://drive-thirdparty.googleusercontent.com/16/type/application/vnd.google-apps.document",
            "fileId": "1AbCdEfG",
        }
    ],
    "extendedProperties": {"private": {"crmId": "42"}, "shared": {}},
    "source": {"url": "https://example.com/tickets/42", "title": "Ticket 42"},
    "anyoneCanAddSelf": False,
}

# A Google-shaped attendee entry as Google returns it (all writable/tolerated).
GOOGLE_ATTENDEE = {
    "id": "attendee-id-1",
    "email": "alice@example.com",
    "displayName": "Alice Smith",
    "responseStatus": "accepted",
    "optional": False,
    "organizer": False,
    "self": False,
    "resource": False,
    "comment": "Joining remotely",
    "additionalGuests": 1,
}


def google_event_as_fetched() -> dict:
    """A Meet event as `GET .../events/{id}` returns it: read-only keys,
    writable-but-undeclared keys, and every declared field."""
    return {
        **GOOGLE_READ_ONLY_EVENT_FIELDS,
        **GOOGLE_WRITABLE_UNDECLARED_FIELDS,
        "status": "confirmed",
        "summary": "Team Standup",
        "description": "Daily standup meeting",
        "location": "Room 4B",
        "start": {"dateTime": "2024-01-15T10:00:00-05:00", "timeZone": "America/New_York"},
        "end": {"dateTime": "2024-01-15T10:30:00-05:00", "timeZone": "America/New_York"},
        "attendees": [GOOGLE_ATTENDEE, {"email": "room@example.com", "resource": True}],
        "reminders": {"useDefault": False, "overrides": [{"method": "popup", "minutes": 10}]},
        "transparency": "opaque",
        "visibility": "default",
        "guestsCanInviteOthers": True,
        "guestsCanModify": False,
        "guestsCanSeeOtherGuests": True,
    }


def round_trip_body() -> dict:
    """The fetched event after the removal step the README documents."""
    return {
        k: v for k, v in google_event_as_fetched().items()
        if k not in GOOGLE_WRITABLE_UNDECLARED_FIELDS
    }


class TestPatchAcceptsEveryCreateField:
    """A field the create model accepts must be patchable too (C4)."""

    @pytest.mark.parametrize(
        "field,value",
        [
            ("transparency", "transparent"),
            ("visibility", "private"),
            ("guestsCanInviteOthers", False),
            ("guestsCanModify", True),
            ("guestsCanSeeOtherGuests", False),
            ("status", "tentative"),
        ],
    )
    def test_patch_forwards_field(self, client, mock_proxy_client, field, value):
        response = client.patch(
            "/calendars/primary/events/event_123", json={field: value}
        )
        assert response.status_code == 200, response.text
        assert mock_proxy_client.patch_event.call_args.kwargs["event_data"] == {field: value}

    @pytest.mark.parametrize("method", ["post", "put"])
    def test_status_is_writable_on_create_and_update(self, client, mock_proxy_client, method):
        path = "/calendars/primary/events" + ("" if method == "post" else "/event_123")
        response = getattr(client, method)(
            path, json={"summary": "Standup", "status": "tentative"}
        )
        assert response.status_code == 200, response.text
        forwarder = mock_proxy_client.create_event if method == "post" else mock_proxy_client.update_event
        assert forwarder.call_args.kwargs["event_data"]["status"] == "tentative"


class TestCancelledStatusIsRejected:
    """`status: "cancelled"` is refused on every write (fix round, item 1).

    A cancel through PUT/PATCH reaches Google with none of the DELETE path's
    re-read verification and no `outcome`. `status` stays declared (a fetched
    event carries it) but accepts only `confirmed` / `tentative`; `cancelled`
    is a 422 that points the caller at DELETE. The rule lives on the shared
    `EventFields` model, so POST and bulk `updates` get it too.
    """

    CANCEL = {"summary": "Standup", "status": "cancelled"}

    @pytest.mark.parametrize(
        "method,path,forwarder",
        [
            ("post", "/calendars/primary/events", "create_event"),
            ("put", "/calendars/primary/events/event_123", "update_event"),
            ("patch", "/calendars/primary/events/event_123", "patch_event"),
        ],
    )
    def test_single_route_rejects_cancelled_and_points_at_delete(
        self, client, mock_proxy_client, method, path, forwarder
    ):
        response = getattr(client, method)(path, json=self.CANCEL)
        assert response.status_code == 422, response.text
        data = response.json()
        assert data["success"] is False
        assert "DELETE" in data["error"]
        assert [err["loc"] for err in data["detail"]] == [["body", "status"]]
        getattr(mock_proxy_client, forwarder).assert_not_called()

    @pytest.mark.parametrize(
        "operation,forwarder", [("update", "update_event"), ("patch", "patch_event")]
    )
    def test_bulk_rejects_cancelled_before_any_operation_runs(
        self, client, mock_proxy_client, operation, forwarder
    ):
        response = client.post("/bulk-actions", json={"operations": [
            {"operation": "delete", "event_id": "event_0", "calendar_id": "primary"},
            {"operation": operation, "event_id": "event_1", "calendar_id": "primary",
             "updates": self.CANCEL},
        ]})
        assert response.status_code == 422, response.text
        assert "DELETE" in response.json()["error"]
        getattr(mock_proxy_client, forwarder).assert_not_called()
        mock_proxy_client.delete_event.assert_not_called()

    @pytest.mark.parametrize("value", ["confirmed", "tentative"])
    def test_confirmed_and_tentative_are_still_forwarded(
        self, client, mock_proxy_client, value
    ):
        response = client.patch(
            "/calendars/primary/events/event_123", json={"status": value}
        )
        assert response.status_code == 200, response.text
        assert mock_proxy_client.patch_event.call_args.kwargs["event_data"] == {"status": value}

        response = client.post("/bulk-actions", json={"operations": [
            {"operation": "update", "event_id": "event_1", "calendar_id": "primary",
             "updates": {"summary": "Standup", "status": value}},
        ]})
        assert response.status_code == 200, response.text
        assert mock_proxy_client.update_event.call_args.kwargs["event_data"]["status"] == value

    def test_any_other_status_value_is_rejected(self, client, mock_proxy_client):
        response = client.patch(
            "/calendars/primary/events/event_123", json={"status": "maybe"}
        )
        assert response.status_code == 422, response.text
        assert [err["loc"] for err in response.json()["detail"]] == [["body", "status"]]
        mock_proxy_client.patch_event.assert_not_called()


class TestEmptySingleRouteWriteIsRejected:
    """A PUT/PATCH body that forwards nothing is a 422, exactly like an empty
    bulk `updates` (fix round, item 2). Before this the single routes sent
    Google an empty body and reported `success: true`. Emptiness is judged on
    the forwarded payload -- `{}`, all-null values and read-only-only keys all
    dump to nothing -- and none may reach the proxy."""

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param({}, id="empty-object"),
            pytest.param({"summary": None, "location": None}, id="all-null-values"),
            pytest.param({"id": "event_1", "etag": '"1"'}, id="read-only-keys-only"),
        ],
    )
    @pytest.mark.parametrize(
        "method,operation,forwarder",
        [("put", "update", "update_event"), ("patch", "patch", "patch_event")],
    )
    def test_nothing_to_forward_is_a_422(
        self, client, mock_proxy_client, method, operation, forwarder, body
    ):
        response = getattr(client, method)("/calendars/primary/events/event_1", json=body)
        assert response.status_code == 422, response.text
        data = response.json()
        assert data["success"] is False
        assert f"'{operation}' requires a non-empty payload" in data["error"]
        assert [err["loc"] for err in data["detail"]] == [["body"]]
        getattr(mock_proxy_client, forwarder).assert_not_called()

    def test_single_and_bulk_use_the_same_message(self, client, mock_proxy_client):
        single = client.put("/calendars/primary/events/event_1", json={}).json()["error"]
        bulk = client.post("/bulk-actions", json={"operations": [
            {"operation": "update", "event_id": "event_1", "calendar_id": "primary",
             "updates": {}},
        ]}).json()["error"]
        # Same text after the `<loc>: ` prefix the envelope adds.
        assert single.split(": ", 1)[1] == bulk.split(": ", 1)[1]


class TestGoogleEventRoundTrip:
    """fetch -> modify -> write of a real Google event succeeds (C3), once the
    caller removes the writable-but-undeclared keys (fix round, item 4)."""

    @pytest.mark.parametrize("method", ["put", "patch"])
    def test_round_trip_succeeds_and_strips_only_read_only_keys(
        self, client, mock_proxy_client, method
    ):
        body = round_trip_body()
        body["summary"] = "Team Standup (moved)"
        response = getattr(client, method)("/calendars/primary/events/meeting_001", json=body)
        assert response.status_code == 200, response.text

        forwarder = mock_proxy_client.update_event if method == "put" else mock_proxy_client.patch_event
        forwarded = forwarder.call_args.kwargs["event_data"]
        for key in GOOGLE_READ_ONLY_EVENT_FIELDS:
            assert key not in forwarded, f"read-only key {key!r} was forwarded"
        assert forwarded["summary"] == "Team Standup (moved)"
        assert forwarded["status"] == "confirmed"
        assert forwarded["attendees"][0] == GOOGLE_ATTENDEE
        assert forwarded["attendees"][1] == {"email": "room@example.com", "resource": True}
        assert forwarded["reminders"]["overrides"][0] == {"method": "popup", "minutes": 10}

    def test_round_trip_through_create_succeeds(self, client, mock_proxy_client):
        """Duplicating an event by POSTing a fetched one works the same way."""
        response = client.post("/calendars/primary/events", json=round_trip_body())
        assert response.status_code == 200, response.text
        forwarded = mock_proxy_client.create_event.call_args.kwargs["event_data"]
        assert not set(forwarded) & set(GOOGLE_READ_ONLY_EVENT_FIELDS)
        assert forwarded["summary"] == "Team Standup"

    @pytest.mark.parametrize("method", ["put", "patch"])
    def test_fetched_event_sent_verbatim_names_exactly_the_undeclared_keys(
        self, client, mock_proxy_client, method
    ):
        """The removal step is real and bounded: sending a fetched Meet event
        as-is is a 422 on precisely the five writable-but-undeclared keys --
        none of them is silently stripped, and nothing else is rejected."""
        response = getattr(client, method)(
            "/calendars/primary/events/meeting_001", json=google_event_as_fetched()
        )
        assert response.status_code == 422, response.text
        rejected = sorted(err["loc"][-1] for err in response.json()["detail"])
        assert rejected == sorted(GOOGLE_WRITABLE_UNDECLARED_FIELDS)
        assert all(err["type"] == "extra_forbidden" for err in response.json()["detail"])
        mock_proxy_client.update_event.assert_not_called()
        mock_proxy_client.patch_event.assert_not_called()

    def test_unknown_key_outside_read_only_set_still_rejected(self, client, mock_proxy_client):
        body = {**round_trip_body(), "titel": "typo"}
        response = client.put("/calendars/primary/events/meeting_001", json=body)
        assert response.status_code == 422
        data = response.json()
        assert data["detail"] == [
            {
                "type": "extra_forbidden",
                "loc": ["body", "titel"],
                "msg": "Extra inputs are not permitted",
                "input": "typo",
            }
        ]
        mock_proxy_client.update_event.assert_not_called()

    @pytest.mark.parametrize("key", sorted(GOOGLE_READ_ONLY_EVENT_FIELDS))
    def test_read_only_keys_are_not_silently_written_by_a_bare_request(
        self, client, mock_proxy_client, key
    ):
        """Stripping is exactly the named set: a read-only key sent alone is
        dropped (200, nothing forwarded for it), never forwarded."""
        response = client.patch(
            "/calendars/primary/events/meeting_001",
            json={key: GOOGLE_READ_ONLY_EVENT_FIELDS[key], "location": "Room B"},
        )
        assert response.status_code == 200, response.text
        assert mock_proxy_client.patch_event.call_args.kwargs["event_data"] == {
            "location": "Room B"
        }


class TestBulkUpdatesAreTyped:
    """A bulk operation's `updates` goes through the same strict event model
    as the single-event routes (C1): an unknown key inside it is a 422, a
    delete does not carry one, and the payload is dumped the same way."""

    def _bulk(self, client, op):
        return client.post("/bulk-actions", json={"operations": [op]})

    def test_delete_with_updates_is_rejected(self, client, mock_proxy_client):
        """A mis-set operation must not DELETE while silently ignoring the payload."""
        response = self._bulk(client, {
            "operation": "delete",
            "event_id": "event_1",
            "calendar_id": "primary",
            "updates": {"summary": "meant to patch this"},
        })
        assert response.status_code == 422, response.text
        assert ["body", "operations", 0, "delete", "updates"] in [
            err["loc"] for err in response.json()["detail"]
        ]
        mock_proxy_client.delete_event.assert_not_called()

    def test_updates_are_dumped_like_the_single_routes(self, client, mock_proxy_client):
        """exclude_none + by_alias + read-only strip, exactly as PUT/PATCH do."""
        response = self._bulk(client, {
            "operation": "patch",
            "event_id": "event_1",
            "calendar_id": "primary",
            "updates": {
                "id": "event_1",
                "etag": '"1"',
                "summary": "Renamed",
                "location": None,
                "attendees": [{"email": "alice@example.com", "self": True}],
            },
        })
        assert response.status_code == 200, response.text
        assert mock_proxy_client.patch_event.call_args.kwargs["event_data"] == {
            "summary": "Renamed",
            "attendees": [{"email": "alice@example.com", "self": True}],
        }

    @pytest.mark.parametrize(
        "updates",
        [
            pytest.param({}, id="empty-object"),
            pytest.param({"summary": None}, id="all-null-values"),
            pytest.param({"id": "event_1", "etag": '"1"'}, id="read-only-keys-only"),
        ],
    )
    def test_empty_updates_payload_rejects_the_whole_batch(
        self, client, mock_proxy_client, updates
    ):
        """An `updates` that forwards nothing is a 422 for the whole request
        (main's F3 rule), judged on the dumped payload rather than on the
        raw object: `{}`, all-null values and read-only-only keys all dump to
        nothing, and none may reach the proxy or be reported per item."""
        response = self._bulk(client, {
            "operation": "update",
            "event_id": "event_1",
            "calendar_id": "primary",
            "updates": updates,
        })
        assert response.status_code == 422, response.text
        assert "'update' requires a non-empty payload" in response.json()["error"]
        mock_proxy_client.update_event.assert_not_called()


class TestBulkOperationTagErrors:
    """A bad `operation` value gets a fixed message naming the allowed
    operations (fix round, item 3). Before, the discriminated union's own
    message leaked Python enum reprs (`<BulkOperationType.DELETE: 'delete'>`)
    into `error`, `detail[].msg` and `detail[].ctx.expected_tags`."""

    ENUM_REPR = re.compile(r"BulkOperationType|<[A-Za-z_.]+: '")
    FIXED = "'operation' must be one of 'update', 'delete', 'patch'"

    @pytest.mark.parametrize(
        "op",
        [
            pytest.param({"operation": "bogus"}, id="unknown-tag"),
            pytest.param({}, id="missing-tag"),
            pytest.param({"operation": None}, id="null-tag"),
            pytest.param({"operation": []}, id="list-tag"),
            pytest.param({"operation": {}}, id="object-tag"),
        ],
    )
    def test_bad_operation_tag_gets_a_fixed_message(self, client, mock_proxy_client, op):
        # An unhashable tag must not turn the 422 into a 500 via `in <set>`.
        response = client.post("/bulk-actions", json={"operations": [
            {"event_id": "event_1", "calendar_id": "primary", **op},
        ]})
        assert response.status_code == 422, response.text
        assert not self.ENUM_REPR.search(response.text), response.text
        data = response.json()
        assert data["success"] is False
        assert self.FIXED in data["error"]
        assert [err["loc"] for err in data["detail"]] == [["body", "operations", 0]]
        for name in ("delete_event", "update_event", "patch_event"):
            getattr(mock_proxy_client, name).assert_not_called()


# ============================================================================
# /search flat-shape fold (issue #8 "shape of the fix" items 2-4)
#
# Earlier callers sent /search with the filter keys at the top level
# ({"calendar_id": ..., "query": ..., "time_min": ...}) rather than nested
# under "filters"; the flat shape is kept for compatibility with them (the
# installed calendar skills nest their filters). With extra="forbid" alone
# that shape went from 200-with-wrong-answer to hard 422. A before-validator folds exactly the
# keys SearchFilters declares into "filters" (consuming them, so a true typo
# still 422s); the allowlist is derived from SearchFilters.model_fields, so
# the fold needs no edit when a filter field is added. The parametrized test
# below still needs a sample value for the new field, and `search_events`
# must forward it -- neither is derived.
# ============================================================================


NESTED_SEARCH = {
    "calendar_id": "primary",
    "filters": {
        "query": "project",
        "time_min": "2024-01-01T00:00:00Z",
        "time_max": "2024-03-31T23:59:59Z",
        "max_results": 20,
        "order_by": "startTime",
    },
}
FLAT_SEARCH = {"calendar_id": "primary", **NESTED_SEARCH["filters"]}


class TestSearchFlatShapeFold:
    """The flat /search shape kept for earlier callers works, and only that shape."""

    def test_flat_shape_gives_the_same_proxy_call_as_nested(self, client, mock_proxy_client):
        client.post("/search", json=NESTED_SEARCH)
        nested_call = mock_proxy_client.list_events.call_args.kwargs
        mock_proxy_client.list_events.reset_mock()

        response = client.post("/search", json=FLAT_SEARCH)
        assert response.status_code == 200, response.text
        assert mock_proxy_client.list_events.call_args.kwargs == nested_call
        assert nested_call["time_min"] == "2024-01-01T00:00:00Z"
        assert nested_call["max_results"] == 20

    def test_flat_shape_with_typo_is_rejected(self, client, mock_proxy_client):
        """The fold consumes only declared keys: Google-style timeMin is still a 422."""
        response = client.post(
            "/search",
            json={"calendar_id": "primary", "query": "project", "timeMin": "2024-01-01T00:00:00Z"},
        )
        assert response.status_code == 422, response.text
        assert response.json()["detail"] == [
            {
                "type": "extra_forbidden",
                "loc": ["body", "timeMin"],
                "msg": "Extra inputs are not permitted",
                "input": "2024-01-01T00:00:00Z",
            }
        ]
        mock_proxy_client.list_events.assert_not_called()

    @pytest.mark.parametrize("field", list(SearchFilters.model_fields))
    def test_every_search_filter_field_is_accepted_flat(self, client, mock_proxy_client, field):
        """Parametrized over SearchFilters.model_fields, so a new filter field
        is picked up by name -- but it needs a sample value in the mapping
        below (and a proxy-key entry if the proxy's name differs), and
        `search_events` must forward it. Until then this test fails for it,
        which is the point: the fold's allowlist is derived, the rest is not."""
        value = {"query": "x", "time_min": "2024-01-01T00:00:00Z",
                 "time_max": "2024-01-02T00:00:00Z", "order_by": "updated"}.get(field)
        if value is None:
            annotation = SearchFilters.model_fields[field].annotation
            value = 7 if annotation is int else True
        response = client.post("/search", json={"calendar_id": "primary", field: value})
        assert response.status_code == 200, response.text
        forwarded = mock_proxy_client.list_events.call_args.kwargs
        # the flat key reached the proxy call under its SearchFilters name
        proxy_key = {"query": "q"}.get(field, field)
        assert forwarded[proxy_key] == value

    def test_nested_wins_per_field_on_conflict(self, client, mock_proxy_client):
        response = client.post(
            "/search",
            json={
                "calendar_id": "primary",
                "time_min": "FLAT-MIN",
                "time_max": "FLAT-MAX",
                "filters": {"time_min": "NESTED-MIN"},
            },
        )
        assert response.status_code == 200, response.text
        forwarded = mock_proxy_client.list_events.call_args.kwargs
        assert forwarded["time_min"] == "NESTED-MIN"
        assert forwarded["time_max"] == "FLAT-MAX"

    def test_explicit_null_flat_keys_mean_not_supplied(self, client, mock_proxy_client):
        """A client that serializes every optional field as null gets defaults, not a 422."""
        response = client.post(
            "/search",
            json={"calendar_id": "primary", "query": None, "time_min": None,
                  "max_results": None, "order_by": None, "show_deleted": None},
        )
        assert response.status_code == 200, response.text
        forwarded = mock_proxy_client.list_events.call_args.kwargs
        assert forwarded["max_results"] == 100
        assert forwarded["show_deleted"] is False

    @pytest.mark.parametrize("bad_filters", ["oops", [], 0, ""])
    def test_malformed_filters_with_flat_keys_is_a_clean_422(
        self, client, mock_proxy_client, bad_filters
    ):
        """PR #1's regressions: a non-mapping `filters` alongside a flat key
        was a 500 (TypeError), and `or {}` turned []/""/0 into defaults."""
        response = client.post(
            "/search",
            json={"calendar_id": "primary", "time_min": "2024-01-01T00:00:00Z",
                  "filters": bad_filters},
        )
        assert response.status_code == 422, response.text
        assert response.json()["success"] is False
        assert ["body", "filters"] in [err["loc"][:2] for err in response.json()["detail"]]
        mock_proxy_client.list_events.assert_not_called()


# ============================================================================
# Unknown query-string keys (C2)
#
# Body fields are strict, but FastAPI ignores undeclared query parameters:
# GET /calendars/x/events?timeMin=..&timeMax=.. ran an unbounded list, and
# ?sendUpdates=all on a write meant nobody was emailed while the caller was
# told success. A shared dependency now 422s any query key the route does
# not declare.
# ============================================================================


class TestUnknownQueryParamsRejected:
    """An undeclared query-string key is a 422, not silently ignored."""

    def test_google_style_time_bounds_on_list_events(self, client, mock_proxy_client):
        response = client.get(
            "/calendars/primary/events",
            params={"timeMin": "2026-01-01T00:00:00Z", "timeMax": "2026-02-01T00:00:00Z",
                    "maxResults": 5},
        )
        assert response.status_code == 422, response.text
        data = response.json()
        assert data["success"] is False
        assert [err["loc"] for err in data["detail"]] == [
            ["query", "timeMin"], ["query", "timeMax"], ["query", "maxResults"]
        ]
        assert all(err["type"] == "extra_forbidden" for err in data["detail"])
        assert "timeMin" in data["error"]
        mock_proxy_client.list_events.assert_not_called()

    @pytest.mark.parametrize(
        "method,path,body",
        [
            pytest.param("post", "/calendars/primary/events", {"summary": "s"}, id="POST create"),
            pytest.param("put", "/calendars/primary/events/e1", {"summary": "s"}, id="PUT update"),
            pytest.param("patch", "/calendars/primary/events/e1", {"summary": "s"}, id="PATCH patch"),
            pytest.param("delete", "/calendars/primary/events/e1", None, id="DELETE delete"),
        ],
    )
    def test_camelcase_send_updates_on_writes(self, client, mock_proxy_client, method, path, body):
        kwargs = {"json": body} if body is not None else {}
        response = getattr(client, method)(path, params={"sendUpdates": "all"}, **kwargs)
        assert response.status_code == 422, response.text
        assert response.json()["detail"][0]["loc"] == ["query", "sendUpdates"]
        for forwarder in ("create_event", "update_event", "patch_event", "delete_event"):
            getattr(mock_proxy_client, forwarder).assert_not_called()

    def test_declared_query_params_still_accepted(self, client, mock_proxy_client):
        response = client.get(
            "/calendars/primary/events",
            params={"time_min": "2026-01-01T00:00:00Z", "max_results": 5, "q": "x",
                    "single_events": "false", "order_by": "updated", "page_token": "t",
                    "time_max": "2026-02-01T00:00:00Z"},
        )
        assert response.status_code == 200, response.text
        response = client.post(
            "/calendars/primary/events", params={"send_updates": "all"}, json={"summary": "s"}
        )
        assert response.status_code == 200, response.text
        assert mock_proxy_client.create_event.call_args.kwargs["send_updates"] == "all"

    def test_route_with_no_query_params_rejects_any_key(self, client, mock_proxy_client):
        response = client.post(
            "/search", params={"max_results": 5}, json={"calendar_id": "primary"}
        )
        assert response.status_code == 422, response.text
        assert response.json()["detail"][0]["loc"] == ["query", "max_results"]
        mock_proxy_client.list_events.assert_not_called()

    def test_query_key_error_is_reported_alone_and_masks_body_errors(
        self, client, mock_proxy_client
    ):
        """The query guard is an app-level dependency and raises before the
        body is validated, so a request with both an undeclared query key and
        a bad body reports only the query key (README "What a 422 looks
        like" says so; fix round, item 6). Fix the query string, resend, and
        the body errors appear."""
        response = client.patch(
            "/calendars/primary/events/event_123?sendUpdates=all",
            json={"titel": "typo", "status": "cancelled"},
        )
        assert response.status_code == 422, response.text
        detail = response.json()["detail"]
        assert [err["loc"] for err in detail] == [["query", "sendUpdates"]]
        assert response.json()["error"] == "query.sendUpdates: Extra inputs are not permitted"
        mock_proxy_client.patch_event.assert_not_called()

    def test_every_route_is_guarded(self, client, subtests):
        """Walk the app's own routes: none of them may accept a bogus query key."""
        from tests.test_readme_documentation import get_all_endpoints

        for method, path in get_all_endpoints():
            with subtests.test(endpoint=f"{method} {path}"):
                url = path.replace("{calendar_id}", "primary").replace("{event_id}", "e1")
                kwargs = {"json": {}} if method in ("POST", "PUT", "PATCH") else {}
                response = getattr(client, method.lower())(
                    url, params={"bogus_query_key": "x"}, **kwargs
                )
                assert response.status_code == 422, f"{method} {path}: {response.text}"
                assert ["query", "bogus_query_key"] in [
                    err["loc"] for err in response.json()["detail"]
                ]
