"""Pytest fixtures and sample data for Calendar Agent tests."""

import base64
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

# ============================================================================
# Sample Calendar Data
# ============================================================================


def _encode_body(text: str) -> str:
    """Encode text as base64url for API format."""
    return base64.urlsafe_b64encode(text.encode()).decode()


SAMPLE_CALENDARS = {
    "primary": {
        "id": "primary",
        "summary": "john.doe@example.com",
        "description": "Primary calendar",
        "timeZone": "America/New_York",
        "primary": True,
    },
    "work": {
        "id": "work_calendar_123",
        "summary": "Work Calendar",
        "description": "Work meetings and deadlines",
        "timeZone": "America/New_York",
        "primary": False,
    },
    "personal": {
        "id": "personal_calendar_456",
        "summary": "Personal",
        "description": None,
        "timeZone": "America/Los_Angeles",
        "primary": False,
    },
}


def get_sample_event(
    event_id: str = "event_123",
    summary: str = "Team Meeting",
    description: str = "Weekly team sync",
    location: str | None = "Conference Room A",
    start_hours_from_now: int = 1,
    duration_hours: int = 1,
    attendees: list[dict[str, Any]] | None = None,
    is_all_day: bool = False,
) -> dict[str, Any]:
    """Generate a sample event with customizable properties."""
    now = datetime.now(UTC).replace(tzinfo=None)
    start = now + timedelta(hours=start_hours_from_now)
    end = start + timedelta(hours=duration_hours)

    if is_all_day:
        start_obj = {"date": start.strftime("%Y-%m-%d")}
        end_obj = {"date": end.strftime("%Y-%m-%d")}
    else:
        start_obj = {
            "dateTime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "timeZone": "UTC",
        }
        end_obj = {
            "dateTime": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "timeZone": "UTC",
        }

    event = {
        "id": event_id,
        "summary": summary,
        "description": description,
        "location": location,
        "start": start_obj,
        "end": end_obj,
        "status": "confirmed",
        "htmlLink": f"https://calendar.google.com/event?eid={event_id}",
        "created": (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updated": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if attendees is not None:
        event["attendees"] = attendees

    return event


SAMPLE_EVENTS = {
    "basic_meeting": get_sample_event(
        event_id="meeting_001",
        summary="Team Standup",
        description="Daily standup meeting",
        location="Zoom",
        start_hours_from_now=1,
        duration_hours=0.5,
        attendees=[
            {"email": "alice@example.com", "displayName": "Alice Smith", "responseStatus": "accepted"},
            {"email": "bob@example.com", "displayName": "Bob Jones", "responseStatus": "tentative"},
        ],
    ),
    "all_day_event": get_sample_event(
        event_id="allday_001",
        summary="Company Holiday",
        description="Office closed",
        location=None,
        start_hours_from_now=24,
        duration_hours=24,
        is_all_day=True,
    ),
    "no_description": get_sample_event(
        event_id="nodesc_001",
        summary="Quick Chat",
        description="",
        location=None,
        start_hours_from_now=2,
        duration_hours=0.25,
    ),
    "long_description": get_sample_event(
        event_id="longdesc_001",
        summary="Project Review",
        description="A" * 5000,  # Very long description
        location="Board Room",
        start_hours_from_now=3,
        duration_hours=2,
    ),
    "with_recurrence": {
        **get_sample_event(
            event_id="recurring_001",
            summary="Weekly 1:1",
            description="Weekly check-in with manager",
            location=None,
            start_hours_from_now=24,
            duration_hours=0.5,
        ),
        "recurrence": ["RRULE:FREQ=WEEKLY;BYDAY=MO"],
    },
    "with_conference": {
        **get_sample_event(
            event_id="conf_001",
            summary="Remote Meeting",
            description="Discussion of Q4 goals",
            location=None,
            start_hours_from_now=4,
            duration_hours=1,
        ),
        "conferenceData": {
            "entryPoints": [
                {"entryPointType": "video", "uri": "https://meet.google.com/abc-defg-hij"},
            ],
        },
    },
    "many_attendees": get_sample_event(
        event_id="many_att_001",
        summary="All Hands",
        description="Company-wide meeting",
        location="Main Auditorium",
        start_hours_from_now=48,
        duration_hours=2,
        attendees=[
            {"email": f"employee{i}@example.com", "responseStatus": "needsAction"}
            for i in range(50)
        ],
    ),
    "past_event": get_sample_event(
        event_id="past_001",
        summary="Yesterday's Meeting",
        description="This meeting already happened",
        start_hours_from_now=-25,
        duration_hours=1,
    ),
}


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_proxy_client():
    """Mock CalendarProxyClient with default responses."""
    with patch("calendar_agent.calendar_server.get_calendar_client") as mock_get:
        mock_client = AsyncMock()

        # Default responses
        mock_client.list_calendars.return_value = {
            "items": list(SAMPLE_CALENDARS.values()),
        }
        mock_client.get_calendar.return_value = SAMPLE_CALENDARS["primary"]
        mock_client.list_events.return_value = {
            "items": [SAMPLE_EVENTS["basic_meeting"]],
        }
        mock_client.get_event.return_value = SAMPLE_EVENTS["basic_meeting"]
        mock_client.create_event.return_value = SAMPLE_EVENTS["basic_meeting"]
        mock_client.update_event.return_value = SAMPLE_EVENTS["basic_meeting"]
        mock_client.patch_event.return_value = SAMPLE_EVENTS["basic_meeting"]
        mock_client.delete_event.return_value = {"success": True}

        mock_get.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_llm_service():
    """Mock LLMService with default responses."""
    with patch("calendar_agent.calendar_server.get_llm_service") as mock_get:
        mock_service = AsyncMock()

        # Default responses
        mock_service.summarize_event.return_value = {
            "event_id": "event_123",
            "summary": "This is a team meeting to discuss project progress.",
        }
        mock_service.ask_about_event.return_value = {
            "event_id": "event_123",
            "question": "What time is the meeting?",
            "answer": "The meeting is scheduled for 2:00 PM.",
        }
        mock_service.batch_summarize.return_value = {
            "results": [
                {"event_id": "event_1", "summary": "Summary 1", "action_type": "meeting"},
            ],
            "total": 1,
        }
        mock_service.find_free_time.return_value = {
            "available_slots": [
                {"start": "2024-01-15T14:00:00Z", "end": "2024-01-15T15:00:00Z", "duration_minutes": 60}
            ],
            "suggestions": "The best time for a 30-minute meeting is 2:00 PM.",
            "duration_requested": 30,
        }
        mock_service.analyze_schedule.return_value = {
            "time_range": "2024-01-15 to 2024-01-22",
            "metrics": {"total_events": 5, "total_hours": 8.5},
            "analysis_type": "overview",
            "insights": "Your schedule looks balanced with good focus time blocks.",
        }
        mock_service.prepare_briefing.return_value = {
            "briefing_type": "daily",
            "period": "daily schedule",
            "event_count": 3,
            "briefing": "Today you have 3 meetings...",
        }

        mock_get.return_value = mock_service
        yield mock_service


@pytest.fixture
def client(mock_proxy_client, mock_llm_service):
    """FastAPI test client with mocked dependencies."""
    from calendar_agent.calendar_server import app
    return TestClient(app)


@pytest.fixture
def client_no_mocks():
    """FastAPI test client without mocked dependencies (for integration tests)."""
    from calendar_agent.calendar_server import app
    return TestClient(app)


# ============================================================================
# Helper Fixtures
# ============================================================================


@pytest.fixture
def sample_event():
    """Return a basic sample event."""
    return SAMPLE_EVENTS["basic_meeting"].copy()


@pytest.fixture
def sample_events_list():
    """Return list of sample events for batch operations."""
    return [
        SAMPLE_EVENTS["basic_meeting"],
        SAMPLE_EVENTS["all_day_event"],
        SAMPLE_EVENTS["no_description"],
    ]


@pytest.fixture
def sample_calendars():
    """Return sample calendars dict."""
    return SAMPLE_CALENDARS.copy()
