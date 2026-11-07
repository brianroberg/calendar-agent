"""Pytest fixtures for Calendar Agent tests.

Sample calendars/events and the event factories live in ``tests/factories.py``.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.factories import SAMPLE_CALENDARS, SAMPLE_EVENTS

# A synthetic token for the suite. The app under test runs in `required` mode
# with this one token configured (issue #15), and the `client` fixtures send
# it, so every existing test exercises its route as an authenticated caller.
TEST_API_TOKEN = "synthetic-test-suite-token-0123456789abcdef"
TEST_AUTH_HEADERS = {"Authorization": f"Bearer {TEST_API_TOKEN}"}


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def auth_settings():
    """Give the app its authentication settings for the test, then take them
    away again.

    The server reads them from the environment in its start-up hook, which a
    bare ``TestClient(app)`` does not run. Tests of the other modes assign
    ``app.state.auth_settings`` themselves; this fixture restores the state
    afterwards either way.
    """
    from calendar_agent import auth
    from calendar_agent.calendar_server import app

    app.state.auth_settings = auth.load_auth_settings(
        {auth.KEYS_ENV: f"test-suite:{TEST_API_TOKEN}"}
    )
    yield
    del app.state.auth_settings


@pytest.fixture(autouse=True)
def clear_authenticated_calendar_id():
    """Clear the process-wide authenticated-calendar-id cache around every
    test.

    ``calendar_server`` reads the authenticated account's own calendar id
    once and keeps it for the life of the process (the proxy's credentials
    never change under a running server). In a test run that would leak one
    test's resolved identity into the next, so tests would pass or fail
    depending on their order.
    """
    from calendar_agent.calendar_server import reset_authenticated_calendar_id

    reset_authenticated_calendar_id()
    yield
    reset_authenticated_calendar_id()


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
        mock_client.respond_to_event.return_value = SAMPLE_EVENTS["basic_meeting"]

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
    """FastAPI test client with mocked dependencies, sending the suite's token."""
    from calendar_agent.calendar_server import app
    return TestClient(app, headers=TEST_AUTH_HEADERS)


@pytest.fixture
def client_no_mocks():
    """FastAPI test client without mocked dependencies (for integration tests),
    sending the suite's token."""
    from calendar_agent.calendar_server import app
    return TestClient(app, headers=TEST_AUTH_HEADERS)


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
