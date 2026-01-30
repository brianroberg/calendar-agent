"""Tests for calendar_utils module."""

from datetime import datetime

from calendar_agent.calendar_utils import (
    find_free_slots,
    format_attendees,
    format_event_time,
    get_event_duration_minutes,
    get_event_summary_text,
    get_event_time,
    get_now_rfc3339,
    get_time_range_rfc3339,
    is_all_day_event,
    parse_attendee_name,
)

# ============================================================================
# Tests for get_event_time
# ============================================================================


def test_get_event_time_with_datetime():
    """Test extracting dateTime from event."""
    event_datetime = {"dateTime": "2024-01-15T10:00:00Z", "timeZone": "UTC"}
    assert get_event_time(event_datetime) == "2024-01-15T10:00:00Z"


def test_get_event_time_with_date():
    """Test extracting date from all-day event."""
    event_datetime = {"date": "2024-01-15"}
    assert get_event_time(event_datetime) == "2024-01-15"


def test_get_event_time_prefers_datetime():
    """Test that dateTime takes precedence over date."""
    event_datetime = {"dateTime": "2024-01-15T10:00:00Z", "date": "2024-01-15"}
    assert get_event_time(event_datetime) == "2024-01-15T10:00:00Z"


def test_get_event_time_none_input():
    """Test handling None input."""
    assert get_event_time(None) == ""


def test_get_event_time_empty_dict():
    """Test handling empty dict."""
    assert get_event_time({}) == ""


# ============================================================================
# Tests for format_event_time
# ============================================================================


def test_format_event_time_datetime_utc():
    """Test formatting UTC datetime."""
    event_datetime = {"dateTime": "2024-01-15T10:00:00Z"}
    result = format_event_time(event_datetime)
    assert "January 15, 2024" in result
    assert "10:00 AM" in result


def test_format_event_time_datetime_positive_offset():
    """Test formatting datetime with positive timezone offset."""
    event_datetime = {"dateTime": "2024-01-15T10:00:00+05:00"}
    result = format_event_time(event_datetime)
    assert "January 15, 2024" in result


def test_format_event_time_datetime_negative_offset():
    """Test formatting datetime with negative timezone offset."""
    event_datetime = {"dateTime": "2024-01-15T10:00:00-05:00"}
    result = format_event_time(event_datetime)
    assert "January 15, 2024" in result
    assert "10:00 AM" in result


def test_format_event_time_all_day():
    """Test formatting all-day event."""
    event_datetime = {"date": "2024-01-15"}
    result = format_event_time(event_datetime)
    assert "January 15, 2024" in result
    assert "(all day)" in result


def test_format_event_time_none_input():
    """Test handling None input."""
    assert format_event_time(None) == "No time specified"


def test_format_event_time_empty_dict():
    """Test handling empty dict."""
    assert format_event_time({}) == "No time specified"


def test_format_event_time_invalid_datetime():
    """Test handling invalid datetime string."""
    event_datetime = {"dateTime": "not-a-date"}
    result = format_event_time(event_datetime)
    assert result == "not-a-date"


# ============================================================================
# Tests for get_event_duration_minutes
# ============================================================================


def test_get_event_duration_minutes_timed_event():
    """Test duration calculation for timed event."""
    start = {"dateTime": "2024-01-15T10:00:00Z"}
    end = {"dateTime": "2024-01-15T11:30:00Z"}
    assert get_event_duration_minutes(start, end) == 90


def test_get_event_duration_minutes_all_day():
    """Test duration calculation for all-day event."""
    start = {"date": "2024-01-15"}
    end = {"date": "2024-01-16"}
    assert get_event_duration_minutes(start, end) == 24 * 60


def test_get_event_duration_minutes_with_timezone():
    """Test duration calculation with timezone offsets."""
    start = {"dateTime": "2024-01-15T10:00:00-05:00"}
    end = {"dateTime": "2024-01-15T11:00:00-05:00"}
    assert get_event_duration_minutes(start, end) == 60


def test_get_event_duration_minutes_none_start():
    """Test handling None start."""
    assert get_event_duration_minutes(None, {"dateTime": "2024-01-15T11:00:00Z"}) is None


def test_get_event_duration_minutes_none_end():
    """Test handling None end."""
    assert get_event_duration_minutes({"dateTime": "2024-01-15T10:00:00Z"}, None) is None


def test_get_event_duration_minutes_invalid_format():
    """Test handling invalid format."""
    start = {"dateTime": "invalid"}
    end = {"dateTime": "invalid"}
    assert get_event_duration_minutes(start, end) is None


# ============================================================================
# Tests for format_attendees
# ============================================================================


def test_format_attendees_with_names():
    """Test formatting attendees with display names."""
    attendees = [
        {"email": "alice@example.com", "displayName": "Alice Smith", "responseStatus": "accepted"},
        {"email": "bob@example.com", "displayName": "Bob Jones", "responseStatus": "tentative"},
    ]
    result = format_attendees(attendees)
    assert "Alice Smith <alice@example.com>" in result
    assert "accepted" in result
    assert "Bob Jones <bob@example.com>" in result


def test_format_attendees_without_names():
    """Test formatting attendees without display names."""
    attendees = [
        {"email": "alice@example.com", "responseStatus": "accepted"},
    ]
    result = format_attendees(attendees)
    assert "alice@example.com (accepted)" in result


def test_format_attendees_empty_list():
    """Test formatting empty attendee list."""
    assert format_attendees([]) == "No attendees"


def test_format_attendees_none():
    """Test formatting None attendees."""
    assert format_attendees(None) == "No attendees"


# ============================================================================
# Tests for parse_attendee_name
# ============================================================================


def test_parse_attendee_name_with_display_name():
    """Test parsing attendee with display name."""
    attendee = {"email": "alice@example.com", "displayName": "Alice Smith"}
    assert parse_attendee_name(attendee) == "Alice Smith"


def test_parse_attendee_name_without_display_name():
    """Test parsing attendee without display name."""
    attendee = {"email": "alice@example.com"}
    assert parse_attendee_name(attendee) == "alice@example.com"


def test_parse_attendee_name_empty():
    """Test parsing empty attendee."""
    assert parse_attendee_name({}) == "Unknown"


# ============================================================================
# Tests for is_all_day_event
# ============================================================================


def test_is_all_day_event_true():
    """Test detecting all-day event."""
    event = {"start": {"date": "2024-01-15"}}
    assert is_all_day_event(event) is True


def test_is_all_day_event_false():
    """Test detecting timed event."""
    event = {"start": {"dateTime": "2024-01-15T10:00:00Z"}}
    assert is_all_day_event(event) is False


def test_is_all_day_event_both_fields():
    """Test event with both date and dateTime (dateTime wins)."""
    event = {"start": {"date": "2024-01-15", "dateTime": "2024-01-15T10:00:00Z"}}
    assert is_all_day_event(event) is False


def test_is_all_day_event_no_start():
    """Test event without start field."""
    event = {}
    assert is_all_day_event(event) is False


# ============================================================================
# Tests for get_event_summary_text
# ============================================================================


def test_get_event_summary_text_full_event():
    """Test building summary text for full event."""
    event = {
        "summary": "Team Meeting",
        "description": "Weekly sync",
        "location": "Conference Room",
        "start": {"dateTime": "2024-01-15T10:00:00Z"},
        "end": {"dateTime": "2024-01-15T11:00:00Z"},
        "attendees": [{"email": "alice@example.com", "responseStatus": "accepted"}],
    }
    result = get_event_summary_text(event)
    assert "Title: Team Meeting" in result
    assert "Weekly sync" in result
    assert "Conference Room" in result
    assert "alice@example.com" in result


def test_get_event_summary_text_minimal_event():
    """Test building summary text for minimal event."""
    event = {
        "start": {"dateTime": "2024-01-15T10:00:00Z"},
        "end": {"dateTime": "2024-01-15T11:00:00Z"},
    }
    result = get_event_summary_text(event)
    assert "Title: Untitled Event" in result
    assert "Time:" in result


def test_get_event_summary_text_long_description():
    """Test that long descriptions are truncated."""
    event = {
        "summary": "Meeting",
        "description": "A" * 5000,
        "start": {"dateTime": "2024-01-15T10:00:00Z"},
        "end": {"dateTime": "2024-01-15T11:00:00Z"},
    }
    result = get_event_summary_text(event)
    assert "..." in result
    assert len(result) < 5500  # Should be truncated


# ============================================================================
# Tests for get_now_rfc3339 and get_time_range_rfc3339
# ============================================================================


def test_get_now_rfc3339_format():
    """Test RFC3339 format of current time."""
    result = get_now_rfc3339()
    assert result.endswith("Z")
    assert "T" in result
    # Should be parseable
    datetime.fromisoformat(result.replace("Z", "+00:00"))


def test_get_time_range_rfc3339_default():
    """Test default 7-day time range."""
    start, end = get_time_range_rfc3339()
    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    delta = end_dt - start_dt
    assert delta.days == 7


def test_get_time_range_rfc3339_custom():
    """Test custom day range."""
    start, end = get_time_range_rfc3339(days_ahead=14)
    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    delta = end_dt - start_dt
    assert delta.days == 14


# ============================================================================
# Tests for find_free_slots
# ============================================================================


def test_find_free_slots_no_events():
    """Test finding free slots when there are no events."""
    slots = find_free_slots(
        events=[],
        time_min="2024-01-15T09:00:00Z",
        time_max="2024-01-15T17:00:00Z",
        min_duration_minutes=30,
        working_hours_only=False,
    )
    assert len(slots) == 1
    assert slots[0]["duration_minutes"] == 8 * 60


def test_find_free_slots_with_event():
    """Test finding free slots with an event in the middle."""
    events = [
        {
            "start": {"dateTime": "2024-01-15T12:00:00Z"},
            "end": {"dateTime": "2024-01-15T13:00:00Z"},
        }
    ]
    slots = find_free_slots(
        events=events,
        time_min="2024-01-15T09:00:00Z",
        time_max="2024-01-15T17:00:00Z",
        min_duration_minutes=30,
        working_hours_only=False,
    )
    assert len(slots) == 2
    # First slot: 9:00 - 12:00 (3 hours)
    assert slots[0]["duration_minutes"] == 180
    # Second slot: 13:00 - 17:00 (4 hours)
    assert slots[1]["duration_minutes"] == 240


def test_find_free_slots_working_hours_only():
    """Test that working hours filter is applied."""
    slots = find_free_slots(
        events=[],
        time_min="2024-01-15T00:00:00Z",
        time_max="2024-01-15T23:59:59Z",
        min_duration_minutes=30,
        working_hours_only=True,
        working_start_hour=9,
        working_end_hour=17,
    )
    # Should only return working hours slot
    assert len(slots) == 1
    assert slots[0]["duration_minutes"] == 8 * 60


def test_find_free_slots_min_duration_filter():
    """Test that min duration filter works."""
    events = [
        {
            "start": {"dateTime": "2024-01-15T10:00:00Z"},
            "end": {"dateTime": "2024-01-15T10:15:00Z"},
        }
    ]
    slots = find_free_slots(
        events=events,
        time_min="2024-01-15T09:00:00Z",
        time_max="2024-01-15T11:00:00Z",
        min_duration_minutes=60,
        working_hours_only=False,
    )
    # Only the 9:00-10:00 slot should be included (60 min)
    # The 10:15-11:00 slot is 45 min, below minimum
    assert len(slots) == 1
    assert slots[0]["duration_minutes"] == 60


def test_find_free_slots_invalid_time_range():
    """Test handling invalid time range."""
    slots = find_free_slots(
        events=[],
        time_min="invalid",
        time_max="invalid",
        min_duration_minutes=30,
    )
    assert slots == []


def test_find_free_slots_all_day_event():
    """Test handling all-day events."""
    events = [
        {
            "start": {"date": "2024-01-15"},
            "end": {"date": "2024-01-16"},
        }
    ]
    slots = find_free_slots(
        events=events,
        time_min="2024-01-15T09:00:00Z",
        time_max="2024-01-15T17:00:00Z",
        min_duration_minutes=30,
        working_hours_only=False,
    )
    # All-day event should block the entire day
    assert len(slots) == 0
