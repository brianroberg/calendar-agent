"""Utility functions for calendar operations."""

from datetime import UTC, datetime, timedelta
from typing import Any


def get_event_time(event_datetime: dict[str, Any] | None) -> str:
    """Extract time string from event datetime object.

    Handles both all-day events (date) and timed events (dateTime).
    """
    if not event_datetime:
        return ""
    return event_datetime.get("dateTime") or event_datetime.get("date") or ""


def format_event_time(event_datetime: dict[str, Any] | None) -> str:
    """Format event datetime for human-readable display."""
    if not event_datetime:
        return "No time specified"

    if "dateTime" in event_datetime and event_datetime["dateTime"]:
        # Timed event - parse and format
        dt_str = event_datetime["dateTime"]
        try:
            # Handle RFC3339 format with timezone
            if "T" in dt_str:
                # Parse ISO format properly (handles +HH:MM, -HH:MM, and Z)
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                return dt.strftime("%B %d, %Y at %I:%M %p")
        except ValueError:
            return dt_str
        return dt_str

    if "date" in event_datetime and event_datetime["date"]:
        # All-day event
        try:
            dt = datetime.strptime(event_datetime["date"], "%Y-%m-%d")
            return dt.strftime("%B %d, %Y (all day)")
        except ValueError:
            return event_datetime["date"]

    return "No time specified"


def get_event_duration_minutes(
    start: dict[str, Any] | None, end: dict[str, Any] | None
) -> int | None:
    """Calculate event duration in minutes."""
    if not start or not end:
        return None

    start_time = get_event_time(start)
    end_time = get_event_time(end)

    if not start_time or not end_time:
        return None

    try:
        # Handle dateTime format
        if "T" in start_time:
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            delta = end_dt - start_dt
            return int(delta.total_seconds() / 60)

        # Handle all-day events (date only)
        start_date = datetime.strptime(start_time, "%Y-%m-%d")
        end_date = datetime.strptime(end_time, "%Y-%m-%d")
        delta = end_date - start_date
        return int(delta.total_seconds() / 60) if delta.days >= 0 else None

    except (ValueError, TypeError):
        return None


def format_attendees(attendees: list[dict[str, Any]] | None) -> str:
    """Format attendee list for display."""
    if not attendees:
        return "No attendees"

    formatted = []
    for attendee in attendees:
        email = attendee.get("email", "")
        name = attendee.get("displayName", "")
        status = attendee.get("responseStatus", "")

        if name:
            formatted.append(f"{name} <{email}> ({status})")
        else:
            formatted.append(f"{email} ({status})")

    return ", ".join(formatted)


def parse_attendee_name(attendee: dict[str, Any]) -> str:
    """Extract display name from attendee, falling back to email."""
    return attendee.get("displayName") or attendee.get("email", "Unknown")


def is_all_day_event(event: dict[str, Any]) -> bool:
    """Check if an event is an all-day event."""
    start = event.get("start", {})
    return "date" in start and "dateTime" not in start


def get_event_summary_text(event: dict[str, Any]) -> str:
    """Build a summary text block from an event for LLM processing."""
    summary = event.get("summary", "Untitled Event")
    description = event.get("description", "")
    location = event.get("location", "")
    start = event.get("start", {})
    end = event.get("end", {})
    attendees = event.get("attendees", [])

    parts = [
        f"Title: {summary}",
        f"Time: {format_event_time(start)} to {format_event_time(end)}",
    ]

    if location:
        parts.append(f"Location: {location}")

    if description:
        # Truncate long descriptions
        max_desc_length = 2000
        if len(description) > max_desc_length:
            description = description[:max_desc_length] + "..."
        parts.append(f"Description: {description}")

    if attendees:
        parts.append(f"Attendees: {format_attendees(attendees)}")

    return "\n".join(parts)


def get_now_rfc3339() -> str:
    """Get current time in RFC3339 format."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_time_range_rfc3339(days_ahead: int = 7) -> tuple[str, str]:
    """Get a time range from now to N days ahead in RFC3339 format."""
    now = datetime.now(UTC)
    end = now + timedelta(days=days_ahead)
    return (
        now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def find_free_slots(
    events: list[dict[str, Any]],
    time_min: str,
    time_max: str,
    min_duration_minutes: int = 30,
    working_hours_only: bool = True,
    working_start_hour: int = 9,
    working_end_hour: int = 17,
) -> list[dict[str, Any]]:
    """Find free time slots between events.

    Args:
        events: List of calendar events (should have singleEvents=true applied)
        time_min: Start of time range (RFC3339)
        time_max: End of time range (RFC3339)
        min_duration_minutes: Minimum slot duration to consider
        working_hours_only: Only return slots during working hours
        working_start_hour: Start of working day (0-23)
        working_end_hour: End of working day (0-23)

    Returns:
        List of free slots with start, end, and duration_minutes
    """
    # Parse boundaries
    try:
        range_start = datetime.fromisoformat(time_min.replace("Z", "+00:00"))
        range_end = datetime.fromisoformat(time_max.replace("Z", "+00:00"))
    except ValueError:
        return []

    # Extract busy times from events
    busy_periods: list[tuple[datetime, datetime]] = []
    for event in events:
        start_str = get_event_time(event.get("start"))
        end_str = get_event_time(event.get("end"))

        if not start_str or not end_str:
            continue

        try:
            # Handle all-day events
            if "T" not in start_str:
                start_dt = datetime.strptime(start_str, "%Y-%m-%d")
                end_dt = datetime.strptime(end_str, "%Y-%m-%d")
            else:
                start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))

            # Make timezone-naive for comparison
            if start_dt.tzinfo is not None:
                start_dt = start_dt.replace(tzinfo=None)
            if end_dt.tzinfo is not None:
                end_dt = end_dt.replace(tzinfo=None)

            busy_periods.append((start_dt, end_dt))
        except ValueError:
            continue

    # Sort by start time
    busy_periods.sort(key=lambda x: x[0])

    # Make range boundaries timezone-naive
    if range_start.tzinfo is not None:
        range_start = range_start.replace(tzinfo=None)
    if range_end.tzinfo is not None:
        range_end = range_end.replace(tzinfo=None)

    # Find gaps
    free_slots: list[dict[str, Any]] = []
    current_time = range_start

    for busy_start, busy_end in busy_periods:
        if busy_start > current_time:
            # There's a gap before this event
            gap_start = current_time
            gap_end = min(busy_start, range_end)

            # Check if within working hours (if required)
            if working_hours_only:
                # Adjust to working hours
                if gap_start.hour < working_start_hour:
                    gap_start = gap_start.replace(
                        hour=working_start_hour, minute=0, second=0
                    )
                elif gap_start.hour >= working_end_hour:
                    # Gap starts after working hours, skip this slot
                    continue
                if gap_end.hour >= working_end_hour:
                    gap_end = gap_end.replace(
                        hour=working_end_hour, minute=0, second=0
                    )
                elif gap_end.hour < working_start_hour:
                    # Gap ends before working hours, skip this slot
                    continue

            # Calculate duration (ensure end > start after adjustments)
            if gap_end > gap_start:
                duration = int((gap_end - gap_start).total_seconds() / 60)
                if duration >= min_duration_minutes:
                    free_slots.append({
                        "start": gap_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "end": gap_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "duration_minutes": duration,
                    })

        # Move current time to end of this event
        current_time = max(current_time, busy_end)

    # Check for gap after last event
    if current_time < range_end:
        gap_start = current_time
        gap_end = range_end

        if working_hours_only:
            if gap_start.hour < working_start_hour:
                gap_start = gap_start.replace(
                    hour=working_start_hour, minute=0, second=0
                )
            elif gap_start.hour >= working_end_hour:
                # Gap starts after working hours, no slot to add
                return free_slots
            if gap_end.hour >= working_end_hour:
                gap_end = gap_end.replace(hour=working_end_hour, minute=0, second=0)
            elif gap_end.hour < working_start_hour:
                # Gap ends before working hours, no slot to add
                return free_slots

        if gap_end > gap_start:
            duration = int((gap_end - gap_start).total_seconds() / 60)
            if duration >= min_duration_minutes:
                free_slots.append({
                    "start": gap_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end": gap_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "duration_minutes": duration,
                })

    return free_slots
