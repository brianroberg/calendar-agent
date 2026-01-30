"""Calendar Agent Server - A privacy-focused FastAPI server for Google Calendar.

This server acts as an intermediary between AI agents (like Claude Code) and the
Google Calendar API via a proxy server. Calendar event details are processed
locally; only metadata and LLM-generated summaries are returned to calling agents.

Key features:
- Calendar and event CRUD operations via proxy
- LLM-powered event summarization and Q&A
- Schedule analysis and free time finding
- Daily/weekly briefing generation
- Bulk operations support

All calendar operations go through the api-proxy server which handles OAuth
and enforces security policies.
"""

import os
from enum import Enum
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .calendar_utils import find_free_slots, get_time_range_rfc3339
from .exceptions import ProxyAuthError, ProxyError, ProxyForbiddenError
from .llm_service import get_llm_service
from .proxy_client import get_calendar_client

load_dotenv()


# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="Calendar Agent",
    description="A privacy-focused FastAPI server for Google Calendar operations with AI agents",
    version=__version__,
)


# ============================================================================
# Pydantic Models - Requests
# ============================================================================


class EventDateTime(BaseModel):
    """DateTime specification for calendar events."""
    date: str | None = Field(None, description="Date for all-day events (YYYY-MM-DD)")
    dateTime: str | None = Field(None, description="DateTime for timed events (RFC3339)")
    timeZone: str | None = Field(None, description="Timezone (e.g., 'America/New_York')")


class EventAttendee(BaseModel):
    """Event attendee."""
    email: str
    displayName: str | None = None
    responseStatus: str | None = None
    optional: bool | None = None
    organizer: bool | None = None
    self_: bool | None = Field(None, alias="self")


class EventReminder(BaseModel):
    """Event reminder."""
    method: str
    minutes: int


class EventReminders(BaseModel):
    """Event reminders configuration."""
    useDefault: bool = True
    overrides: list[EventReminder] | None = None


class EventCreateRequest(BaseModel):
    """Request body for creating a new event."""
    summary: str | None = Field(None, description="Event title")
    description: str | None = Field(None, description="Event description")
    location: str | None = Field(None, description="Event location")
    start: EventDateTime | None = Field(None, description="Start time")
    end: EventDateTime | None = Field(None, description="End time")
    attendees: list[EventAttendee] | None = Field(None, description="Event attendees")
    reminders: EventReminders | None = Field(None, description="Reminder settings")
    recurrence: list[str] | None = Field(None, description="Recurrence rules (RRULE)")
    colorId: str | None = Field(None, description="Color ID")
    transparency: str | None = Field(None, description="'opaque' or 'transparent'")
    visibility: str | None = Field(None, description="'default', 'public', 'private'")
    guestsCanInviteOthers: bool | None = None
    guestsCanModify: bool | None = None
    guestsCanSeeOtherGuests: bool | None = None


class EventUpdateRequest(EventCreateRequest):
    """Request body for updating an event (full replacement)."""
    pass


class EventPatchRequest(BaseModel):
    """Request body for partially updating an event."""
    summary: str | None = None
    description: str | None = None
    location: str | None = None
    start: EventDateTime | None = None
    end: EventDateTime | None = None
    attendees: list[EventAttendee] | None = None
    reminders: EventReminders | None = None
    recurrence: list[str] | None = None
    colorId: str | None = None


class SummarizeRequest(BaseModel):
    """Request to summarize an event."""
    calendar_id: str = Field(..., description="Calendar ID containing the event")
    event_id: str = Field(..., description="Event ID to summarize")
    format: str = Field("brief", description="'brief' or 'detailed'")


class AskAboutRequest(BaseModel):
    """Request to ask a question about an event."""
    calendar_id: str = Field(..., description="Calendar ID containing the event")
    event_id: str = Field(..., description="Event ID to ask about")
    question: str = Field(..., description="Question to ask about the event")


class BatchSummarizeRequest(BaseModel):
    """Request to summarize multiple events."""
    calendar_id: str = Field(..., description="Calendar ID containing the events")
    event_ids: list[str] = Field(..., description="List of event IDs to summarize")
    triage: bool = Field(False, description="Include action type classification")


class FindFreeTimeRequest(BaseModel):
    """Request to find free time slots."""
    calendar_id: str = Field(..., description="Calendar ID to check")
    time_min: str = Field(..., description="Start of search range (RFC3339)")
    time_max: str = Field(..., description="End of search range (RFC3339)")
    duration_minutes: int = Field(..., gt=0, description="Required meeting duration")
    working_hours_only: bool = Field(True, description="Only consider 9am-5pm")
    buffer_minutes: int = Field(0, ge=0, description="Buffer between meetings")
    prefer_morning: bool = Field(False, description="Prefer morning times")
    prefer_afternoon: bool = Field(False, description="Prefer afternoon times")


class AnalyzeScheduleRequest(BaseModel):
    """Request to analyze schedule patterns."""
    calendar_id: str = Field(..., description="Calendar ID to analyze")
    time_min: str = Field(..., description="Start of analysis period (RFC3339)")
    time_max: str = Field(..., description="End of analysis period (RFC3339)")
    analysis_type: str = Field(
        "overview",
        description="Type: 'overview', 'workload', 'patterns', 'conflicts'"
    )


class PrepareBriefingRequest(BaseModel):
    """Request to prepare a schedule briefing."""
    calendar_id: str = Field(..., description="Calendar ID for briefing")
    briefing_type: str = Field("daily", description="'daily' or 'weekly'")
    time_min: str | None = Field(None, description="Start time (defaults to now)")
    time_max: str | None = Field(None, description="End time (defaults based on type)")


class SearchFilters(BaseModel):
    """Filters for event search."""
    query: str | None = Field(None, description="Free text search")
    time_min: str | None = Field(None, description="Start of time range (RFC3339)")
    time_max: str | None = Field(None, description="End of time range (RFC3339)")
    max_results: int = Field(100, ge=1, le=500, description="Maximum results")
    order_by: str | None = Field(None, description="'startTime' or 'updated'")
    show_deleted: bool = Field(False, description="Include deleted events")


class SearchRequest(BaseModel):
    """Request to search events."""
    calendar_id: str = Field(..., description="Calendar ID to search")
    filters: SearchFilters = Field(default_factory=SearchFilters)


class BulkOperationType(str, Enum):
    """Types of bulk operations."""
    UPDATE = "update"
    DELETE = "delete"
    PATCH = "patch"


class BulkOperation(BaseModel):
    """A single operation in a bulk request."""
    operation: BulkOperationType
    event_id: str
    calendar_id: str
    updates: dict[str, Any] | None = Field(
        None, description="Update data (for update/patch operations)"
    )
    send_updates: str | None = Field(None, description="'all', 'externalOnly', 'none'")


class BulkActionsRequest(BaseModel):
    """Request for bulk operations on events."""
    operations: list[BulkOperation] = Field(..., min_length=1)


# ============================================================================
# Pydantic Models - Responses
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = __version__


class CalendarSummary(BaseModel):
    """Summary of a calendar."""
    id: str
    summary: str
    description: str | None = None
    timeZone: str | None = None
    primary: bool = False


class CalendarsResponse(BaseModel):
    """Response for listing calendars."""
    success: bool
    calendars: list[CalendarSummary]
    error: str | None = None


class CalendarDetailResponse(BaseModel):
    """Response for getting a single calendar."""
    success: bool
    calendar: dict[str, Any] | None = None
    error: str | None = None


class EventSummary(BaseModel):
    """Summary of an event (metadata only, no body)."""
    id: str
    calendar_id: str
    summary: str
    start: str
    end: str
    location: str | None = None
    attendee_count: int
    is_all_day: bool
    status: str | None = None
    html_link: str | None = None


class EventsListResponse(BaseModel):
    """Response for listing events."""
    success: bool
    events: list[EventSummary]
    next_page_token: str | None = None
    error: str | None = None


class EventDetailResponse(BaseModel):
    """Full event details (for get/create/update operations)."""
    success: bool
    event: dict[str, Any] | None = None
    error: str | None = None


class ActionResponse(BaseModel):
    """Response for action endpoints."""
    success: bool
    message: str
    error: str | None = None


class LLMResponse(BaseModel):
    """Response from LLM-powered endpoints."""
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None


class BulkOperationResult(BaseModel):
    """Result of a single bulk operation."""
    event_id: str
    operation: str
    success: bool
    error: str | None = None


class BulkActionsResponse(BaseModel):
    """Response for bulk operations."""
    success: bool
    results: list[BulkOperationResult]
    success_count: int
    error_count: int
    error: str | None = None


# ============================================================================
# Helper Functions
# ============================================================================


def format_proxy_error(e: Exception) -> str:
    """Format a proxy error for user-friendly display."""
    if isinstance(e, ProxyAuthError):
        return f"Authentication error: {e}"
    if isinstance(e, ProxyForbiddenError):
        return f"Operation blocked: {e}"
    if isinstance(e, ProxyError):
        return f"Proxy error: {e}"
    return str(e)


def event_to_summary(event: dict[str, Any], calendar_id: str) -> EventSummary:
    """Convert a full event to a summary (metadata only)."""
    start = event.get("start", {})
    end = event.get("end", {})
    attendees = event.get("attendees", [])

    # Get time string (prefer dateTime, fall back to date for all-day)
    start_str = start.get("dateTime") or start.get("date") or ""
    end_str = end.get("dateTime") or end.get("date") or ""
    is_all_day = "date" in start and "dateTime" not in start

    return EventSummary(
        id=event.get("id", ""),
        calendar_id=calendar_id,
        summary=event.get("summary", "Untitled Event"),
        start=start_str,
        end=end_str,
        location=event.get("location"),
        attendee_count=len(attendees) if attendees else 0,
        is_all_day=is_all_day,
        status=event.get("status"),
        html_link=event.get("htmlLink"),
    )


# ============================================================================
# Health Endpoint
# ============================================================================


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint. Returns server status and version."""
    return HealthResponse()


# ============================================================================
# Calendar Endpoints
# ============================================================================


@app.get("/calendars", response_model=CalendarsResponse, tags=["calendars"])
async def list_calendars(
    max_results: int | None = None,
    page_token: str | None = None,
):
    """List all calendars for the authenticated user."""
    try:
        client = get_calendar_client()
        result = await client.list_calendars(
            max_results=max_results,
            page_token=page_token,
        )

        calendars = [
            CalendarSummary(
                id=cal.get("id", ""),
                summary=cal.get("summary", ""),
                description=cal.get("description"),
                timeZone=cal.get("timeZone"),
                primary=cal.get("primary", False),
            )
            for cal in result.get("items", [])
        ]

        return CalendarsResponse(success=True, calendars=calendars)
    except Exception as e:
        return CalendarsResponse(success=False, calendars=[], error=format_proxy_error(e))


@app.get("/calendars/{calendar_id}", response_model=CalendarDetailResponse, tags=["calendars"])
async def get_calendar(calendar_id: str):
    """Get metadata for a specific calendar."""
    try:
        client = get_calendar_client()
        calendar = await client.get_calendar(calendar_id)
        return CalendarDetailResponse(success=True, calendar=calendar)
    except Exception as e:
        return CalendarDetailResponse(success=False, calendar=None, error=format_proxy_error(e))


# ============================================================================
# Event CRUD Endpoints
# ============================================================================


@app.get(
    "/calendars/{calendar_id}/events",
    response_model=EventsListResponse,
    tags=["events"]
)
async def list_events(
    calendar_id: str,
    max_results: int = 100,
    page_token: str | None = None,
    time_min: str | None = None,
    time_max: str | None = None,
    q: str | None = None,
    single_events: bool = True,
    order_by: str | None = None,
):
    """List events in a calendar.

    By default, recurring events are expanded to individual instances (singleEvents=true).
    """
    try:
        client = get_calendar_client()
        result = await client.list_events(
            calendar_id=calendar_id,
            max_results=max_results,
            page_token=page_token,
            time_min=time_min,
            time_max=time_max,
            q=q,
            single_events=single_events,
            order_by=order_by,
        )

        events = [
            event_to_summary(event, calendar_id)
            for event in result.get("items", [])
        ]

        return EventsListResponse(
            success=True,
            events=events,
            next_page_token=result.get("nextPageToken"),
        )
    except Exception as e:
        return EventsListResponse(success=False, events=[], error=format_proxy_error(e))


@app.post(
    "/calendars/{calendar_id}/events",
    response_model=EventDetailResponse,
    tags=["events"]
)
async def create_event(
    calendar_id: str,
    event: EventCreateRequest,
    send_updates: str | None = None,
):
    """Create a new event in a calendar."""
    try:
        client = get_calendar_client()
        # Convert Pydantic model to dict, excluding None values
        event_data = event.model_dump(exclude_none=True, by_alias=True)

        result = await client.create_event(
            calendar_id=calendar_id,
            event_data=event_data,
            send_updates=send_updates,
        )
        return EventDetailResponse(success=True, event=result)
    except Exception as e:
        return EventDetailResponse(success=False, event=None, error=format_proxy_error(e))


@app.get(
    "/calendars/{calendar_id}/events/{event_id}",
    response_model=EventDetailResponse,
    tags=["events"]
)
async def get_event(
    calendar_id: str,
    event_id: str,
    time_zone: str | None = None,
):
    """Get a specific event by ID."""
    try:
        client = get_calendar_client()
        event = await client.get_event(
            calendar_id=calendar_id,
            event_id=event_id,
            time_zone=time_zone,
        )
        return EventDetailResponse(success=True, event=event)
    except Exception as e:
        return EventDetailResponse(success=False, event=None, error=format_proxy_error(e))


@app.put(
    "/calendars/{calendar_id}/events/{event_id}",
    response_model=EventDetailResponse,
    tags=["events"]
)
async def update_event(
    calendar_id: str,
    event_id: str,
    event: EventUpdateRequest,
    send_updates: str | None = None,
):
    """Update an event (full replacement)."""
    try:
        client = get_calendar_client()
        event_data = event.model_dump(exclude_none=True, by_alias=True)

        result = await client.update_event(
            calendar_id=calendar_id,
            event_id=event_id,
            event_data=event_data,
            send_updates=send_updates,
        )
        return EventDetailResponse(success=True, event=result)
    except Exception as e:
        return EventDetailResponse(success=False, event=None, error=format_proxy_error(e))


@app.patch(
    "/calendars/{calendar_id}/events/{event_id}",
    response_model=EventDetailResponse,
    tags=["events"]
)
async def patch_event(
    calendar_id: str,
    event_id: str,
    event: EventPatchRequest,
    send_updates: str | None = None,
):
    """Partially update an event."""
    try:
        client = get_calendar_client()
        event_data = event.model_dump(exclude_none=True, by_alias=True)

        result = await client.patch_event(
            calendar_id=calendar_id,
            event_id=event_id,
            event_data=event_data,
            send_updates=send_updates,
        )
        return EventDetailResponse(success=True, event=result)
    except Exception as e:
        return EventDetailResponse(success=False, event=None, error=format_proxy_error(e))


@app.delete(
    "/calendars/{calendar_id}/events/{event_id}",
    response_model=ActionResponse,
    tags=["events"]
)
async def delete_event(
    calendar_id: str,
    event_id: str,
    send_updates: str | None = None,
):
    """Delete an event.

    Note: The proxy may require confirmation for delete operations.
    If confirmation is required, this endpoint will return an error with
    details on how to confirm the deletion.
    """
    try:
        client = get_calendar_client()
        await client.delete_event(
            calendar_id=calendar_id,
            event_id=event_id,
            send_updates=send_updates,
        )
        return ActionResponse(success=True, message="Event deleted successfully")
    except ProxyForbiddenError as e:
        # Pass through confirmation requirements
        return ActionResponse(
            success=False,
            message="Deletion requires confirmation",
            error=str(e),
        )
    except Exception as e:
        return ActionResponse(
            success=False,
            message="Failed to delete event",
            error=format_proxy_error(e),
        )


# ============================================================================
# LLM-Powered Endpoints - Basic
# ============================================================================


@app.post("/summarize", response_model=LLMResponse, tags=["llm"])
async def summarize_event(request: SummarizeRequest):
    """Summarize a calendar event using AI.

    The event is fetched from the calendar and processed locally.
    Only the summary is returned to the calling agent.
    """
    try:
        client = get_calendar_client()
        event = await client.get_event(
            calendar_id=request.calendar_id,
            event_id=request.event_id,
        )

        llm_service = get_llm_service()
        result = await llm_service.summarize_event(event, format=request.format)

        return LLMResponse(success=True, data=result)
    except Exception as e:
        return LLMResponse(success=False, data=None, error=format_proxy_error(e))


@app.post("/ask-about", response_model=LLMResponse, tags=["llm"])
async def ask_about_event(request: AskAboutRequest):
    """Ask a question about a specific calendar event.

    The event is fetched and processed locally. Only the answer is returned.
    """
    try:
        client = get_calendar_client()
        event = await client.get_event(
            calendar_id=request.calendar_id,
            event_id=request.event_id,
        )

        llm_service = get_llm_service()
        result = await llm_service.ask_about_event(event, request.question)

        return LLMResponse(success=True, data=result)
    except Exception as e:
        return LLMResponse(success=False, data=None, error=format_proxy_error(e))


@app.post("/batch-summarize", response_model=LLMResponse, tags=["llm"])
async def batch_summarize_events(request: BatchSummarizeRequest):
    """Summarize multiple events, optionally with triage classification.

    When triage=true, events are classified by action type (meeting, deadline, etc.)
    """
    try:
        client = get_calendar_client()

        # Fetch all events
        events = []
        for event_id in request.event_ids:
            try:
                event = await client.get_event(
                    calendar_id=request.calendar_id,
                    event_id=event_id,
                )
                events.append(event)
            except Exception:
                # Continue on individual failures
                events.append({"id": event_id, "error": "Failed to fetch event"})

        llm_service = get_llm_service()
        result = await llm_service.batch_summarize(events, triage=request.triage)

        return LLMResponse(success=True, data=result)
    except Exception as e:
        return LLMResponse(success=False, data=None, error=format_proxy_error(e))


# ============================================================================
# LLM-Powered Endpoints - Calendar-Specific
# ============================================================================


@app.post("/find-free-time", response_model=LLMResponse, tags=["llm"])
async def find_free_time(request: FindFreeTimeRequest):
    """Find available time slots and get AI suggestions for scheduling.

    Analyzes the calendar for the specified time range, identifies free slots
    that meet the duration requirement, and provides AI-powered recommendations.
    """
    try:
        client = get_calendar_client()

        # Get events in the time range
        result = await client.list_events(
            calendar_id=request.calendar_id,
            time_min=request.time_min,
            time_max=request.time_max,
            single_events=True,  # Expand recurring events
            order_by="startTime",
        )
        events = result.get("items", [])

        # Find free slots
        free_slots = find_free_slots(
            events=events,
            time_min=request.time_min,
            time_max=request.time_max,
            min_duration_minutes=request.duration_minutes,
            working_hours_only=request.working_hours_only,
        )

        # Get AI suggestions
        llm_service = get_llm_service()
        preferences = {}
        if request.prefer_morning:
            preferences["prefer_morning"] = True
        if request.prefer_afternoon:
            preferences["prefer_afternoon"] = True
        if request.buffer_minutes:
            preferences["buffer_minutes"] = request.buffer_minutes

        suggestions = await llm_service.find_free_time(
            free_slots=free_slots,
            duration_minutes=request.duration_minutes,
            preferences=preferences if preferences else None,
        )

        return LLMResponse(success=True, data=suggestions)
    except Exception as e:
        return LLMResponse(success=False, data=None, error=format_proxy_error(e))


@app.post("/analyze-schedule", response_model=LLMResponse, tags=["llm"])
async def analyze_schedule(request: AnalyzeScheduleRequest):
    """Analyze schedule patterns and get AI-powered insights.

    Provides analysis of meeting load, patterns, potential conflicts,
    and recommendations for schedule optimization.
    """
    try:
        client = get_calendar_client()

        # Get events in the analysis period
        result = await client.list_events(
            calendar_id=request.calendar_id,
            time_min=request.time_min,
            time_max=request.time_max,
            single_events=True,
            order_by="startTime",
        )
        events = result.get("items", [])

        # Build human-readable time range description
        time_range = f"{request.time_min} to {request.time_max}"

        llm_service = get_llm_service()
        analysis = await llm_service.analyze_schedule(
            events=events,
            time_range=time_range,
            analysis_type=request.analysis_type,
        )

        return LLMResponse(success=True, data=analysis)
    except Exception as e:
        return LLMResponse(success=False, data=None, error=format_proxy_error(e))


@app.post("/prepare-briefing", response_model=LLMResponse, tags=["llm"])
async def prepare_briefing(request: PrepareBriefingRequest):
    """Generate an AI-powered schedule briefing.

    Creates a comprehensive overview of the upcoming schedule including
    key meetings, preparation notes, and potential issues.
    """
    try:
        client = get_calendar_client()

        # Determine time range based on briefing type
        if request.time_min and request.time_max:
            time_min = request.time_min
            time_max = request.time_max
        else:
            # Default time ranges
            if request.briefing_type == "weekly":
                time_min, time_max = get_time_range_rfc3339(days_ahead=7)
            else:
                time_min, time_max = get_time_range_rfc3339(days_ahead=1)

        # Get events for the briefing period
        result = await client.list_events(
            calendar_id=request.calendar_id,
            time_min=time_min,
            time_max=time_max,
            single_events=True,
            order_by="startTime",
        )
        events = result.get("items", [])

        # Build description
        date_description = f"{request.briefing_type} schedule"

        llm_service = get_llm_service()
        briefing = await llm_service.prepare_briefing(
            events=events,
            briefing_type=request.briefing_type,
            date_description=date_description,
        )

        return LLMResponse(success=True, data=briefing)
    except Exception as e:
        return LLMResponse(success=False, data=None, error=format_proxy_error(e))


# ============================================================================
# Operations Endpoints
# ============================================================================


@app.post("/search", response_model=EventsListResponse, tags=["operations"])
async def search_events(request: SearchRequest):
    """Search events in a calendar with structured filters."""
    try:
        client = get_calendar_client()

        result = await client.list_events(
            calendar_id=request.calendar_id,
            max_results=request.filters.max_results,
            time_min=request.filters.time_min,
            time_max=request.filters.time_max,
            q=request.filters.query,
            single_events=True,
            order_by=request.filters.order_by,
            show_deleted=request.filters.show_deleted,
        )

        events = [
            event_to_summary(event, request.calendar_id)
            for event in result.get("items", [])
        ]

        return EventsListResponse(
            success=True,
            events=events,
            next_page_token=result.get("nextPageToken"),
        )
    except Exception as e:
        return EventsListResponse(success=False, events=[], error=format_proxy_error(e))


@app.post("/bulk-actions", response_model=BulkActionsResponse, tags=["operations"])
async def bulk_actions(request: BulkActionsRequest):
    """Execute multiple operations on events in a single request.

    Supports update, patch, and delete operations. Operations are executed
    sequentially, and the response includes results for each operation.

    Note: Delete operations may require confirmation from the proxy.
    """
    try:
        client = get_calendar_client()
        results: list[BulkOperationResult] = []
        success_count = 0
        error_count = 0

        for op in request.operations:
            try:
                if op.operation == BulkOperationType.DELETE:
                    await client.delete_event(
                        calendar_id=op.calendar_id,
                        event_id=op.event_id,
                        send_updates=op.send_updates,
                    )
                    results.append(BulkOperationResult(
                        event_id=op.event_id,
                        operation="delete",
                        success=True,
                    ))
                    success_count += 1

                elif op.operation == BulkOperationType.UPDATE:
                    if not op.updates:
                        results.append(BulkOperationResult(
                            event_id=op.event_id,
                            operation="update",
                            success=False,
                            error="No update data provided",
                        ))
                        error_count += 1
                        continue

                    await client.update_event(
                        calendar_id=op.calendar_id,
                        event_id=op.event_id,
                        event_data=op.updates,
                        send_updates=op.send_updates,
                    )
                    results.append(BulkOperationResult(
                        event_id=op.event_id,
                        operation="update",
                        success=True,
                    ))
                    success_count += 1

                elif op.operation == BulkOperationType.PATCH:
                    if not op.updates:
                        results.append(BulkOperationResult(
                            event_id=op.event_id,
                            operation="patch",
                            success=False,
                            error="No update data provided",
                        ))
                        error_count += 1
                        continue

                    await client.patch_event(
                        calendar_id=op.calendar_id,
                        event_id=op.event_id,
                        event_data=op.updates,
                        send_updates=op.send_updates,
                    )
                    results.append(BulkOperationResult(
                        event_id=op.event_id,
                        operation="patch",
                        success=True,
                    ))
                    success_count += 1

            except Exception as e:
                results.append(BulkOperationResult(
                    event_id=op.event_id,
                    operation=op.operation.value,
                    success=False,
                    error=format_proxy_error(e),
                ))
                error_count += 1

        return BulkActionsResponse(
            success=True,
            results=results,
            success_count=success_count,
            error_count=error_count,
        )
    except Exception as e:
        return BulkActionsResponse(
            success=False,
            results=[],
            success_count=0,
            error_count=0,
            error=format_proxy_error(e),
        )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("CALENDAR_AGENT_PORT", "8082"))
    uvicorn.run(app, host="0.0.0.0", port=port)
