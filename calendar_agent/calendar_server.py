"""Calendar Agent Server - A privacy-focused FastAPI server for Google Calendar.

This server acts as an intermediary between AI agents (like Claude Code) and the
Google Calendar API via a proxy server. The LLM endpoints process event content
locally and return generated text. List and search return EventSummary rows:
metadata plus the organizer's and creator's email addresses, no description and
no attendee list. The single-event detail routes (.../events/{event_id} and
.../respond) return the full Google event, description and attendee addresses
included.

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
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from . import __version__
from .calendar_utils import (
    READ_RESPONSE_STATUSES,
    RSVP_RESPONSES,
    RsvpResponse,
    RsvpState,
    attendee_entries,
    calendar_perspective,
    find_free_slots,
    get_event_time,
    get_time_range_rfc3339,
    is_all_day_event,
)
from .exceptions import (
    LLMError,
    ProxyAuthError,
    ProxyError,
    ProxyForbiddenError,
    ProxyNotFoundError,
    ProxyRequestError,
    ProxyTimeoutError,
    RsvpCalendarRefusedError,
)
from .llm_service import get_llm_service
from .proxy_client import get_calendar_client

load_dotenv()


# ============================================================================
# FastAPI App Setup
# ============================================================================

async def reject_unknown_query_params(request: Request) -> None:
    """422 any query-string key the matched route does not declare.

    FastAPI ignores undeclared query parameters, which is the same silent
    wrong answer as an ignored body field (issue #8): `?timeMin=..` on the
    events list ran an unbounded list, `?sendUpdates=all` on a write emailed
    nobody while reporting success. The allowed set is derived from the
    route's own declared query parameters, so there is no list to maintain.
    Applied to every route via the app-level dependency below.
    """
    route = request.scope.get("route")
    if route is None:
        return
    allowed = {param.alias for param in route.dependant.query_params}
    unknown = [key for key in request.query_params if key not in allowed]
    if unknown:
        raise RequestValidationError([
            {
                "type": "extra_forbidden",
                "loc": ("query", key),
                "msg": "Extra inputs are not permitted",
                "input": request.query_params[key],
            }
            for key in unknown
        ])


app = FastAPI(
    title="Calendar Agent",
    description="A privacy-focused FastAPI server for Google Calendar operations with AI agents",
    version=__version__,
    dependencies=[Depends(reject_unknown_query_params)],
)


@app.exception_handler(RequestValidationError)
async def validation_error_envelope(request: Request, exc: RequestValidationError):
    """Return 422s in the same success/error envelope as every other failure.

    FastAPI's default 422 body is {"detail": [...]} with no `success` or
    `error` key. With unknown fields rejected (issue #8) a 422 is the main way
    a misnamed field now fails, and callers that read the envelope (e.g. the
    briefing skills' `jq 'if .success then .briefing else "Error: " + .error
    end'`) would render a blank error. `detail` is kept for clients that read
    FastAPI's structure.
    """
    errors = jsonable_encoder(exc.errors())
    message = "; ".join(
        f"{'.'.join(str(part) for part in err.get('loc', ()))}: {err.get('msg', '')}"
        for err in errors
    )
    return JSONResponse(
        status_code=422,
        content={"success": False, "error": message, "detail": errors},
    )


# ============================================================================
# Pydantic Models - Requests
# ============================================================================


class StrictRequestModel(BaseModel):
    """Base for every request body model, top-level and nested: an unknown key
    is a 422, not a silent drop. Rationale in README "Error Responses"; issue #8.
    """

    model_config = ConfigDict(extra="forbid")


class EventDateTime(StrictRequestModel):
    """DateTime specification for calendar events."""
    date: str | None = Field(None, description="Date for all-day events (YYYY-MM-DD)")
    dateTime: str | None = Field(None, description="DateTime for timed events (RFC3339)")
    timeZone: str | None = Field(None, description="Timezone (e.g., 'America/New_York')")


class EventAttendee(StrictRequestModel):
    """Event attendee, as Google returns it (so a fetched list round-trips)."""
    email: str
    id: str | None = Field(None, description="Google's attendee id (round-tripped)")
    displayName: str | None = None
    responseStatus: str | None = None
    optional: bool | None = None
    organizer: bool | None = None
    self_: bool | None = Field(None, alias="self")
    resource: bool | None = Field(None, description="True for a booked room/resource")
    comment: str | None = Field(None, description="The attendee's response comment")
    additionalGuests: int | None = Field(None, ge=0, description="Extra guests")


class EventReminder(StrictRequestModel):
    """Event reminder."""
    method: str
    minutes: int


class EventReminders(StrictRequestModel):
    """Event reminders configuration."""
    useDefault: bool = True
    overrides: list[EventReminder] | None = None


# Server-populated, read-only keys Google puts on the events it returns. A
# caller doing fetch -> modify -> write sends them back verbatim; Google
# tolerates that, so these -- and ONLY these -- are stripped before the
# extra="forbid" check instead of being rejected. Documented in README.md
# next to the "Unknown fields are rejected" paragraph; keep the two in sync.
# Writable keys Google also returns but this server does not declare
# (`conferenceData`, `attachments`, `extendedProperties`, `source`,
# `anyoneCanAddSelf`) are deliberately NOT here: stripping them would make
# a write silently drop what the caller sent, and declaring them is new
# write surface. The README tells callers to remove them.
GOOGLE_READ_ONLY_EVENT_FIELDS: frozenset[str] = frozenset({
    "kind",
    "etag",
    "id",
    "htmlLink",
    "hangoutLink",
    "created",
    "updated",
    "creator",
    "organizer",
    "iCalUID",
    "sequence",
    "eventType",
    "recurringEventId",
    "originalStartTime",
    "privateCopy",
    "locked",
    "attendeesOmitted",
    "endTimeUnspecified",
    "outOfOfficeProperties",
    "workingLocationProperties",
})


class EventFields(StrictRequestModel):
    """Every writable Google event field this server forwards.

    Shared by the create, update (PUT) and patch bodies so that a field one
    write route accepts is accepted by all of them (issue #8, review round 2).
    """
    summary: str | None = Field(None, description="Event title")
    description: str | None = Field(None, description="Event description")
    location: str | None = Field(None, description="Event location")
    start: EventDateTime | None = Field(None, description="Start time")
    end: EventDateTime | None = Field(None, description="End time")
    attendees: list[EventAttendee] | None = Field(None, description="Event attendees")
    reminders: EventReminders | None = Field(None, description="Reminder settings")
    recurrence: list[str] | None = Field(None, description="Recurrence rules (RRULE)")
    colorId: str | None = Field(None, description="Color ID")
    # Declared so a fetched event round-trips, but `cancelled` is refused: a
    # cancel through an update reaches Google with none of the DELETE path's
    # re-read verification and no `outcome` (PR #12 review, item 1).
    status: Literal["confirmed", "tentative"] | None = Field(
        None,
        description="'confirmed' or 'tentative'. To cancel an event use DELETE, which "
        "verifies the deletion; 'cancelled' here is a 422",
    )
    transparency: str | None = Field(None, description="'opaque' or 'transparent'")
    visibility: str | None = Field(None, description="'default', 'public', 'private'")
    guestsCanInviteOthers: bool | None = None
    guestsCanModify: bool | None = None
    guestsCanSeeOtherGuests: bool | None = None

    @field_validator("status", mode="before")
    @classmethod
    def _cancel_goes_through_delete(cls, value: Any) -> Any:
        """Name the right route instead of a bare "not a permitted value"."""
        if value == "cancelled":
            raise ValueError(
                "'status: cancelled' is not accepted on a write. Cancel the event "
                "with DELETE /calendars/{calendar_id}/events/{event_id}, which "
                "verifies the deletion and reports an outcome"
            )
        return value

    @model_validator(mode="before")
    @classmethod
    def _strip_google_read_only_fields(cls, data: Any) -> Any:
        """Drop Google's server-populated keys so a fetched event round-trips.

        Runs before the extra="forbid" check. Only the named set is dropped;
        any other undeclared key is still rejected.
        """
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k not in GOOGLE_READ_ONLY_EVENT_FIELDS}
        return data

    def forwarded_data(self) -> dict[str, Any]:
        """The payload as sent upstream: nulls and read-only keys are gone.

        The one dump every write route and bulk `updates` uses, so "does this
        body forward anything" is judged the same way everywhere.
        """
        return self.model_dump(exclude_none=True, by_alias=True)


def empty_write_payload_message(operation: str) -> str:
    """The 422 text for an update/patch that would forward nothing. Shared by
    the single routes and bulk `updates` so both paths say the same thing."""
    return (
        f"'{operation}' requires a non-empty payload: nothing would be sent once "
        "null values and Google's read-only keys are dropped"
    )


def _require_forwardable_payload(event: EventFields, operation: str) -> EventFields:
    # Validated with the request, so it is a 422 before anything is sent
    # upstream. Judged on the forwarded payload, not the model object (which
    # is always truthy): `{}`, all-null values and read-only-only keys are
    # empty. Before PR #12's fix round the single routes sent Google an empty
    # body and reported success; bulk already refused (main's F3 rule).
    if not event.forwarded_data():
        raise ValueError(empty_write_payload_message(operation))
    return event


class EventCreateRequest(EventFields):
    """Request body for creating a new event."""


class EventUpdateRequest(EventCreateRequest):
    """Request body for updating an event (full replacement). Must forward
    at least one field; used by PUT and by bulk `update`."""

    @model_validator(mode="after")
    def _needs_a_payload(self) -> "EventUpdateRequest":
        return _require_forwardable_payload(self, "update")


class EventPatchRequest(EventFields):
    """Request body for partially updating an event. Must forward at least
    one field; used by PATCH and by bulk `patch`."""

    @model_validator(mode="after")
    def _needs_a_payload(self) -> "EventPatchRequest":
        return _require_forwardable_payload(self, "patch")


# Rendered into the field descriptions below so /openapi.json lists exactly
# the vocabularies the code enforces.
_RSVP_RESPONSE_LIST = ", ".join(f"'{v}'" for v in RSVP_RESPONSES)
_READ_STATUS_LIST = ", ".join(f"'{v}'" for v in READ_RESPONSE_STATUSES)


class RespondRequest(StrictRequestModel):
    """Request body for RSVPing to an event.

    The proxy writes the AUTHENTICATED USER's own attendee entry, found by
    email address (it does not trust Google's ``self`` flag). Because that is
    the opposite perspective from the read side -- whose ``calendar_*``
    fields describe the calendar being read -- this route accepts only the
    authenticated user's own calendar_id (``primary`` or the account's own
    calendar id) and refuses any other with 400.
    """
    response_status: RsvpResponse = Field(
        ...,
        description=(
            f"One of {_RSVP_RESPONSE_LIST}: the values of "
            "EventSummary.calendar_rsvp_state that can be written back "
            "('needsAction' and the derived states cannot). Written to the "
            "AUTHENTICATED USER's own attendee entry. The route accepts only "
            "the user's own calendar_id ('primary' or the account's own "
            "calendar id); any other calendar_id is refused with 400 and "
            "nothing is written."
        ),
    )


class SummarizeRequest(StrictRequestModel):
    """Request to summarize an event."""
    calendar_id: str = Field(..., description="Calendar ID containing the event")
    event_id: str = Field(..., description="Event ID to summarize")
    format: str = Field("brief", description="'brief' or 'detailed'")


class AskAboutRequest(StrictRequestModel):
    """Request to ask a question about an event."""
    calendar_id: str = Field(..., description="Calendar ID containing the event")
    event_id: str = Field(..., description="Event ID to ask about")
    question: str = Field(..., description="Question to ask about the event")


class BatchSummarizeRequest(StrictRequestModel):
    """Request to summarize multiple events."""
    calendar_id: str = Field(..., description="Calendar ID containing the events")
    event_ids: list[str] = Field(..., description="List of event IDs to summarize")
    triage: bool = Field(False, description="Include action type classification")


class FindFreeTimeRequest(StrictRequestModel):
    """Request to find free time slots."""
    calendar_id: str = Field(..., description="Calendar ID to check")
    time_min: str = Field(..., description="Start of search range (RFC3339)")
    time_max: str = Field(..., description="End of search range (RFC3339)")
    duration_minutes: int = Field(..., gt=0, description="Required meeting duration")
    working_hours_only: bool = Field(True, description="Only consider 9am-5pm")
    buffer_minutes: int = Field(0, ge=0, description="Buffer between meetings")
    prefer_morning: bool = Field(False, description="Prefer morning times")
    prefer_afternoon: bool = Field(False, description="Prefer afternoon times")


class AnalyzeScheduleRequest(StrictRequestModel):
    """Request to analyze schedule patterns."""
    calendar_id: str = Field(..., description="Calendar ID to analyze")
    time_min: str = Field(..., description="Start of analysis period (RFC3339)")
    time_max: str = Field(..., description="End of analysis period (RFC3339)")
    analysis_type: str = Field(
        "overview",
        description="Type: 'overview', 'workload', 'patterns', 'conflicts'"
    )


class PrepareBriefingRequest(StrictRequestModel):
    """Request to prepare a schedule briefing."""
    calendar_id: str = Field(..., description="Calendar ID for briefing")
    briefing_type: str = Field("daily", description="'daily' or 'weekly'")
    time_min: str | None = Field(None, description="Start time (defaults to now)")
    time_max: str | None = Field(None, description="End time (defaults based on type)")


class SearchFilters(StrictRequestModel):
    """Filters for event search."""
    query: str | None = Field(None, description="Free text search")
    time_min: str | None = Field(None, description="Start of time range (RFC3339)")
    time_max: str | None = Field(None, description="End of time range (RFC3339)")
    max_results: int = Field(100, ge=1, le=500, description="Maximum results")
    order_by: str | None = Field(None, description="'startTime' or 'updated'")
    show_deleted: bool = Field(False, description="Include deleted events")


class SearchRequest(StrictRequestModel):
    """Request to search events.

    Accepts the filter keys either nested under `filters` (the shape
    /openapi.json describes) or flat at the top level (kept for compatibility
    with earlier callers; the installed skills nest theirs); see
    `_fold_flat_filter_keys`.
    """
    calendar_id: str = Field(..., description="Calendar ID to search")
    filters: SearchFilters = Field(default_factory=SearchFilters)

    @model_validator(mode="before")
    @classmethod
    def _fold_flat_filter_keys(cls, data: Any) -> Any:
        """Move top-level filter keys into `filters` (issue #8, items 2-4).

        - The allowlist is `SearchFilters.model_fields`, never a hand copy,
          so adding a filter field keeps the flat shape working.
        - Folded keys are consumed from the top level; anything else left
          there still hits extra="forbid", so a true typo (`timeMin`) is a 422.
        - Merge is per field, nested wins: a key present in `filters` beats
          the same key at the top level.
        - An explicit null at the top level means "not supplied".
        - `filters` is only touched when it is absent/null or a mapping; any
          other value is left for SearchFilters to reject cleanly (no 500).
        """
        if not isinstance(data, dict):
            return data
        flat = {k: v for k, v in data.items() if k in SearchFilters.model_fields}
        if not flat:
            return data
        rest = {k: v for k, v in data.items() if k not in flat}
        nested = rest.get("filters")
        if nested is None:
            nested = {}
        elif not isinstance(nested, dict):
            return data  # let the field validator report the bad `filters`
        merged = {k: v for k, v in flat.items() if v is not None}
        merged.update(nested)
        return {**rest, "filters": merged}


class BulkOperationType(str, Enum):
    """Types of bulk operations."""
    UPDATE = "update"
    DELETE = "delete"
    PATCH = "patch"


class _BulkOperationBase(StrictRequestModel):
    """Fields every bulk operation carries."""
    event_id: str
    calendar_id: str
    send_updates: str | None = Field(None, description="'all', 'externalOnly', 'none'")


class BulkDeleteOperation(_BulkOperationBase):
    """Delete one event. Carries no `updates` -- sending one is a 422, so a
    mis-set operation can't delete while its payload is silently ignored."""
    operation: Literal[BulkOperationType.DELETE]


class _BulkWriteOperation(_BulkOperationBase):
    """An update or patch: `updates` is required. Subclasses narrow `updates`
    to the matching single-event body, so an unknown key inside it is rejected
    at the same depth as on the single-event routes (issue #8) and an empty
    payload is refused by that body's own validator -- with the request, so a
    malformed operation anywhere in the batch is a 422 before any operation
    runs (F3: discovering it mid-loop left the envelope's status depending on
    operation order)."""
    operation: BulkOperationType
    updates: EventFields = Field(..., description="Update data")

    @property
    def event_data(self) -> dict[str, Any]:
        """The payload as forwarded: same dump as the single-event routes."""
        return self.updates.forwarded_data()


class BulkUpdateOperation(_BulkWriteOperation):
    """Full replacement of one event; `updates` is validated exactly like PUT."""
    operation: Literal[BulkOperationType.UPDATE]
    updates: EventUpdateRequest = Field(..., description="Full event body")


class BulkPatchOperation(_BulkWriteOperation):
    """Partial update of one event; `updates` is validated exactly like PATCH."""
    operation: Literal[BulkOperationType.PATCH]
    updates: EventPatchRequest = Field(..., description="Fields to change")


_BULK_OPERATION_NAMES = ", ".join(f"'{op.value}'" for op in BulkOperationType)
_BULK_OPERATION_VALUES = frozenset(op.value for op in BulkOperationType)


def _check_bulk_operation_tag(item: Any) -> Any:
    """Report an absent or unknown `operation` in fixed words.

    Runs before the discriminated union below, whose own message for a bad
    tag spells the expected tags as Python enum reprs
    (`<BulkOperationType.DELETE: 'delete'>`) -- in `error`, `detail[].msg`
    and `ctx.expected_tags` alike (PR #12 review, item 3). Anything that is
    not a mapping is left for the union to reject.
    """
    if not isinstance(item, dict):
        return item
    tag = item.get("operation")
    # The `isinstance` guard keeps an unhashable tag (a list or object) from
    # raising TypeError out of the set lookup, which would be a 500 not a 422.
    if not (isinstance(tag, str) and tag in _BULK_OPERATION_VALUES):
        got = f"got {tag!r}" if "operation" in item else "it is missing"
        raise ValueError(f"'operation' must be one of {_BULK_OPERATION_NAMES}; {got}")
    return item


# Discriminated on `operation`, so `updates` is typed by the operation it
# accompanies and an unknown key inside it is rejected at the same depth as
# on the single-event routes (issue #8).
BulkOperation = Annotated[
    BulkDeleteOperation | BulkUpdateOperation | BulkPatchOperation,
    Field(discriminator="operation"),
    BeforeValidator(_check_bulk_operation_tag),
]


class BulkActionsRequest(StrictRequestModel):
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
    """Summary of an event: metadata plus the organizer's and creator's
    addresses; no description and no attendee list.

    The ``calendar_*`` fields describe the calendar named by ``calendar_id``,
    because Google's ``organizer.self`` / ``attendees[].self`` flags mark
    "the calendar on which this copy of the event appears" -- not the
    caller. They describe the authenticated user only when reading the
    user's own calendar (``primary`` or its own address); on a colleague's
    or group calendar they describe that calendar. POST .../respond takes
    the opposite perspective -- it writes the authenticated user's own entry
    -- and for that reason accepts only the user's own calendar_id, refusing
    any other with 400.
    """
    id: str
    calendar_id: str
    summary: str
    start: str
    end: str
    location: str | None = None
    attendee_count: int
    is_all_day: bool
    status: str | None = Field(
        None,
        description=(
            "Google event status: 'confirmed', 'tentative', or 'cancelled'. "
            "On a plain GET .../events with single_events=false, cancelled "
            "rows are the stubs Google keeps for deleted instances of a "
            "recurring series: empty start/end, no organizer, no attendees, "
            "calendar_rsvp_state 'unknown'. POST /search with "
            "filters.show_deleted=true forwards showDeleted to Google (with "
            "singleEvents fixed to true); the cancelled rows it returns are "
            "whatever Google sends for them, and are classified like any "
            "other row -- they may carry real start/end, an organizer and "
            "attendees."
        ),
    )
    html_link: str | None = None
    # Organizer and creator addresses are exposed by decision (2026-09-03);
    # attendee addresses are not -- attendee_count stays a count.
    organizer_email: str | None = Field(
        None,
        description=(
            "The organizer's address. For an event created directly on a "
            "group calendar this is the group calendar's own id (Google makes "
            "the calendar the organizer); see creator_email for the person. "
            "null when the event carries no organizer (e.g. a cancelled "
            "recurring-instance stub)."
        ),
    )
    creator_email: str | None = Field(
        None,
        description=(
            "The address of the account that created the event, as Google "
            "reports it. Usually equals organizer_email; differs for events "
            "created on a group calendar (organizer = the calendar) or moved "
            "between calendars. null when absent."
        ),
    )
    calendar_is_organizer: bool = Field(
        ...,
        description=(
            "Whether the calendar named by calendar_id organizes this event "
            "(Google's organizer.self). On a colleague's calendar this is "
            "about the colleague; it is about the authenticated user only "
            "when calendar_id is the user's own calendar."
        ),
    )
    calendar_rsvp_state: RsvpState = Field(
        ...,
        description=(
            "The RSVP of the attendee entry belonging to the calendar named "
            "by calendar_id (Google's attendees[].self), classified. "
            f"{_READ_STATUS_LIST}: Google's responseStatus, verbatim. "
            "'organizer_no_rsvp': the calendar organizes the event and has no "
            "responseStatus (no entry of its own, or an entry without one) -- "
            "its own event, nothing to answer. 'not_attendee': the calendar "
            "neither organizes the event nor appears in its attendee list. "
            "'unknown': the calendar's entry carries a responseStatus this "
            "service does not recognise (organizer or not); or the calendar "
            "does not organize the event and its own entry has no "
            "responseStatus at all; or the event has no organizer and no "
            "attendees (a cancelled recurring-instance stub). On a "
            "colleague's calendar this is the colleague's RSVP, "
            f"not the authenticated user's. Only {_RSVP_RESPONSE_LIST} can be "
            "sent back to POST .../respond, and only a value read from the "
            "user's own calendar: that route refuses any other calendar_id."
        ),
    )


class EventsListResponse(BaseModel):
    """Response for listing events."""
    success: bool
    events: list[EventSummary]
    next_page_token: str | None = None
    error: str | None = None


class EventDetailResponse(BaseModel):
    """Full event details (for get/create/update/patch/respond).

    ``event`` is the Google event as the proxy returned it, including
    ``description`` and the ``attendees`` list with addresses -- the
    summary-only privacy rule applies to list/search rows, not here.
    """
    success: bool
    event: dict[str, Any] | None = None
    error: str | None = None
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Caveats about a successful result. Nothing in this service "
            "appends to it today, so it is empty on every route: POST "
            ".../respond used to warn when calendar_id was not the literal "
            "'primary', and now refuses a calendar_id that is not the "
            "authenticated user's own with 400 instead of completing it."
        ),
    )


class OperationOutcome(str, Enum):
    """What is actually known about one mutation.

    ``UNKNOWN`` is a first-class outcome, not a flavour of failure: a mutation
    that timed out may still be applied when the operator approves it later
    (issue #4). Treating it as failed is what produced the duplicate-event
    incident. ``NOT_ATTEMPTED`` marks a bulk operation that was never sent
    because an earlier one in the same batch came back unknown.
    """
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"
    NOT_ATTEMPTED = "not_attempted"


class ActionResponse(BaseModel):
    """Response for action endpoints.

    ``outcome`` puts the tri-state on the single-event envelope too, so a
    caller reading only the body can tell "rejected" from "may still apply".
    """
    success: bool
    outcome: OperationOutcome
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
    outcome: OperationOutcome
    error: str | None = None


class BulkActionsResponse(BaseModel):
    """Response for bulk operations.

    ``success`` is true only when every operation is known to have happened;
    ``unknown_count`` covers operations whose outcome must be verified by
    re-reading the event before anything is done about them;
    ``not_attempted_count`` covers operations never sent because an earlier
    one came back unknown.
    """
    success: bool
    results: list[BulkOperationResult]
    success_count: int
    error_count: int
    unknown_count: int = 0
    not_attempted_count: int = 0
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
    if isinstance(e, ProxyNotFoundError):
        return f"Not found: {e}"
    if isinstance(e, ProxyTimeoutError):
        return f"Outcome unknown: {e}"
    if isinstance(e, ProxyRequestError):
        return f"Proxy rejected the request ({e.status_code}): {e}"
    if isinstance(e, ProxyError):
        return f"Proxy error: {e}"
    return str(e)


def error_status_code(e: Exception) -> int:
    """Map an error to the HTTP status this server should return.

    The success/error envelope stays in the body; the status code must agree
    with it (issue #4): 403 passes through an operator rejection or policy
    block; 404 an absent calendar or event (ProxyNotFoundError); 400 passes
    through a proxy 400 with its message (ProxyRequestError, e.g. /respond
    when the authenticated user is not an attendee) and also covers this
    server's own refusal to RSVP through a calendar that is not the
    authenticated user's (RsvpCalendarRefusedError); 504 marks a timed-out
    call whose outcome is unknown; 502 covers upstream proxy/LLM failures,
    including a proxy 401 (this service's own key rejected) and any other
    proxy 4xx; 500 anything unexpected.
    """
    if isinstance(e, ProxyForbiddenError):
        return 403
    if isinstance(e, ProxyNotFoundError):
        return 404
    if isinstance(e, RsvpCalendarRefusedError):
        return 400
    if isinstance(e, ProxyRequestError) and e.status_code == 400:
        return 400
    if isinstance(e, ProxyTimeoutError):
        return 504
    if isinstance(e, ProxyAuthError | ProxyError | LLMError):
        return 502
    return 500


def error_response(body: BaseModel, e: Exception) -> JSONResponse:
    """Return an error envelope with a status code that agrees with it."""
    return JSONResponse(status_code=error_status_code(e), content=body.model_dump())


# Most to least urgent for the caller. An unknown outcome outranks every
# definite failure, however many of each there are: the timed-out operation is
# the one that can still change the calendar, so the caller must verify before
# issuing anything compensating (issue #4). A rejection outranks an upstream
# fault, which outranks an absent event. Position in the batch never matters.
BULK_STATUS_PRECEDENCE = (504, 403, 502, 500, 404)


def bulk_status_code(codes: list[int]) -> int:
    """Pick the status for a bulk response from its per-operation statuses."""
    failures = [code for code in codes if code != 200]
    if not failures:
        return 200
    for status in BULK_STATUS_PRECEDENCE:
        if status in failures:
            return status
    return failures[0]


@dataclass(frozen=True)
class OperationVerdict:
    """What one mutation established, with the status code that agrees."""
    outcome: OperationOutcome
    status_code: int
    message: str
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.outcome is OperationOutcome.SUCCEEDED


async def event_presence(client, calendar_id: str, event_id: str) -> str:
    """Re-read an event: ``"gone"``, ``"present"`` or ``"inconclusive"``.

    Google keeps a deleted event readable with ``status: cancelled`` for a
    while before answering 404/410, so both count as gone.
    """
    try:
        event = await client.get_event(calendar_id, event_id)
    except ProxyNotFoundError:
        return "gone"
    except Exception:
        return "inconclusive"
    status = event.get("status") if isinstance(event, dict) else None
    if status is None:
        return "inconclusive"
    return "gone" if status == "cancelled" else "present"


async def verified_delete(
    client, calendar_id: str, event_id: str, send_updates: str | None = None
) -> OperationVerdict:
    """Delete an event and decide by re-reading it, not by the proxy's answer.

    The proxy's response to a gated delete is a claim: it can be a success for
    work that has not happened (the 2026-08-07 incident) or a timeout for work
    that is still going to happen. Only the re-read is evidence (issue #4).
    A 403 or 404 needs no re-read: the proxy dropped the request, or there was
    nothing to delete.
    """
    try:
        await client.delete_event(
            calendar_id=calendar_id, event_id=event_id, send_updates=send_updates
        )
    except ProxyForbiddenError as e:
        return OperationVerdict(
            OperationOutcome.FAILED, 403,
            "Deletion blocked or rejected by operator", format_proxy_error(e),
        )
    except ProxyNotFoundError as e:
        return OperationVerdict(
            OperationOutcome.FAILED, 404,
            "No such event: nothing was deleted", format_proxy_error(e),
        )
    except ProxyTimeoutError as e:
        claim, claim_error = "unknown", format_proxy_error(e)
        claim_status = 504
    except Exception as e:
        claim, claim_error = "failed", format_proxy_error(e)
        claim_status = error_status_code(e)
    else:
        claim, claim_error, claim_status = "deleted", None, 200

    presence = await event_presence(client, calendar_id, event_id)

    if presence == "gone":
        message = "Event deleted successfully"
        if claim != "deleted":
            message += f" (confirmed by re-read; the delete itself reported: {claim_error})"
        return OperationVerdict(OperationOutcome.SUCCEEDED, 200, message)

    if presence == "present":
        if claim == "unknown":
            return OperationVerdict(
                OperationOutcome.UNKNOWN, 504,
                "Deletion outcome unknown: no response before timeout and the "
                "event is still present; it may yet be applied when the operator "
                "approves it. Re-verify before acting",
                claim_error,
            )
        if claim == "failed":
            return OperationVerdict(
                OperationOutcome.FAILED, claim_status, "Failed to delete event", claim_error
            )
        return OperationVerdict(
            OperationOutcome.FAILED, 502,
            "Proxy reported the deletion complete, but the event is still present",
            "Proxy claimed success for a deletion that has not happened: the event "
            "is still present on re-read",
        )

    return OperationVerdict(
        OperationOutcome.UNKNOWN, 504,
        "Deletion outcome unknown: could not re-read the event, so nothing is "
        "established. Re-verify before acting",
        claim_error or "The delete was accepted but the verifying re-read failed",
    )


async def gated_write(
    client, op: BulkUpdateOperation | BulkPatchOperation
) -> OperationVerdict:
    """Run a bulk update/patch and map its answer to a verdict."""
    write = (
        client.update_event
        if op.operation == BulkOperationType.UPDATE
        else client.patch_event
    )
    try:
        await write(
            calendar_id=op.calendar_id,
            event_id=op.event_id,
            event_data=op.event_data,
            send_updates=op.send_updates,
        )
    except ProxyTimeoutError as e:
        return OperationVerdict(
            OperationOutcome.UNKNOWN, 504,
            f"{op.operation.value} outcome unknown", format_proxy_error(e),
        )
    except Exception as e:
        return OperationVerdict(
            OperationOutcome.FAILED, error_status_code(e),
            f"{op.operation.value} failed", format_proxy_error(e),
        )
    return OperationVerdict(OperationOutcome.SUCCEEDED, 200, f"{op.operation.value} applied")


def bulk_error_summary(results: list[BulkOperationResult]) -> str | None:
    """One sentence for the envelope's ``error`` when a batch did not fully succeed."""
    counts = {
        outcome: sum(1 for r in results if r.outcome is outcome)
        for outcome in OperationOutcome
    }
    not_ok = len(results) - counts[OperationOutcome.SUCCEEDED]
    if not_ok == 0:
        return None
    parts = []
    if counts[OperationOutcome.UNKNOWN]:
        parts.append(
            f"{counts[OperationOutcome.UNKNOWN]} outcome unknown "
            "(may still be applied; verify before acting)"
        )
    if counts[OperationOutcome.FAILED]:
        parts.append(f"{counts[OperationOutcome.FAILED]} failed")
    if counts[OperationOutcome.NOT_ATTEMPTED]:
        parts.append(f"{counts[OperationOutcome.NOT_ATTEMPTED]} not attempted")
    return (
        f"{not_ok} of {len(results)} operations did not succeed: "
        f"{', '.join(parts)}; see results"
    )


# The authenticated account's own calendar id, as the proxy reports it
# (GET /calendars/primary answers with the account's address in ``id``).
# Cached for the life of the process: the credentials the proxy holds do not
# change under a running server, so this costs ONE proxy read in total, not
# one per RSVP.
_authenticated_calendar_id: str | None = None


def reset_authenticated_calendar_id() -> None:
    """Drop the cached authenticated calendar id.

    Only the tests call this; nothing in the running server invalidates the
    cache, because the proxy's credentials are fixed for the process.
    """
    global _authenticated_calendar_id
    _authenticated_calendar_id = None


async def get_authenticated_calendar_id() -> str:
    """Return the authenticated account's own calendar id, reading it once.

    Raises ProxyError if the proxy answers without an ``id`` -- there is then
    nothing to compare a calendar_id against, and guessing would defeat the
    check that calls this. Proxy failures propagate as their own exceptions.
    """
    global _authenticated_calendar_id
    if _authenticated_calendar_id is None:
        calendar = await get_calendar_client().get_calendar("primary")
        calendar_id = calendar.get("id") if isinstance(calendar, dict) else None
        if not isinstance(calendar_id, str) or not calendar_id:
            raise ProxyError(
                "The proxy's primary calendar carries no 'id', so the "
                "authenticated account's own calendar cannot be identified."
            )
        _authenticated_calendar_id = calendar_id
    return _authenticated_calendar_id


async def require_own_calendar_for_rsvp(calendar_id: str) -> None:
    """Raise RsvpCalendarRefusedError unless ``calendar_id`` is the
    authenticated user's own calendar (Brian's decision, 2026-09-04).

    Read and write take opposite perspectives on this route's calendar_id:
    the read fields describe the calendar named, POST .../respond always
    writes the authenticated user's own attendee entry. On a group or
    colleague calendar those are different people, so a "you have not
    responded" read there says nothing about the entry an RSVP would change.
    Restricting the route to the user's own calendar is what keeps the two
    talking about the same attendee entry.

    ``primary`` is accepted without a lookup, so the ordinary path makes no
    extra proxy call. Any other id needs the account's own calendar id, read
    once per process from GET /calendars/primary. If that read fails -- for
    any reason, timeout included -- the failure is re-raised as a plain
    ProxyError (502): it happened before anything was sent, so it must not
    be reported in the mutation's vocabulary (a timeout there is not an
    "outcome unknown", and a 404 there is not "no such event").
    """
    if calendar_id.casefold() == "primary":
        return
    try:
        own_calendar_id = await get_authenticated_calendar_id()
    except ProxyError as e:
        raise ProxyError(
            "The ownership check for this RSVP could not be performed: reading "
            f"the authenticated account's own calendar (GET /calendars/primary) "
            f"failed ({e}). The RSVP was not attempted and nothing was sent."
        ) from e
    if calendar_id.casefold() == own_calendar_id.casefold():
        return
    # ESCAPE HATCH (deliberately not implemented): if a deliberate RSVP on a
    # shared calendar is ever wanted, add an explicit opt-in field to
    # RespondRequest and check it here -- never widen the comparison above.
    raise RsvpCalendarRefusedError(
        f"Refusing to RSVP through calendar '{calendar_id}': it is not the "
        "authenticated user's own calendar. This route always writes the "
        "authenticated user's attendee entry, while that calendar's read "
        "fields (calendar_rsvp_state, calendar_is_organizer) describe the "
        "calendar itself -- so an RSVP state read there is not the entry this "
        "would change. Send the RSVP on the authenticated user's own calendar "
        "instead: POST /calendars/primary/events/{event_id}/respond."
    )


def event_to_summary(event: dict[str, Any], calendar_id: str) -> EventSummary:
    """Convert a full event to an EventSummary: metadata plus the organizer's
    and creator's addresses; no description, no attendee list.

    The ``calendar_*`` fields are derived from Google's ``self`` flags and so
    describe ``calendar_id`` -- the calendar this copy of the event sits on.
    """
    organizer = event.get("organizer")
    creator = event.get("creator")
    perspective = calendar_perspective(event)

    return EventSummary(
        id=event.get("id", ""),
        calendar_id=calendar_id,
        summary=event.get("summary", "Untitled Event"),
        # A cancelled recurring-instance stub has no start/end: empty strings.
        start=get_event_time(event.get("start")),
        end=get_event_time(event.get("end")),
        location=event.get("location"),
        # A non-dict organizer/creator reads as absent and non-dict attendee
        # rows are skipped, rather than raising; the count uses the same
        # definition of "attendee" as calendar_rsvp_state.
        attendee_count=len(attendee_entries(event)),
        is_all_day=is_all_day_event(event),
        status=event.get("status"),
        html_link=event.get("htmlLink"),
        organizer_email=organizer.get("email") if isinstance(organizer, dict) else None,
        creator_email=creator.get("email") if isinstance(creator, dict) else None,
        calendar_is_organizer=perspective.is_organizer,
        calendar_rsvp_state=perspective.rsvp_state,
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
        return error_response(
            CalendarsResponse(success=False, calendars=[], error=format_proxy_error(e)), e
        )


@app.get("/calendars/{calendar_id}", response_model=CalendarDetailResponse, tags=["calendars"])
async def get_calendar(calendar_id: str):
    """Get metadata for a specific calendar."""
    try:
        client = get_calendar_client()
        calendar = await client.get_calendar(calendar_id)
        return CalendarDetailResponse(success=True, calendar=calendar)
    except Exception as e:
        return error_response(
            CalendarDetailResponse(success=False, calendar=None, error=format_proxy_error(e)), e
        )


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
        return error_response(
            EventsListResponse(success=False, events=[], error=format_proxy_error(e)), e
        )


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
        event_data = event.forwarded_data()

        result = await client.create_event(
            calendar_id=calendar_id,
            event_data=event_data,
            send_updates=send_updates,
        )
        return EventDetailResponse(success=True, event=result)
    except Exception as e:
        return error_response(
            EventDetailResponse(success=False, event=None, error=format_proxy_error(e)), e
        )


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
        return error_response(
            EventDetailResponse(success=False, event=None, error=format_proxy_error(e)), e
        )


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
        event_data = event.forwarded_data()

        result = await client.update_event(
            calendar_id=calendar_id,
            event_id=event_id,
            event_data=event_data,
            send_updates=send_updates,
        )
        return EventDetailResponse(success=True, event=result)
    except Exception as e:
        return error_response(
            EventDetailResponse(success=False, event=None, error=format_proxy_error(e)), e
        )


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
        event_data = event.forwarded_data()

        result = await client.patch_event(
            calendar_id=calendar_id,
            event_id=event_id,
            event_data=event_data,
            send_updates=send_updates,
        )
        return EventDetailResponse(success=True, event=result)
    except Exception as e:
        return error_response(
            EventDetailResponse(success=False, event=None, error=format_proxy_error(e)), e
        )


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
    """Delete an event, and verify it by re-reading before answering.

    The proxy requires operator confirmation for deletes: the request blocks
    while a human approves it. A 403 here means the operator rejected the
    deletion (or never answered); a 504 means the outcome is unknown — this
    server got no response before its timeout and the event is still present,
    or the verifying re-read failed. A 200 is only ever returned after the
    re-read shows the event gone (404/410, or ``status: cancelled``).
    """
    try:
        client = get_calendar_client()
        verdict = await verified_delete(client, calendar_id, event_id, send_updates)
    except Exception as e:
        verdict = OperationVerdict(
            OperationOutcome.FAILED, error_status_code(e),
            "Failed to delete event", format_proxy_error(e),
        )
    body = ActionResponse(
        success=verdict.success,
        outcome=verdict.outcome,
        message=verdict.message,
        error=verdict.error,
    )
    if verdict.status_code == 200:
        return body
    return JSONResponse(status_code=verdict.status_code, content=body.model_dump())


@app.post(
    "/calendars/{calendar_id}/events/{event_id}/respond",
    response_model=EventDetailResponse,
    tags=["events"]
)
async def respond_to_event(
    calendar_id: str,
    event_id: str,
    request: RespondRequest,
):
    """RSVP to an event by setting the authenticated user's own responseStatus.

    ``calendar_id`` must be the authenticated user's own calendar -- the
    literal ``primary`` or the account's own calendar id (its address).
    Any other calendar is refused with 400 and the request is not forwarded
    (Brian's decision, 2026-09-04). The reason is that the two sides take
    opposite perspectives on the same ``calendar_id``: the read fields
    (``calendar_rsvp_state``, ``calendar_is_organizer``) describe the
    calendar being read, while this route always writes the authenticated
    user's own attendee entry. On a group or colleague calendar those are
    different people, so a ``needsAction`` read there is not the entry an
    RSVP would change. Refusing keeps read and write on one calendar, where
    they agree. ``primary`` is accepted without any lookup; other ids are
    compared against the account's own calendar id, read once per process
    from GET /calendars/primary. If that read fails, the answer is 502 with
    a message saying the ownership check could not be performed and nothing
    was sent -- never 504 "outcome unknown" or 404, because no RSVP was
    attempted.

    Forwards to the proxy's dedicated /respond route. The proxy resolves the
    authenticated account's email address (from its primary calendar), finds
    that address in the event's attendee list -- it does not trust Google's
    ``self`` flag -- patches only that entry, and sends no invitations or
    notifications. Valid values for response_status are 'accepted',
    'declined', or 'tentative'.

    If the authenticated user is not an attendee, the proxy answers 400
    ("You are not an attendee of this event; cannot RSVP."), which this
    server passes through as 400 with that message in ``error``. Like other
    mutations, the proxy blocks while a human operator approves the RSVP:
    403 means it was rejected (or the operator never answered); 504 means
    no response before this server's timeout and the outcome is unknown --
    verify by re-reading the event.
    """
    try:
        await require_own_calendar_for_rsvp(calendar_id)
        client = get_calendar_client()
        result = await client.respond_to_event(
            calendar_id,
            event_id,
            request.response_status,
        )
        return EventDetailResponse(success=True, event=result)
    except Exception as e:
        return error_response(
            EventDetailResponse(success=False, event=None, error=format_proxy_error(e)), e
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
        return error_response(
            LLMResponse(success=False, data=None, error=format_proxy_error(e)), e
        )


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
        return error_response(
            LLMResponse(success=False, data=None, error=format_proxy_error(e)), e
        )


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
        return error_response(
            LLMResponse(success=False, data=None, error=format_proxy_error(e)), e
        )


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
        return error_response(
            LLMResponse(success=False, data=None, error=format_proxy_error(e)), e
        )


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
        return error_response(
            LLMResponse(success=False, data=None, error=format_proxy_error(e)), e
        )


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
        return error_response(
            LLMResponse(success=False, data=None, error=format_proxy_error(e)), e
        )


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
        return error_response(
            EventsListResponse(success=False, events=[], error=format_proxy_error(e)), e
        )


@app.post("/bulk-actions", response_model=BulkActionsResponse, tags=["operations"])
async def bulk_actions(request: BulkActionsRequest):
    """Execute multiple operations on events in a single request.

    Supports update, patch, and delete operations. Operations are executed
    sequentially; deletes are verified by re-reading the event before they
    are reported as succeeded.

    Every mutation blocks while a human operator approves it, so a bulk
    request can legitimately take several minutes (up to ~330s per gated
    operation). Once one operation comes back with an unknown outcome, the
    operator is not answering: the remaining operations are not sent (each
    would hold the connection another full timeout and enqueue another
    approval) and are reported as ``not_attempted``.
    """
    try:
        client = get_calendar_client()
        results: list[BulkOperationResult] = []
        codes: list[int] = []
        stop_reason: str | None = None

        for op in request.operations:
            if stop_reason is not None:
                results.append(BulkOperationResult(
                    event_id=op.event_id,
                    operation=op.operation.value,
                    success=False,
                    outcome=OperationOutcome.NOT_ATTEMPTED,
                    error=f"Not attempted: {stop_reason}",
                ))
                continue

            if op.operation == BulkOperationType.DELETE:
                verdict = await verified_delete(
                    client, op.calendar_id, op.event_id, op.send_updates
                )
            else:
                verdict = await gated_write(client, op)

            results.append(BulkOperationResult(
                event_id=op.event_id,
                operation=op.operation.value,
                success=verdict.success,
                outcome=verdict.outcome,
                error=verdict.error,
            ))
            codes.append(verdict.status_code)
            if verdict.outcome is OperationOutcome.UNKNOWN:
                stop_reason = (
                    f"the {op.operation.value} of {op.event_id} timed out with its "
                    "outcome unknown (the operator is not answering), so nothing "
                    "further was queued for approval"
                )

        body = BulkActionsResponse(
            success=all(r.success for r in results),
            results=results,
            success_count=sum(r.outcome is OperationOutcome.SUCCEEDED for r in results),
            error_count=sum(r.outcome is OperationOutcome.FAILED for r in results),
            unknown_count=sum(r.outcome is OperationOutcome.UNKNOWN for r in results),
            not_attempted_count=sum(
                r.outcome is OperationOutcome.NOT_ATTEMPTED for r in results
            ),
            error=bulk_error_summary(results),
        )
        status = bulk_status_code(codes)
        if status == 200:
            return body
        return JSONResponse(status_code=status, content=body.model_dump())
    except Exception as e:
        return error_response(
            BulkActionsResponse(
                success=False,
                results=[],
                success_count=0,
                error_count=0,
                unknown_count=0,
                error=format_proxy_error(e),
            ),
            e,
        )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("CALENDAR_AGENT_PORT", "8082"))
    uvicorn.run(app, host="0.0.0.0", port=port)
