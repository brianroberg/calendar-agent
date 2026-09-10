# Calendar Agent

A privacy-focused FastAPI server that wraps the Google Calendar API for use with AI agents.

## Overview

Calendar Agent acts as an intermediary between AI orchestrators (like Claude Code) and the Google Calendar API via a proxy server. Event content is processed locally for the LLM endpoints. The list and search endpoints return `EventSummary` rows -- metadata plus the organizer's and creator's email addresses, no description and no attendee list; the single-event detail routes (`GET`/`POST`/`PUT`/`PATCH .../events/{event_id}` and `POST .../respond`) return the full Google event, including `description` and `attendees[]` with addresses (see Key Privacy Features).

**Key Privacy Features:**
- The LLM endpoints (`/summarize`, `/ask-about`, `/batch-summarize`,
  `/prepare-briefing`, ...) process event content locally and return only
  the generated text.
- List and search rows (`EventSummary`) expose metadata (IDs, dates,
  titles, location, status, the Google Calendar link, attendee counts) plus
  the **organizer's and creator's email addresses** -- and nothing else
  about attendees: no attendee list, no attendee addresses. This is pinned
  by tests on the `EventSummary` field set.
- The single-event detail routes are **not** summaries: they return the
  Google event as the proxy sent it, description and attendee addresses
  included (see their response examples below).
  The two addresses are included by decision (2026-09-03) because on a
  group calendar Google makes the calendar itself the organizer, so the
  creator's address is the only way to know which person created an event.
- LLM processing happens locally via MLX or can use hosted APIs

**Architecture:**
```
Orchestrator Agent <-> Calendar Agent (local) <-> API Proxy <-> Google Calendar API
                            |
                      Local LLM (MLX)
```

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Access to an api-proxy server instance
- (Optional) Local LLM server (MLX-based) for AI features

### Installation

```bash
# Clone the repository
cd calendar-agent

# Install dependencies with uv
uv sync

# Install dev dependencies
uv sync --dev
```

### Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Required environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `PROXY_URL` | URL of the api-proxy server | `http://localhost:8000` |
| `PROXY_API_KEY` | API key for proxy authentication | (required) |
| `LLM_URL` | URL of the local LLM server | `http://localhost:8080/v1/chat/completions` |
| `LLM_MODEL` | Model name for LLM requests | `qwen/qwen3-14b` |
| `CALENDAR_AGENT_PORT` | Port for the calendar agent server | `8082` |
| `PROXY_CONFIRMATION_WINDOW` | How long (seconds) api-proxy waits for a human to approve a mutation — **must mirror the proxy's `--confirmation-timeout` by hand** (see [Coupling with api-proxy](#coupling-with-api-proxy)). Must be finite and positive | `300` |
| `PROXY_CONFIRM_TIMEOUT` | Client timeout (seconds) for mutations, which block in the proxy while a human operator approves them. Must be at least `PROXY_CONFIRMATION_WINDOW` + 30s; the server refuses to start otherwise, and refuses `nan`/`inf` | `330` (window + 30) |

#### Coupling with api-proxy

Mutations block inside api-proxy until a human approves them, for at most the
proxy's `--confirmation-timeout` (300s by default; `0` or less means *wait
forever*). This server's mutation timeout must outlive that window, or every
approval and rejection arrives after the client has given up and the
operation completes unobserved (issue #4). The proxy does not publish its
window on `/health`, so the check is only as honest as its inputs: **if you
change `--confirmation-timeout` on the proxy, change `PROXY_CONFIRMATION_WINDOW`
here to match** (and set `PROXY_CONFIRM_TIMEOUT` at least 30s above it, or
leave it unset to follow automatically). A proxy configured to wait forever
cannot be mirrored — set a finite window on both sides.

### Running the Server

```bash
# Using uv
uv run python -m calendar_agent.calendar_server

# Or with uvicorn directly
uv run uvicorn calendar_agent.calendar_server:app --host 0.0.0.0 --port 8082
```

The server will be available at `http://localhost:8082`. API documentation is at `http://localhost:8082/docs`.

## API Endpoints

### Error Responses

Every endpoint returns a body with a `success` field, and on failure an
`error` message; the HTTP status code always agrees with the body
([issue #4](https://github.com/brianroberg/calendar-agent/issues/4)):

| Status | Meaning |
|--------|---------|
| `200` | The operation succeeded (`success: true`) |
| `400` | The proxy rejected the request as malformed or not applicable, and its message is in `error` (e.g. `/respond` when the authenticated user is not an attendee of the event); also this service's own refusal to RSVP through a calendar that is not the authenticated user's own, which is not forwarded to the proxy at all |
| `403` | The proxy blocked the operation by policy, or the human operator rejected it (mutations block in the proxy until an operator approves them) |
| `404` | The calendar or event does not exist (the proxy's message is in `error`). When verifying a deletion this is the *expected* answer: the event is gone |
| `422` | Request validation failed: an unknown field or query key, a missing required field, or a bad value — nothing was sent upstream. Same `success: false` / `error` envelope as every other failure, plus FastAPI's `detail` list (see below). A `PUT`/`PATCH` body that would forward nothing (`{}`, all-null values, or only read-only keys) is rejected this way too, and a bulk `update`/`patch` with a missing or empty `updates` payload rejects the **whole batch**, before any operation runs |
| `502` | The proxy or LLM backend failed; the proxy answered 401 (this service's own key was rejected) or a 4xx other than 400/403/404 (its message is in `error`); or, on a delete, the proxy claimed success for an event that is still present on re-read |
| `504` | **The outcome is unknown**: no response before this server's timeout and the resource is still present, or the verifying re-read itself failed. A confirmation-gated mutation may still complete if approved later. Verify by re-reading the resource; never issue a compensating mutation on the strength of a `504` |

Mutation envelopes (`DELETE …/events/{id}` and each `/bulk-actions` result)
also carry an `outcome` of `succeeded` / `failed` / `unknown` (bulk only:
`not_attempted`), so a caller reading only the body can tell "rejected" from
"may still apply". The exception is a request-validation `422` (for example
`DELETE …?sendUpdates=all`, an undeclared query key): it carries the
validation envelope described below and no `outcome`, since nothing was
attempted.
| `500` | Unexpected internal error |

#### Validation: unknown fields are rejected, not ignored

Every request body model uses `extra="forbid"`
([issue #8](https://github.com/brianroberg/calendar-agent/issues/8)): a key the
model doesn't declare -- a typo, or a Google/JS-style name like `timeMin`
instead of `time_min`, or `title` instead of `summary` -- is a `422`, at any
nesting depth (inside `attendees`, `filters`, a bulk `operation`'s `updates`,
...), not a silently-dropped no-op. Before this, such a request could return
`success: true` while quietly doing something other than what was asked (an
unbounded search when time bounds were misnamed, an event titled "Untitled
Event" when `title` was sent instead of `summary`).

The same rule covers the **query string**: a query key the route does not
declare (`?timeMin=...` on the events list, `?sendUpdates=all` on a write) is
a `422` with `loc: ["query", "<key>"]`, not ignored.

**Which fields are accepted.** Each route accepts exactly the fields its
section below lists. The three event-write routes -- `POST` create, `PUT`
update and `PATCH` -- all accept the same set (the one listed under
`POST /calendars/{calendar_id}/events`), so anything you can create you can
also update or patch. The one value-level exception is `status`: it accepts
`"confirmed"` and `"tentative"` only. `"cancelled"` is a `422` that names the
right route -- cancelling goes through `DELETE`, which verifies the deletion
and reports an `outcome`; an update path would not. A bulk `update`/`patch`
operation's `updates` object is validated with
the same model as `PUT`/`PATCH`; a bulk `delete` must not carry `updates`.

**Round-tripping a fetched event.** Google adds server-populated, read-only
keys to the events it returns. So that fetch -> modify -> write works, the
event-write routes strip exactly this named set before checking for unknown
fields, and never forward them:

`kind`, `etag`, `id`, `htmlLink`, `hangoutLink`, `created`, `updated`,
`creator`, `organizer`, `iCalUID`, `sequence`, `eventType`,
`recurringEventId`, `originalStartTime`, `privateCopy`, `locked`,
`attendeesOmitted`, `endTimeUnspecified`, `outOfOfficeProperties`,
`workingLocationProperties`

Anything else undeclared is still rejected -- including five keys Google
returns that *are* writable but this server does not declare:
`conferenceData` (the object `hangoutLink` derives from, present on Meet
events), `attachments`, `extendedProperties`, `source` and
`anyoneCanAddSelf`. A caller round-tripping a fetched event removes those
before writing. They are not stripped because that would silently drop
something the caller sent, and not declared because that is new write
surface (and without `conferenceDataVersion`, which this server does not
send, Google ignores `conferenceData` on a write anyway). Attendee entries
accept everything Google returns for an attendee (`id`, `resource`,
`comment`, `additionalGuests`, `self`, ...).

**`/search` accepts two shapes** -- filter keys nested under `filters`, or the
same keys flat at the top level -- see its section for the rules.

**What a 422 looks like.** The body carries the standard envelope; for body
errors, `error` names each failing field and `detail` is FastAPI's structured
list. The query-string check runs first and on its own: a request with an
undeclared query key reports only that key, and any body errors surface once
the query string is fixed and the request is resent.

```json
{
  "success": false,
  "error": "body.title: Extra inputs are not permitted",
  "detail": [
    {
      "type": "extra_forbidden",
      "loc": ["body", "title"],
      "msg": "Extra inputs are not permitted",
      "input": "Standup"
    }
  ]
}
```

### GET /health

Health check endpoint. Returns server status and version.

```bash
curl http://localhost:8082/health
```

Response:
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

---

## Calendar Endpoints

### GET /calendars

List all calendars for the authenticated user.

```bash
curl http://localhost:8082/calendars
```

Response:
```json
{
  "success": true,
  "calendars": [
    {
      "id": "john.doe@example.com",
      "summary": "john.doe@example.com",
      "description": "Primary calendar",
      "timeZone": "America/New_York",
      "primary": true
    }
  ],
  "error": null
}
```

### GET /calendars/{calendar_id}

Get metadata for a specific calendar.

```bash
curl http://localhost:8082/calendars/primary
```

Response:
```json
{
  "success": true,
  "calendar": {
    "id": "john.doe@example.com",
    "summary": "john.doe@example.com",
    "timeZone": "America/New_York"
  }
}
```

---

## Event CRUD Endpoints

### GET /calendars/{calendar_id}/events

List events in a calendar. By default, recurring events are expanded to individual instances.

Query parameters:
- `max_results` (int): Maximum number of events to return (default: 100)
- `page_token` (string): Token for pagination
- `time_min` (string): Start of time range (RFC3339)
- `time_max` (string): End of time range (RFC3339)
- `q` (string): Free text search query
- `single_events` (bool): Expand recurring events (default: true)
- `order_by` (string): "startTime" or "updated"

```bash
curl "http://localhost:8082/calendars/primary/events?time_min=2024-01-01T00:00:00Z&max_results=10"
```

Response:
```json
{
  "success": true,
  "events": [
    {
      "id": "event123",
      "calendar_id": "primary",
      "summary": "Team Meeting",
      "start": "2024-01-15T10:00:00Z",
      "end": "2024-01-15T11:00:00Z",
      "location": "Conference Room A",
      "attendee_count": 5,
      "is_all_day": false,
      "status": "confirmed",
      "html_link": "https://www.google.com/calendar/event?eid=ZXZlbnQxMjM",
      "organizer_email": "alice@example.com",
      "creator_email": "alice@example.com",
      "calendar_is_organizer": false,
      "calendar_rsvp_state": "accepted"
    }
  ],
  "next_page_token": null,
  "error": null
}
```

#### Organizer and RSVP fields -- whose perspective they report

**The `calendar_*` fields describe the calendar named by `calendar_id`,
not the authenticated user.** They are computed from Google's `self` flags,
which the Events reference defines relative to the calendar:
`attendees[].self` is "whether this entry represents **the calendar on
which this copy of the event appears**", and `organizer.self` is "whether
the organizer corresponds to **the calendar on which this copy of the event
appears**". So:

- Reading your own calendar (`primary`, or your own address), they describe
  you.
- Reading a colleague's calendar (`GET /calendars/colleague@example.com/events`),
  they describe **the colleague**: `calendar_rsvp_state` is *their* RSVP,
  and `calendar_is_organizer` says whether *they* organize it. Your own RSVP
  on that event is not reported -- it is only knowable from your own calendar.
- Reading a group calendar, they describe the group calendar: an event
  created directly on one has the calendar itself as organizer
  (`organizer_email` is the calendar's id, `calendar_is_organizer` is `true`).
  `creator_email` is then the person who created it.

Fields (the authoritative per-value descriptions are the `EventSummary`
schema in `/docs` / `/openapi.json`, generated from the code; this list is
the short form):

- `organizer_email`: the organizer's address as Google reports it. For an
  event created on a group calendar this is the group calendar's id. `null`
  when the event carries no organizer (e.g. a cancelled recurring-instance
  stub -- see `status`).
- `creator_email`: the address of the account that created the event.
  Usually the same as `organizer_email`; differs on group calendars (above)
  and for events moved between calendars. `null` when absent.
- `calendar_is_organizer`: Google's `organizer.self` -- the calendar being
  read organizes this event.
- `calendar_rsvp_state`: the RSVP of the attendee entry Google marks `self`
  on this copy -- the calendar's own entry -- classified so the caller never
  has to decode a missing value. Either one of Google's four values
  (`"accepted"`, `"declined"`, `"tentative"`, `"needsAction"`) verbatim, or
  one of this service's own: `"organizer_no_rsvp"` (the calendar's own
  event, nothing to answer), `"not_attendee"` (neither organizer nor
  invited -- e.g. an event copied onto the calendar, or an invitation
  addressed to a group), `"unknown"` (a `responseStatus` this service does
  not recognise; or the calendar's own entry has no `responseStatus` and
  the calendar is not the organizer; or a stub with no organizer and no
  attendees). The raw
  Google string is not exposed separately: it is either one of the four
  values above or something this service cannot classify.
- `status`: Google's event status -- `"confirmed"`, `"tentative"`, or
  `"cancelled"`. On a plain `GET` with `single_events=false`, cancelled rows
  are the stubs Google keeps for deleted instances of a recurring series
  (the default `single_events=true` expansion omits them): empty
  `start`/`end`, no organizer, no attendees, so `calendar_rsvp_state:
  "unknown"`. `POST /search` is different: `filters.show_deleted: true`
  forwards `showDeleted` to Google (with `singleEvents` fixed to `true`
  there), and the cancelled rows it returns are whatever Google sends for
  them -- possibly with real times, an organizer and attendees -- classified
  like any other row.

Only `"accepted"`, `"declined"` and `"tentative"` can be sent back to `POST
.../respond`; a `"needsAction"` or derived state cannot be echoed to it. Note
that the read side and `/respond` take **different perspectives**: these
fields report the calendar's own entry, while `/respond` always writes the
**authenticated user's** entry (matched by email address, never by the
`self` flag). On your own calendar the two coincide; on a colleague's or
group calendar they do not, which is why `/respond` accepts only your own
`calendar_id` and refuses the rest -- see that endpoint's note below.

### POST /calendars/{calendar_id}/events

Create a new event in a calendar.

Accepted fields (the same set applies to `PUT` and `PATCH` below; anything
else is a `422`, except the read-only keys listed under *Validation* above,
which are stripped):

| Field | Type |
|-------|------|
| `summary`, `description`, `location`, `colorId` | string |
| `start`, `end` | `{"dateTime": ..., "timeZone": ...}` or `{"date": "YYYY-MM-DD"}` |
| `attendees` | list of `{"email": ..., "displayName", "responseStatus", "optional", "organizer", "self", "id", "resource", "comment", "additionalGuests"}` |
| `reminders` | `{"useDefault": bool, "overrides": [{"method": ..., "minutes": ...}]}` |
| `recurrence` | list of RRULE strings |
| `status` | `"confirmed"` or `"tentative"` (`"cancelled"` is a `422`: cancel with `DELETE`) |
| `transparency` | `"opaque"` or `"transparent"` |
| `visibility` | `"default"`, `"public"` or `"private"` |
| `guestsCanInviteOthers`, `guestsCanModify`, `guestsCanSeeOtherGuests` | bool |

Query parameter: `send_updates` (`all`, `externalOnly`, `none`).

```bash
curl -X POST http://localhost:8082/calendars/primary/events \
  -H "Content-Type: application/json" \
  -d '{
    "summary": "Project Review",
    "description": "Quarterly project review meeting",
    "location": "Board Room",
    "start": {"dateTime": "2024-01-20T14:00:00Z"},
    "end": {"dateTime": "2024-01-20T15:00:00Z"},
    "attendees": [
      {"email": "alice@example.com"},
      {"email": "bob@example.com"}
    ]
  }'
```

Response:
```json
{
  "success": true,
  "event": {
    "id": "newEvent123",
    "summary": "Project Review",
    "status": "confirmed"
  },
  "error": null
}
```

### GET /calendars/{calendar_id}/events/{event_id}

Get a specific event by ID.

```bash
curl http://localhost:8082/calendars/primary/events/event123
```

Response:
```json
{
  "success": true,
  "event": {
    "id": "event123",
    "summary": "Team Meeting",
    "description": "Weekly team sync",
    "start": {"dateTime": "2024-01-15T10:00:00Z", "timeZone": "UTC"},
    "end": {"dateTime": "2024-01-15T11:00:00Z", "timeZone": "UTC"},
    "attendees": [
      {"email": "alice@example.com", "responseStatus": "accepted"}
    ]
  },
  "error": null
}
```

### PUT /calendars/{calendar_id}/events/{event_id}

Update an event (full replacement). Accepts the same fields as `POST`; a
body that would forward nothing (`{}`, all-null values, only read-only keys)
is a `422` rather than an empty write. A
fetched event round-trips once the caller removes the writable keys this
server does not declare -- `conferenceData`, `attachments`,
`extendedProperties`, `source`, `anyoneCanAddSelf` (Meet and attachment
events are the common case) -- Google's read-only keys (`id`, `etag`,
`htmlLink`, `privateCopy`, ...) are stripped, not rejected. See
*Round-tripping a fetched event* above. Query parameter: `send_updates`.

```bash
curl -X PUT http://localhost:8082/calendars/primary/events/event123 \
  -H "Content-Type: application/json" \
  -d '{
    "summary": "Updated Meeting Title",
    "start": {"dateTime": "2024-01-15T14:00:00Z"},
    "end": {"dateTime": "2024-01-15T15:00:00Z"}
  }'
```

Response:
```json
{
  "success": true,
  "event": {
    "id": "event123",
    "summary": "Updated Meeting Title"
  },
  "error": null
}
```

### PATCH /calendars/{calendar_id}/events/{event_id}

Partially update an event. Accepts every field `POST` does (including
`status`, `transparency`, `visibility` and the `guestsCan*` flags); only the
fields sent are changed. A body that would forward nothing is a `422`, as on
`PUT`. Query parameter: `send_updates`.

```bash
curl -X PATCH http://localhost:8082/calendars/primary/events/event123 \
  -H "Content-Type: application/json" \
  -d '{"location": "New Conference Room"}'
```

Response:
```json
{
  "success": true,
  "event": {
    "id": "event123",
    "location": "New Conference Room"
  },
  "error": null
}
```

### DELETE /calendars/{calendar_id}/events/{event_id}

Delete an event, **verified by re-reading it**. The proxy requires operator
confirmation for deletes: the request blocks while a human approves it. The
proxy's answer is treated as a claim, not evidence — this server re-reads the
event afterwards and decides from that:

| Proxy said | Re-read shows | Status | `outcome` |
|------------|---------------|--------|-----------|
| anything but 403/404 | gone (`404`/`410`, or `status: cancelled`) | `200` | `succeeded` |
| success | still present | `502` | `failed` (the 2026-08-07 incident shape: a claim of success for work that has not happened) |
| timed out | still present | `504` | `unknown` — may yet be applied when the operator approves; re-verify before acting |
| anything | re-read failed | `504` | `unknown` — nothing established |
| error (`5xx`, `4xx`) | still present | that error's code | `failed` |
| `403` rejected | *(not re-read — the proxy dropped it)* | `403` | `failed` |
| `404` no such event | *(not re-read — nothing to delete)* | `404` | `failed` |

```bash
curl -X DELETE http://localhost:8082/calendars/primary/events/event123
```

Response:
```json
{
  "success": true,
  "outcome": "succeeded",
  "message": "Event deleted successfully",
  "error": null
}
```

If the operator rejects the deletion (HTTP `403`):
```json
{
  "success": false,
  "outcome": "failed",
  "message": "Deletion blocked or rejected by operator",
  "error": "Operation blocked: Request rejected by operator"
}
```

If no response arrives before the timeout and the event is still there (HTTP
`504` — outcome unknown; never create a replacement until a re-read shows it
gone):
```json
{
  "success": false,
  "outcome": "unknown",
  "message": "Deletion outcome unknown: no response before timeout and the event is still present; ...",
  "error": "Outcome unknown: No response from proxy after 330s; ..."
}
```

#### Deleting from a script: `scripts/calendar-delete-event.sh`

The server now verifies deletes itself (above), but a caller that reads only
`curl`'s status code, or whose `curl` is killed by a harness timeout, still
cannot tell the cases apart. The wrapper script captures body and status,
**re-reads the event itself** rather than trusting the delete's answer, and
reports through its exit status:

| Exit | Result | Meaning |
|------|--------|---------|
| `0` | `SUCCESS` | The event is gone — re-read returned `404`, or `status: cancelled` |
| `1` | `FAILURE` | The event is still there and nothing is outstanding (rejected `403`, other `4xx`, or `success: false`) |
| `2` | `UNKNOWN` | The event is still there but the deletion may yet be applied (the DELETE timed out, or answered `408`/**any `5xx`** — a `502` can be a transport fault *after* the request reached the proxy, where it stays queued), or the re-read established nothing, or the response carried no calendar-agent envelope (a bare router `404` from a wrong `CALENDAR_AGENT_URL`) |
| `3` | `NOT FOUND` | The DELETE itself answered `404`/`410` *with* calendar-agent's envelope — the event id (or calendar id) didn't exist before this ran, so nothing was deleted. Check the id; a re-read that also 404s is not evidence of a completed deletion |
| `4` | `USAGE` | Bad arguments or configuration (missing url, `python3` not on `PATH`, a deadline that does not outlive the server's budget). Nothing was attempted |

```bash
CALENDAR_AGENT_URL=http://localhost:8082 \
  scripts/calendar-delete-event.sh event123 primary
```

```
DELETE primary event event123 (deadline 340s) -> HTTP 200 (body success: true)
VERIFY event123 (verify deadline 35s) -> HTTP 404 (event status: <absent>)
RESULT: SUCCESS - event no longer present
```

| Variable | Meaning | Default |
|----------|---------|---------|
| `CALENDAR_AGENT_CONFIRM_TIMEOUT` | calendar-agent's own mutation budget (its `PROXY_CONFIRM_TIMEOUT`); the script cannot read it across the container boundary, so keep them in step by hand | `330` |
| `CALENDAR_DELETE_MAX_TIME` | `curl` deadline for the DELETE. **Must exceed the budget** — the script refuses (exit `4`) otherwise, because a shorter deadline abandons the DELETE while the operator can still approve it, re-creating one hop out the very mismatch the server guards against | budget + 10 = `340` |
| `CALENDAR_DELETE_VERIFY_MAX_TIME` | `curl` deadline for the verifying GET (an ordinary 30s-bounded read) | `35` |

**Worst case the script runs for 375s** (340 + 35). The *calling* harness must
allow at least that — pass a tool timeout of 400s or more. If the caller kills
the script earlier, it dies before its verification step, which is the one
part that establishes anything. The script never retries — a retry enqueues a
second operator approval for the same operation.

**Exit `2` means do not act.** In particular, never create a replacement event
until a deletion has been observed complete.

> **Deployment note.** This script lives in this repository. The copy the
> assistant actually runs is the separate whitelisted script in the workspace
> repo, and the `calendar-delete-event` skill still does `curl -X DELETE | jq .`.
> Merging this PR changes neither; both must be replaced with this script
> (and the skill taught the exit codes and the 400s tool timeout) before any
> of it takes effect operationally.

### POST /calendars/{calendar_id}/events/{event_id}/respond

RSVP to an event by setting the response status of **the authenticated
user's own** attendee entry. Forwards to the proxy's dedicated respond route,
which reads the event, finds the authenticated account's entry in the
attendee list **by email address** (it resolves that address from the
account's primary calendar and does not trust Google's `self` flag), patches
only that entry, and sends no invitations or notifications.

> **`calendar_id` must be your own calendar.** The route accepts the literal
> `primary` (any capitalisation) and the authenticated account's own calendar
> id (its address), and refuses anything else with **`400`**; the RSVP is not
> forwarded. The one proxy call a non-`primary` id does make is the identity
> lookup described below.
>
> **Why.** Whose RSVP this route changes is always the authenticated user's
> -- the account the proxy holds credentials for. That is the opposite
> perspective from the read side, where `calendar_rsvp_state` and
> `calendar_is_organizer` describe *the calendar being read*. On a group or
> colleague calendar those are two different attendee entries, so a
> `"needsAction"` read there says nothing about the entry an RSVP would
> write -- a read-then-respond loop across calendars answers on the strength
> of a state that was about someone else. Refusing keeps reading and
> answering on one calendar, where the two agree. `primary` is accepted with
> no lookup; any other id is compared against the account's own calendar id,
> which this service reads once per process from `GET /calendars/primary`
> (cached for the life of the process, so the read happens on the first
> non-`primary` RSVP after a start). If that lookup fails -- timeout, 404, 403
> or any other proxy error -- the route answers **`502`** with a message
> saying the ownership check could not be performed, and nothing is sent: not
> `504` outcome-unknown and not `404`, because no RSVP was attempted.
>
> If the authenticated user is not an attendee, the proxy answers
> `400 You are not an attendee of this event; cannot RSVP.` This service
> passes that through as **`400`** with the proxy's message in `error` (see
> Error Responses for which proxy statuses pass through and which map to 502).

Request body:
- `response_status` (string): one of `accepted`, `declined`, `tentative`

```bash
curl -X POST http://localhost:8082/calendars/primary/events/event123/respond \
  -H "Content-Type: application/json" \
  -d '{"response_status": "accepted"}'
```

Response:
```json
{
  "success": true,
  "event": {
    "id": "event123",
    "attendees": [
      {"email": "you@example.com", "responseStatus": "accepted", "self": true}
    ]
  },
  "error": null,
  "warnings": []
}
```

On any other calendar -- a colleague's, or a group calendar such as
`POST /calendars/team_calendar@group.calendar.google.com/events/event123/respond`
-- the RSVP is refused with `400` and is not forwarded. The proxy sees only
the `GET /calendars/primary` identity lookup the check needs (once per
process); if that lookup fails the answer is `502` with nothing sent, rather
than the refusal below:
```json
{
  "success": false,
  "event": null,
  "error": "Refusing to RSVP through calendar 'team_calendar@group.calendar.google.com': it is not the authenticated user's own calendar. This route always writes the authenticated user's attendee entry, while that calendar's read fields (calendar_rsvp_state, calendar_is_organizer) describe the calendar itself -- so an RSVP state read there is not the entry this would change. Send the RSVP on the authenticated user's own calendar instead: POST /calendars/primary/events/{event_id}/respond.",
  "warnings": []
}
```

---

## LLM-Powered Endpoints

These endpoints use a local LLM to process calendar data and generate insights.

### POST /summarize

Summarize a calendar event using AI.

```bash
curl -X POST http://localhost:8082/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "calendar_id": "primary",
    "event_id": "event123",
    "format": "brief"
  }'
```

Response:
```json
{
  "success": true,
  "data": {
    "event_id": "event123",
    "summary": "This is a weekly team sync meeting scheduled for Monday at 10 AM with 5 attendees. The meeting is held in Conference Room A and typically covers project updates and blockers."
  },
  "error": null
}
```

### POST /ask-about

Ask a question about a specific calendar event.

```bash
curl -X POST http://localhost:8082/ask-about \
  -H "Content-Type: application/json" \
  -d '{
    "calendar_id": "primary",
    "event_id": "event123",
    "question": "Who are the attendees and what are their response statuses?"
  }'
```

Response:
```json
{
  "success": true,
  "data": {
    "event_id": "event123",
    "question": "Who are the attendees and what are their response statuses?",
    "answer": "There are 5 attendees: Alice Smith (accepted), Bob Jones (tentative), Carol White (accepted), David Brown (needs action), and Eve Davis (declined)."
  },
  "error": null
}
```

### POST /batch-summarize

Summarize multiple events, optionally with triage classification.

```bash
curl -X POST http://localhost:8082/batch-summarize \
  -H "Content-Type: application/json" \
  -d '{
    "calendar_id": "primary",
    "event_ids": ["event1", "event2", "event3"],
    "triage": true
  }'
```

Response:
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "event_id": "event1",
        "summary": "Quarterly planning meeting with leadership team",
        "action_type": "meeting",
        "deadline": null
      },
      {
        "event_id": "event2",
        "summary": "Project deadline reminder",
        "action_type": "deadline",
        "deadline": "2024-01-20"
      }
    ],
    "total": 2
  },
  "error": null
}
```

### POST /find-free-time

Find available time slots and get AI suggestions for scheduling.

```bash
curl -X POST http://localhost:8082/find-free-time \
  -H "Content-Type: application/json" \
  -d '{
    "calendar_id": "primary",
    "time_min": "2024-01-15T09:00:00Z",
    "time_max": "2024-01-15T17:00:00Z",
    "duration_minutes": 30,
    "working_hours_only": true,
    "prefer_morning": true
  }'
```

Response:
```json
{
  "success": true,
  "data": {
    "available_slots": [
      {
        "start": "2024-01-15T09:00:00Z",
        "end": "2024-01-15T10:00:00Z",
        "duration_minutes": 60
      },
      {
        "start": "2024-01-15T14:00:00Z",
        "end": "2024-01-15T15:30:00Z",
        "duration_minutes": 90
      }
    ],
    "suggestions": "Based on your preference for morning meetings, I recommend the 9:00 AM slot. It gives you a full hour before your first scheduled meeting and allows time for preparation.",
    "duration_requested": 30
  },
  "error": null
}
```

### POST /analyze-schedule

Analyze schedule patterns and get AI-powered insights.

```bash
curl -X POST http://localhost:8082/analyze-schedule \
  -H "Content-Type: application/json" \
  -d '{
    "calendar_id": "primary",
    "time_min": "2024-01-15T00:00:00Z",
    "time_max": "2024-01-22T00:00:00Z",
    "analysis_type": "overview"
  }'
```

Response:
```json
{
  "success": true,
  "data": {
    "time_range": "2024-01-15T00:00:00Z to 2024-01-22T00:00:00Z",
    "metrics": {
      "total_events": 15,
      "total_hours": 22.5
    },
    "analysis_type": "overview",
    "insights": "Your week has 15 scheduled events totaling 22.5 hours. Key observations:\n\n1. Meeting load is moderate at ~4.5 hours/day\n2. Tuesday and Thursday are meeting-heavy days\n3. You have good focus time blocks on Monday and Friday mornings\n\nRecommendations:\n- Consider consolidating Tuesday meetings to create longer focus blocks\n- The back-to-back meetings on Thursday afternoon may cause fatigue"
  },
  "error": null
}
```

### POST /prepare-briefing

Generate an AI-powered schedule briefing.

```bash
curl -X POST http://localhost:8082/prepare-briefing \
  -H "Content-Type: application/json" \
  -d '{
    "calendar_id": "primary",
    "briefing_type": "daily"
  }'
```

Response:
```json
{
  "success": true,
  "data": {
    "briefing_type": "daily",
    "period": "daily schedule",
    "event_count": 5,
    "briefing": "Today's Schedule Overview:\n\nYou have 5 events scheduled today:\n\n1. 9:00 AM - Team Standup (15 min)\n   Quick daily sync with your team\n\n2. 10:00 AM - Project Review (1 hour)\n   Prepare: Review Q4 metrics document\n\n3. 12:00 PM - Lunch with Client (1.5 hours)\n   Location: Restaurant downtown\n\n4. 3:00 PM - 1:1 with Manager (30 min)\n   Prepare: Weekly status update\n\n5. 4:30 PM - Tech Talk (1 hour)\n   Optional attendance\n\nKey Preparation:\n- Review Q4 metrics before the Project Review\n- Allow 30 min travel time to lunch location"
  },
  "error": null
}
```

---

## Operations Endpoints

### POST /search

Search events in a calendar with structured filters. `filters` accepts
`query`, `time_min`, `time_max`, `max_results` (1-500, default 100),
`order_by` (`startTime` or `updated`) and `show_deleted` (default `false`;
forwards Google's `showDeleted`, so cancelled events come back as
`status: "cancelled"` rows with whatever fields Google sends for them --
see `status` under `GET /calendars/{calendar_id}/events`). Recurring events
are always expanded (`singleEvents=true`).

The filter keys (`query`, `time_min`, `time_max`, `max_results`, `order_by`,
`show_deleted`) may be nested under `filters` (the shape `/openapi.json`
describes) **or** sent flat at the top level next to `calendar_id`. The flat
shape is kept for compatibility with earlier callers; the installed calendar
skills nest their filters. Both requests below are equivalent. Rules for the
flat shape:

- Only the keys `filters` declares are folded in; anything else at the top
  level (a typo, `timeMin`) is still a `422`.
- If a key appears both flat and inside `filters`, the nested value wins,
  field by field -- the two are merged, not replaced.
- A flat key sent as `null` means "not supplied" (the default applies).

```bash
curl -X POST http://localhost:8082/search \
  -H "Content-Type: application/json" \
  -d '{
    "calendar_id": "primary",
    "filters": {
      "query": "project review",
      "time_min": "2024-01-01T00:00:00Z",
      "time_max": "2024-03-31T23:59:59Z",
      "max_results": 20,
      "order_by": "startTime"
    }
  }'

# Equivalent flat shape
curl -X POST http://localhost:8082/search \
  -H "Content-Type: application/json" \
  -d '{
    "calendar_id": "primary",
    "query": "project review",
    "time_min": "2024-01-01T00:00:00Z",
    "time_max": "2024-03-31T23:59:59Z",
    "max_results": 20,
    "order_by": "startTime"
  }'
```

Response:
```json
{
  "success": true,
  "events": [
    {
      "id": "event456",
      "calendar_id": "primary",
      "summary": "Q1 Project Review",
      "start": "2024-01-20T14:00:00Z",
      "end": "2024-01-20T15:00:00Z",
      "attendee_count": 8,
      "is_all_day": false,
      "status": "confirmed",
      "organizer_email": "john.doe@example.com",
      "creator_email": "john.doe@example.com",
      "calendar_is_organizer": true,
      "calendar_rsvp_state": "organizer_no_rsvp"
    }
  ],
  "next_page_token": null,
  "error": null
}
```

### POST /bulk-actions

Execute multiple operations on events in a single request.

Supported operations:
- `update`: Full event replacement -- `updates` is validated exactly like a
  `PUT` body
- `patch`: Partial event update -- `updates` is validated exactly like a
  `PATCH` body
- `delete`: Delete event -- must not carry `updates` (a `422` if it does, so a
  mis-set operation can't delete while its payload is silently ignored)

An unknown key inside `updates` is a `422` for the whole request, and so is
a missing or empty `updates` on `update`/`patch` (see below). Each operation
may also carry `send_updates`.

Operations run sequentially. Deletes are verified by re-reading the event,
exactly as the single-event `DELETE` is. An `update`/`patch` without a
non-empty `updates` payload fails request validation (`422`) and **no
operation in the batch runs**.

Each result carries an `outcome`:

| `outcome` | Meaning |
|-----------|---------|
| `succeeded` | Known to have happened (deletes: the re-read showed the event gone) |
| `failed` | Known not to have happened (rejected, absent, upstream error, or a delete whose re-read still shows the event) |
| `unknown` | Timed out and **may still be applied** when the operator approves it — re-read the event before doing anything about it |
| `not_attempted` | Never sent: an earlier operation in the batch came back `unknown`, which means the operator is not answering, and each further gated operation would have held the connection another full timeout and queued another approval. Re-issue these in a new request once the unknown one is settled |

The envelope's `success` is true only when every operation succeeded;
otherwise `error` summarises the counts (`"2 of 3 operations did not
succeed: 1 outcome unknown (...), 1 not attempted; see results"`). The status
code agrees with the body and is ranked by severity, **never by position in
the batch**: `504` if any outcome is unknown (the unknown operation is the one
that can still change the calendar), else `403` if any was rejected, else
`502`, else `500`, else `404`.

```bash
curl -X POST http://localhost:8082/bulk-actions \
  -H "Content-Type: application/json" \
  -d '{
    "operations": [
      {
        "operation": "patch",
        "event_id": "event1",
        "calendar_id": "primary",
        "updates": {"location": "Room A"}
      },
      {
        "operation": "delete",
        "event_id": "event2",
        "calendar_id": "primary"
      }
    ]
  }'
```

Response:
```json
{
  "success": true,
  "results": [
    {
      "event_id": "event1",
      "operation": "patch",
      "success": true,
      "outcome": "succeeded",
      "error": null
    },
    {
      "event_id": "event2",
      "operation": "delete",
      "success": true,
      "outcome": "succeeded",
      "error": null
    }
  ],
  "success_count": 2,
  "error_count": 0,
  "unknown_count": 0,
  "not_attempted_count": 0,
  "error": null
}
```

---

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=calendar_agent

# Run specific test file
uv run pytest tests/test_calendar_server.py

# Run documentation tests
uv run pytest tests/test_readme_documentation.py
```

### Linting

```bash
# Check linting
uv run ruff check .

# Fix linting issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Refreshing the API Proxy Spec Snapshot

`docs/api-proxy-openapi-doc.json` is a snapshot of the
[api-proxy](https://github.com/brianroberg/api-proxy) OpenAPI spec, stamped
with the commit and time it was generated from (`x-generated-from`). It records
the proxy contract this agent was built against; `tests/test_proxy_contract.py`
fails if `proxy_client.py` calls a route the snapshot doesn't contain, so
refresh it whenever the proxy contract changes:

```bash
# From a local api-proxy checkout
uv run python scripts/refresh_openapi.py --checkout ../api-proxy

# From a running proxy instance
uv run python scripts/refresh_openapi.py --url http://localhost:8000
```

### Project Structure

```
calendar-agent/
├── calendar_agent/
│   ├── __init__.py
│   ├── calendar_server.py    # FastAPI application with all endpoints
│   ├── proxy_client.py       # HTTP client for api-proxy
│   ├── calendar_utils.py     # Utility functions
│   ├── llm_service.py        # LLM provider abstraction and implementation
│   └── exceptions.py         # Custom exceptions
├── tests/
│   ├── conftest.py           # Test fixtures
│   ├── test_calendar_server.py
│   ├── test_delete_event_script.py  # End-to-end tests for the delete wrapper
│   ├── test_proxy_contract.py    # Client routes vs. the api-proxy spec snapshot
│   └── test_readme_documentation.py
├── docs/
│   └── api-proxy-openapi-doc.json    # Stamped snapshot of the api-proxy spec
├── scripts/
│   ├── calendar-delete-event.sh  # Verified delete wrapper (success/failure/unknown)
│   └── refresh_openapi.py    # Regenerates the api-proxy spec snapshot
├── pyproject.toml
├── README.md
└── .env.example
```

## LLM Provider Extensibility

The calendar agent is designed to support multiple LLM providers. Currently, it uses a local MLX-based server, but the architecture allows easy swapping to hosted APIs.

To add a new LLM provider:

1. Create a new class implementing `LLMProvider` in `llm_service.py`:

```python
class AnthropicProvider(LLMProvider):
    async def generate(self, system_prompt, user_content, max_tokens=1024, temperature=0.3):
        # Implementation using Anthropic API
        ...
```

2. Update the `get_llm_service()` function to use your provider based on configuration.

## License

MIT
