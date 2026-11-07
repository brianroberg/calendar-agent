#!/usr/bin/env bash
# Delete a single calendar event via calendar-agent, and prove what happened.
#
# Usage: calendar-delete-event.sh <event_id> [calendar_id]
#
# Environment:
#   CALENDAR_AGENT_URL              base URL of calendar-agent (required)
#   CALENDAR_AGENT_API_KEY          this caller's bearer token, sent as
#                                   `Authorization: Bearer <token>` on both
#                                   requests. Needed when calendar-agent
#                                   requires one (CALENDAR_AGENT_AUTH=required);
#                                   when unset, no Authorization header is sent.
#   CALENDAR_AGENT_CONFIRM_TIMEOUT  calendar-agent's own mutation budget in
#                                   seconds — its PROXY_CONFIRM_TIMEOUT
#                                   (default 330). The script cannot read it
#                                   across the container boundary; keep them
#                                   in step by hand.
#   CALENDAR_DELETE_MAX_TIME        curl deadline for the DELETE, seconds
#                                   (default: budget + 10 = 340). Must be
#                                   greater than the budget: refused otherwise.
#   CALENDAR_DELETE_VERIFY_MAX_TIME curl deadline for the verifying GET,
#                                   seconds (default 35; the re-read is an
#                                   ordinary 30s-bounded read, not a gated
#                                   mutation).
#
# Worst case the script runs for MAX_TIME + VERIFY_MAX_TIME (375s by default).
# Callers must allow at least that — a harness that kills the script earlier
# kills it before the verification step, which is the only part that
# establishes anything.
#
# Deletes are gated on a human operator, who may take minutes to answer, so the
# response to the DELETE is not evidence of anything: it can be a claim of
# success for work that has not happened, or a timeout for work that is still
# going to happen. This script therefore decides by re-reading the event, and
# reports distinct outcomes (issue #4):
#
#   exit 0  SUCCESS   the event is gone (404, or status "cancelled")
#   exit 1  FAILURE   the event is still there and nothing is outstanding
#   exit 2  UNKNOWN   the event is still there but the deletion may yet land,
#                     or the re-read could not establish either
#   exit 3  NOT FOUND the DELETE itself answered 404/410 - the event id (or
#                     calendar id) did not exist before this script ran, so
#                     nothing was deleted. A re-read that also 404s is not
#                     evidence of a completed deletion; check the id.
#   exit 4  USAGE     bad arguments or configuration; nothing was attempted.
#                     Includes calendar-agent refusing the bearer token (its
#                     401 comes before the request reaches any route).
#
# Callers must treat exit 2 as "do not act": in particular, never create a
# replacement event until a delete is observed complete.
#
# Deliberately narrow: it can only ever issue one DELETE and one GET against
# calendar-agent's /calendars/{id}/events/{id} route. It never retries, because
# a retry enqueues a second approval request for the same operation.

set -uo pipefail

usage() { echo "$*" >&2; exit 4; }

[ $# -ge 1 ] || usage "usage: calendar-delete-event.sh <event_id> [calendar_id]"
EVENT_ID="$1"
CAL_ID="${2:-robergb@dm.org}"
[ -n "${CALENDAR_AGENT_URL:-}" ] || usage "CALENDAR_AGENT_URL is not set"

# ---------------------------------------------------------------------------
# Deadlines. The DELETE must outlive calendar-agent's mutation budget, which
# in turn outlives the proxy's operator window: otherwise an approval given at
# minute four lands after curl has hung up, the deletion completes unobserved,
# and this script reports UNKNOWN for something it could have watched finish.
# ---------------------------------------------------------------------------
CONFIRM_BUDGET="${CALENDAR_AGENT_CONFIRM_TIMEOUT:-330}"
MAX_TIME="${CALENDAR_DELETE_MAX_TIME:-$((CONFIRM_BUDGET + 10))}"
VERIFY_MAX_TIME="${CALENDAR_DELETE_VERIFY_MAX_TIME:-35}"

for v in CONFIRM_BUDGET MAX_TIME VERIFY_MAX_TIME; do
    case "${!v}" in
        ''|*[!0-9.]*|.|*.*.*) usage "$v must be a number of seconds, got '${!v}'" ;;
    esac
done
if ! python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)' \
        "$MAX_TIME" "$CONFIRM_BUDGET" 2>/dev/null; then
    command -v python3 >/dev/null 2>&1 || usage "python3 is required and was not found on PATH"
    usage "CALENDAR_DELETE_MAX_TIME=${MAX_TIME}s does not outlive calendar-agent's" \
          "${CONFIRM_BUDGET}s mutation budget (CALENDAR_AGENT_CONFIRM_TIMEOUT):" \
          "the DELETE would be abandoned while the operator can still approve it"
fi

enc() { python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$1"; }

# A failed enc() must be fatal: under `set -u` alone it is not, and the URL
# silently loses both ids (the DELETE would go to /calendars//events/).
enc_cal=$(enc "$CAL_ID") || usage "could not URL-encode the calendar id (is python3 on PATH?)"
enc_evt=$(enc "$EVENT_ID") || usage "could not URL-encode the event id (is python3 on PATH?)"
[ -n "$enc_cal" ] && [ -n "$enc_evt" ] || usage "could not URL-encode the ids (is python3 on PATH?)"

URL="$CALENDAR_AGENT_URL/calendars/$enc_cal/events/$enc_evt"

# Every request to calendar-agent goes through here. The token reaches curl on
# stdin (`-H @-`, curl 7.55+), never on its command line, which any process on
# the machine can read. The script does not print it, but tracing (`bash -x`)
# prints each command as expanded, the token included: do not trace a run
# with a real token.
API_KEY="${CALENDAR_AGENT_API_KEY:-}"
API_KEY="${API_KEY//[[:space:]]/}"
agent_curl() {
    if [ -n "$API_KEY" ]; then
        printf 'Authorization: Bearer %s\n' "$API_KEY" | curl -H @- "$@"
    else
        curl "$@"
    fi
}

# Read a field out of a JSON body without assuming it parses at all.
json_field() {
    printf '%s' "$1" | python3 -c '
import json, sys
path = sys.argv[1].split(".")
try:
    node = json.load(sys.stdin)
except Exception:
    print("<unparseable>")
    raise SystemExit(0)
for key in path:
    if not isinstance(node, dict) or key not in node:
        print("<absent>")
        raise SystemExit(0)
    node = node[key]
if node is True:
    print("true")
elif node is False:
    print("false")
elif node is None:
    print("<absent>")
else:
    print(node)
' "$2"
}

# calendar-agent always answers with a {success, ...} envelope. A body without
# one is some other server — most likely the FastAPI router's bare 404 for a
# route that does not exist, i.e. a wrong CALENDAR_AGENT_URL — and its status
# code says nothing about the event.
is_envelope() { case "$1" in true|false) return 0 ;; *) return 1 ;; esac; }

# ---------------------------------------------------------------------------
# 1. Ask for the deletion. Capture body *and* status; a discarded body is how
#    a 200/success:false got read as success on 2026-08-07.
# ---------------------------------------------------------------------------
resp=$(agent_curl -s --max-time "$MAX_TIME" -w '\n%{http_code}' -X DELETE "$URL" 2>/dev/null)
curl_rc=$?

if [ "$curl_rc" -ne 0 ]; then
    claim="unknown"
    echo "DELETE $CAL_ID event $EVENT_ID (deadline ${MAX_TIME}s) -> no response (curl exit $curl_rc)"
else
    code="${resp##*$'\n'}"
    payload="${resp%$'\n'*}"
    body_success=$(json_field "$payload" "success")
    echo "DELETE $CAL_ID event $EVENT_ID (deadline ${MAX_TIME}s) -> HTTP $code (body success: $body_success)"

    if ! is_envelope "$body_success"; then
        # Not calendar-agent's answer: nothing is known, including whether
        # the request reached anything that could act on it.
        claim="unknown"
        echo "  no calendar-agent envelope in the response - is CALENDAR_AGENT_URL right?"
    else
        case "$code" in
            204)                    claim="deleted" ;;
            200) [ "$body_success" = "true" ] && claim="deleted" || claim="failed" ;;
            404|410)                claim="absent" ;;
            # calendar-agent checks the token before the request reaches a
            # route, so its 401 means nothing was attempted. No re-read: it
            # would be refused the same way and establish nothing.
            401)
                usage "calendar-agent refused the bearer token (HTTP 401): set" \
                      "CALENDAR_AGENT_API_KEY to a token this calendar-agent is" \
                      "configured with. Nothing was attempted."
                ;;
            # 408/504: calendar-agent gave up waiting for the operator.
            # Other 5xx: calendar-agent hit a fault *after* the request may
            # have reached the proxy, where it can stay queued for approval.
            # Neither is "nothing outstanding".
            408|5??)                claim="unknown" ;;
            *)                      claim="failed" ;;
        esac
    fi
fi

# ---------------------------------------------------------------------------
# 2. Re-read the event. This, not the answer above, is what decides.
# ---------------------------------------------------------------------------
vresp=$(agent_curl -s --max-time "$VERIFY_MAX_TIME" -w '\n%{http_code}' "$URL" 2>/dev/null)
vcurl_rc=$?

if [ "$vcurl_rc" -ne 0 ]; then
    verify="inconclusive"
    echo "VERIFY $EVENT_ID (verify deadline ${VERIFY_MAX_TIME}s) -> no response (curl exit $vcurl_rc)"
else
    vcode="${vresp##*$'\n'}"
    vpayload="${vresp%$'\n'*}"
    vsuccess=$(json_field "$vpayload" "success")
    vstatus=$(json_field "$vpayload" "event.status")
    echo "VERIFY $EVENT_ID (verify deadline ${VERIFY_MAX_TIME}s) -> HTTP $vcode (event status: $vstatus)"

    if ! is_envelope "$vsuccess"; then
        verify="inconclusive"
        echo "  no calendar-agent envelope in the response - is CALENDAR_AGENT_URL right?"
    else
        case "$vcode" in
            404|410) verify="gone" ;;
            200)
                case "$vstatus" in
                    cancelled)              verify="gone" ;;
                    "<absent>"|"<unparseable>") verify="inconclusive" ;;
                    *)                      verify="present" ;;
                esac
                ;;
            *) verify="inconclusive" ;;
        esac
    fi
fi

# ---------------------------------------------------------------------------
# 3. Reconcile. "Still present" only means failure when nothing is pending.
# ---------------------------------------------------------------------------
case "$verify" in
    gone)
        if [ "$claim" = "absent" ]; then
            echo "RESULT: NOT FOUND - no such event before the delete was" \
                 "attempted (nothing was deleted)"
            exit 3
        fi
        echo "RESULT: SUCCESS - event no longer present"
        exit 0
        ;;
    present)
        if [ "$claim" = "unknown" ]; then
            echo "RESULT: UNKNOWN - event still present, but the deletion had no" \
                 "answer and may still be applied when the operator approves it." \
                 "Re-verify before creating anything in its place."
            exit 2
        fi
        echo "RESULT: FAILURE - event still present (delete reported: $claim)"
        exit 1
        ;;
    *)
        echo "RESULT: UNKNOWN - could not re-read the event, so the outcome is" \
             "unestablished (delete reported: $claim). Re-verify before acting."
        exit 2
        ;;
esac
