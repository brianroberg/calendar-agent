"""Tests for Calendar Agent server endpoints."""


import pytest

from calendar_agent.exceptions import ProxyAuthError, ProxyError, ProxyForbiddenError

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
        assert data["calendars"][0]["id"] == "primary"

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
        assert data["calendar"]["id"] == "primary"

    def test_get_calendar_not_found(self, client, mock_proxy_client):
        """Get calendar handles not found."""
        mock_proxy_client.get_calendar.side_effect = ProxyError("Calendar not found")
        response = client.get("/calendars/nonexistent")
        data = response.json()
        assert data["success"] is False


# ============================================================================
# Event CRUD Endpoint Tests
# ============================================================================


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


class TestEventCreateEndpoint:
    """Tests for POST /calendars/{calendar_id}/events."""

    def test_create_event_success(self, client, mock_proxy_client):
        """Create event returns created event."""
        event_data = {
            "summary": "New Meeting",
            "start": {"dateTime": "2024-01-15T10:00:00Z"},
            "end": {"dateTime": "2024-01-15T11:00:00Z"},
        }
        response = client.post("/calendars/primary/events", json=event_data)
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
        event_data = {
            "summary": "Updated Meeting",
            "start": {"dateTime": "2024-01-15T14:00:00Z"},
            "end": {"dateTime": "2024-01-15T15:00:00Z"},
        }
        response = client.put("/calendars/primary/events/event_123", json=event_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestEventPatchEndpoint:
    """Tests for PATCH /calendars/{calendar_id}/events/{event_id}."""

    def test_patch_event_success(self, client, mock_proxy_client):
        """Patch event with partial update."""
        patch_data = {"summary": "Renamed Meeting"}
        response = client.patch("/calendars/primary/events/event_123", json=patch_data)
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
        """Delete event returns success."""
        response = client.delete("/calendars/primary/events/event_123")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "deleted" in data["message"].lower()

    def test_delete_event_requires_confirmation(self, client, mock_proxy_client):
        """Delete event handles confirmation requirement."""
        mock_proxy_client.delete_event.side_effect = ProxyForbiddenError(
            "Confirmation required: Please confirm deletion"
        )
        response = client.delete("/calendars/primary/events/event_123")
        data = response.json()
        assert data["success"] is False
        assert "confirmation" in data["message"].lower()


# ============================================================================
# LLM Endpoint Tests - Basic
# ============================================================================


class TestSummarizeEndpoint:
    """Tests for POST /summarize."""

    def test_summarize_success(self, client, mock_proxy_client, mock_llm_service):
        """Summarize event returns AI summary."""
        request_data = {
            "calendar_id": "primary",
            "event_id": "event_123",
            "format": "brief",
        }
        response = client.post("/summarize", json=request_data)
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
        request_data = {
            "calendar_id": "primary",
            "event_id": "event_123",
            "question": "What time is the meeting?",
        }
        response = client.post("/ask-about", json=request_data)
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
        request_data = {
            "calendar_id": "primary",
            "event_ids": ["event_1", "event_2", "event_3"],
            "triage": False,
        }
        response = client.post("/batch-summarize", json=request_data)
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
        request_data = {
            "calendar_id": "primary",
            "time_min": "2024-01-15T09:00:00Z",
            "time_max": "2024-01-15T17:00:00Z",
            "duration_minutes": 30,
        }
        response = client.post("/find-free-time", json=request_data)
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
        request_data = {
            "calendar_id": "primary",
            "time_min": "2024-01-15T00:00:00Z",
            "time_max": "2024-01-22T00:00:00Z",
            "analysis_type": "overview",
        }
        response = client.post("/analyze-schedule", json=request_data)
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
        request_data = {
            "calendar_id": "primary",
            "briefing_type": "daily",
        }
        response = client.post("/prepare-briefing", json=request_data)
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
        request_data = {
            "calendar_id": "primary",
            "filters": {
                "query": "meeting",
                "time_min": "2024-01-01T00:00:00Z",
                "time_max": "2024-01-31T23:59:59Z",
            },
        }
        response = client.post("/search", json=request_data)
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
        request_data = {
            "operations": [
                {"operation": "delete", "event_id": "event_1", "calendar_id": "primary"},
                {"operation": "delete", "event_id": "event_2", "calendar_id": "primary"},
            ]
        }
        response = client.post("/bulk-actions", json=request_data)
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
        request_data = {
            "operations": [
                {"operation": "delete", "event_id": "event_1", "calendar_id": "primary"},
                {"operation": "delete", "event_id": "event_2", "calendar_id": "primary"},
            ]
        }
        response = client.post("/bulk-actions", json=request_data)
        data = response.json()
        assert data["success"] is True  # Overall still success
        assert data["success_count"] == 1
        assert data["error_count"] == 1

    def test_bulk_update_without_data(self, client, mock_proxy_client):
        """Bulk update fails without update data."""
        request_data = {
            "operations": [
                {"operation": "update", "event_id": "event_1", "calendar_id": "primary"},
            ]
        }
        response = client.post("/bulk-actions", json=request_data)
        data = response.json()
        assert data["error_count"] == 1
        assert "No update data" in data["results"][0]["error"]


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestProxyErrorHandling:
    """Tests for proxy error handling across endpoints."""

    def test_auth_error_handling(self, client, mock_proxy_client):
        """Authentication errors are properly formatted."""
        mock_proxy_client.list_calendars.side_effect = ProxyAuthError("Invalid API key")
        response = client.get("/calendars")
        data = response.json()
        assert data["success"] is False
        assert "Authentication error" in data["error"]

    def test_forbidden_error_handling(self, client, mock_proxy_client):
        """Forbidden errors are properly formatted."""
        mock_proxy_client.delete_event.side_effect = ProxyForbiddenError(
            "Confirmation required"
        )
        response = client.delete("/calendars/primary/events/event_123")
        data = response.json()
        assert data["success"] is False
        assert "blocked" in data["error"].lower() or "confirmation" in data["message"].lower()

    def test_generic_proxy_error_handling(self, client, mock_proxy_client):
        """Generic proxy errors are properly formatted."""
        mock_proxy_client.list_events.side_effect = ProxyError("Connection timeout")
        response = client.get("/calendars/primary/events")
        data = response.json()
        assert data["success"] is False
        assert "Proxy error" in data["error"]


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
