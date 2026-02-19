"""Tests for LLM service API key authentication support."""

import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from calendar_agent.llm_service import LocalMLXProvider, LLMService


# ============================================================================
# LocalMLXProvider API Key Tests
# ============================================================================


class TestLocalMLXProviderInit:
    """Tests for LocalMLXProvider constructor API key handling."""

    def test_explicit_api_key_stored(self):
        """Provider stores an explicitly passed API key."""
        provider = LocalMLXProvider(api_key="sk-test-123")
        assert provider.api_key == "sk-test-123"

    def test_empty_string_api_key_from_env(self):
        """Provider uses empty string when env var is unset."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLM_API_KEY", None)
            # Re-import to pick up env change
            import importlib
            import calendar_agent.llm_service as mod
            importlib.reload(mod)
            provider = mod.LocalMLXProvider()
            assert provider.api_key == ""

    def test_api_key_from_env_var(self):
        """Provider reads LLM_API_KEY from environment when no arg passed."""
        with patch.dict(os.environ, {"LLM_API_KEY": "sk-from-env"}):
            import importlib
            import calendar_agent.llm_service as mod
            importlib.reload(mod)
            provider = mod.LocalMLXProvider()
            assert provider.api_key == "sk-from-env"

    def test_explicit_api_key_overrides_env(self):
        """Explicit api_key argument takes precedence over env var."""
        with patch.dict(os.environ, {"LLM_API_KEY": "sk-from-env"}):
            import importlib
            import calendar_agent.llm_service as mod
            importlib.reload(mod)
            provider = mod.LocalMLXProvider(api_key="sk-explicit")
            assert provider.api_key == "sk-explicit"

    def test_explicit_empty_string_overrides_env(self):
        """Passing api_key='' explicitly disables auth even if env is set."""
        with patch.dict(os.environ, {"LLM_API_KEY": "sk-from-env"}):
            import importlib
            import calendar_agent.llm_service as mod
            importlib.reload(mod)
            provider = mod.LocalMLXProvider(api_key="")
            assert provider.api_key == ""


class TestLocalMLXProviderAuthHeader:
    """Tests for Authorization header in generate() requests."""

    @pytest.fixture
    def mock_response(self):
        """Create a mock httpx response with valid LLM output."""
        response = AsyncMock(spec=httpx.Response)
        response.status_code = 200
        response.raise_for_status = lambda: None
        response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        return response

    async def test_sends_bearer_token_when_api_key_set(self, mock_response):
        """Generate sends Authorization: Bearer header when api_key is provided."""
        provider = LocalMLXProvider(
            url="http://fake-llm/v1/chat/completions",
            api_key="sk-novita-key",
        )

        with patch("calendar_agent.llm_service.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await provider.generate("system prompt", "user content")

            mock_client.post.assert_called_once()
            call_kwargs = mock_client.post.call_args
            headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
            assert headers["Authorization"] == "Bearer sk-novita-key"

    async def test_no_auth_header_when_api_key_empty(self, mock_response):
        """Generate omits Authorization header when api_key is empty."""
        provider = LocalMLXProvider(
            url="http://fake-llm/v1/chat/completions",
            api_key="",
        )

        with patch("calendar_agent.llm_service.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await provider.generate("system prompt", "user content")

            call_kwargs = mock_client.post.call_args
            headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
            assert "Authorization" not in headers

    async def test_sends_correct_url(self, mock_response):
        """Generate posts to the configured URL."""
        provider = LocalMLXProvider(
            url="http://novita.ai/v1/chat/completions",
            api_key="sk-test",
        )

        with patch("calendar_agent.llm_service.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await provider.generate("sys", "user")

            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://novita.ai/v1/chat/completions"

    async def test_sends_correct_model_in_body(self, mock_response):
        """Generate includes the configured model in the request body."""
        provider = LocalMLXProvider(
            url="http://fake/v1/chat/completions",
            model="deepseek/deepseek-v3-0324",
            api_key="sk-test",
        )

        with patch("calendar_agent.llm_service.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await provider.generate("sys", "user")

            call_kwargs = mock_client.post.call_args
            body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert body["model"] == "deepseek/deepseek-v3-0324"

    async def test_content_type_always_set(self, mock_response):
        """Generate always sets Content-Type: application/json."""
        provider = LocalMLXProvider(
            url="http://fake/v1/chat/completions",
            api_key="sk-test",
        )

        with patch("calendar_agent.llm_service.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await provider.generate("sys", "user")

            call_kwargs = mock_client.post.call_args
            headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
            assert headers["Content-Type"] == "application/json"


# ============================================================================
# LLMService Integration with API Key
# ============================================================================


class TestLLMServiceWithApiKey:
    """Tests that LLMService correctly passes through to provider with API key."""

    async def test_service_uses_provider_with_api_key(self):
        """LLMService delegates to provider that has API key configured."""
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = "Summary of event"

        service = LLMService(provider=mock_provider)
        result = await service.summarize_event(
            {"id": "e1", "summary": "Meeting", "start": {"dateTime": "2024-01-15T10:00:00Z"}},
            format="brief",
        )

        mock_provider.generate.assert_called_once()
        assert "summary" in result
