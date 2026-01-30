# Claude Code Guidelines for Calendar Agent

## Project Overview

Calendar Agent is a FastAPI server that provides a privacy-focused interface between AI orchestrators and the Google Calendar API. It processes calendar data locally and returns only metadata and LLM-generated summaries to calling agents.

## Key Architecture Decisions

1. **Minimal Architecture**: Single-module design mirroring the email-agent pattern
2. **Stateless Operations**: No database; all state lives in the Google Calendar backend
3. **LLM Abstraction**: Provider interface allows swapping between local MLX and hosted APIs
4. **Privacy-First**: Event content processed locally; only summaries returned to cloud

## File Structure

- `calendar_agent/calendar_server.py` - Main FastAPI app with all endpoints
- `calendar_agent/proxy_client.py` - HTTP client for api-proxy server
- `calendar_agent/llm_service.py` - LLM provider abstraction
- `calendar_agent/calendar_utils.py` - Utility functions
- `calendar_agent/exceptions.py` - Custom exception classes

## Running the Server

```bash
uv run python -m calendar_agent.calendar_server
```

## Running Tests

```bash
uv run pytest                    # All tests
uv run pytest -v                 # Verbose output
uv run pytest --cov=calendar_agent  # With coverage
```

## Code Style

- Use ruff for linting and formatting
- Follow existing patterns in the codebase
- All async operations use httpx
- Response models follow success/error pattern

## Common Tasks

### Adding a New Endpoint

1. Define request/response Pydantic models in `calendar_server.py`
2. Add the endpoint function with FastAPI decorators
3. Add tests in `test_calendar_server.py`
4. Document in README.md (required by documentation tests)

### Modifying LLM Behavior

- System prompts are in `llm_service.py`
- Always include security warnings about untrusted content
- Use THINKING_PATTERN regex to strip Qwen3 thinking tags

### Error Handling

- Use `ProxyAuthError` for 401 responses
- Use `ProxyForbiddenError` for 403 responses (including confirmations)
- Use `ProxyError` for other proxy errors
- Use `LLMError` for LLM failures
- Always return `{"success": false, "error": "..."}` pattern

## Testing Guidelines

- Mock `get_calendar_client()` and `get_llm_service()` in tests
- Use `subtests` for documentation verification tests
- Test both success and error paths
- Sample data is in `tests/conftest.py`

## Dependencies

Production:
- fastapi, uvicorn - Web framework
- httpx - Async HTTP client
- pydantic - Data validation
- python-dotenv - Environment variables

Development:
- pytest, pytest-asyncio - Testing
- pytest-subtests - Documentation tests
- ruff - Linting and formatting
