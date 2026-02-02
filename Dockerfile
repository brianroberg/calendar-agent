FROM python:3.12-slim

WORKDIR /app

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy dependency files first (for layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev deps for production)
RUN uv sync --frozen --no-dev

# Copy source code
COPY calendar_agent/ ./calendar_agent/

# Default port
EXPOSE 8082

# Run the FastAPI server
CMD ["uv", "run", "uvicorn", "calendar_agent.calendar_server:app", \
     "--host", "0.0.0.0", \
     "--port", "8082"]
