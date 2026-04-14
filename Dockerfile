# =============================================================================
# Dockerfile — code-reviewer-app:latest
# =============================================================================
# Builds the FastAPI application image.
# Separate from docker/sandbox/Dockerfile (which builds the code sandbox).
#
# Build:
#   docker build -t code-reviewer-app:latest .
#
# Run (standalone, needs external Postgres + ChromaDB):
#   docker run --env-file .env -p 8000:8000 code-reviewer-app:latest
#
# Production: use docker-compose.yml or ECS task definition.
# =============================================================================

# ── Stage 1: Build dependencies ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build deps (needed for psycopg, cryptography, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first (layer cache optimisation)
COPY pyproject.toml ./

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv==0.4.30

# Install all project dependencies into /build/.venv
RUN uv venv /build/.venv && \
    uv pip install --python /build/.venv/bin/python \
        --no-cache \
        -e . 2>/dev/null || \
    uv pip install --python /build/.venv/bin/python \
        --no-cache \
        fastapi uvicorn[standard] \
        sqlalchemy asyncpg alembic \
        psycopg psycopg-pool \
        langgraph langchain-google-genai \
        langsmith \
        chromadb \
        httpx requests \
        python-dotenv pydantic-settings \
        docker \
        gitpython \
        streamlit


# ── Stage 2: Runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Advanced GitHub Code Reviewer"
LABEL org.opencontainers.image.description="Agentic AI PR review — FastAPI + LangGraph + Gemini"
LABEL org.opencontainers.image.version="4.0.0"

WORKDIR /app

# Runtime system deps (libpq for asyncpg, git for repo cloning)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /build/.venv /app/.venv

# Copy application source
COPY app/          ./app/
COPY run.py        ./run.py
COPY pyproject.toml ./pyproject.toml

# Copy Streamlit app (served separately but packaged together)
COPY streamlit_app/ ./streamlit_app/

# Make venv Python the default
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Production logging — emit JSON
ENV LOG_FORMAT=json
ENV LOG_LEVEL=INFO

# FastAPI listens on 0.0.0.0:8000
EXPOSE 8000

# Streamlit listens on 8501 (started separately or via docker-compose)
EXPOSE 8501

# Health check — FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: start FastAPI app
# Override in docker-compose.yml to start Streamlit instead
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8000"]