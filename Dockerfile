# ---- Stage 1: builder — installs dependencies into an isolated venv ----
FROM python:3.11-slim AS builder
WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: runtime — only the venv + app code, no build tools ----
FROM python:3.11-slim
WORKDIR /app

# Dedicated non-root user; never run the app as root
RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Only app code + the baked-in model artifact — no tests, no data, no .git
COPY app/ ./app/
COPY artifacts/ ./artifacts/

# The venv was copied in as root; hand ownership to appuser
RUN chown -R appuser:appuser /app /opt/venv
USER appuser

ENV MODEL_DIR=artifacts
EXPOSE 8080

# Useful for local `docker run`; hosted platforms use their own probe instead
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8080}/health')" || exit 1

# Shell form so ${PORT} expands — Render/Cloud Run/HF each inject a different port
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
