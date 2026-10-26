# Pin the base image by digest so it can never change under us.
ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70

# ---------- Stage 1: builder (installs packages and trains the model) ----------
FROM ${PYTHON_IMAGE} AS builder
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY data ./data
RUN python -m src.train.train --data data/churn_v1.csv --model models/model.joblib

# ---------- Stage 2: runtime (small, non-root, only what we need) ----------
FROM ${PYTHON_IMAGE}
ARG GIT_SHA=dev
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH=/app/models/model.joblib \
    MODEL_VERSION=${GIT_SHA}
RUN useradd --system --uid 10001 --no-create-home appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/models ./models
COPY src ./src
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
