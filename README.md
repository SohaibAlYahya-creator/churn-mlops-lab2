# churn-mlops-lab2

[![CI](https://github.com/SohaibAlYahya-creator/churn-mlops-lab2/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SohaibAlYahya-creator/churn-mlops-lab2/actions/workflows/ci.yml)

A customer churn model served as a tested, containerised FastAPI service.
Every pull request is linted and tested. Every merge to `main` builds the
image, scans it with Trivy and pushes it to GHCR, tagged with the commit SHA.

## Run it (one command)

Requires Docker.

    docker compose up --build

The service listens on http://localhost:8000 (interactive docs at `/docs`).

## Try it

    curl -s localhost:8000/health
    curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" \
      -d '{"tenure_months":5,"monthly_charges":89.9,"contract":"month-to-month","internet_service":"fiber","support_calls":3}'

The prediction returns `churn_probability` 0.866. Invalid input, for example
`"contract": "weekly"`, is rejected with HTTP 422.

## Use the published image

    docker run --rm -p 8000:8000 ghcr.io/sohaibalyahya-creator/churn-mlops-lab2@sha256:6a758396f7b031cac765ff6767d5cfc7f2dcdbbe20aa4eac3929a65177e49a38

Every image is tagged with the full commit SHA of the code that built it, and
`/health` reports the same value as `model_version`.

## Develop

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements-dev.txt
    python -m src.train.train
    python -m pytest -q
    ruff check . && ruff format --check .

## Layout

- `src/train/train.py`: trains and saves the model.
- `src/app/main.py`: FastAPI service (`/health`, `/predict`) with pydantic validation.
- `tests/`: pytest suite.
- `Dockerfile`: multi-stage, non-root, pinned base image, HEALTHCHECK.
- `.github/workflows/ci.yml`: lint, test, build, Trivy scan, push to GHCR.
