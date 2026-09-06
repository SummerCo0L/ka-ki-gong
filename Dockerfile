# Ka-Ki-Gong bot container.
#
# The heavy, GPU-oriented dependencies (torch/torchaudio/omnivoice) used only
# by the archived local Hokkien-TTS downloader/server (scripts/archive/) are
# NOT installed here — see the `archive-hokkien-tts` extra in pyproject.toml.
# This keeps the image to pure-Python + numpy + Pipecat/FastAPI/aiortc
# dependencies, all installable from prebuilt wheels (no compiler needed).
#
# All external AI services (OpenRouter, MERaLiON) are called over HTTPS at
# runtime with API keys supplied via environment variables — no model
# weights or secrets are baked into this image.

FROM python:3.12-slim-bookworm

# Official uv static binary (see https://docs.astral.sh/uv/guides/integration/docker/).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

# Install dependencies first (separate layer from app code for build caching).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Now copy the application code and finish the sync (installs the project itself).
COPY bots/ bots/
COPY services/ services/
COPY retrieval/ retrieval/
COPY processors/ processors/
COPY prompt/ prompt/
COPY knowledge_base/ knowledge_base/
COPY data/ data/
COPY scripts/build_index.py scripts/evaluate_retrieval.py scripts/scrape_supportgowhere_schemes.py scripts/
COPY eval/README.md eval/retrieval_eval_set.json eval/
COPY env.example_cpp ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Run as a non-root user.
RUN groupadd --system app && useradd --system --gid app --create-home app \
    && chown -R app:app /app
USER app

EXPOSE 7860

# Bind to 0.0.0.0 so the WebRTC demo is reachable from outside the container;
# Pipecat's runner defaults to --host localhost otherwise. Requires
# OPENROUTER_API_KEY / MERALION_API_KEY to be supplied via environment (see
# env.example_cpp and docker-compose.yml).
CMD ["uv", "run", "bots/bot.py", "--host", "0.0.0.0", "--port", "7860"]
