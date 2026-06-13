FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock .python-version README.md models.yaml voice_models.yaml ./
COPY apps/gradio-space/pyproject.toml apps/gradio-space/README.md apps/gradio-space/
COPY libs/inference/pyproject.toml libs/inference/README.md libs/inference/
COPY libs/agent/pyproject.toml libs/agent/README.md libs/agent/
COPY libs/echocoach/pyproject.toml libs/echocoach/README.md libs/echocoach/
COPY apps/gradio-space/src apps/gradio-space/src
COPY apps/gradio-space/static apps/gradio-space/static
COPY libs/inference/src libs/inference/src
COPY libs/agent/src libs/agent/src
COPY libs/echocoach/src libs/echocoach/src
COPY skills skills

RUN useradd -m -u 1000 user && \
    uv sync --frozen --no-dev --package gradio-space && \
    chown -R user:user /app

USER user
ENV HOME=/home/user \
    PATH="/app/.venv/bin:$PATH" \
    AGENT_OUTPUTS_DIR=/tmp/agent_outputs \
    AGENT_TRACES_DIR=/tmp/agent_traces

RUN mkdir -p /tmp/agent_outputs /tmp/agent_traces

EXPOSE 7860

CMD ["uv", "run", "--package", "gradio-space", "python", "-m", "gradio_space.app"]
