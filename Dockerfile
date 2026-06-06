FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
COPY apps/gradio-space/pyproject.toml apps/gradio-space/
COPY libs/inference/pyproject.toml libs/inference/
COPY apps/gradio-space/src apps/gradio-space/src
COPY libs/inference/src libs/inference/src

RUN useradd -m -u 1000 user && \
    uv sync --frozen --no-dev --package gradio-space && \
    chown -R user:user /app

USER user
ENV HOME=/home/user \
    PATH="/app/.venv/bin:$PATH"

EXPOSE 7860

CMD ["uv", "run", "--package", "gradio-space", "python", "-m", "gradio_space.app"]
