FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

WORKDIR /app

# install dependencies first, so Docker caches this layer
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# then copy your code
COPY . .
RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m", "portfolio_copilot"]
