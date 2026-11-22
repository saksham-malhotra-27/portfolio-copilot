FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# install dependencies first, so Docker caches this layer
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# then copy your code
COPY . .
RUN uv sync --frozen

CMD ["uv", "run", "python", "main.py"]
