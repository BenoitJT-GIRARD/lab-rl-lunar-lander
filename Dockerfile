# --- build: resolve and install, with the tooling that does it -------------
FROM python:3.12-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv==0.11.6

COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src

# `--extra ui` because the second service of the compose is a Streamlit page, and one image
# serves both. The torch that lands here is the CPU build: `tool.uv.sources` routes it to
# PyTorch's CPU index everywhere but Windows, which is what keeps this image to a gigabyte
# of tensor library instead of pulling a CUDA runtime no container of this project can use.
RUN uv sync --frozen --no-dev --extra ui


# --- runtime: the environment, the model, and nothing that built them ------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"
# pygame builds the replay frames through SDL, which looks for a display and aborts without
# one. The image has no X server and never will: the frames are pixels in memory, and they
# leave over HTTP.
ENV SDL_VIDEODRIVER=dummy

WORKDIR /app

# uid 1000, named `lander`. Everything this image serves is read: one checkpoint, two CSVs.
# Nothing it does asks for an owner, and a bind mount would come back owned by root.
RUN useradd --create-home --uid 1000 lander

COPY --from=build --chown=lander:lander /app/.venv /app/.venv
COPY --chown=lander:lander pyproject.toml README.md /app/
COPY --chown=lander:lander src /app/src
COPY --chown=lander:lander models /app/models
COPY --chown=lander:lander reports /app/reports
COPY --chown=lander:lander .streamlit /app/.streamlit
COPY --chown=lander:lander .env.example /app/.env.example

EXPOSE 8000 8501

# /health, and not /ready: `docs/operations.md` says what each of the two answers, and what
# Docker does to a container probed on the wrong one.
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

USER lander

CMD ["uvicorn", "rl_lander.api:app", "--host", "0.0.0.0", "--port", "8000"]
