# The demo as one runnable image: `docker run -p 8050:8050 <image>`.
#
# The app is installed editable into /app rather than as a wheel on purpose. It locates
# `data/curves.npz` relative to its own source file, so a wheel dropped into site-packages
# would look for the extract three directories above the wrong place and fail to start.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Everything below runs as this user, including the installs. Creating it first and
# building into its home is what keeps the virtualenv out of a later `chown -R`, which
# would otherwise copy the whole torch install into a second layer.
RUN useradd --create-home --uid 1000 demo && mkdir /app && chown demo:demo /app
USER demo

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/home/demo/venv \
    PATH="/home/demo/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so a code change does not re-download torch.
COPY --chown=demo:demo pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra deploy --no-install-project

COPY --chown=demo:demo . .
RUN uv sync --frozen --no-dev --extra deploy

# Hosts that pick their own port set PORT; everything else gets 8050. WEB_CONCURRENCY is
# the same convention: two workers is right on a normal host, but the whole process tree
# holds roughly 570 MB idle and 640 MB while an attribution is being computed, so a small
# instance wants one.
ENV PORT=8050 \
    WEB_CONCURRENCY=2
EXPOSE 8050

# gunicorn, not `spp2422-demo serve`: that runs Flask's development server, which is not
# meant to face anyone but its author. Callbacks keep no server-side state, so workers are
# interchangeable and a request may land on any of them.
CMD gunicorn --bind "0.0.0.0:${PORT}" --workers "${WEB_CONCURRENCY}" --threads 4 \
    --timeout 120 spp2422_demo.app:server
