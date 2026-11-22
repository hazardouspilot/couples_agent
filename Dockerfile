# The CLI app in a container. Interactive, so start it with:
#   docker compose run --rm --service-ports app
# Secrets come from .env at RUN time (env_file in docker-compose.yml); they
# are never copied into the image (.dockerignore excludes .env).
FROM python:3.11.4-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install first, copy cli.py last: editing cli.py doesn't redo the install layer.
COPY pyproject.toml ./
COPY app ./app
RUN pip install .
COPY cli.py ./

# Run as a non-root user; databases live on a volume at /data so memory
# survives container restarts (and stays out of the image).
RUN useradd --create-home appuser && mkdir /data && chown appuser /data
USER appuser
ENV COUPLES_AGENT_DB=/data/couples_agent.db \
    COUPLES_AGENT_CHECKPOINTS=/data/checkpoints.db \
    METRICS_PORT=8000

EXPOSE 8000
CMD ["python", "cli.py"]
