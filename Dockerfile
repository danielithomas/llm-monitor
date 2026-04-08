FROM python:3.12-slim

RUN pip install --no-cache-dir llm-monitor

# Create non-root user
RUN useradd --create-home --shell /bin/bash monitor
USER monitor

# Default data directory
ENV LLM_MONITOR_DATA_DIR=/data
ENV LLM_MONITOR_CACHE_DIR=/data/cache
ENV LLM_MONITOR_CONTAINER=1

VOLUME /data

ENTRYPOINT ["llm-monitor", "daemon", "run"]
