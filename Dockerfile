FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY . /tmp/build
RUN pip install --no-cache-dir build && python -m build --wheel /tmp/build -o /tmp/dist

FROM python:3.12-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash monitor

COPY --from=builder /tmp/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl

# Create data directory owned by monitor before switching user
RUN mkdir -p /data/cache && chown -R monitor:monitor /data

USER monitor

# Default data directory
ENV CLAWMETER_DATA_DIR=/data
ENV CLAWMETER_CACHE_DIR=/data/cache
ENV CLAWMETER_CONTAINER=1

VOLUME /data

ENTRYPOINT ["clawmeter", "daemon", "run"]
