FROM python:3.12-slim

RUN pip install --no-cache-dir clawmeter

# Create non-root user
RUN useradd --create-home --shell /bin/bash monitor
USER monitor

# Default data directory
ENV CLAWMETER_DATA_DIR=/data
ENV CLAWMETER_CACHE_DIR=/data/cache
ENV CLAWMETER_CONTAINER=1

VOLUME /data

ENTRYPOINT ["clawmeter", "daemon", "run"]
