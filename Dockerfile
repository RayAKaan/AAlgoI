FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY core/ core/
COPY algorithms/ algorithms/
COPY interface/ interface/
COPY pipeline.py config/config.json ./

RUN pip install --no-cache-dir build && \
    python -m build --wheel

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r aalgoi && useradd -r -g aalgoi -d /app -s /sbin/nologin aalgoi

WORKDIR /app

COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm -rf /tmp/*.whl

RUN rm -rf /root/.cache/pip

COPY config/config.json config.json

USER aalgoi

EXPOSE 7860 8000

ENTRYPOINT ["aalgoi"]
CMD ["--help"]
