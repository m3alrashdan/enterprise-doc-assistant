# ---- Build stage: install dependencies into an isolated venv ----------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
# CPU-only torch keeps the image several GB smaller than the default CUDA build.
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /tmp/requirements.txt

# ---- Final stage: slim runtime image, non-root user --------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/srv/app/data/hf-cache

RUN groupadd -g 1001 app && useradd -u 1001 -g app -m app

WORKDIR /srv/app
COPY --from=builder /opt/venv /opt/venv
COPY app ./app
COPY static ./static
RUN mkdir -p data && chown -R app:app /srv/app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
