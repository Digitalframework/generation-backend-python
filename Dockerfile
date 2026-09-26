# ---------------------------------------------------------------------------
# ImageGenBackend — Flask resource server, started by gunicorn on container
# start. Python 3.10 is not incidental: requirements.txt caps rembg at 2.0.69
# because 2.0.81+ needs >=3.11, so moving up the base image breaks that pin.
# ---------------------------------------------------------------------------
FROM python:3.10-slim-bookworm

# opencv is imported at module scope by app/vggt_scene.py, which app/
# generate_routes.py imports at module scope — so libGL has to be here or the
# app does not even start. git is for the `vggt @ git+https://…` requirement.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Set to 0 for a ~300MB image that starts and serves every route except a real
# generation: torch, vggt and rembg are all imported lazily, so leaving them
# out costs nothing until /api/generate actually runs the scene solver.
# The full build pulls several GB of CPU torch wheels.
ARG WITH_ML=1

COPY requirements.txt .
RUN if [ "$WITH_ML" = "1" ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    else \
        grep -vE '^(torch|vggt|rembg)' requirements.txt > /tmp/req-web.txt && \
        pip install --no-cache-dir -r /tmp/req-web.txt; \
    fi

COPY . .

# config.ROOM_IMAGE_PATH is relative, so it resolves under WORKDIR. The repo
# ships assets/ empty; mount a room.png over this or POST `scene` per request.
RUN mkdir -p assets

# Nothing here writes to the image at runtime.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/api/health', timeout=4).status == 200 else 1)"

# A generation holds its worker for the whole FLUX round trip, which
# FLUX_TIMEOUT bounds at 180s — so the gunicorn timeout has to sit above that
# or it kills the worker mid-poll. Threads, not more workers: each worker that
# touches VGGT loads its own copy of the model.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--threads", "4", \
     "--timeout", "300", \
     "--access-logfile", "-", \
     "wsgi:app"]
