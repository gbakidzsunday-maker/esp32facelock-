FROM python:3.11-slim

# OpenCV (even the "headless" build) still links against a few system
# libraries at import time. Without these you'll get an
# "ImportError: libGL.so.1: cannot open shared object file" on startup.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY face_server.py .

# Render sets $PORT at runtime and routes traffic to it - the app reads
# this env var itself, so we don't hardcode a port here.
ENV PYTHONUNBUFFERED=1

CMD ["python", "face_server.py"]
