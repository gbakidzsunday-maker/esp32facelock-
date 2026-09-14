FROM python:3.11-slim

# opencv-contrib-python-headless still dynamically links a few system
# shared libraries even without GUI support - install the small set it
# actually needs instead of the full python:3.11 image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY face_server.py .

# Render sets $PORT itself and routes traffic to it - do not hardcode a
# port here, gunicorn's bind below reads it from the environment.
ENV PORT=10000
EXPOSE 10000

# Single worker: recognizer state (id_to_name / trained model) lives in
# process memory and is protected by a threading.Lock, not shared across
# worker processes. Use --threads for concurrency instead of more workers.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 60 face_server:app"]
