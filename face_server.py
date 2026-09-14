"""
face_server.py - Face recognition backend for the ESP32-CAM door lock
(push architecture - the ESP32 sends frames to us, we don't poll it)

The ESP32 sketch (esp32cam_wifi_relay_node.ino) makes all outbound calls:

  POST /register   multipart fields: key, name, image(jpg)
                    -> one call per "Capture Sample" click on the ESP32's
                       /register page. Saves the sample, retrains, replies
                       with a short plain-text status.

  POST /recognize   multipart fields: key, image(jpg)
                    -> called by the ESP32 every RECOGNITION_INTERVAL_MS.
                       Replies with JSON: {"match": true/false, "name": "..."}
                       The ESP32 does a substring check for "match":true, so
                       keep responses exactly in this shape.

Admin (protected by the same API key, via query string or header):
  GET  /list?key=...             -> {"names": [...]}
  POST /delete?key=...&name=...  -> removes a person and retrains

Auth: every route above requires the key to match API_KEY. Set API_KEY as
a Render environment variable rather than hardcoding it - it must match
API_KEY in the .ino sketch.

Storage note (important for Render): the container filesystem is EPHEMERAL.
DATASET_DIR contents (registered faces + labels.json) are lost on every
redeploy, restart, or scale event unless you attach a Render Persistent
Disk mounted at DATASET_DIR. For a hobby project this is often fine (just
re-register faces after a redeploy); for anything you don't want to
re-register repeatedly, add a persistent disk in the Render dashboard.

Local run (no Docker):
    pip install -r requirements.txt
    export API_KEY=changeme123
    python face_server.py
"""

import os
import json
import threading

import cv2
import numpy as np
from flask import Flask, request, jsonify

# ============================= USER CONFIG =================================

API_KEY = os.environ.get("API_KEY", "changeme123")  # must match API_KEY in the .ino

MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "70"))  # LBPH distance, lower = better match
DATASET_DIR = os.environ.get("DATASET_DIR", "dataset")
FACE_SIZE = (200, 200)

# ============================================================================

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
recognizer = cv2.face.LBPHFaceRecognizer_create()

lock = threading.Lock()
id_to_name = {}   # {"0": "alice", "1": "bob", ...}
trained = False


# ------------------------------- helpers ------------------------------------

def labels_path():
    return os.path.join(DATASET_DIR, "labels.json")


def load_labels():
    path = labels_path()
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_labels():
    with open(labels_path(), "w") as f:
        json.dump(id_to_name, f)


def train_recognizer():
    """Rebuild the recognizer from every image currently in dataset/."""
    global trained
    faces, labels = [], []
    for id_str, name in id_to_name.items():
        person_dir = os.path.join(DATASET_DIR, name)
        if not os.path.isdir(person_dir):
            continue
        for fname in os.listdir(person_dir):
            img = cv2.imread(os.path.join(person_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(int(id_str))
    if faces:
        recognizer.train(faces, np.array(labels))
        trained = True
        print(f"Trained on {len(faces)} images across {len(id_to_name)} people.", flush=True)
    else:
        trained = False
        print("No training data yet - register at least one face.", flush=True)


def decode_upload(file_storage):
    """Decode an uploaded JPEG (werkzeug FileStorage) into a BGR OpenCV image."""
    arr = np.frombuffer(file_storage.read(), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def detect_face(gray):
    """Returns the largest detected face as a cropped grayscale image, or None."""
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    return gray[y:y + h, x:x + w]


def check_key():
    key = request.form.get("key") or request.args.get("key")
    return key == API_KEY


# ------------------------------- routes --------------------------------------

@app.route("/")
def health():
    with lock:
        people = sorted(set(id_to_name.values()))
    return jsonify(status="ok", trained=trained, people=people)


@app.route("/register", methods=["POST"])
def register():
    if not check_key():
        return "bad key", 403

    name = request.form.get("name", "").strip()
    image = request.files.get("image")
    if not name:
        return "missing name", 400
    if image is None:
        return "missing image", 400

    img = decode_upload(image)
    if img is None:
        return "could not decode image", 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = detect_face(gray)
    if face is None:
        return "no face detected - try again, closer to the camera", 200
    face = cv2.resize(face, FACE_SIZE)

    with lock:
        person_dir = os.path.join(DATASET_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        existing = len(os.listdir(person_dir))
        cv2.imwrite(os.path.join(person_dir, f"{existing}.png"), face)

        existing_ids = [int(k) for k in id_to_name.keys()]
        new_id = None
        for k, v in id_to_name.items():
            if v == name:
                new_id = int(k)
                break
        if new_id is None:
            new_id = max(existing_ids) + 1 if existing_ids else 0
            id_to_name[str(new_id)] = name

        save_labels()
        train_recognizer()
        total = len(os.listdir(person_dir))

    return f"sample saved ({total} total for {name})", 200


@app.route("/recognize", methods=["POST"])
def recognize():
    if not check_key():
        return jsonify(match=False, error="bad key"), 403

    image = request.files.get("image")
    if image is None:
        return jsonify(match=False, error="missing image"), 400

    img = decode_upload(image)
    if img is None:
        return jsonify(match=False, error="could not decode image"), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = detect_face(gray)
    if face is None:
        return jsonify(match=False)
    face = cv2.resize(face, FACE_SIZE)

    with lock:
        if not trained:
            return jsonify(match=False)
        label, confidence = recognizer.predict(face)
        name = id_to_name.get(str(label))

    if name and confidence < MATCH_THRESHOLD:
        return jsonify(match=True, name=name)
    return jsonify(match=False)


@app.route("/list")
def list_faces():
    if not check_key():
        return jsonify(error="bad key"), 403
    with lock:
        return jsonify(names=sorted(set(id_to_name.values())))


@app.route("/delete", methods=["POST"])
def delete_face():
    if not check_key():
        return jsonify(error="bad key"), 403
    name = request.args.get("name", "").strip()
    with lock:
        for k in [k for k, v in id_to_name.items() if v == name]:
            del id_to_name[k]
        save_labels()
        person_dir = os.path.join(DATASET_DIR, name)
        if os.path.isdir(person_dir):
            for f in os.listdir(person_dir):
                os.remove(os.path.join(person_dir, f))
            os.rmdir(person_dir)
        train_recognizer()
    return jsonify(ok=True)


# ------------------------------- startup --------------------------------------

os.makedirs(DATASET_DIR, exist_ok=True)
id_to_name.update(load_labels())
train_recognizer()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Listening on 0.0.0.0:{port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
