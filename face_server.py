"""
face_server.py - Face recognition server for the ESP32-CAM door lock (Option B)

What this does:
  - Repeatedly fetches a JPEG snapshot from the ESP32-CAM at /capture
  - Detects faces with OpenCV's Haar cascade, recognizes them with LBPH
  - On a confident match, calls the ESP32's /grant endpoint to trigger the
    LED + lock sequence
  - Serves a small web page (http://<this-pc's-ip>:5000) for registering
    new faces and managing the list of registered people

Install (Windows/Mac/Linux, no compiler needed):
    pip install -r requirements.txt

Before running:
  - Flash esp32cam_relay_node.ino to the ESP32-CAM, get its IP address from
    the Serial Monitor, and set ESP32_IP below.
  - Make sure GRANT_KEY here matches GRANT_KEY in the .ino file.
  - Make sure this PC and the ESP32-CAM are on the same WiFi network.

Run:
    python face_server.py
Then open http://localhost:5000 in a browser on this PC (or
http://<this-pc's-LAN-IP>:5000 from another device on the network) to
register faces.
"""

import os
import json
import time
import threading

import cv2
import numpy as np
import requests
from flask import Flask, Response, request, jsonify, render_template_string

# ============================= USER CONFIG =================================
# All of these can be overridden with environment variables so the same image
# works locally and on Render. Set them in the Render dashboard under
# Environment, or in a local .env / `docker run -e ...`.

ESP32_IP = os.environ.get("ESP32_IP", "192.168.1.50")     # <-- your ESP32's IP or tunnel hostname
GRANT_KEY = os.environ.get("GRANT_KEY", "changeme123")    # <-- must match GRANT_KEY in the .ino sketch

POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", 1.0))   # seconds between recognition attempts
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", 70))  # LBPH distance - LOWER is a better match
GRANT_COOLDOWN = float(os.environ.get("GRANT_COOLDOWN", 20))   # seconds between grants

# On Render's free tier the filesystem is EPHEMERAL - it's wiped on every
# deploy/restart, so registered faces will be lost. Point this at a Render
# "Disk" mount (e.g. /data) if you add one, otherwise treat registrations as
# temporary.
DATASET_DIR = os.environ.get("DATASET_DIR", "dataset")
SAMPLES_PER_PERSON = int(os.environ.get("SAMPLES_PER_PERSON", 15))
FACE_SIZE = (200, 200)

PORT = int(os.environ.get("PORT", 5000))  # Render injects PORT - don't hardcode 5000 in production

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
        print(f"Trained on {len(faces)} images across {len(id_to_name)} people.")
    else:
        trained = False
        print("No training data yet - register at least one face.")


def fetch_frame():
    r = requests.get(f"http://{ESP32_IP}/capture", timeout=5)
    r.raise_for_status()
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def detect_face(gray):
    """Returns the largest detected face as a cropped grayscale image, or None."""
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    return gray[y:y + h, x:x + w]


# ------------------------------- recognition loop ---------------------------

def recognition_loop():
    last_grant = 0.0
    while True:
        time.sleep(POLL_INTERVAL)
        try:
            img = fetch_frame()
        except Exception as e:
            print("capture fetch failed:", e)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = detect_face(gray)
        if face is None:
            continue
        face = cv2.resize(face, FACE_SIZE)

        with lock:
            if not trained:
                continue
            label, confidence = recognizer.predict(face)
            name = id_to_name.get(str(label))

        print(f"Detected -> {name} (confidence {confidence:.1f}, lower=better)")

        if name and confidence < MATCH_THRESHOLD:
            now = time.time()
            if now - last_grant > GRANT_COOLDOWN:
                last_grant = now
                try:
                    requests.get(
                        f"http://{ESP32_IP}/grant",
                        params={"key": GRANT_KEY},
                        timeout=5,
                    )
                    print(f"ACCESS GRANTED: {name}")
                except Exception as e:
                    print("grant request failed:", e)


# ------------------------------- web UI --------------------------------------

INDEX_HTML = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Face Lock Registration</title>
<style>
  body{font-family:sans-serif;max-width:420px;margin:20px auto;text-align:center}
  img{width:100%;border-radius:8px;border:2px solid #444}
  input{padding:8px;width:70%;font-size:16px}
  button{padding:8px 16px;font-size:16px;margin:4px}
  button.danger{background:#c33;color:#fff;border:none;border-radius:4px}
  #status{margin:10px 0;min-height:20px}
</style></head><body>
  <h2>Face Lock Registration</h2>
  <img id="snap" src="/proxy_capture">
  <p><input id="name" placeholder="Person's name"></p>
  <p><button onclick="doRegister()">Register Face</button></p>
  <div id="status"></div>
  <h3>Registered faces</h3>
  <ul id="list"></ul>
<script>
setInterval(()=>{document.getElementById('snap').src='/proxy_capture?t='+Date.now();}, 800);

function doRegister(){
  const name = document.getElementById('name').value.trim();
  if(!name){ alert('Enter a name first'); return; }
  document.getElementById('status').innerText = 'Registering... look at the camera';
  fetch('/register?name='+encodeURIComponent(name), {method:'POST'})
    .then(r=>r.json())
    .then(j=>{
      document.getElementById('status').innerText = j.message;
      refreshList();
    });
}

function refreshList(){
  fetch('/list').then(r=>r.json()).then(j=>{
    const ul = document.getElementById('list');
    ul.innerHTML = '';
    j.names.forEach(n=>{
      const li = document.createElement('li');
      li.innerHTML = n + ' <button class="danger" onclick="del(\\''+n+'\\')">Delete</button>';
      ul.appendChild(li);
    });
  });
}
function del(n){
  fetch('/delete?name='+encodeURIComponent(n), {method:'POST'}).then(refreshList);
}
refreshList();
</script>
</body></html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/proxy_capture")
def proxy_capture():
    try:
        r = requests.get(f"http://{ESP32_IP}/capture", timeout=5)
        return Response(r.content, mimetype="image/jpeg")
    except Exception:
        return Response(status=502)


@app.route("/register", methods=["POST"])
def register():
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify(ok=False, message="missing name"), 400

    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    got, attempts = 0, 0
    while got < SAMPLES_PER_PERSON and attempts < SAMPLES_PER_PERSON * 6:
        attempts += 1
        try:
            img = fetch_frame()
        except Exception:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = detect_face(gray)
        if face is None:
            continue
        face = cv2.resize(face, FACE_SIZE)
        cv2.imwrite(os.path.join(person_dir, f"{got}.png"), face)
        got += 1
        time.sleep(0.3)

    if got == 0:
        return jsonify(ok=False, message="No face detected - try again, closer to the camera")

    with lock:
        existing_ids = [int(k) for k in id_to_name.keys()]
        new_id = max(existing_ids) + 1 if existing_ids else 0
        for k, v in list(id_to_name.items()):
            if v == name:
                new_id = int(k)  # reuse the same id if re-registering the same name
        id_to_name[str(new_id)] = name
        save_labels()
        train_recognizer()

    return jsonify(ok=True, message=f"Registered {got} samples for '{name}'")


@app.route("/list")
def list_faces():
    return jsonify(names=sorted(set(id_to_name.values())))


@app.route("/delete", methods=["POST"])
def delete_face():
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


# ------------------------------- entrypoint ----------------------------------

if __name__ == "__main__":
    os.makedirs(DATASET_DIR, exist_ok=True)
    id_to_name.update(load_labels())
    train_recognizer()

    t = threading.Thread(target=recognition_loop, daemon=True)
    t.start()

    print(f"Open http://localhost:{PORT} to register faces.")
    print(f"Polling ESP32-CAM at http://{ESP32_IP}/capture every {POLL_INTERVAL}s")
    app.run(host="0.0.0.0", port=PORT, debug=False)
