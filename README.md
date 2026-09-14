# Face Lock Backend (Render / Docker)

Flask API the ESP32-CAM pushes frames to. Matches `esp32cam_wifi_relay_node.ino`:

- `POST /register` - multipart fields `key`, `name`, `image` -> saves one face sample, retrains, returns plain text.
- `POST /recognize` - multipart fields `key`, `image` -> returns `{"match": true/false, "name": "..."}`.
- `GET /list?key=...` - list registered names.
- `POST /delete?key=...&name=...` - remove a person and retrain.
- `GET /` - health check.

## Deploy to Render

**Option A - Blueprint (render.yaml), fastest:**
1. Push this folder to a GitHub repo.
2. In Render: **New > Blueprint**, point it at the repo. It reads `render.yaml` and creates the service.
3. In the service's **Environment** tab, set `API_KEY` to the same value as `API_KEY` in the `.ino` sketch.
4. Deploy. Render builds the `Dockerfile` automatically.

**Option B - Manual web service:**
1. Push this folder to a GitHub repo.
2. In Render: **New > Web Service**, connect the repo.
3. Environment: **Docker** (auto-detected from the `Dockerfile`).
4. Add environment variable `API_KEY` (must match the ESP32 sketch).
5. Deploy.

Either way, once live, set `BACKEND_BASE_URL` in the `.ino` to your Render URL, e.g. `https://face-lock-backend.onrender.com` (no trailing slash), and re-flash.

## Storage caveat

Render's default filesystem is **ephemeral** - registered faces (`dataset/`) are wiped on every redeploy, restart, or scale event. Fine for a hobby setup (just re-register after a redeploy); if you don't want that, add a Render **Persistent Disk** mounted at `/app/dataset` (uncomment the `disk:` block in `render.yaml` - requires a paid plan) or swap the storage layer for S3/similar.

## Free-plan cold starts

Render's free web services spin down after inactivity and take 30-60s to wake on the next request. Your ESP32's `/recognize` calls (every ~2s) will actually keep it warm during active use, but the *first* request after a period of silence will be slow - the timeouts already added on the ESP32 side (`setConnectTimeout`/`setTimeout`) will make that call fail fast rather than hang, but recognition just won't work for that first ~30-60s. Upgrade to a paid plan if you need it always warm.

## Local test without Docker

```bash
pip install -r requirements.txt
export API_KEY=changeme123
python face_server.py
```

Then, e.g.:
```bash
curl -F "key=changeme123" -F "name=alice" -F "image=@sample.jpg" http://localhost:5000/register
curl -F "key=changeme123" -F "image=@sample.jpg" http://localhost:5000/recognize
```
