# DFL Control Plane — Backend

Thin FastAPI wrapper that launches DeepFaceLab CLI stages as subprocesses,
streams their output via Server-Sent Events, and exposes a simple REST API
for a browser UI to consume.

---

## Setup

This backend runs in its **own** Python environment, separate from DFL's
`venv/` (which stays on Python 3.7).  Use Python 3.9 or newer.
On this machine, Python 3.10 is available via the `py` launcher:

```bat
cd backend
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Start the server

```bat
cd backend
.venv\Scripts\activate
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The `--reload` flag restarts on code changes.  Drop it in production.

Interactive API docs are served at:  http://127.0.0.1:8000/docs

---

## Endpoints

| Method | Path             | Purpose                          |
|--------|------------------|----------------------------------|
| GET    | `/health`        | Liveness probe                   |
| GET    | `/state`         | Current process state + log count|
| POST   | `/run/{stage}`   | Launch a DFL stage               |
| POST   | `/stop`          | Kill the active process          |
| GET    | `/logs/stream`   | Live log stream (SSE)            |

Valid stage names: `extract-src`, `extract-dst`, `train`, `merge`

---

## Test steps

All commands use PowerShell / Windows terminal with the server running.

### 1 — Health

```bat
curl http://127.0.0.1:8000/health
```
Expected: `{"ok":true}`

### 2 — State (idle)

```bat
curl http://127.0.0.1:8000/state
```
Expected:
```json
{"stage":null,"status":"idle","pid":null,"start_time":null,"log_lines":0}
```

### 3 — Run a stage

```bat
curl -X POST http://127.0.0.1:8000/run/train
```
Expected:
```json
{"ok":true,"pid":12345,"stage":"train"}
```

Check state while running:
```bat
curl http://127.0.0.1:8000/state
```
Expected: `"status":"running"`

### 4 — Stream logs (open a second terminal)

**Option A — curl (works on Windows 10+ / PowerShell 7):**
```bat
curl -N http://127.0.0.1:8000/logs/stream
```

**Option B — Python one-liner:**
```bat
python -c "import urllib.request; [print(l.decode(), end='') for l in urllib.request.urlopen('http://127.0.0.1:8000/logs/stream')]"
```

You will see log lines prefixed with `data: ` and a final
`event: close\ndata: process-ended` when the stage finishes.

### 5 — Stop process (open a third terminal, or run while streaming)

```bat
curl -X POST http://127.0.0.1:8000/stop
```
Expected: `{"ok":true}`

Check state:
```bat
curl http://127.0.0.1:8000/state
```
Expected: `"status":"stopped"`

### 6 — Reject double-launch

While a stage is running, try to start another:
```bat
curl -X POST http://127.0.0.1:8000/run/merge
```
Expected: `422` / `409` with error message about the running stage.

---

## Wiring real DFL commands

Edit `command_registry.py`.  Each stage maps to an argv list:

```python
"train": [
    DFL_PYTHON, str(DFL_ROOT / "debug_train.py"),
],
```

The commented-out blocks above each entry show the full DFL CLI form.
Uncomment and adjust paths/flags, then restart the server.

---

## Windows process kill notes

`POST /stop` uses `taskkill /F /T /PID <pid>` on Windows.

- `/F` — force (equivalent to SIGKILL)
- `/T` — terminates the full process **tree**

The `/T` flag is essential for the `train` stage, which spawns
TensorFlow and sample-generator subprocesses.  Without it, those
child processes continue running after the parent is killed.

Edge case: if DFL spawns grandchildren with `CREATE_NEW_PROCESS_GROUP`,
`/T` may not reach them.  This is rare in practice.  If you observe
orphaned GPU processes after stop, kill them with Task Manager or:

```bat
tasklist /FI "IMAGENAME eq python.exe"
taskkill /F /PID <orphan-pid>
```

---

## Architecture notes

```
main.py              FastAPI app, routes, SSE endpoint
process_manager.py   ProcessManager class — all state and subprocess logic
command_registry.py  Stage → argv mapping — the only file to edit for DFL commands
```

`ProcessManager` is a singleton created at module load.  The event loop
is captured in the FastAPI `lifespan` startup hook (not at import time),
which ensures it matches the loop uvicorn actually runs.

SSE clients each receive a private `asyncio.Queue`.  The stdout reader
thread pushes lines via `loop.call_soon_threadsafe`, which is the
correct way to hand data from a thread to the async event loop.
