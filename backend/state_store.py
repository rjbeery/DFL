"""
state_store.py
--------------
Persists job state across server restarts.

File: <workspace>/.dfl_wrapper_state.json
Structure:
  {
    "was_running_at_shutdown": false,
    "last_stage":              "train",
    "last_status":             "stopped",
    "last_start_time":         1713271200.0,
    "shutdown_time":           1713271234.0,
    "recent_jobs":             [...]      # newest first, max 10 entries
  }

Writes are atomic: write to a .tmp sibling then os.replace() it into place.
os.replace() is atomic on Windows (same drive) since Vista.

Public API:
    load()                              -> dict
    write_job(record_dict)              -> None
    write_shutdown(stage, status, t)    -> None
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

_lock = threading.Lock()

_MAX_RECENT_JOBS = 50


# ── File location ─────────────────────────────────────────────────────────────

def _state_file() -> Path:
    """
    Same resolution logic as config_manager._config_file():
      1. DFL_WORKSPACE env var   → <ws>/.dfl_wrapper_state.json
      2. Fallback                → <repo_root>/workspace/.dfl_wrapper_state.json
    """
    ws = os.environ.get("DFL_WORKSPACE")
    if ws:
        return Path(ws) / ".dfl_wrapper_state.json"
    return Path(__file__).parent.parent / "workspace" / ".dfl_wrapper_state.json"


# ── Low-level I/O ─────────────────────────────────────────────────────────────

def _read() -> dict:
    path = _state_file()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _write(data: dict) -> None:
    """Atomic write: temp file in same dir → os.replace()."""
    path = _state_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, path)    # atomic on Windows (same drive) and POSIX
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise
    except Exception:
        pass    # best-effort; never crash the server over a persistence failure


# ── Public API ────────────────────────────────────────────────────────────────

def load() -> dict:
    """
    Return the persisted state dict.  Returns {} if the file is absent or
    unreadable.  Callers must not mutate the returned dict.
    """
    with _lock:
        return _read()


def write_job(record: dict) -> None:
    """
    Persist a completed job record (called after each run ends).

    Updates:
      - last_stage / last_status / last_start_time
      - was_running_at_shutdown → False  (job completed cleanly)
      - recent_jobs             → prepend, cap at _MAX_RECENT_JOBS
    """
    with _lock:
        data = _read()
        data["last_stage"]              = record.get("stage")
        data["last_status"]             = record.get("status")
        data["last_start_time"]         = record.get("start_time")
        data["was_running_at_shutdown"] = False

        recent: list = data.get("recent_jobs", [])
        recent.insert(0, record)
        data["recent_jobs"] = recent[:_MAX_RECENT_JOBS]
        _write(data)


def replace_history(records: list[dict]) -> None:
    """Overwrite the recent_jobs list atomically (newest-first order)."""
    with _lock:
        data = _read()
        data["recent_jobs"] = records[:_MAX_RECENT_JOBS]
        _write(data)


def write_shutdown(
    stage:      Optional[str],
    status:     str,
    start_time: Optional[float],
) -> None:
    """
    Called during server shutdown lifespan cleanup.

    Records whether the server is stopping while a job is still running.
    On the next startup, the lifespan hook reads was_running_at_shutdown
    and synthesises an "interrupted" history record.
    """
    with _lock:
        data = _read()
        data["shutdown_time"]           = time.time()
        data["was_running_at_shutdown"] = (status == "running")
        if status == "running":
            data["last_stage"]      = stage
            data["last_status"]     = "running"
            data["last_start_time"] = start_time
        _write(data)
