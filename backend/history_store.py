"""
history_store.py
----------------
Persistent JSON-backed store for job history.

Separate from state_store.py (which handles shutdown/recovery flags).
This file is the single source of truth for the full job run log.

File path resolution (first match wins):
  1. DFL_HISTORY_PATH env var          (explicit override)
  2. <DFL_WORKSPACE>/state/job_history.json
  3. <repo_root>/workspace/state/job_history.json

Writes are atomic (tempfile + os.replace).
Read failures log a warning and return [] — never crash the server.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

_lock       = threading.Lock()
_MAX_ENTRIES = 200


# ── File location ──────────────────────────────────────────────────────────────

def _history_file() -> Path:
    explicit = os.environ.get("DFL_HISTORY_PATH")
    if explicit:
        return Path(explicit)
    ws = os.environ.get("DFL_WORKSPACE")
    if ws:
        return Path(ws) / "state" / "job_history.json"
    return Path(__file__).parent.parent / "workspace" / "state" / "job_history.json"


# ── I/O ───────────────────────────────────────────────────────────────────────

def load() -> list[dict]:
    """
    Return stored records (newest first).
    Returns [] if the file is absent, unreadable, or malformed.
    """
    path = _history_file()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except Exception as exc:
        print(f"[history_store] Warning: could not load {path}: {exc}", flush=True)
    return []


def save(records: list[dict]) -> None:
    """Atomically write records (newest first), capped at _MAX_ENTRIES."""
    path = _history_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(records[:_MAX_ENTRIES], f, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise
    except Exception as exc:
        print(f"[history_store] Warning: could not save to {path}: {exc}", flush=True)


def append(record: dict) -> None:
    """Prepend one record and persist the updated list. Thread-safe."""
    with _lock:
        records = load()
        records.insert(0, record)
        save(records)
