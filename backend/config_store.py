"""
config_store.py
---------------
Persistent JSON-backed store for the subset of backend config fields
needed to locate workspace data and select the execution mode.

Stores only five fields — everything else is derived at runtime:
  workspace_dir   → workspace root
  data_src_dir    → raw source frames
  data_dst_dir    → raw destination frames
  model_dir       → trained model files
  exec_mode       → "docker" | "direct"

Separate from config_manager.py (which owns the full in-memory Config
dataclass) and from history_store.py (job run log).

File path resolution (first match wins):
  1. DFL_CONFIG_PATH env var
  2. <DFL_WORKSPACE>/state/backend_config.json
  3. <repo_root>/workspace/state/backend_config.json

Writes are atomic (tempfile + os.replace).
Read failures log a warning and return {} — never crash the server.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

_FIELDS = frozenset(("workspace_dir", "data_src_dir", "data_dst_dir", "model_dir", "exec_mode"))


# ── File location ──────────────────────────────────────────────────────────────

def _config_file() -> Path:
    explicit = os.environ.get("DFL_CONFIG_PATH")
    if explicit:
        return Path(explicit)
    ws = os.environ.get("DFL_WORKSPACE")
    if ws:
        return Path(ws) / "state" / "backend_config.json"
    return Path(__file__).parent.parent / "workspace" / "state" / "backend_config.json"


# ── I/O ───────────────────────────────────────────────────────────────────────

def load() -> dict:
    """
    Return stored config fields as a dict.
    Returns {} if the file is absent, unreadable, or malformed.
    Only recognised field names are returned.
    """
    path = _config_file()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {k: v for k, v in data.items()
                        if k in _FIELDS and isinstance(v, str)}
    except Exception as exc:
        print(f"[config_store] Warning: could not load {path}: {exc}", flush=True)
    return {}


def save(fields: dict) -> None:
    """
    Atomically write only the recognised config fields.
    Unknown keys are silently dropped.
    """
    path = _config_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in fields.items() if k in _FIELDS}
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise
    except Exception as exc:
        print(f"[config_store] Warning: could not save to {path}: {exc}", flush=True)
