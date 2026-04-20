"""
preset_manager.py
-----------------
Named preset persistence for the DFL control panel.

A preset captures the path-related and runtime-identity config fields that
differ between DFL projects.  It deliberately does NOT store machine-level
settings (dfl_root, storage_backend, backup_dir) which belong to the host,
not the project.

Storage:  <workspace>/.dfl_wrapper_presets.json
Format:
    {
      "_active": "project-alpha",
      "project-alpha": {
        "name":             "project-alpha",
        "workspace":        "C:/DFL/workspace",
        "model_dir":        "C:/DFL/workspace/model",
        "data_src":         "C:/DFL/workspace/data_src",
        "data_src_aligned": "C:/DFL/workspace/data_src/aligned",
        "data_dst":         "C:/DFL/workspace/data_dst",
        "data_dst_aligned": "C:/DFL/workspace/data_dst/aligned",
        "output_dir":       "C:/DFL/workspace/output",
        "output_mask_dir":  "C:/DFL/workspace/output_mask",
        "docker_image":     "dfl:latest",
        "train_model_name": "poc",
        "merge_model_class":"SAEHD",
        "created_at":       1713271200.0,
        "updated_at":       1713271200.0
      }
    }

Atomic writes use the same tempfile + os.replace() pattern as state_store.py.
The _active key is set only after a preset has been successfully applied.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from config_manager import get_config, update_config

# ── Fields that a preset captures (subset of Config) ─────────────────────────

PRESET_FIELDS = [
    "workspace",
    "model_dir",
    "data_src",
    "data_src_aligned",
    "data_dst",
    "data_dst_aligned",
    "output_dir",
    "output_mask_dir",
    "docker_image",
    "train_model_name",
    "merge_model_class",
]

# Required fields — save/apply is rejected if any of these are empty.
_REQUIRED = [
    "workspace", "model_dir",
    "data_src", "data_src_aligned",
    "data_dst", "data_dst_aligned",
    "output_dir",
]

# ── Thread-safe storage ────────────────────────────────────────────────────────

_lock = threading.Lock()


def _presets_file() -> Path:
    ws = os.environ.get("DFL_WORKSPACE")
    if ws:
        return Path(ws) / ".dfl_wrapper_presets.json"
    return Path(__file__).parent.parent / "workspace" / ".dfl_wrapper_presets.json"


def _load_raw() -> dict:
    """Read raw presets dict from disk.  Caller must hold _lock."""
    p = _presets_file()
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_raw(data: dict) -> None:
    """Atomically write *data* to the presets file.  Best-effort."""
    p = _presets_file()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, p)
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise
    except Exception:
        pass  # I/O failure is non-fatal — worst case the preset isn't persisted


# ── Public API ─────────────────────────────────────────────────────────────────

def list_presets() -> list[dict]:
    """Return all saved presets as a list, sorted alphabetically by name."""
    with _lock:
        raw = _load_raw()
    return sorted(
        [v for k, v in raw.items() if k != "_active" and isinstance(v, dict)],
        key=lambda p: p.get("name", "").lower(),
    )


def get_active() -> Optional[str]:
    """Return the name of the last successfully applied preset, or None."""
    with _lock:
        return _load_raw().get("_active")


def save_preset(name: str, fields: dict) -> list[str]:
    """
    Create or overwrite the preset named *name* with *fields*.
    Returns a (possibly empty) list of error strings.
    """
    errors = _validate_name(name)
    if errors:
        return errors
    errors = _validate_fields(fields)
    if errors:
        return errors

    now = time.time()
    with _lock:
        raw = _load_raw()
        existing = raw.get(name, {})
        raw[name] = {
            "name":       name,
            **{f: str(fields.get(f, "")).strip() for f in PRESET_FIELDS},
            "created_at": existing.get("created_at", now),
            "updated_at": now,
        }
        _save_raw(raw)
    return []


def apply_preset(name: str) -> list[str]:
    """
    Apply preset *name* to the live config.
    Returns errors; if non-empty the config is NOT modified (all-or-nothing).
    """
    with _lock:
        raw = _load_raw()

    preset = raw.get(name)
    if not preset or not isinstance(preset, dict):
        return [f"Preset '{name}' not found."]

    # Merge preset fields on top of current config so that non-preset fields
    # (dfl_root, storage_backend, backup_dir) are preserved unchanged.
    current = dataclasses.asdict(get_config())
    merged  = {
        **current,
        **{f: preset.get(f, current.get(f, "")) for f in PRESET_FIELDS},
    }

    errors = update_config(merged)
    if errors:
        return errors

    # Record _active only after config was successfully updated.
    with _lock:
        raw = _load_raw()
        raw["_active"] = name
        _save_raw(raw)

    return []


def delete_preset(name: str) -> list[str]:
    """Delete preset *name*.  Returns errors; empty list = success."""
    with _lock:
        raw = _load_raw()
        if name not in raw or not isinstance(raw.get(name), dict):
            return [f"Preset '{name}' not found."]
        del raw[name]
        if raw.get("_active") == name:
            raw["_active"] = None
        _save_raw(raw)
    return []


def replace_all_presets(presets: list[dict]) -> None:
    """
    Atomically overwrite all saved presets with *presets*.
    The _active pointer is kept only if that preset name is still present.
    """
    with _lock:
        raw    = _load_raw()
        active = raw.get("_active")
        new_raw: dict = {}
        for p in presets:
            if isinstance(p, dict) and str(p.get("name", "")).strip():
                new_raw[p["name"]] = p
        new_raw["_active"] = active if (active and active in new_raw) else None
        _save_raw(new_raw)


def current_config_snapshot() -> dict:
    """Return the current config's preset-relevant fields as a plain dict."""
    cfg = dataclasses.asdict(get_config())
    return {f: cfg.get(f, "") for f in PRESET_FIELDS}


# ── Validation helpers ─────────────────────────────────────────────────────────

def _validate_name(name: str) -> list[str]:
    name = (name or "").strip()
    if not name:
        return ["Preset name must not be empty."]
    if len(name) > 64:
        return ["Preset name must be 64 characters or fewer."]
    if not re.match(r'^[\w][\w\- ]*$', name):
        return [
            "Preset name may only contain letters, digits, spaces, "
            "underscores, and hyphens."
        ]
    return []


def _validate_fields(fields: dict) -> list[str]:
    errors = []
    for f in _REQUIRED:
        if not str(fields.get(f, "")).strip():
            errors.append(f"'{f}' must not be empty.")
    return errors
