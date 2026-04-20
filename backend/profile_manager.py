"""
profile_manager.py
------------------
Named environment profiles for the DFL control panel.

A profile captures machine-specific settings that differ between environments
(local workstation vs remote GPU VM):
    dfl_root     path to the DFL code root
    workspace    workspace root (all sub-paths derived automatically)
    backup_dir   where snapshots/backups are written
    docker_image DFL Docker image tag

Applying a profile calls update_config() with the profile's fields merged onto
the current config.  All nine workspace-based paths (data_src, model_dir, etc.)
are re-derived from the new workspace root — matching the logic in
config_manager._default_config() — so the user only needs to set the workspace
root, not all nine paths individually.

Non-environment config fields (extract_detector, train_model_name,
merge_model_class, storage_backend) are left unchanged.

Storage:
    <backend_dir>/profiles.json
    Stored in the backend directory, NOT in workspace, so profiles survive
    workspace path changes and are machine-local configuration.

Format:
    {
      "_active": "local",
      "local": {
        "name":         "local",
        "label":        "Local workstation",
        "dfl_root":     "C:/Users/Rod/Documents/DFL",
        "workspace":    "C:/Users/Rod/Documents/DFL/workspace",
        "backup_dir":   "",
        "docker_image": "dfl:latest",
        "created_at":   1713271200.0,
        "updated_at":   1713271200.0
      },
      "remote": {
        "name":         "remote",
        "label":        "Cloud GPU VM",
        "dfl_root":     "/opt/dfl",
        "workspace":    "/mnt/disks/workspace",
        "backup_dir":   "/mnt/disks/backups",
        "docker_image": "dfl:latest",
        ...
      }
    }

Presets and history are GLOBAL, not profile-scoped.
Presets capture what project you are working on; profiles capture which
machine you are using.  You should be able to load "project-alpha" on either
the local or remote machine without recreating presets.
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


# ── Profile fields (what a profile stores) ────────────────────────────────────

# These four are explicit inputs. All other Config path fields are derived
# automatically from dfl_root / workspace when the profile is applied.
PROFILE_FIELDS = ["dfl_root", "workspace", "backup_dir", "docker_image"]

_REQUIRED_PROFILE_FIELDS = ["dfl_root", "workspace"]


# ── Storage ───────────────────────────────────────────────────────────────────

_PROFILES_FILE = Path(__file__).parent / "profiles.json"
_lock = threading.Lock()


def _load_raw() -> dict:
    """Read raw profiles dict from disk.  Caller should hold _lock."""
    try:
        if _PROFILES_FILE.exists():
            data = json.loads(_PROFILES_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_raw(data: dict) -> None:
    """Atomically write *data* to profiles.json.  Best-effort."""
    try:
        _PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=_PROFILES_FILE.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, _PROFILES_FILE)
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise
    except Exception:
        pass  # non-fatal — worst case the profile isn't persisted


# ── Public API ─────────────────────────────────────────────────────────────────

def list_profiles() -> list[dict]:
    """Return all saved profiles as a list, sorted alphabetically by name."""
    with _lock:
        raw = _load_raw()
    return sorted(
        [v for k, v in raw.items() if k != "_active" and isinstance(v, dict)],
        key=lambda p: p.get("name", "").lower(),
    )


def get_active() -> Optional[str]:
    """Return the name of the currently active profile, or None."""
    with _lock:
        return _load_raw().get("_active")


def get_active_profile() -> Optional[dict]:
    """Return the active profile dict, or None if no profile is active."""
    with _lock:
        raw = _load_raw()
    active = raw.get("_active")
    if active and isinstance(raw.get(active), dict):
        return raw[active]
    return None


def save_profile(name: str, fields: dict) -> list[str]:
    """
    Create or overwrite the profile named *name* with *fields*.
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
            "name":         name,
            "label":        str(fields.get("label", "")).strip(),
            "dfl_root":     str(fields.get("dfl_root", "")).strip(),
            "workspace":    str(fields.get("workspace", "")).strip(),
            "backup_dir":   str(fields.get("backup_dir", "")).strip(),
            "docker_image": str(fields.get("docker_image", "dfl:latest")).strip() or "dfl:latest",
            "created_at":   existing.get("created_at", now),
            "updated_at":   now,
        }
        _save_raw(raw)
    return []


def apply_profile(name: str) -> list[str]:
    """
    Apply profile *name* to the live config.
    Derives all workspace-based paths from the profile's workspace root.
    Non-environment config fields are preserved unchanged.
    Returns errors; if non-empty the config is NOT modified.
    """
    with _lock:
        raw = _load_raw()

    profile = raw.get(name)
    if not profile or not isinstance(profile, dict):
        return [f"Profile '{name}' not found."]

    ws = Path(profile["workspace"])
    derived = {
        "dfl_root":         profile["dfl_root"],
        "workspace":        str(ws),
        "data_src":         str(ws / "data_src"),
        "data_src_aligned": str(ws / "data_src" / "aligned"),
        "data_dst":         str(ws / "data_dst"),
        "data_dst_aligned": str(ws / "data_dst" / "aligned"),
        "model_dir":        str(ws / "model"),
        "output_dir":       str(ws / "output"),
        "output_mask_dir":  str(ws / "output_mask"),
        "backup_dir":       profile.get("backup_dir", ""),
        "docker_image":     profile.get("docker_image", "dfl:latest") or "dfl:latest",
    }

    # Merge derived env fields onto current config; preserve project-level fields.
    current = dataclasses.asdict(get_config())
    merged  = {**current, **derived}

    errors = update_config(merged)
    if errors:
        return errors

    # Mark active only after config update succeeds.
    with _lock:
        raw = _load_raw()
        raw["_active"] = name
        _save_raw(raw)

    return []


def delete_profile(name: str) -> list[str]:
    """Delete profile *name*.  Rejects deleting the active profile."""
    with _lock:
        raw = _load_raw()
        if name not in raw or not isinstance(raw.get(name), dict):
            return [f"Profile '{name}' not found."]
        if raw.get("_active") == name:
            return [
                f"Cannot delete the active profile '{name}'. "
                "Switch to a different profile first."
            ]
        del raw[name]
        _save_raw(raw)
    return []


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_name(name: str) -> list[str]:
    name = (name or "").strip()
    if not name:
        return ["Profile name must not be empty."]
    if len(name) > 64:
        return ["Profile name must be 64 characters or fewer."]
    if not re.match(r'^[\w][\w\- ]*$', name):
        return [
            "Profile name may only contain letters, digits, spaces, "
            "underscores, and hyphens."
        ]
    return []


def _validate_fields(fields: dict) -> list[str]:
    errors = []
    for f in _REQUIRED_PROFILE_FIELDS:
        if not str(fields.get(f, "")).strip():
            errors.append(f"'{f}' must not be empty.")
    return errors
