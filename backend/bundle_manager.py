"""
bundle_manager.py
-----------------
Export and import the portable DFL state bundle.

Bundle schema (v1):
    {
      "bundle_version": 1,
      "exported_at":    <float timestamp>,
      "config":         {<all Config fields as strings>},
      "presets":        [{...}, ...],        # newest/all saved presets
      "recent_jobs":    [{...}, ...],        # newest-first, capped at 50
      "model_meta":     {<scan_model_dir result>}  # informational only
    }

On import, model_meta is ignored (no file operations on model files).
Config, presets, and recent_jobs are all replaced atomically.

Public API:
    export_bundle(manager)          -> dict
    import_bundle(data, manager)    -> list[str]   (errors; empty = success)
    get_last_result()               -> dict | None
    bundle_summary(manager)         -> dict
"""
from __future__ import annotations

import dataclasses
import threading
import time
from typing import Optional

import preset_manager
import state_store
from config_manager import get_config, update_config
from model_meta import scan_model_dir

BUNDLE_VERSION    = 1
_MAX_HISTORY_EXPORT = 50

_lock        = threading.Lock()
_last_result: Optional[dict] = None


# ── Public API ────────────────────────────────────────────────────────────────

def export_bundle(manager) -> dict:
    """Build and return the full export bundle dict."""
    cfg        = get_config()
    model_meta = scan_model_dir(cfg.model_dir) if cfg.model_dir else {}

    bundle = {
        "bundle_version": BUNDLE_VERSION,
        "exported_at":    time.time(),
        "config":         dataclasses.asdict(cfg),
        "presets":        preset_manager.list_presets(),
        "recent_jobs":    manager.get_history()[:_MAX_HISTORY_EXPORT],
        "model_meta":     model_meta,
    }
    _set_last_result({
        "action":       "export",
        "ok":           True,
        "timestamp":    bundle["exported_at"],
        "preset_count": len(bundle["presets"]),
        "job_count":    len(bundle["recent_jobs"]),
        "error":        None,
    })
    return bundle


def import_bundle(data: object, manager) -> list[str]:
    """
    Validate *data* and, if valid, atomically replace config / presets / history.

    Returns a list of error strings.  Empty list = success.
    Nothing is modified if validation fails.
    model_meta is informational and is never applied.
    """
    errors = _validate(data)
    if errors:
        _set_last_result({"action": "import", "ok": False,
                          "error": "; ".join(errors), "timestamp": time.time()})
        return errors

    assert isinstance(data, dict)

    # Config — merge with current values so fields absent in older bundles get defaults.
    config_errors = _apply_config(data.get("config", {}))
    if config_errors:
        _set_last_result({"action": "import", "ok": False,
                          "error": "; ".join(config_errors), "timestamp": time.time()})
        return config_errors

    preset_manager.replace_all_presets(data.get("presets", []))

    jobs = data.get("recent_jobs", [])[:_MAX_HISTORY_EXPORT]
    manager.replace_history(jobs)
    state_store.replace_history(jobs)

    _set_last_result({
        "action":       "import",
        "ok":           True,
        "timestamp":    time.time(),
        "preset_count": len(data.get("presets", [])),
        "job_count":    len(jobs),
        "error":        None,
    })
    return []


def get_last_result() -> Optional[dict]:
    with _lock:
        return dict(_last_result) if _last_result else None


def bundle_summary(manager) -> dict:
    """Cheap counts for the UI pre-export summary — no filesystem scanning."""
    cfg = get_config()
    return {
        "preset_count": len(preset_manager.list_presets()),
        "job_count":    len(manager.get_history()),
        "model_dir":    cfg.model_dir,
    }


# ── Private helpers ───────────────────────────────────────────────────────────

def _validate(data: object) -> list[str]:
    if not isinstance(data, dict):
        return ["Bundle must be a JSON object."]
    version = data.get("bundle_version")
    if version != BUNDLE_VERSION:
        return [f"Unsupported bundle_version: {version!r}. Expected {BUNDLE_VERSION}."]
    if "config" not in data:
        return ["Bundle is missing required 'config' key."]
    if not isinstance(data.get("presets", []), list):
        return ["'presets' must be a JSON array."]
    if not isinstance(data.get("recent_jobs", []), list):
        return ["'recent_jobs' must be a JSON array."]
    return []


def _apply_config(config_dict: object) -> list[str]:
    if not isinstance(config_dict, dict):
        return ["'config' must be a JSON object."]
    # Merge onto current values so fields missing in older bundles get their defaults.
    current = dataclasses.asdict(get_config())
    merged  = {**current, **{k: str(v) for k, v in config_dict.items() if v is not None}}
    return update_config(merged)


def _set_last_result(r: dict) -> None:
    global _last_result
    with _lock:
        _last_result = r
