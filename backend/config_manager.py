"""
config_manager.py
-----------------
Single source of truth for runtime-mutable configuration.

Design:
- Config is a frozen dataclass (immutable once created).
  update_config() produces a new instance and swaps the singleton atomically.
- Defaults are derived from environment variables so they resolve correctly
  regardless of the working directory at server startup.

Persistence (new in Task 20):
- Config is saved to /workspace/.dfl_wrapper_config.json on every update_config() call.
- On startup, saved values override env-var defaults.
- Precedence (lowest → highest):
    1. Built-in defaults (hardcoded fallbacks)
    2. Environment variables (DFL_WORKSPACE, DFL_BACKUP, etc.)
    3. Persisted config file (/workspace/.dfl_wrapper_config.json)
- If the file is missing or malformed, env-var defaults are used unchanged.
- The file lives on the host-mounted /workspace volume, so it survives container restarts.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import config_store


# ── Config schema ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    # ── Paths ─────────────────────────────────────────────────────────────────
    dfl_root:           str   # DFL code root (contains main.py, venv/)
    workspace:          str   # workspace root
    data_src:           str   # raw source frames
    data_src_aligned:   str   # source aligned faces
    data_dst:           str   # raw destination frames
    data_dst_aligned:   str   # destination aligned faces
    model_dir:          str   # trained model files
    output_dir:         str   # merged output frames

    # ── Backup / storage ──────────────────────────────────────────────────────
    storage_backend:    str = "local"  # "local" | future: "s3", "gcs"
    backup_dir:         str = ""       # local backend: root of backup target

    # ── Merge output ──────────────────────────────────────────────────────────
    output_mask_dir:    str = ""       # where merge mask images are written

    # ── Docker ────────────────────────────────────────────────────────────────
    docker_image:       str = "dfl:latest"  # image built from Dockerfile

    # ── Stage arguments ───────────────────────────────────────────────────────
    extract_detector:   str = "s3fd"   # "s3fd" | "manual"
    train_model_name:   str = "poc"    # must match a model in model_dir
    merge_model_class:  str = "SAEHD"  # DFL model class used for merge: SAEHD | AMP | Quick96
    video_fps:          str = "25"     # frames-per-second for video-from-seq output


# ── Persistence helpers ────────────────────────────────────────────────────────

def _config_file() -> Path:
    """
    Path of the persisted config file.
    Always rooted at DFL_WORKSPACE env var (or the default workspace path),
    not the runtime config's workspace field, to avoid chicken-and-egg on startup.
    """
    ws = os.environ.get("DFL_WORKSPACE")
    if ws:
        return Path(ws) / ".dfl_wrapper_config.json"
    return Path(__file__).parent.parent / "workspace" / ".dfl_wrapper_config.json"


def _load_persisted() -> dict:
    """
    Read the config file and return a flat dict of field overrides.
    Returns {} if the file doesn't exist or cannot be parsed.
    """
    path = _config_file()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if isinstance(v, str)}
    except Exception:
        pass
    return {}


def _save_config(cfg: Config) -> None:
    """
    Write *cfg* to the persisted config file.  Best-effort: silently ignores
    errors (e.g. workspace not mounted yet, read-only filesystem).
    """
    path = _config_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(dataclasses.asdict(cfg), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


# ── Defaults ──────────────────────────────────────────────────────────────────

def _default_config() -> Config:
    """
    Build Config from environment variables.
    Precedence at this stage: built-in defaults < env vars.
    Persisted file overrides are applied separately in _init_config().
    """
    root        = Path(__file__).parent.parent        # …/DFL/ or /app/dfl_code/
    ws_env      = os.environ.get("DFL_WORKSPACE")
    ws          = Path(ws_env) if ws_env else (root / "workspace")
    backup_env  = os.environ.get("DFL_BACKUP", "")
    backend_env = os.environ.get("DFL_STORAGE_BACKEND", "local")
    return Config(
        dfl_root         = str(root),
        workspace        = str(ws),
        data_src         = str(ws / "data_src"),
        data_src_aligned = str(ws / "data_src" / "aligned"),
        data_dst         = str(ws / "data_dst"),
        data_dst_aligned = str(ws / "data_dst" / "aligned"),
        model_dir        = str(ws / "model"),
        output_dir       = str(ws / "output"),
        output_mask_dir  = str(ws / "output_mask"),
        storage_backend  = backend_env,
        backup_dir       = backup_env,
        docker_image     = os.environ.get("DFL_DOCKER_IMAGE", "dfl:latest"),
    )


def _init_config() -> Config:
    """
    Build the startup Config applying the full precedence chain:
        built-in defaults < env vars < persisted file.
    Falls back to env-var defaults if the file is missing or malformed.

    Container note: when DFL_WORKSPACE is set as an env var (e.g. /workspace
    inside Docker), all workspace-derived paths are re-derived from it so that
    persisted Windows host paths never leak into a Linux container.
    """
    defaults  = dataclasses.asdict(_default_config())
    overrides = _load_persisted()
    merged    = {**defaults, **overrides}

    # config_store holds a small set of user-facing fields persisted at
    # /workspace/state/backend_config.json.  Apply them on top of the full
    # persisted config so they take priority, but before the DFL_WORKSPACE
    # env-var override (which always wins inside a container).
    cs = config_store.load()
    if cs:
        _CS_MAP = {
            "workspace_dir": "workspace",
            "data_src_dir":  "data_src",
            "data_dst_dir":  "data_dst",
            "model_dir":     "model_dir",
        }
        for cs_key, cfg_key in _CS_MAP.items():
            if cs_key in cs:
                merged[cfg_key] = cs[cs_key]

    # If DFL_WORKSPACE is explicitly set, force all workspace-derived paths to
    # use it.  This ensures container paths (/workspace) always win over any
    # Windows paths that may have been saved while running on the host.
    ws_env = os.environ.get("DFL_WORKSPACE")
    if ws_env:
        ws = Path(ws_env)
        merged.update({
            "dfl_root":         str(Path(__file__).parent.parent),
            "workspace":        str(ws),
            "data_src":         str(ws / "data_src"),
            "data_src_aligned": str(ws / "data_src" / "aligned"),
            "data_dst":         str(ws / "data_dst"),
            "data_dst_aligned": str(ws / "data_dst" / "aligned"),
            "model_dir":        str(ws / "model"),
            "output_dir":       str(ws / "output"),
            "output_mask_dir":  str(ws / "output_mask"),
        })

    try:
        return Config(**merged)
    except Exception:
        return _default_config()


# ── Singleton ─────────────────────────────────────────────────────────────────

_lock:   threading.Lock = threading.Lock()
_config: Config         = _init_config()


def get_config() -> Config:
    """Return the current Config.  Callers must not mutate the returned object."""
    with _lock:
        return _config


# ── Update ────────────────────────────────────────────────────────────────────

_PATH_FIELDS = [
    "dfl_root", "workspace",
    "data_src", "data_src_aligned",
    "data_dst", "data_dst_aligned",
    "model_dir", "output_dir", "output_mask_dir",
]

_VALID_MODEL_NAME       = re.compile(r'^[\w\-]+$')
_KNOWN_STORAGE_BACKENDS = {"local"}       # extend when new backends are added
_KNOWN_MODEL_CLASSES    = {"SAEHD", "AMP", "Quick96"}  # DFL model architectures


def update_config(data: dict) -> list[str]:
    """
    Validate *data* and, if valid, atomically replace the current Config
    and persist it to the config file.

    Returns a list of error strings.  An empty list means the update was applied.
    *data* is expected to be a flat dict of str→str (e.g. from an HTML form).
    """
    global _config

    errors = _validate(data)
    if errors:
        return errors

    new_cfg = _build(data)
    with _lock:
        _config = new_cfg
    _save_config(new_cfg)   # persist full config (best-effort I/O)
    config_store.save({     # persist minimal subset to /workspace/state/
        "workspace_dir": new_cfg.workspace,
        "data_src_dir":  new_cfg.data_src,
        "data_dst_dir":  new_cfg.data_dst,
        "model_dir":     new_cfg.model_dir,
        "exec_mode":     os.environ.get("DFL_EXEC_MODE", "docker"),
    })
    return []


def _validate(data: dict) -> list[str]:
    errors: list[str] = []

    for key in _PATH_FIELDS:
        if not data.get(key, "").strip():
            errors.append(f"'{_label(key)}' must not be empty.")

    backend_name = data.get("storage_backend", "local").strip()
    if backend_name not in _KNOWN_STORAGE_BACKENDS:
        errors.append(
            f"Unknown storage backend '{backend_name}'. "
            f"Supported: {', '.join(sorted(_KNOWN_STORAGE_BACKENDS))}."
        )

    detector = data.get("extract_detector", "").strip()
    if detector not in ("s3fd", "manual"):
        errors.append("Extract detector must be 's3fd' or 'manual'.")

    model_name = data.get("train_model_name", "").strip()
    if not model_name:
        errors.append("Train model name must not be empty.")
    elif not _VALID_MODEL_NAME.match(model_name):
        errors.append(
            "Train model name may only contain letters, digits, underscores, and hyphens."
        )

    model_class = data.get("merge_model_class", "").strip()
    if model_class not in _KNOWN_MODEL_CLASSES:
        errors.append(
            f"Merge model class must be one of: {', '.join(sorted(_KNOWN_MODEL_CLASSES))}."
        )

    fps_str = data.get("video_fps", "25").strip()
    if not fps_str.isdigit() or int(fps_str) < 1:
        errors.append("Video FPS must be a positive integer.")

    docker_image = data.get("docker_image", "").strip()
    if not docker_image:
        errors.append("Docker image must not be empty.")

    return errors


def _build(data: dict) -> Config:
    """Construct a new Config from validated form data."""
    def norm(key: str) -> str:
        return str(Path(data[key].strip()))

    raw_backup = data.get("backup_dir", "").strip()
    return Config(
        dfl_root         = norm("dfl_root"),
        workspace        = norm("workspace"),
        data_src         = norm("data_src"),
        data_src_aligned = norm("data_src_aligned"),
        data_dst         = norm("data_dst"),
        data_dst_aligned = norm("data_dst_aligned"),
        model_dir        = norm("model_dir"),
        output_dir       = norm("output_dir"),
        output_mask_dir  = norm("output_mask_dir"),
        storage_backend   = data.get("storage_backend", "local").strip() or "local",
        backup_dir        = str(Path(raw_backup)) if raw_backup else "",
        extract_detector  = data.get("extract_detector", "s3fd").strip(),
        train_model_name  = data.get("train_model_name", "poc").strip(),
        merge_model_class = data.get("merge_model_class", "SAEHD").strip(),
        docker_image      = data.get("docker_image", "dfl:latest").strip() or "dfl:latest",
        video_fps         = data.get("video_fps", "25").strip() or "25",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

_LABELS = {
    "dfl_root":          "DFL root",
    "workspace":         "Workspace",
    "data_src":          "Source frames dir",
    "data_src_aligned":  "Source aligned dir",
    "data_dst":          "Destination frames dir",
    "data_dst_aligned":  "Destination aligned dir",
    "model_dir":         "Model dir",
    "output_dir":        "Output dir",
    "output_mask_dir":   "Output mask dir",
    "storage_backend":   "Storage backend",
    "backup_dir":        "Backup dir",
    "merge_model_class": "Merge model class",
    "docker_image":      "Docker image",
    "video_fps":         "Video FPS",
}

def _label(key: str) -> str:
    return _LABELS.get(key, key)


def field_label(key: str) -> str:
    """Public accessor used by templates."""
    return _label(key)
