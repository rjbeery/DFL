"""
snapshot_service.py
-------------------
Create, list, and restore snapshots of critical DFL data.

Critical data (aligned faces are expensive to regenerate; model weights
represent hours or days of training):
  - model directory
  - source aligned faces  (data_src_aligned)
  - destination aligned faces  (data_dst_aligned)

Snapshot layout on disk:
  <backup_dir>/
    <YYYY-MM-DD_HHmmss>[_<label>]/
      model/          ← copy of model_dir
      aligned_src/    ← copy of data_src_aligned  (if present)
      aligned_dst/    ← copy of data_dst_aligned  (if present)
      .meta.json      ← written LAST (signals snapshot is complete)

.meta.json schema:
  {
    "created_at":      float,
    "label":           str,
    "components":      ["model", "aligned_src", "aligned_dst"],
    "src_model_dir":   str,
    "src_aligned_src": str,
    "src_aligned_dst": str
  }

Writing .meta.json after all copies ensures an incomplete snapshot
(e.g., power loss mid-copy) is detected and labelled "incomplete" in listings.

Public API:
    create_snapshot(cfg, label="")                                 -> dict
    list_snapshots(backup_root)                                    -> list[dict]
    restore_snapshot(snap_path, cfg, restore_model, restore_aligned) -> dict
    get_last_result()                                              -> dict | None
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from config_manager import Config

_lock        = threading.Lock()
_last_result: Optional[dict] = None


# ── Public API ────────────────────────────────────────────────────────────────

def create_snapshot(cfg: "Config", label: str = "") -> dict:
    """
    Copy model_dir, data_src_aligned, and data_dst_aligned into a new
    timestamped folder under cfg.backup_dir.

    Aligned dirs are optional: skipped (not an error) if the directory does
    not exist.  model_dir is required; the call fails if it is absent.

    Returns a result dict:
        {ok, snapshot_path, components, skipped, error}
    """
    err = _validate_backup_root(cfg.backup_dir)
    if err:
        result = {"ok": False, "snapshot_path": "", "components": [],
                  "skipped": [], "error": err}
        _set_last(result)
        return result

    if not Path(cfg.model_dir).is_dir():
        result = {"ok": False, "snapshot_path": "", "components": [],
                  "skipped": [],
                  "error": f"Model directory does not exist: {cfg.model_dir}"}
        _set_last(result)
        return result

    # ── Build timestamped snapshot folder name ────────────────────────────────
    ts        = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe_label = _safe_label(label)
    folder    = ts + (f"_{safe_label}" if safe_label else "")
    snap_dir  = Path(cfg.backup_dir) / folder

    try:
        snap_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        result = {"ok": False, "snapshot_path": str(snap_dir), "components": [],
                  "skipped": [],
                  "error": f"Snapshot folder already exists (retry in 1 s): {snap_dir}"}
        _set_last(result)
        return result
    except OSError as exc:
        result = {"ok": False, "snapshot_path": str(snap_dir), "components": [],
                  "skipped": [],
                  "error": f"Cannot create snapshot folder {snap_dir}: {exc}"}
        _set_last(result)
        return result

    components: list[str] = []
    skipped:    list[str] = []

    # ── Copy model (required) ─────────────────────────────────────────────────
    try:
        shutil.copytree(cfg.model_dir, str(snap_dir / "model"))
        components.append("model")
    except Exception as exc:
        shutil.rmtree(snap_dir, ignore_errors=True)   # clean up partial folder
        result = {"ok": False, "snapshot_path": str(snap_dir), "components": [],
                  "skipped": [],
                  "error": f"Failed to copy model directory: {exc}"}
        _set_last(result)
        return result

    # ── Copy aligned src (optional) ───────────────────────────────────────────
    if Path(cfg.data_src_aligned).is_dir():
        try:
            shutil.copytree(cfg.data_src_aligned, str(snap_dir / "aligned_src"))
            components.append("aligned_src")
        except Exception as exc:
            skipped.append(f"aligned_src: copy failed ({exc})")
    else:
        skipped.append(f"aligned_src: not found at {cfg.data_src_aligned}")

    # ── Copy aligned dst (optional) ───────────────────────────────────────────
    if Path(cfg.data_dst_aligned).is_dir():
        try:
            shutil.copytree(cfg.data_dst_aligned, str(snap_dir / "aligned_dst"))
            components.append("aligned_dst")
        except Exception as exc:
            skipped.append(f"aligned_dst: copy failed ({exc})")
    else:
        skipped.append(f"aligned_dst: not found at {cfg.data_dst_aligned}")

    # ── Write .meta.json (last — signals completion) ──────────────────────────
    _write_meta(snap_dir, {
        "created_at":      time.time(),
        "label":           safe_label,
        "components":      components,
        "src_model_dir":   cfg.model_dir,
        "src_aligned_src": cfg.data_src_aligned,
        "src_aligned_dst": cfg.data_dst_aligned,
    })

    result = {"ok": True, "snapshot_path": str(snap_dir),
              "components": components, "skipped": skipped, "error": None}
    _set_last(result)
    return result


def list_snapshots(backup_root: str) -> list[dict]:
    """
    Return all snapshots under *backup_root*, newest-first.

    Fast: one os.scandir + one .meta.json read per entry.
    Each entry:
        {name, path, created_at, label, components, complete}
    """
    if not backup_root or not Path(backup_root).is_dir():
        return []

    entries: list[dict] = []
    try:
        with os.scandir(backup_root) as it:
            for entry in it:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                snap = _read_snapshot_entry(entry)
                entries.append(snap)
    except (PermissionError, OSError):
        return []

    entries.sort(key=lambda e: e["created_at"], reverse=True)
    return entries


def restore_snapshot(
    snap_path: str,
    cfg: "Config",
    restore_model: bool   = True,
    restore_aligned: bool = True,
) -> dict:
    """
    Restore components from a snapshot into the current config paths.

    restore_model:   copy snapshot/model/   → cfg.model_dir
    restore_aligned: copy snapshot/aligned_src/ → cfg.data_src_aligned
                     copy snapshot/aligned_dst/ → cfg.data_dst_aligned

    Uses dirs_exist_ok=True: files in dst not in src are left untouched;
    conflicts are overwritten.  Returns error without partial state
    if source components are missing.

    Returns {ok, restored, skipped, error}.
    """
    p = Path(snap_path)
    if not p.is_dir():
        result = {"ok": False, "restored": [], "skipped": [],
                  "error": f"Snapshot path does not exist: {snap_path}"}
        _set_last(result)
        return result

    meta = _read_meta(p)

    # Decide what to restore and validate sources first (all-or-nothing check).
    plan: list[tuple[str, str, str]] = []   # (component, src_path, dst_path)
    if restore_model:
        src = str(p / "model")
        if Path(src).is_dir():
            plan.append(("model", src, cfg.model_dir))
        else:
            result = {"ok": False, "restored": [], "skipped": [],
                      "error": "Snapshot does not contain a model component."}
            _set_last(result)
            return result

    if restore_aligned:
        for comp, subdir, dst in [
            ("aligned_src", "aligned_src", cfg.data_src_aligned),
            ("aligned_dst", "aligned_dst", cfg.data_dst_aligned),
        ]:
            src = str(p / subdir)
            if Path(src).is_dir():
                plan.append((comp, src, dst))
            # else: skip silently (aligned dirs are optional)

    if not plan:
        result = {"ok": False, "restored": [], "skipped": [],
                  "error": "Nothing to restore (no matching components selected)."}
        _set_last(result)
        return result

    # ── Execute copies ────────────────────────────────────────────────────────
    restored: list[str] = []
    for comp, src, dst in plan:
        try:
            Path(dst).mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            restored.append(comp)
        except Exception as exc:
            result = {
                "ok":       False,
                "restored": restored,
                "skipped":  [],
                "error":    f"Failed to restore '{comp}': {exc} "
                            f"(already restored: {restored})",
            }
            _set_last(result)
            return result

    result = {"ok": True, "restored": restored, "skipped": [], "error": None}
    _set_last(result)
    return result


def get_last_result() -> Optional[dict]:
    with _lock:
        return dict(_last_result) if _last_result else None


# ── Private helpers ───────────────────────────────────────────────────────────

def _validate_backup_root(backup_root: str) -> Optional[str]:
    if not backup_root or not backup_root.strip():
        return ("Backup directory is not configured. "
                "Set 'Backup dir' in Configuration.")
    p = Path(backup_root)
    if not p.exists():
        return f"Backup directory does not exist: {backup_root}"
    if not p.is_dir():
        return f"Backup path is not a directory: {backup_root}"
    return None


def _safe_label(label: str) -> str:
    """Strip/sanitize a label so it is safe for a directory name."""
    import re
    s = label.strip()[:40]
    s = re.sub(r'[^\w\-]', '_', s)
    return s.strip('_')


def _write_meta(snap_dir: Path, meta: dict) -> None:
    """Atomically write .meta.json to mark snapshot as complete."""
    dst = snap_dir / ".meta.json"
    try:
        fd, tmp = tempfile.mkstemp(dir=snap_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            os.replace(tmp, dst)
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
    except Exception:
        pass  # non-fatal; snapshot still usable, just not marked complete


def _read_meta(snap_dir: Path) -> dict:
    try:
        text = (snap_dir / ".meta.json").read_text(encoding="utf-8")
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_snapshot_entry(entry: os.DirEntry) -> dict:
    """Build a listing row for one snapshot directory."""
    p    = Path(entry.path)
    meta = _read_meta(p)
    complete = bool(meta)   # .meta.json present = complete snapshot

    # Detect components from directory contents as fallback.
    if meta.get("components") is not None:
        components = meta["components"]
    else:
        components = [
            c for c in ("model", "aligned_src", "aligned_dst")
            if (p / c).is_dir()
        ]

    # Fall back to directory mtime for created_at if .meta.json is absent.
    created_at = meta.get("created_at")
    if created_at is None:
        try:
            created_at = entry.stat().st_mtime
        except OSError:
            created_at = 0.0

    return {
        "name":       entry.name,
        "path":       entry.path,
        "created_at": created_at,
        "label":      meta.get("label", ""),
        "components": components,
        "complete":   complete,
    }


def _set_last(result: dict) -> None:
    global _last_result
    with _lock:
        _last_result = result
