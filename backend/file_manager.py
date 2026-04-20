"""
file_manager.py
---------------
Path-safe helpers for file listing, upload, download, and bounded zip.

All operations are restricted to a small set of roots derived from the
current Config.  Arbitrary filesystem access is prevented by resolving paths
and verifying the resolved result is still inside the declared root
(Path.resolve() + relative_to() idiom).

Roots:
    data_src         cfg.data_src          Source media          upload allowed
    data_dst         cfg.data_dst          Dest media            upload allowed
    data_src_aligned cfg.data_src_aligned  Source aligned faces  zip allowed
    data_dst_aligned cfg.data_dst_aligned  Dest aligned faces    zip allowed
    model            cfg.model_dir         Model weights         zip allowed
    output           cfg.output_dir        Merged output         zip allowed
    backup           cfg.backup_dir        Backups/snapshots     read-only
    workspace        cfg.workspace         Workspace root        read-only

Limits (enforced by the endpoints, not here):
    MAX_UPLOAD_BYTES  2 GB per file
    MAX_LIST_ENTRIES  500 entries per shallow listing
    MAX_ZIP_FILES     1 000 files per zip operation
    MAX_ZIP_BYTES     2 GB uncompressed total per zip operation
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from config_manager import get_config


# ── Root definitions ──────────────────────────────────────────────────────────
# key → (config_field, display_label, upload_ok, zip_ok)

_ROOTS: dict[str, tuple[str, str, bool, bool]] = {
    "data_src":         ("data_src",         "Source media (raw)",     True,  False),
    "data_dst":         ("data_dst",         "Dest media (raw)",       True,  False),
    "data_src_aligned": ("data_src_aligned", "Source faces (aligned)", False, True),
    "data_dst_aligned": ("data_dst_aligned", "Dest faces (aligned)",   False, True),
    "model":            ("model_dir",        "Model weights",          False, True),
    "output":           ("output_dir",       "Merged output",          False, True),
    "backup":           ("backup_dir",       "Backups / Snapshots",    False, False),
    "workspace":        ("workspace",        "Workspace root",         False, False),
}

# ── Limits ────────────────────────────────────────────────────────────────────

MAX_UPLOAD_BYTES = 2  * 1024 ** 3   # 2 GB per file
MAX_LIST_ENTRIES = 500
MAX_ZIP_FILES    = 1_000
MAX_ZIP_BYTES    = 2  * 1024 ** 3   # 2 GB uncompressed total


# ── Root access ───────────────────────────────────────────────────────────────

def get_roots() -> dict[str, dict]:
    """
    Return metadata for every configured non-empty root.
    Result: {root_key: {key, path, label, upload_ok, zip_ok, exists}}
    """
    cfg = dataclasses.asdict(get_config())
    result = {}
    for key, (field, label, upload_ok, zip_ok) in _ROOTS.items():
        path = cfg.get(field, "")
        if path:
            result[key] = {
                "key":       key,
                "path":      path,
                "label":     label,
                "upload_ok": upload_ok,
                "zip_ok":    zip_ok,
                "exists":    Path(path).is_dir(),
            }
    return result


def _root_path(root_key: str) -> Optional[str]:
    cfg = dataclasses.asdict(get_config())
    entry = _ROOTS.get(root_key)
    if entry:
        return cfg.get(entry[0], "") or None
    return None


# ── Path safety ───────────────────────────────────────────────────────────────

def resolve_safe(root_key: str, rel_path: str = "") -> Path:
    """
    Resolve root_key/rel_path to an absolute Path.

    Raises ValueError if:
    - root_key is unknown or not configured
    - the resolved path escapes the root (path traversal)
    """
    rp = _root_path(root_key)
    if not rp:
        raise ValueError(f"Unknown or unconfigured root: '{root_key}'")

    root_resolved = Path(rp).resolve()

    # Normalise rel_path: strip leading slashes/dots, normalise separators.
    rel_clean = rel_path.replace("\\", "/").lstrip("/")
    target = (root_resolved / rel_clean).resolve() if rel_clean else root_resolved

    try:
        target.relative_to(root_resolved)
    except ValueError:
        raise ValueError("Path escapes allowed root.")

    return target


# ── Listing ───────────────────────────────────────────────────────────────────

def list_dir(root_key: str, rel_path: str = "") -> list[dict]:
    """
    Return a shallow listing of root_key/rel_path, capped at MAX_LIST_ENTRIES.

    Each entry: {name, type, size, mtime, rel_path}
    rel_path is relative to the root (not to rel_path), for use in API calls.
    """
    target = resolve_safe(root_key, rel_path)
    if not target.is_dir():
        raise ValueError(f"Not a directory: {rel_path!r}")

    prefix = rel_path.replace("\\", "/").lstrip("/")
    entries = []
    try:
        items = sorted(target.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    except PermissionError:
        return []

    for entry in items:
        try:
            st    = entry.stat()
            epath = (prefix + "/" + entry.name).lstrip("/") if prefix else entry.name
            entries.append({
                "name":     entry.name,
                "type":     "file" if entry.is_file() else "dir",
                "size":     st.st_size if entry.is_file() else None,
                "mtime":    st.st_mtime,
                "rel_path": epath,
            })
        except OSError:
            continue
        if len(entries) >= MAX_LIST_ENTRIES:
            break

    return entries


# ── Zip helpers ───────────────────────────────────────────────────────────────

def count_dir(target: Path) -> tuple[int, int]:
    """
    Walk *target* recursively counting files and total uncompressed bytes.
    Stops early once either limit is exceeded.
    Returns (file_count, total_bytes).
    """
    count = 0
    total = 0
    for f in target.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
                count += 1
            except OSError:
                pass
            if count > MAX_ZIP_FILES or total > MAX_ZIP_BYTES:
                return count, total
    return count, total


def create_zip(target: Path) -> str:
    """
    Create a zip of *target* directory in a temp file.
    Returns the temp file path.  Caller must delete it (use BackgroundTask).
    Raises RuntimeError on failure.
    """
    fd, tmp = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for f in sorted(target.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(target))
    except Exception as exc:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise RuntimeError(f"Failed to create zip: {exc}") from exc
    return tmp
