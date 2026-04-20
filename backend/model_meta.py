"""
model_meta.py
-------------
Lightweight model directory metadata — no file content reads, one level only.

Public API:
    scan_model_dir(path_str) -> dict
"""
from __future__ import annotations

import os
from pathlib import Path


def scan_model_dir(path_str: str) -> dict:
    """
    Stat and classify files in a model directory (one level, no recursion).

    Returns:
        {
            "path":        str,
            "exists":      bool,
            "initialized": bool,   # has a <name>_options.pkl (model was saved at least once)
            "has_weights": bool,   # has .npy weight files
            "checkpoint":  bool,   # has TF checkpoint index/data files
            "backup_zip":  bool,   # has a <name>_backup.zip
            "file_count":  int | None,
            "mtime":       float | None,  # directory mtime
            "pkl_files":   list[str],     # basenames of _options.pkl files found
        }
    """
    result: dict = {
        "path":        path_str,
        "exists":      False,
        "initialized": False,
        "has_weights": False,
        "checkpoint":  False,
        "backup_zip":  False,
        "file_count":  None,
        "mtime":       None,
        "pkl_files":   [],
    }

    p = Path(path_str)
    if not p.is_dir():
        return result

    result["exists"] = True
    try:
        result["mtime"] = p.stat().st_mtime
    except OSError:
        pass

    count = 0
    try:
        with os.scandir(p) as it:
            for entry in it:
                if not entry.is_file(follow_symlinks=False):
                    continue
                count += 1
                name = entry.name
                if name.endswith("_options.pkl"):
                    result["initialized"] = True
                    result["pkl_files"].append(name)
                elif name.endswith(".npy"):
                    result["has_weights"] = True
                elif name.endswith(".index") or name.endswith(".data-00000-of-00001"):
                    result["checkpoint"] = True
                elif name.endswith("_backup.zip"):
                    result["backup_zip"] = True
    except (PermissionError, OSError):
        pass

    result["file_count"] = count
    return result
