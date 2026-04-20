"""
artifact_scanner.py
-------------------
Lightweight directory scanner for artifact visibility.

Public API:
    scan_dir_shallow(path_str, max_count=500) -> dict

Returns:
    {
      "path":       str,   # the input path
      "exists":     bool,
      "file_count": int | None,   # immediate file children only
      "mtime":      float | None, # directory mtime (not deepest child)
      "truncated":  bool,         # True if scan was stopped at max_count
    }

Design constraints:
- One level only (os.scandir, no recursion).
- Stops counting after *max_count* entries to protect large aligned dirs.
- Uses is_file(follow_symlinks=False) — does not chase symlinks.
- Silently tolerates PermissionError / OSError (returns partial result).
"""

from __future__ import annotations

import os
from pathlib import Path


def scan_dir_shallow(path_str: str, max_count: int = 500) -> dict:
    """
    Stat and count files in one directory level.
    Safe to call on workspace/aligned/ even when it contains thousands of files.
    """
    result: dict = {
        "path":       path_str,
        "exists":     False,
        "file_count": None,
        "mtime":      None,
        "truncated":  False,
    }

    p = Path(path_str)
    if not p.exists():
        return result

    result["exists"] = True

    try:
        result["mtime"] = p.stat().st_mtime
    except OSError:
        pass

    count = 0
    truncated = False
    try:
        with os.scandir(p) as it:
            for i, entry in enumerate(it):
                if i >= max_count:
                    truncated = True
                    break
                if entry.is_file(follow_symlinks=False):
                    count += 1
    except (PermissionError, OSError):
        pass

    result["file_count"] = count
    result["truncated"]  = truncated
    return result
