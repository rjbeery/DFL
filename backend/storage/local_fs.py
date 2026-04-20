"""
storage/local_fs.py
-------------------
LocalFilesystemStorageBackend: copies DFL artifacts into a host-mounted directory.

Each backup call creates a fresh timestamped sub-folder under the configured root:
    <root>/YYYY-MM-DD_HHmmss/model/
    <root>/YYYY-MM-DD_HHmmss/output/   (skipped if source is absent)

Source files are never moved or deleted (copy-out, not move-out).
No deduplication and no partial-resume — shutil.copytree is used for simplicity.

The destination URI for this backend is an absolute filesystem path.
For cloud backends it would be a URI such as "s3://bucket/prefix/timestamp".
"""

from __future__ import annotations

import datetime
import shutil
from pathlib import Path

from storage.base import BackupResult, StorageBackend


class LocalFilesystemStorageBackend(StorageBackend):
    """Copies artifacts to a directory on the local (host-mounted) filesystem."""

    def __init__(self, root: str) -> None:
        """
        Args:
            root: absolute path to the backup root directory.
                  Must be an existing, writable directory at backup time.
                  Empty string is allowed here; run_backup() will reject it cleanly.
        """
        self._root = root

    def run_backup(self, model_dir: str, output_dir: str) -> BackupResult:
        # ── Validate backup root ──────────────────────────────────────────────
        if not self._root or not self._root.strip():
            return BackupResult(
                ok=False, destination="", copied=[], skipped=[],
                error="Backup target directory is not configured. "
                      "Set 'Backup dir' in Configuration.",
            )

        br = Path(self._root)
        if not br.exists():
            return BackupResult(
                ok=False, destination="", copied=[], skipped=[],
                error=f"Backup target directory does not exist: {self._root}",
            )
        if not br.is_dir():
            return BackupResult(
                ok=False, destination="", copied=[], skipped=[],
                error=f"Backup target path is not a directory: {self._root}",
            )

        # ── Validate model source ─────────────────────────────────────────────
        src_model = Path(model_dir)
        if not src_model.exists():
            return BackupResult(
                ok=False, destination="", copied=[], skipped=[],
                error=f"Model directory does not exist: {model_dir}",
            )

        # ── Create timestamped destination ────────────────────────────────────
        ts  = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        dst = br / ts
        try:
            dst.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            return BackupResult(
                ok=False, destination=str(dst), copied=[], skipped=[],
                error=f"Destination already exists (try again in 1 s): {dst}",
            )
        except OSError as exc:
            return BackupResult(
                ok=False, destination=str(dst), copied=[], skipped=[],
                error=f"Cannot create backup destination {dst}: {exc}",
            )

        copied:  list[str] = []
        skipped: list[str] = []

        # ── Copy model (required) ─────────────────────────────────────────────
        try:
            shutil.copytree(str(src_model), str(dst / "model"))
            copied.append("model")
        except Exception as exc:
            return BackupResult(
                ok=False, destination=str(dst), copied=copied, skipped=skipped,
                error=f"Failed to copy model directory: {exc}",
            )

        # ── Copy output (optional) ────────────────────────────────────────────
        src_output = Path(output_dir) if output_dir else None
        if src_output and src_output.exists() and src_output.is_dir():
            try:
                shutil.copytree(str(src_output), str(dst / "output"))
                copied.append("output")
            except Exception as exc:
                # Non-fatal: model is already safe.
                skipped.append(f"output: copy failed ({exc})")
        else:
            skipped.append(f"output: not found at {output_dir}")

        return BackupResult(
            ok=True,
            destination=str(dst),
            copied=copied,
            skipped=skipped,
        )
