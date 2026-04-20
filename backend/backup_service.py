"""
backup_service.py
-----------------
Thin service layer: selects the configured StorageBackend and runs a backup.

Public interface:
    run_backup(cfg) -> BackupResult
    get_last_backup() -> BackupResult | None

The last result is kept in module-level state so the UI can display it after
the HTTP response has been sent. It resets on container restart (in-memory only),
but the backup folders themselves persist on the host-mounted volume.

Safety contract (enforced by callers, not here):
    Do NOT call run_backup() while a DFL stage is running. Model weights may be
    mid-write during training, producing a corrupt checkpoint copy.
    The /backup endpoint and /ui/backup route both enforce this guard.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from storage import get_backend
from storage.base import BackupResult

if TYPE_CHECKING:
    from config_manager import Config


_last_result: Optional[BackupResult] = None


def get_last_backup() -> Optional[BackupResult]:
    """Return the result of the most recent backup call, or None if none yet."""
    return _last_result


def run_backup(cfg: "Config") -> BackupResult:
    """
    Run a backup using the storage backend selected in *cfg*.

    Selects and instantiates the backend from cfg.storage_backend, then
    delegates to backend.run_backup(cfg.model_dir, cfg.output_dir).
    """
    global _last_result
    backend = get_backend(cfg)
    result  = backend.run_backup(cfg.model_dir, cfg.output_dir)
    _last_result = result
    return result
