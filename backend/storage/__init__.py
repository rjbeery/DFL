"""
storage/__init__.py
-------------------
Factory: resolve the configured storage backend from a Config object.

To add a new backend:
1. Implement StorageBackend in storage/<name>.py.
2. Import it below and add an elif branch in get_backend().
3. Add its name to _KNOWN_STORAGE_BACKENDS in config_manager.py.

Current backends:
    "local"  →  LocalFilesystemStorageBackend  (host-mounted folder)

Planned (not yet implemented):
    "s3"     →  S3StorageBackend               (AWS S3 / S3-compatible)
    "gcs"    →  GCSStorageBackend              (Google Cloud Storage)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from storage.base import BackupResult, StorageBackend
from storage.local_fs import LocalFilesystemStorageBackend

if TYPE_CHECKING:
    from config_manager import Config

__all__ = ["BackupResult", "StorageBackend", "get_backend"]


def get_backend(cfg: "Config") -> StorageBackend:
    """
    Return a StorageBackend instance configured from *cfg*.

    The backend type is selected by cfg.storage_backend.
    Backend-specific parameters are read from cfg as needed.
    """
    if cfg.storage_backend == "local":
        return LocalFilesystemStorageBackend(root=cfg.backup_dir)

    # ── Future backends ───────────────────────────────────────────────────────
    # elif cfg.storage_backend == "s3":
    #     from storage.s3 import S3StorageBackend
    #     return S3StorageBackend(
    #         bucket=cfg.s3_bucket,
    #         prefix=cfg.s3_prefix,
    #         region=cfg.s3_region,
    #     )
    #
    # elif cfg.storage_backend == "gcs":
    #     from storage.gcs import GCSStorageBackend
    #     return GCSStorageBackend(
    #         bucket=cfg.gcs_bucket,
    #         prefix=cfg.gcs_prefix,
    #     )

    raise ValueError(
        f"Unknown storage backend: {cfg.storage_backend!r}. "
        "Check storage_backend in config."
    )
