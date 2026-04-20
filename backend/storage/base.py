"""
storage/base.py
---------------
Shared result type and abstract interface for backup storage backends.

Adding a new backend (e.g. S3):
1. Create storage/s3.py implementing StorageBackend.
2. Register it in storage/__init__.py inside get_backend().
3. Add any extra config fields to config_manager.py (e.g. s3_bucket, s3_prefix).
4. Add the new name to _KNOWN_STORAGE_BACKENDS in config_manager.py.
5. Add a <option> entry in templates/partials/config_form.html.

BackupResult is intentionally backend-agnostic:
- destination is a URI string: filesystem path for local, "s3://bucket/prefix/ts" for S3.
- copied / skipped use plain artifact names ("model", "output").
- error is a plain string, never an exception object.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Optional


@dataclasses.dataclass
class BackupResult:
    ok:          bool
    destination: str         # URI of the backup location (path, s3://..., gs://..., …)
    copied:      list[str]   # artifact names written, e.g. ["model", "output"]
    skipped:     list[str]   # not written with reason, e.g. ["output: not found at /…"]
    error:       Optional[str] = None  # set only when ok is False


class StorageBackend(ABC):
    """
    Interface for backup storage targets.

    Each concrete implementation takes its own configuration in __init__
    (e.g. LocalFilesystemStorageBackend takes a root path; an S3 backend
    would take bucket + prefix).

    run_backup() is the only required method.  It must never raise — all
    errors must be captured and returned in BackupResult.error.
    """

    @abstractmethod
    def run_backup(self, model_dir: str, output_dir: str) -> BackupResult:
        """
        Copy model_dir (required) and output_dir (optional) to the backend.

        Args:
            model_dir:  absolute path to the source model directory (must exist).
            output_dir: absolute path to the source output directory
                        (skipped gracefully if absent or empty).

        Returns:
            BackupResult — never raises.
        """
        ...
