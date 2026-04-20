"""
run_context.py
--------------
Snapshot job-launch metadata for a single run.

Called once per run() invocation so main.py / ui.py do not need to
know about filesystem layouts or preset state.

Public API:
    build_run_context(stage) -> dict
        {preset, config_snapshot, artifact_dirs, pretrained_bootstrap}
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import preset_manager
from config_manager import get_config


# Config field names whose dirs to scan after each stage finishes.
_STAGE_ARTIFACT_FIELDS: dict[str, list[str]] = {
    "extract-src": ["data_src_aligned"],
    "extract-dst": ["data_dst_aligned"],
    "train":       ["model_dir"],
    "merge":       ["output_dir"],
}


def build_run_context(stage: str) -> dict:
    """
    Snapshot run-launch metadata.

    Returns:
        preset               : active preset name or None
        config_snapshot      : dict of preset-relevant config fields
        artifact_dirs        : list[str] — paths to scan at finish time
        pretrained_bootstrap : bool — whether a pretrained SAEHD pack is present
    """
    cfg      = get_config()
    cfg_dict = dataclasses.asdict(cfg)

    field_names   = _STAGE_ARTIFACT_FIELDS.get(stage, [])
    artifact_dirs = [cfg_dict[f] for f in field_names if cfg_dict.get(f)]

    pretrained_dir = Path(cfg.workspace) / "pretrained" / "SAEHD"
    pretrained_ok  = (
        pretrained_dir.exists()
        and bool(list(pretrained_dir.glob("*encoder.npy")))
    )

    return {
        "preset":               preset_manager.get_active(),
        "config_snapshot":      preset_manager.current_config_snapshot(),
        "artifact_dirs":        artifact_dirs,
        "pretrained_bootstrap": pretrained_ok,
    }
