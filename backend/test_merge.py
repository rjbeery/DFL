"""
test_merge.py
-------------
Targeted tests for the merge stage: config validation, command construction,
and registry registration.  Runs without a live server or DFL data.

Usage:
    cd backend
    .venv\\Scripts\\python test_merge.py
"""

import os
import sys
import tempfile
from pathlib import Path

# -- Result tracking ----------------------------------------------------------─

_passed: list[str] = []
_failed: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        _passed.append(label)
        print(f"  PASS  {label}")
    else:
        _failed.append(label)
        msg = f"  FAIL  {label}"
        if detail:
            msg += f"\n        detail: {detail}"
        print(msg)


# -- Config fixture ------------------------------------------------------------

def _make_cfg(tmp: str, **overrides):
    """Build a Config using a temp directory as workspace."""
    from config_manager import Config
    base = dict(
        dfl_root          = str(Path(__file__).parent.parent),
        workspace         = tmp,
        data_src          = f"{tmp}/data_src",
        data_src_aligned  = f"{tmp}/data_src/aligned",
        data_dst          = f"{tmp}/data_dst",
        data_dst_aligned  = f"{tmp}/data_dst/aligned",
        model_dir         = f"{tmp}/model",
        output_dir        = f"{tmp}/output",
        output_mask_dir   = f"{tmp}/output_mask",
        merge_model_class = "SAEHD",
        train_model_name  = "poc",
    )
    base.update(overrides)
    return Config(**base)


# -- Tests --------------------------------------------------------------------─

def test_merge_registered():
    """Merge must appear in build_stages() with a real cmd (not None)."""
    print("\n-- test_merge_registered --")
    from config_manager import Config
    from command_registry import build_stages

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_cfg(tmp)
        stages = build_stages(cfg)

        check("merge key exists in stages", "merge" in stages)
        check("merge cmd is not None", stages["merge"].cmd is not None,
              "merge is still marked as not-wired")
        if stages["merge"].cmd is not None:
            cmd = stages["merge"].cmd
            check("debug_merge.py in merge cmd", any("debug_merge.py" in c for c in cmd))
            check("--model-class in merge cmd",  "--model-class"     in cmd)
            check("--model-name in merge cmd",   "--model-name"      in cmd)
            check("--input-dir in merge cmd",    "--input-dir"       in cmd)
            check("--output-dir in merge cmd",   "--output-dir"      in cmd)
            check("--output-mask-dir in merge cmd", "--output-mask-dir" in cmd)
            check("--aligned-dir in merge cmd",  "--aligned-dir"     in cmd)
            # Check values are resolved from config
            check("model class is SAEHD", "SAEHD" in cmd)
            check("model name is poc",    "poc"   in cmd)


def test_merge_validation_missing_dst():
    """Validation must fail when destination frames directory is missing."""
    print("\n-- test_merge_validation_missing_dst --")
    from command_registry import validate_for_stage

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_cfg(tmp)
        # data_dst does not exist (temp dir is empty)
        errors = validate_for_stage("merge", cfg)
        check("error when data_dst missing",
              any("Destination frames" in e for e in errors),
              f"errors = {errors}")


def test_merge_validation_missing_aligned():
    """Validation must fail when data_dst exists but aligned dir is missing."""
    print("\n-- test_merge_validation_missing_aligned --")
    from command_registry import validate_for_stage

    with tempfile.TemporaryDirectory() as tmp:
        # Create data_dst with a dummy frame but no aligned dir
        dst = Path(tmp) / "data_dst"
        dst.mkdir()
        (dst / "frame_001.jpg").write_bytes(b"dummy")

        cfg = _make_cfg(tmp)
        errors = validate_for_stage("merge", cfg)
        check("error when aligned dir missing",
              any("aligned" in e.lower() for e in errors),
              f"errors = {errors}")


def test_merge_validation_missing_model():
    """Validation must fail when the trained model data file is absent."""
    print("\n-- test_merge_validation_missing_model --")
    from command_registry import validate_for_stage

    with tempfile.TemporaryDirectory() as tmp:
        # Create data_dst + aligned with dummy files
        dst         = Path(tmp) / "data_dst"
        dst_aligned = dst / "aligned"
        model_dir   = Path(tmp) / "model"
        dst.mkdir()
        dst_aligned.mkdir(parents=True)
        model_dir.mkdir()
        (dst         / "frame_001.jpg").write_bytes(b"dummy")
        (dst_aligned / "00001.jpg"    ).write_bytes(b"dummy")
        # model_dir exists but has no .dat files

        cfg = _make_cfg(tmp)
        errors = validate_for_stage("merge", cfg)
        check("error when model data file missing",
              any("model" in e.lower() for e in errors),
              f"errors = {errors}")


def test_merge_validation_all_present():
    """Validation must pass when all prerequisites exist."""
    print("\n-- test_merge_validation_all_present --")
    from command_registry import validate_for_stage

    with tempfile.TemporaryDirectory() as tmp:
        dst         = Path(tmp) / "data_dst"
        dst_aligned = dst / "aligned"
        model_dir   = Path(tmp) / "model"
        dst.mkdir()
        dst_aligned.mkdir(parents=True)
        model_dir.mkdir()
        (dst         / "frame_001.jpg").write_bytes(b"dummy")
        (dst_aligned / "00001.jpg"    ).write_bytes(b"dummy")
        # Create the expected model data file: poc_SAEHD_data.dat
        (model_dir / "poc_SAEHD_data.dat").write_bytes(b"dummy")

        cfg = _make_cfg(tmp)
        errors = validate_for_stage("merge", cfg)
        check("no validation errors when all prerequisites present",
              errors == [],
              f"errors = {errors}")


def test_config_new_fields():
    """Config must include output_mask_dir and merge_model_class with correct defaults."""
    print("\n-- test_config_new_fields --")
    from config_manager import _default_config

    cfg = _default_config()
    check("output_mask_dir field exists",   hasattr(cfg, "output_mask_dir"))
    check("merge_model_class field exists", hasattr(cfg, "merge_model_class"))
    check("output_mask_dir default non-empty", bool(cfg.output_mask_dir))
    check("merge_model_class default is SAEHD", cfg.merge_model_class == "SAEHD")
    check("output_mask_dir ends with output_mask",
          cfg.output_mask_dir.replace("\\", "/").endswith("output_mask"))


def test_config_validation_bad_class():
    """update_config must reject unknown merge model classes."""
    print("\n-- test_config_validation_bad_class --")
    from config_manager import _validate

    data = {
        "dfl_root": "/app", "workspace": "/ws",
        "data_src": "/ws/a", "data_src_aligned": "/ws/b",
        "data_dst": "/ws/c", "data_dst_aligned": "/ws/d",
        "model_dir": "/ws/e", "output_dir": "/ws/f", "output_mask_dir": "/ws/g",
        "storage_backend": "local",
        "extract_detector": "s3fd",
        "train_model_name": "poc",
        "merge_model_class": "UnknownNet",
    }
    errors = _validate(data)
    check("validation error for unknown merge_model_class",
          any("Merge model class" in e for e in errors),
          f"errors = {errors}")


def test_config_validation_good_classes():
    """update_config must accept all known merge model classes."""
    print("\n-- test_config_validation_good_classes --")
    from config_manager import _validate

    base = {
        "dfl_root": "/app", "workspace": "/ws",
        "data_src": "/ws/a", "data_src_aligned": "/ws/b",
        "data_dst": "/ws/c", "data_dst_aligned": "/ws/d",
        "model_dir": "/ws/e", "output_dir": "/ws/f", "output_mask_dir": "/ws/g",
        "storage_backend": "local",
        "extract_detector": "s3fd",
        "train_model_name": "poc",
    }
    for cls in ("SAEHD", "AMP", "Quick96"):
        data = {**base, "merge_model_class": cls}
        errors = _validate(data)
        class_errors = [e for e in errors if "Merge model class" in e]
        check(f"no class error for merge_model_class={cls}", class_errors == [],
              f"errors = {errors}")


def test_debug_merge_script_exists():
    """debug_merge.py must exist at the DFL root."""
    print("\n-- test_debug_merge_script_exists --")
    script = Path(__file__).parent.parent / "debug_merge.py"
    check("debug_merge.py exists", script.is_file(), str(script))


# -- Runner --------------------------------------------------------------------

if __name__ == "__main__":
    # Add backend dir to sys.path so imports work when run directly
    backend_dir = str(Path(__file__).parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    test_merge_registered()
    test_merge_validation_missing_dst()
    test_merge_validation_missing_aligned()
    test_merge_validation_missing_model()
    test_merge_validation_all_present()
    test_config_new_fields()
    test_config_validation_bad_class()
    test_config_validation_good_classes()
    test_debug_merge_script_exists()

    print(f"\n{'='*50}")
    print(f"Results: {len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        print("Failed tests:")
        for f in _failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All tests passed.")
