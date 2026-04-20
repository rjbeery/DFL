"""
command_registry.py
-------------------
Maps stage names to subprocess commands for each DFL pipeline stage.

Two execution modes are supported, selected by the DFL_EXEC_MODE env var:

  docker  (default) — wraps each stage in `docker run --rm dfl:latest …`
                      Used for local development with the two-image compose setup.

  direct            — runs DFL scripts directly via python3.8 subprocess.
                      Used in the all-in-one cloud/RunPod image (Dockerfile.cloud)
                      where Docker-in-Docker is not available.

Docker-mode knobs (env vars, docker mode only):
  DFL_DOCKER_IMAGE     Image to run           (default: dfl:latest)
  DFL_DOCKER_GPU       GPU flag(s) to pass    (default: --gpus all)
                       Set to "" to disable GPU passthrough.
  DFL_DOCKER_EXTRA     Any extra docker flags (default: empty)
  DFL_WORKSPACE_HOST   Real host path for -v mount when backend is itself
                       inside Docker (compose setup).

Direct-mode knobs (env vars, direct mode only):
  DFL_PYTHON           Python interpreter for DFL scripts (default: python3.8)

Container/image layout (both modes):
  /app/       DFL source code (main.py, debug_train.py, debug_merge.py, …)
  /workspace/ persistent data (host-mounted in docker mode, pod volume in cloud)
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import NamedTuple, Optional

from config_manager import Config


# ── Execution mode ─────────────────────────────────────────────────────────────
_EXEC_MODE  = os.environ.get("DFL_EXEC_MODE",  "docker").lower()   # "docker" | "direct"
_DFL_PYTHON = os.environ.get("DFL_PYTHON",     "python3.8")        # interpreter for direct mode

# ── Docker-mode knobs ──────────────────────────────────────────────────────────
_GPU_FLAG = os.environ.get("DFL_DOCKER_GPU",  "--gpus all")
_EXTRA    = os.environ.get("DFL_DOCKER_EXTRA", "")

# Fixed paths inside the DFL image / cloud container
_C_WORKSPACE   = "/workspace"
_C_MAIN        = "/app/main.py"
_C_DBG_TRAIN   = "/app/debug_train.py"
_C_DBG_MERGE   = "/app/debug_merge.py"


# ── Stage config type ─────────────────────────────────────────────────────────

class StageCmd(NamedTuple):
    """
    Configuration for a single pipeline stage.

    cmd:              argv list passed to subprocess.Popen.
                      None = not yet wired (backend returns HTTP 501).
    cwd:              Working directory for the subprocess (host side).
    container_name:   Docker container name embedded in cmd; used for
                      `docker stop` during graceful shutdown.
    not_wired_reason: Human-readable explanation when cmd is None.
    """
    cmd:              Optional[list[str]]
    cwd:              str
    container_name:   Optional[str] = None
    not_wired_reason: str           = ""


# ── Path conversion ───────────────────────────────────────────────────────────

def _to_container(host_path: str, host_workspace: str) -> str:
    """
    Convert a host workspace sub-path to its /workspace equivalent inside
    the container.

    Example (Windows):
      host_workspace = "C:\\Users\\Rod\\Documents\\DFL\\workspace"
      host_path      = "C:\\Users\\Rod\\Documents\\DFL\\workspace\\data_src"
      returns        → "/workspace/data_src"
    """
    try:
        rel = Path(host_path).relative_to(Path(host_workspace))
        return str(PurePosixPath(_C_WORKSPACE) / rel)
    except ValueError:
        # host_path is not under workspace — pass through unchanged
        return host_path


# ── Docker prefix builder ─────────────────────────────────────────────────────

def _docker_prefix(cfg: Config, name: str) -> list[str]:
    """
    Build the `docker run` argument list up to (but not including) the
    in-container command.  Callers append [python, -u, script, args…].

    Windows note: Docker Desktop on Windows requires forward-slash volume
    paths, so we normalise the host workspace path before the colon.
    """
    args = ["docker", "run", "--rm"]

    if _GPU_FLAG:
        args += _GPU_FLAG.split()        # "--gpus all" → ["--gpus", "all"]

    if _EXTRA:
        args += _EXTRA.split()

    # Mount the host workspace directory as /workspace in the pipeline container.
    # When the backend itself runs inside Docker (compose setup), cfg.workspace is
    # the in-container path (e.g. /workspace).  DFL_WORKSPACE_HOST must be set to
    # the actual host path so the Docker daemon (which runs on the host) can create
    # the bind mount correctly.  Falls back to cfg.workspace for bare-metal installs.
    host_ws = os.environ.get("DFL_WORKSPACE_HOST") or cfg.workspace
    host_ws = host_ws.replace("\\", "/")
    args += ["-v", f"{host_ws}:{_C_WORKSPACE}"]

    # env var overrides the persisted config value (useful for CI / one-off overrides)
    image = os.environ.get("DFL_DOCKER_IMAGE") or cfg.docker_image
    args += ["--name", name, image]
    return args


def _container_name(stage: str) -> str:
    """
    Unique container name for this run.  The name is embedded in the command
    AND stored in StageCmd so stop() can call `docker stop <name>`.
    """
    return f"dfl-{stage}-{int(time.time())}"


# ── Stage builder ─────────────────────────────────────────────────────────────

def build_stages(cfg: Config) -> dict[str, StageCmd]:
    """
    Build the complete stage dict from the current Config.
    Called at request time so config changes take effect immediately.
    Dispatches to docker or direct mode based on DFL_EXEC_MODE env var.
    """
    if _EXEC_MODE == "direct":
        return _build_direct_stages(cfg)
    return _build_docker_stages(cfg)


def _build_docker_stages(cfg: Config) -> dict[str, StageCmd]:
    """
    Docker-spawn mode (local dev).
    Each stage wraps the DFL command in `docker run --rm dfl:latest`.
    """
    ws  = cfg.workspace
    cwd = cfg.dfl_root

    def cp(host_path: str) -> str:
        return _to_container(host_path, ws)

    def stage(name: str, subcmd: list[str]) -> StageCmd:
        cname = _container_name(name)
        cmd   = _docker_prefix(cfg, cname) + subcmd
        return StageCmd(cmd=cmd, cwd=cwd, container_name=cname)

    return {

        # ── extract-src ───────────────────────────────────────────────────────
        "extract-src": stage("extract-src", [
            "python", "-u", _C_MAIN, "extract",
            "--input-dir",  cp(cfg.data_src),
            "--output-dir", cp(cfg.data_src_aligned),
            "--detector",   cfg.extract_detector,
            "--no-output-debug",
        ]),

        # ── extract-dst ───────────────────────────────────────────────────────
        "extract-dst": stage("extract-dst", [
            "python", "-u", _C_MAIN, "extract",
            "--input-dir",  cp(cfg.data_dst),
            "--output-dir", cp(cfg.data_dst_aligned),
            "--detector",   cfg.extract_detector,
            "--no-output-debug",
        ]),

        # ── train ─────────────────────────────────────────────────────────────
        "train": stage("train", [
            "python", "-u", _C_DBG_TRAIN,
            "--model-name",      cfg.train_model_name,
            "--model-dir",       cp(cfg.model_dir),
            "--src-aligned-dir", cp(cfg.data_src_aligned),
            "--dst-aligned-dir", cp(cfg.data_dst_aligned),
        ]),

        # ── merge ─────────────────────────────────────────────────────────────
        "merge": stage("merge", [
            "python", "-u", _C_DBG_MERGE,
            "--model-class",     cfg.merge_model_class,
            "--model-name",      cfg.train_model_name,
            "--model-dir",       cp(cfg.model_dir),
            "--input-dir",       cp(cfg.data_dst),
            "--output-dir",      cp(cfg.output_dir),
            "--output-mask-dir", cp(cfg.output_mask_dir) if cfg.output_mask_dir else "",
            "--aligned-dir",     cp(cfg.data_dst_aligned),
        ]),

        # ── video-from-seq ────────────────────────────────────────────────────
        "video-from-seq": stage("video-from-seq", [
            "python", "-u", _C_MAIN, "videoed", "video-from-sequence",
            "--input-dir",   cp(cfg.output_dir),
            "--output-file", f"{_C_WORKSPACE}/result.mp4",
            "--fps",         cfg.video_fps,
        ]),
    }


def _build_direct_stages(cfg: Config) -> dict[str, StageCmd]:
    """
    Direct subprocess mode (cloud / RunPod).
    DFL scripts are called directly with python3.8 — no Docker wrapper.
    Paths come straight from cfg (already correct container paths when
    DFL_WORKSPACE is set).
    """
    py  = _DFL_PYTHON
    cwd = cfg.dfl_root   # /app/ in the cloud image

    def stage(name: str, subcmd: list[str]) -> StageCmd:
        # container_name=None → stop() falls back to PID-based kill
        return StageCmd(cmd=subcmd, cwd=cwd)

    mask_dir = cfg.output_mask_dir or str(Path(cfg.workspace) / "output_mask")
    result   = str(Path(cfg.workspace) / "result.mp4")

    return {

        # ── extract-src ───────────────────────────────────────────────────────
        "extract-src": stage("extract-src", [
            py, "-u", _C_MAIN, "extract",
            "--input-dir",  cfg.data_src,
            "--output-dir", cfg.data_src_aligned,
            "--detector",   cfg.extract_detector,
            "--no-output-debug",
        ]),

        # ── extract-dst ───────────────────────────────────────────────────────
        "extract-dst": stage("extract-dst", [
            py, "-u", _C_MAIN, "extract",
            "--input-dir",  cfg.data_dst,
            "--output-dir", cfg.data_dst_aligned,
            "--detector",   cfg.extract_detector,
            "--no-output-debug",
        ]),

        # ── train ─────────────────────────────────────────────────────────────
        "train": stage("train", [
            py, "-u", _C_DBG_TRAIN,
            "--model-name",      cfg.train_model_name,
            "--model-dir",       cfg.model_dir,
            "--src-aligned-dir", cfg.data_src_aligned,
            "--dst-aligned-dir", cfg.data_dst_aligned,
        ]),

        # ── merge ─────────────────────────────────────────────────────────────
        "merge": stage("merge", [
            py, "-u", _C_DBG_MERGE,
            "--model-class",     cfg.merge_model_class,
            "--model-name",      cfg.train_model_name,
            "--model-dir",       cfg.model_dir,
            "--input-dir",       cfg.data_dst,
            "--output-dir",      cfg.output_dir,
            "--output-mask-dir", mask_dir,
            "--aligned-dir",     cfg.data_dst_aligned,
        ]),

        # ── video-from-seq ────────────────────────────────────────────────────
        "video-from-seq": stage("video-from-seq", [
            py, "-u", _C_MAIN, "videoed", "video-from-sequence",
            "--input-dir",   cfg.output_dir,
            "--output-file", result,
            "--fps",         cfg.video_fps,
        ]),
    }


# ── Pre-launch validation ─────────────────────────────────────────────────────

def validate_for_stage(stage: str, cfg: Config) -> list[str]:
    """
    Sanity-check the environment for *stage* before launching.
    Returns a list of error strings; empty = all checks passed.
    """
    errors: list[str] = []

    # Docker daemon check — only relevant in docker mode.
    # In direct mode there is no Docker daemon (e.g. RunPod), so skip entirely.
    if _EXEC_MODE == "docker":
        try:
            subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
                check=True,
            )
        except Exception as exc:
            errors.append(f"Docker is not available: {exc}")
            return errors   # no point checking further

    if stage in ("extract-src", "extract-dst"):
        src_dir = cfg.data_src if stage == "extract-src" else cfg.data_dst
        label   = "Source" if stage == "extract-src" else "Destination"
        if not Path(src_dir).is_dir():
            errors.append(f"{label} frames directory missing: {src_dir}")

    elif stage == "train":
        for label, path in [
            ("Source aligned",      cfg.data_src_aligned),
            ("Destination aligned", cfg.data_dst_aligned),
        ]:
            p = Path(path)
            if not p.is_dir():
                errors.append(f"{label} faces directory missing: {p}")
            elif next(p.iterdir(), None) is None:
                errors.append(f"{label} faces directory is empty: {p}")

    elif stage == "video-from-seq":
        out = Path(cfg.output_dir)
        if not out.is_dir():
            errors.append(f"Output directory missing: {out}. Run merge first.")
        else:
            frames = list(out.glob("*.png"))[:1] or list(out.glob("*.jpg"))[:1]
            if not frames:
                errors.append(f"No image frames found in {out}. Run merge first.")

    elif stage == "merge":
        dst = Path(cfg.data_dst)
        if not dst.is_dir():
            errors.append(f"Destination frames directory missing: {dst}")
        elif next(dst.iterdir(), None) is None:
            errors.append(f"Destination frames directory is empty: {dst}")

        dst_aligned = Path(cfg.data_dst_aligned)
        if not dst_aligned.is_dir():
            errors.append(
                f"Destination aligned directory missing: {dst_aligned}. "
                "Run extract-dst first."
            )
        elif next(dst_aligned.iterdir(), None) is None:
            errors.append(
                f"Destination aligned directory is empty: {dst_aligned}. "
                "Run extract-dst first."
            )

        model_dir = Path(cfg.model_dir)
        if not model_dir.is_dir():
            errors.append(f"Model directory missing: {model_dir}")
        else:
            expected = f"{cfg.train_model_name}_{cfg.merge_model_class}_data.dat"
            if not (model_dir / expected).exists():
                errors.append(
                    f"Trained model file not found: {model_dir / expected}. "
                    "Train the model first, or check that 'Train model name' and "
                    "'Merge model class' match what was used during training."
                )

    return errors
