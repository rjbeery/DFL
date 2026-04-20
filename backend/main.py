"""
main.py
-------
FastAPI application for the DFL control plane.

Endpoints:
    GET  /health         liveness probe           (public)
    GET  /state          current process state    (public)
    GET  /history        job history              (public)
    GET  /logs/stream    live log SSE stream      (public)
    GET  /backup/last    last backup result       (public)
    POST /run/{stage}    launch a DFL stage       (auth-protected)
    POST /stop           kill the active process  (auth-protected)
    POST /backup         copy artifacts to backup (auth-protected)

Auth:
    Set DFL_AUTH_ENABLED=true + DFL_AUTH_PASSWORD + DFL_SESSION_SECRET.
    When disabled (default), all routes are open — safe for local-only use.
    See auth.py for the full protection map.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from starlette.background import BackgroundTask
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.middleware.sessions import SessionMiddleware

import auth
import backup_service
import history_store as _hs
import bundle_manager
import file_manager
import preset_manager
import profile_manager
import snapshot_service
import state_store
import ui as ui_module
from artifact_scanner import scan_dir_shallow
from auth import NotAuthenticated, require_auth
from command_registry import build_stages, validate_for_stage
from config_manager import get_config, update_config
from process_manager import ProcessManager
from run_context import build_run_context


# ── Upload allowlist ──────────────────────────────────────────────────────────

_UPLOAD_ALLOWED_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".jpg", ".jpeg", ".png", ".zip"}
_ZIP_IMAGE_EXTS      = {".jpg", ".jpeg", ".png"}


def _extract_zip_to(zip_path: Path, dest_dir: Path) -> tuple:
    """
    Extract only image files from *zip_path* into *dest_dir*.
    Paths are flattened to basename-only (zip-slip protection).
    Existing files are skipped and reported in errors (no overwrite).
    Returns (extracted: list[dict], errors: list[str], had_nested: bool).
    had_nested is True when any entry carried a directory component.
    """
    extracted  = []
    errors     = []
    had_nested = False
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                # Detect nested paths before stripping directory components.
                if member.filename != Path(member.filename).name:
                    had_nested = True
                name = Path(member.filename).name   # basename only — zip-slip protection
                if not name:
                    continue
                if Path(name).suffix.lower() not in _ZIP_IMAGE_EXTS:
                    continue
                dest = dest_dir / name
                # Conflict guard — never overwrite an existing file.
                if dest.exists():
                    errors.append(
                        f"'{name}': already exists in {dest_dir} — skipped"
                        " (remove the existing file first to re-extract)"
                    )
                    continue
                size = 0
                try:
                    with zf.open(member) as src, dest.open("wb") as out:
                        while True:
                            chunk = src.read(65536)
                            if not chunk:
                                break
                            size += len(chunk)
                            out.write(chunk)
                    extracted.append({"filename": name, "saved_path": str(dest), "size": size})
                except OSError as e:
                    errors.append(f"'{name}': write error — {e}")
    except zipfile.BadZipFile:
        errors.append(f"'{zip_path.name}' is not a valid zip file.")
    return extracted, errors, had_nested

# Upload destination buckets: UI name → Config field (None = workspace/uploads special case).
_UPLOAD_DESTS: dict = {
    "uploads":  None,        # → workspace/uploads
    "data_src": "data_src",  # → cfg.data_src
    "data_dst": "data_dst",  # → cfg.data_dst
}


# ── Singleton ─────────────────────────────────────────────────────────────────

manager = ProcessManager(max_log_lines=200)
ui_module.init(manager)   # share the same instance with the HTML UI router


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.set_loop(asyncio.get_running_loop())

    # ── Startup: load persisted state ─────────────────────────────────────────
    stored = state_store.load()

    # Re-populate job history from disk (history_store is the primary source;
    # fall back to state_store recent_jobs for backwards compatibility).
    prior_history = _hs.load()
    if prior_history:
        manager.load_history(prior_history)
    elif stored.get("recent_jobs"):
        manager.load_history(stored["recent_jobs"])

    # If the server was killed while a job was running, synthesise an
    # "interrupted" record and set the recovered flag so /state can surface it.
    if stored.get("was_running_at_shutdown"):
        last_stage = stored.get("last_stage", "unknown")
        shutdown_t = stored.get("shutdown_time")
        manager.load_history([{
            "stage":      last_stage,
            "status":     "interrupted",
            "start_time": stored.get("last_start_time"),
            "end_time":   shutdown_t or time.time(),
            "duration":   None,
            "exit_code":  -1,
            "summary":    f"{last_stage} interrupted — server restarted",
            "progress":   {},
        }])
        manager.mark_recovered()

    yield

    # ── Shutdown: persist final state, then stop worker thread ───────────────
    s = manager.get_state()
    state_store.write_shutdown(s["stage"], s["status"], s["start_time"])
    manager.shutdown()   # stops active job + exits worker thread (5 s grace)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DFL Control Plane",
    description="Thin wrapper that launches DeepFaceLab CLI stages as subprocesses.",
    version="0.1.0",
    lifespan=lifespan,
)

# Session middleware — only added when auth is enabled.
# SessionMiddleware uses SameSite=lax by default, which prevents cross-site
# POST requests from carrying the session cookie (CSRF protection).
if auth.auth_enabled():
    secret = os.environ.get("DFL_SESSION_SECRET", "")
    if not secret:
        raise RuntimeError(
            "DFL_SESSION_SECRET must be set when DFL_AUTH_ENABLED=true.\n"
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    https_only = os.environ.get("DFL_HTTPS_ONLY", "").lower() == "true"
    app.add_middleware(
        SessionMiddleware,
        secret_key=secret,
        same_site="lax",
        https_only=https_only,   # set DFL_HTTPS_ONLY=true when behind TLS
        max_age=86400,           # 24-hour session
    )

app.include_router(auth.router)
app.include_router(ui_module.router)


# ── Exception handler for unauthenticated requests ────────────────────────────

@app.exception_handler(NotAuthenticated)
async def not_authenticated_handler(request: Request, exc: NotAuthenticated):
    if exc.htmx:
        # HTMX request: tell the client to redirect the whole page to /login.
        return Response(status_code=401, headers={"HX-Redirect": "/login"})
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/login", status_code=302)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Liveness probe")
def health():
    return {"ok": True}


@app.get("/state", summary="Current process state")
def state():
    s = manager.get_state()
    # Enrich with persisted config and most recent completed job
    s["config"]   = dataclasses.asdict(get_config())
    history       = manager.get_history()
    s["last_job"] = history[0] if history else None
    return s


@app.get("/config", summary="Read current configuration as JSON")
def read_config():
    return dataclasses.asdict(get_config())


@app.patch("/config", summary="Update configuration fields",
           dependencies=[Depends(require_auth)])
def patch_config(data: dict = Body(...)):
    """
    Accepts a JSON object with any subset of Config fields.
    Merges with current config before validating, so callers only need to
    send fields they want to change.
    """
    if manager.get_state()["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="Cannot update config while a job is running.",
        )
    # Merge partial update onto current config so unchanged fields stay valid
    current = dataclasses.asdict(get_config())
    current.update(data)
    errors = update_config(current)
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})
    return {"ok": True, "config": dataclasses.asdict(get_config())}


@app.post("/run/{stage}", summary="Launch a DFL stage (or enqueue if one is already running)",
          dependencies=[Depends(require_auth)])
def run_stage(stage: str):
    cfg    = get_config()
    stages = build_stages(cfg)

    if stage not in stages:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown stage '{stage}'. Valid stages: {sorted(stages)}",
        )

    stage_cfg = stages[stage]

    if stage_cfg.cmd is None:
        raise HTTPException(
            status_code=501,
            detail=f"Stage '{stage}' is not yet wired: {stage_cfg.not_wired_reason}",
        )

    errors = validate_for_stage(stage, cfg)
    if errors:
        raise HTTPException(
            status_code=422,
            detail={"misconfiguration": errors},
        )

    ctx    = build_run_context(stage)
    result = manager.run(
        stage, stage_cfg.cmd, stage_cfg.cwd,
        container_name        = stage_cfg.container_name,
        preset                = ctx["preset"],
        config_snapshot       = ctx["config_snapshot"],
        artifact_dirs         = ctx["artifact_dirs"],
        pretrained_bootstrap  = ctx["pretrained_bootstrap"],
    )
    if not result["ok"]:
        raise HTTPException(status_code=409, detail=result["error"])
    return result


@app.get("/history", summary="Completed job history (newest first)")
def history():
    return manager.get_history()


@app.post("/stop", summary="Stop the active subprocess",
          dependencies=[Depends(require_auth)])
def stop():
    result = manager.stop()
    if not result["ok"]:
        raise HTTPException(status_code=409, detail=result["error"])
    return result


@app.post("/backup", summary="Copy model and output artifacts to the backup target",
          dependencies=[Depends(require_auth)])
def run_backup():
    """
    Copies model_dir (and output_dir if present) to a new timestamped folder
    under backup_dir.  Rejected while a stage is running.
    """
    if manager.get_state()["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="Cannot backup while a stage is running — model files may be mid-write.",
        )

    cfg    = get_config()
    result = backup_service.run_backup(cfg)

    if not result.ok:
        raise HTTPException(status_code=422, detail=result.error)

    return {
        "ok":          result.ok,
        "destination": result.destination,
        "copied":      result.copied,
        "skipped":     result.skipped,
    }


@app.get("/backup/last", summary="Result of the most recent backup call")
def last_backup():
    result = backup_service.get_last_backup()
    if result is None:
        return {"ok": None, "destination": None, "copied": [], "skipped": [], "error": None}
    return {
        "ok":          result.ok,
        "destination": result.destination,
        "copied":      result.copied,
        "skipped":     result.skipped,
        "error":       result.error,
    }


# ── Presets ───────────────────────────────────────────────────────────────────

@app.get("/presets", summary="List all saved presets and the active preset name")
def presets_list():
    return {
        "active":  preset_manager.get_active(),
        "presets": preset_manager.list_presets(),
    }


@app.post("/presets", summary="Create or update a named preset",
          dependencies=[Depends(require_auth)])
async def presets_save(request: Request):
    """
    Body: {name: str, workspace: str, model_dir: str, ...preset fields...}
    Creates or overwrites the preset with that name.
    """
    body = await request.json()
    name = str(body.get("name", "")).strip()
    errors = preset_manager.save_preset(name, body)
    if errors:
        return JSONResponse({"ok": False, "errors": errors}, status_code=400)
    return {"ok": True, "presets": preset_manager.list_presets()}


@app.post("/presets/{name}/apply", summary="Apply a preset to the current config",
          dependencies=[Depends(require_auth)])
def presets_apply(name: str):
    """
    Merges the preset's fields onto the current config (non-preset fields are
    preserved).  Rejected while a stage is running.
    """
    if manager.get_state()["status"] == "running":
        return JSONResponse(
            {"ok": False, "errors": ["Cannot change config while a job is running."]},
            status_code=409,
        )
    errors = preset_manager.apply_preset(name)
    if errors:
        return JSONResponse({"ok": False, "errors": errors}, status_code=400)
    return {"ok": True}


@app.delete("/presets/{name}", summary="Delete a named preset",
            dependencies=[Depends(require_auth)])
def presets_delete(name: str):
    errors = preset_manager.delete_preset(name)
    if errors:
        return JSONResponse({"ok": False, "errors": errors}, status_code=404)
    return {"ok": True}


# ── Path validation ────────────────────────────────────────────────────────────

@app.post("/paths/validate", summary="Check whether path strings exist on the host filesystem")
async def validate_paths(request: Request):
    """
    Body: {field_name: path_string, ...}
    Returns {field_name: "ok" | "empty" | "not_found"} for each entry.
    Does not require auth — reads only the filesystem, modifies nothing.
    """
    body = await request.json()
    results = {}
    for field, raw_path in body.items():
        path = (raw_path or "").strip()
        if not path:
            results[field] = "empty"
        elif Path(path).exists():
            results[field] = "ok"
        else:
            results[field] = "not_found"
    return results


@app.post("/backup/create", summary="Create a snapshot of model and aligned face directories",
          dependencies=[Depends(require_auth)])
def backup_create():
    """
    Copies model_dir + aligned face dirs into a new timestamped folder under
    backup_dir.  Aligned dirs are skipped gracefully if absent; model_dir is
    required.  Rejected while a stage is running.
    """
    if manager.get_state()["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="Cannot create a snapshot while a stage is running — files may be mid-write.",
        )
    cfg    = get_config()
    import preset_manager as _pm
    label  = _pm.get_active() or ""
    result = snapshot_service.create_snapshot(cfg, label=label)
    if not result["ok"]:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@app.get("/backup/list", summary="List available snapshots under backup_dir")
def backup_list():
    """Returns snapshots sorted newest-first.  Fast: reads .meta.json only."""
    cfg = get_config()
    return snapshot_service.list_snapshots(cfg.backup_dir)


@app.post("/backup/restore", summary="Restore model and/or aligned faces from a snapshot",
          dependencies=[Depends(require_auth)])
async def backup_restore(request: Request):
    """
    Body: {
      "path":            str,   # absolute path to snapshot folder
      "restore_model":   bool,  # default true
      "restore_aligned": bool   # default true
    }
    Rejected while a stage is running.
    """
    if manager.get_state()["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="Cannot restore while a stage is running.",
        )
    body = await request.json()
    snap_path = str(body.get("path", "")).strip()
    if not snap_path:
        raise HTTPException(status_code=422, detail="'path' is required.")

    cfg    = get_config()
    result = snapshot_service.restore_snapshot(
        snap_path,
        cfg,
        restore_model   = bool(body.get("restore_model",   True)),
        restore_aligned = bool(body.get("restore_aligned", True)),
    )
    if not result["ok"]:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@app.get("/bundle/summary", summary="Lightweight counts for the export bundle (no filesystem scanning)")
def bundle_summary_route():
    return bundle_manager.bundle_summary(manager)


@app.get("/bundle/export", summary="Export a portable JSON state bundle")
def bundle_export():
    """
    Returns config, presets, job history, and lightweight model metadata as JSON.
    Intended to be downloaded by the browser as a .json file.
    """
    data = bundle_manager.export_bundle(manager)
    return JSONResponse(content=data)


@app.post("/bundle/import", summary="Import and apply a previously exported bundle",
          dependencies=[Depends(require_auth)])
async def bundle_import(request: Request):
    """
    Body: raw JSON bundle (as produced by GET /bundle/export).
    Replaces config, presets, and job history if the bundle is valid.
    Rejected while a stage is running.
    """
    if manager.get_state()["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="Cannot import while a stage is running.",
        )
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Request body is not valid JSON.")

    errors = bundle_manager.import_bundle(data, manager)
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})
    return {"ok": True}


# ── Profiles ──────────────────────────────────────────────────────────────────

@app.get("/profiles", summary="List all profiles and the active profile name")
def profiles_list():
    return {
        "active":   profile_manager.get_active(),
        "profiles": profile_manager.list_profiles(),
    }


@app.post("/profiles", summary="Create or update a named environment profile",
          dependencies=[Depends(require_auth)])
async def profiles_save(request: Request):
    """
    Body: {name, label?, dfl_root, workspace, backup_dir?, docker_image?}
    Creates or overwrites the profile with that name.
    """
    body = await request.json()
    name = str(body.get("name", "")).strip()
    errors = profile_manager.save_profile(name, body)
    if errors:
        return JSONResponse({"ok": False, "errors": errors}, status_code=400)
    return {"ok": True, "profiles": profile_manager.list_profiles()}


@app.post("/profiles/{name}/apply", summary="Switch to a named environment profile",
          dependencies=[Depends(require_auth)])
def profiles_apply(name: str):
    """
    Merges the profile's environment fields onto the current config.
    All workspace sub-paths are re-derived from the profile's workspace root.
    Blocked while a stage is running.
    """
    if manager.get_state()["status"] == "running":
        return JSONResponse(
            {"ok": False, "errors": ["Cannot switch profiles while a job is running."]},
            status_code=409,
        )
    errors = profile_manager.apply_profile(name)
    if errors:
        return JSONResponse({"ok": False, "errors": errors}, status_code=400)
    return {"ok": True}


@app.delete("/profiles/{name}", summary="Delete a named profile",
            dependencies=[Depends(require_auth)])
def profiles_delete(name: str):
    errors = profile_manager.delete_profile(name)
    if errors:
        return JSONResponse({"ok": False, "errors": errors}, status_code=400)
    return {"ok": True}


@app.post("/artifacts/scan", summary="Scan artifact directories for file counts and mtimes")
async def artifacts_scan(request: Request):
    """
    Body: {"dirs": ["path1", "path2", ...]}
    Returns {"scans": [{path, exists, file_count, mtime, truncated}, ...]}.
    Capped at 10 directories per request.  Does not require auth.
    """
    body = await request.json()
    dirs = body.get("dirs", [])
    if not isinstance(dirs, list):
        raise HTTPException(status_code=422, detail="dirs must be a list")
    return {"scans": [scan_dir_shallow(str(d)) for d in dirs[:10]]}


# ── Files ─────────────────────────────────────────────────────────────────────

@app.get("/files/roots", summary="List configured file roots")
def files_roots():
    """Returns metadata for every non-empty configured root, including paths and capabilities."""
    return file_manager.get_roots()


@app.get("/files/list", summary="Shallow directory listing within a safe root")
def files_list(root: str = Query(...), path: str = Query("")):
    """
    Returns one level of entries under root/path.
    Capped at 500 entries.  Subdirectory sizes are not computed.
    """
    try:
        entries = file_manager.list_dir(root, path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "root":    root,
        "path":    path,
        "entries": entries,
        "capped":  len(entries) >= file_manager.MAX_LIST_ENTRIES,
    }


@app.get("/files/download", summary="Download a single file from a safe root")
def files_download(root: str = Query(...), path: str = Query(...)):
    """
    Streams a single file.  Path must be within the declared root.
    Listing and download are public (workspace paths contain project data, not secrets).
    """
    try:
        target = file_manager.resolve_safe(root, path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path=str(target), filename=target.name)


@app.post("/files/upload", summary="Upload files into an upload-allowed root",
          dependencies=[Depends(require_auth)])
async def files_upload(
    root: str = Query(...),
    path: str = Query(""),
    files: list[UploadFile] = File(...),
):
    """
    Accepts multipart/form-data.  Files are written to root/path/.
    Only data_src and data_dst roots accept uploads.
    Each file is limited to 2 GB.
    """
    roots = file_manager.get_roots()
    if root not in roots:
        raise HTTPException(status_code=400, detail=f"Unknown root '{root}'.")
    if not roots[root].get("upload_ok"):
        raise HTTPException(status_code=403, detail=f"Upload not allowed to root '{root}'.")

    try:
        target_dir = file_manager.resolve_safe(root, path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    target_dir.mkdir(parents=True, exist_ok=True)

    if not files:
        raise HTTPException(status_code=422, detail="No files provided.")

    uploaded: list[str] = []
    limit_gb = file_manager.MAX_UPLOAD_BYTES // 1024 ** 3

    for upload in files:
        if not upload.filename:
            continue
        safe_name = Path(upload.filename).name   # strip any path component in filename
        if not safe_name:
            continue
        dest = target_dir / safe_name
        size = 0
        try:
            with dest.open("wb") as out:
                while True:
                    chunk = await upload.read(65536)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > file_manager.MAX_UPLOAD_BYTES:
                        out.close()
                        dest.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=422,
                            detail=f"'{safe_name}' exceeds the {limit_gb} GB upload limit.",
                        )
                    out.write(chunk)
        except HTTPException:
            raise
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Write error for '{safe_name}': {e}")
        uploaded.append(safe_name)

    if not uploaded:
        raise HTTPException(status_code=422, detail="No valid files in the upload.")

    return {"ok": True, "uploaded": uploaded, "count": len(uploaded)}


@app.get("/files/zip", summary="Download a directory as a bounded zip archive")
def files_zip(root: str = Query(...), path: str = Query("")):
    """
    Creates a zip of root/path and returns it for download.
    Limits: 1 000 files and 2 GB uncompressed.
    Only zip-allowed roots (aligned, model, output) support this.
    """
    roots = file_manager.get_roots()
    if root not in roots:
        raise HTTPException(status_code=400, detail=f"Unknown root '{root}'.")
    if not roots[root].get("zip_ok"):
        raise HTTPException(status_code=403, detail=f"ZIP not allowed for root '{root}'.")

    try:
        target = file_manager.resolve_safe(root, path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Target is not a directory.")

    file_count, total_bytes = file_manager.count_dir(target)
    if file_count == 0:
        raise HTTPException(status_code=422, detail="Directory is empty.")
    if file_count > file_manager.MAX_ZIP_FILES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Too many files ({file_count} > {file_manager.MAX_ZIP_FILES}). "
                "Use the Data Snapshots feature for large directories."
            ),
        )
    if total_bytes > file_manager.MAX_ZIP_BYTES:
        mb = file_manager.MAX_ZIP_BYTES // 1024 ** 2
        raise HTTPException(
            status_code=422,
            detail=(
                f"Directory too large ({total_bytes // 1024**2} MB > {mb} MB). "
                "Use the Data Snapshots feature instead."
            ),
        )

    try:
        tmp = file_manager.create_zip(target)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    zip_name = (target.name or root) + ".zip"
    return FileResponse(
        path=tmp,
        media_type="application/zip",
        filename=zip_name,
        background=BackgroundTask(os.unlink, tmp),
    )


@app.get("/result/download", summary="Download the most recent result.mp4")
def result_download():
    cfg  = get_config()
    path = Path(cfg.workspace) / "result.mp4"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="result.mp4 not found. Run video-from-seq first.")
    return FileResponse(path=str(path), filename="result.mp4", media_type="video/mp4")


# ── Upload bucket mapping ─────────────────────────────────────────────────────
# Maps UI bucket names to Config field names.

_MOVE_BUCKETS: dict = {
    "data_src": "data_src",
    "data_dst": "data_dst",
    "model":    "model_dir",
}


@app.get("/uploads", summary="List files in /workspace/uploads")
def list_uploads():
    """Returns all files currently in the uploads staging area."""
    cfg = get_config()
    uploads_dir = Path(cfg.workspace) / "uploads"
    if not uploads_dir.is_dir():
        return {"files": []}
    files = []
    for p in sorted(uploads_dir.iterdir()):
        if p.is_file():
            try:
                st = p.stat()
                files.append({
                    "filename": p.name,
                    "path":     str(p),
                    "size":     st.st_size,
                    "mtime":    st.st_mtime,
                })
            except OSError:
                continue
    return {"files": files}


@app.post("/uploads/move", summary="Move a file from /workspace/uploads to a DFL folder")
async def move_upload(request: Request):
    """
    Body: {"filename": "video.mp4", "bucket": "data_src"}
    Bucket values: data_src | data_dst | model
    Moves the file out of /workspace/uploads into the mapped workspace folder.
    """
    body = await request.json()
    filename = (body.get("filename") or "").strip()
    bucket   = (body.get("bucket")   or "").strip()

    if not filename:
        raise HTTPException(status_code=422, detail="'filename' is required.")
    if bucket not in _MOVE_BUCKETS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid bucket '{bucket}'. Choose: {list(_MOVE_BUCKETS)}",
        )

    cfg = get_config()
    uploads_dir = Path(cfg.workspace) / "uploads"

    # Guard: filename must be a plain name — no path separators, no traversal.
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name:
        raise HTTPException(status_code=422, detail="Invalid filename.")

    src = uploads_dir / safe_name
    if not src.is_file():
        raise HTTPException(status_code=404, detail=f"Not found in uploads: {safe_name}")

    dest_dir = Path(dataclasses.asdict(cfg)[_MOVE_BUCKETS[bucket]])
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / safe_name

    if dest.exists():
        raise HTTPException(
            status_code=409,
            detail=f"'{safe_name}' already exists in {bucket} ({dest}). "
                   "Rename or remove it first.",
        )

    src.rename(dest)

    return {"ok": True, "saved_path": str(dest), "filename": safe_name, "bucket": bucket}


@app.post("/upload", summary="Upload one or more files to a workspace folder")
async def upload_file(
    dest:        str              = Form("uploads"),
    files:       list[UploadFile] = File(...),
    extract_zip: bool             = Form(False),
):
    """
    Accepts multipart/form-data.
    dest:        uploads | data_src | data_dst   (default: uploads)
    files:       one or more files (videos, images, .zip)
    extract_zip: if True, .zip files are extracted (images only) instead of stored as-is
    extract_zip is rejected when dest=data_dst.
    All filenames/extensions are validated before any file is written.
    Returns {ok, dest, files: [{filename, saved_path, size}], extracted_files: [...], errors: [...]}
    """
    if dest not in _UPLOAD_DESTS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid destination '{dest}'. Choose: {list(_UPLOAD_DESTS)}",
        )
    if extract_zip and dest == "data_dst":
        raise HTTPException(
            status_code=422,
            detail="extract_zip is not allowed when dest=data_dst. "
                   "Upload the zip to 'uploads' and move individual files instead.",
        )
    if not files:
        raise HTTPException(status_code=422, detail="No files provided.")

    cfg = get_config()
    dest_dir = (
        Path(cfg.workspace) / "uploads"
        if dest == "uploads"
        else Path(dataclasses.asdict(cfg)[_UPLOAD_DESTS[dest]])
    )
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Validate all filenames/extensions before writing anything.
    validated = []
    for upload in files:
        if not upload.filename:
            raise HTTPException(status_code=422, detail="One or more files has an empty filename.")
        safe_name = Path(upload.filename).name
        ext = Path(safe_name).suffix.lower()
        if ext not in _UPLOAD_ALLOWED_EXTS:
            raise HTTPException(
                status_code=422,
                detail=f"'{safe_name}': file type '{ext}' is not allowed. "
                       f"Allowed: {', '.join(sorted(_UPLOAD_ALLOWED_EXTS))}",
            )
        validated.append((upload, safe_name))

    # Stream each file to disk; collect OS errors rather than aborting mid-batch.
    saved           = []
    extracted_files = []
    errors          = []
    notes           = []
    for upload, safe_name in validated:
        file_dest = dest_dir / safe_name
        size = 0
        try:
            with file_dest.open("wb") as out:
                while True:
                    chunk = await upload.read(65536)
                    if not chunk:
                        break
                    size += len(chunk)
                    out.write(chunk)
        except OSError as e:
            errors.append(f"'{safe_name}': write error — {e}")
            continue

        # If extract_zip is set and this is a zip, expand it now then remove the archive.
        if extract_zip and Path(safe_name).suffix.lower() == ".zip":
            imgs, zip_errors, had_nested = _extract_zip_to(file_dest, dest_dir)
            errors.extend(zip_errors)
            extracted_files.extend(imgs)
            if had_nested:
                notes.append(
                    f"'{safe_name}': zip contained nested folder paths — "
                    "only filenames were used (directory structure flattened)."
                )
            try:
                file_dest.unlink()
            except OSError:
                pass
        else:
            saved.append({"filename": safe_name, "saved_path": str(file_dest), "size": size})

    ok = bool(saved or extracted_files)
    return {
        "ok":              ok,
        "dest":            dest,
        "files":           saved,
        "extracted_files": extracted_files,
        "errors":          errors,
        "notes":           notes,
    }


@app.get("/logs/stream", summary="Live log stream (Server-Sent Events)")
async def logs_stream():
    """
    Streams log lines as SSE events.

    - Sends buffered history immediately on connect.
    - Pushes new lines live while the process runs.
    - Sends a heartbeat comment every 15 s to keep the connection alive.
    - Sends a final 'close' event when the process exits, then closes.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        history, q = manager.subscribe_with_history()
        try:
            for line in history:
                yield _sse_data(line)
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue

                if item is None:
                    yield _sse_event("close", "process-ended")
                    break

                yield _sse_data(item)
        finally:
            manager.unsubscribe(q)
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse_data(line: str) -> str:
    return f"data: {line}\n\n"


def _sse_event(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"
