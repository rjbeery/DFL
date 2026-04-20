"""
ui.py
-----
HTML UI routes for the DFL control panel.

GET /                   → full page (index.html)            [auth-protected]
GET /state/partial      → state panel fragment (polled)     [public]
GET /history/partial    → history fragment (polled)         [public]
POST /ui/run/{stage}    → launch a stage; returns fragment  [auth-protected]
POST /ui/stop           → stop active process; returns frag [auth-protected]
GET /config/partial     → config form fragment              [auth-protected]
POST /config            → apply config update               [auth-protected]
POST /ui/backup         → trigger backup; returns fragment  [auth-protected]
GET /backup/partial     → backup card fragment              [auth-protected]

Kept separate from the JSON API in main.py.
The ProcessManager singleton is injected via init() at startup.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from markupsafe import escape

import auth
import backup_service
import bundle_manager
import profile_manager
import snapshot_service
from auth import require_auth
from command_registry import build_stages, validate_for_stage
from config_manager import get_config, update_config
from process_manager import ProcessManager
from run_context import build_run_context

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

def _ts_to_hhmm(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def _tojson(obj) -> str:
    """Jinja2 filter: serialize *obj* to HTML-safe JSON for embedding in attributes."""
    return escape(json.dumps(obj, ensure_ascii=True, default=str))


templates.env.filters["strftime"] = _ts_to_hhmm
templates.env.filters["tojson"]   = _tojson

router = APIRouter()

# Injected by main.py so this module shares the same ProcessManager instance.
_manager: Optional[ProcessManager] = None

# Stages shown as action buttons in the UI.
_UI_STAGES = ["extract-src", "extract-dst", "train", "merge", "video-from-seq"]


def init(manager: ProcessManager) -> None:
    """Bind the shared ProcessManager singleton. Call once before serving."""
    global _manager
    _manager = manager


# ── Helpers ───────────────────────────────────────────────────────────────────

def _state_ctx() -> dict:
    """Build the template context dict from current manager state."""
    s   = _manager.get_state()
    cfg = get_config()
    start_str: Optional[str] = None
    if s["start_time"] is not None:
        start_str = datetime.datetime.fromtimestamp(s["start_time"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    result_mp4 = Path(cfg.workspace) / "result.mp4"
    return {
        "status":              s["status"],
        "stage":               s["stage"],
        "pid":                 s["pid"],
        "start_time":          start_str,
        "log_lines":           s["log_lines"],
        "is_running":          s["status"] == "running",
        "progress":            s["progress"],
        "recovered":           s["recovered"],
        "docker_image":        cfg.docker_image,
        "result_video_exists": result_mp4.is_file(),
    }


def _msg(request: Request, text: str, kind: str) -> HTMLResponse:
    """Render the message fragment (replaces #messages via HTMX)."""
    return templates.TemplateResponse(
        request,
        "partials/message.html",
        {"message": text, "kind": kind},
    )


def _config_form(
    request: Request,
    errors: list[str] | None = None,
    save_ok: bool = False,
) -> HTMLResponse:
    """Render the config form partial."""
    return templates.TemplateResponse(
        request,
        "partials/config_form.html",
        {
            "cfg":     get_config(),
            "errors":  errors or [],
            "save_ok": save_ok,
        },
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
def index(request: Request):
    ctx = _state_ctx()
    cfg = get_config()
    ctx["stages"]       = _UI_STAGES
    ctx["cfg"]          = cfg
    ctx["errors"]       = []
    ctx["save_ok"]      = False
    ctx["records"]      = _manager.get_history()
    ctx["auth_enabled"]    = auth.auth_enabled()
    ctx["active_profile"]  = profile_manager.get_active_profile()

    # Pretrained bootstrap visibility — filesystem check, no new backend logic.
    pretrained_dir = Path(cfg.workspace) / "pretrained" / "SAEHD"
    matches = list(pretrained_dir.glob("*encoder.npy")) if pretrained_dir.exists() else []
    ctx["pretrained_available"] = bool(matches)
    # Strip trailing "_encoder.npy" and any trailing "_" to get a display label.
    ctx["pretrained_label"] = (
        matches[0].name[: -len("encoder.npy")].rstrip("_") if matches else ""
    )

    return templates.TemplateResponse(request, "index.html", ctx)


@router.get("/state/partial", response_class=HTMLResponse)
def state_partial(request: Request):
    ctx = _state_ctx()
    return templates.TemplateResponse(request, "partials/state.html", ctx)


@router.post("/ui/run/{stage}", response_class=HTMLResponse,
             dependencies=[Depends(require_auth)])
def ui_run(request: Request, stage: str):
    """Launch a DFL stage and return a message fragment."""
    cfg    = get_config()
    stages = build_stages(cfg)

    if stage not in stages:
        return _msg(request, f"Unknown stage: '{stage}'.", "error")

    stage_cfg = stages[stage]
    if stage_cfg.cmd is None:
        return _msg(
            request,
            f"'{stage}' is not yet configured — {stage_cfg.not_wired_reason}",
            "warn",
        )

    errors = validate_for_stage(stage, cfg)
    if errors:
        return _msg(request, "Misconfiguration: " + " | ".join(errors), "error")

    ctx    = build_run_context(stage)
    result = _manager.run(
        stage, stage_cfg.cmd, stage_cfg.cwd,
        container_name        = stage_cfg.container_name,
        preset                = ctx["preset"],
        config_snapshot       = ctx["config_snapshot"],
        artifact_dirs         = ctx["artifact_dirs"],
        pretrained_bootstrap  = ctx["pretrained_bootstrap"],
    )
    if not result["ok"]:
        return _msg(request, result["error"], "error")

    return _msg(request, f"Started '{stage}' — PID {result['pid']}.", "ok")


@router.post("/ui/stop", response_class=HTMLResponse,
             dependencies=[Depends(require_auth)])
def ui_stop(request: Request):
    """Stop the active process and return a message fragment."""
    result = _manager.stop()
    if not result["ok"]:
        return _msg(request, result["error"], "error")
    return _msg(request, "Stop signal sent.", "ok")


@router.get("/history/partial", response_class=HTMLResponse)
def history_partial(request: Request):
    return templates.TemplateResponse(
        request,
        "partials/history.html",
        {"records": _manager.get_history()},
    )


@router.get("/config/partial", response_class=HTMLResponse,
            dependencies=[Depends(require_auth)])
def config_partial(request: Request):
    return _config_form(request)


@router.post("/ui/backup", response_class=HTMLResponse,
             dependencies=[Depends(require_auth)])
def ui_backup(request: Request):
    """Trigger a backup and return a message fragment."""
    if _manager.get_state()["status"] == "running":
        return _msg(
            request,
            "Cannot backup while a stage is running — model files may be mid-write.",
            "error",
        )

    cfg    = get_config()
    result = backup_service.run_backup(cfg)

    if not result.ok:
        return _msg(request, f"Backup failed: {result.error}", "error")

    parts = []
    if result.copied:
        parts.append("Copied: " + ", ".join(result.copied))
    if result.skipped:
        parts.append("Skipped: " + "; ".join(result.skipped))
    summary = " | ".join(parts) if parts else "Nothing copied."
    return _msg(request, f"Backup complete → {result.destination}  ({summary})", "ok")


@router.get("/snapshots/partial", response_class=HTMLResponse)
def snapshots_partial(request: Request):
    """Render the snapshots card (list + last result)."""
    cfg = get_config()
    return templates.TemplateResponse(
        request,
        "partials/snapshots.html",
        {
            "snapshots":   snapshot_service.list_snapshots(cfg.backup_dir),
            "last_result": snapshot_service.get_last_result(),
            "backup_dir":  cfg.backup_dir,
        },
    )


@router.get("/bundle/partial", response_class=HTMLResponse)
def bundle_partial(request: Request):
    """Render the export/import card (last result + summary counts)."""
    return templates.TemplateResponse(
        request,
        "partials/bundle.html",
        {
            "last_result": bundle_manager.get_last_result(),
            "summary":     bundle_manager.bundle_summary(_manager),
        },
    )


@router.get("/backup/partial", response_class=HTMLResponse,
            dependencies=[Depends(require_auth)])
def backup_partial(request: Request):
    """Render the backup card partial (button + last result)."""
    return templates.TemplateResponse(
        request,
        "partials/backup.html",
        {"last_backup": backup_service.get_last_backup()},
    )


@router.get("/uploads/partial", response_class=HTMLResponse, include_in_schema=False)
def uploads_partial(request: Request):
    """Render the uploads staging list as an HTML fragment."""
    cfg = get_config()
    uploads_dir = Path(cfg.workspace) / "uploads"
    files = []
    if uploads_dir.is_dir():
        for p in sorted(uploads_dir.iterdir()):
            if p.is_file():
                try:
                    st = p.stat()
                    files.append({"filename": p.name, "size": st.st_size})
                except OSError:
                    continue
    return templates.TemplateResponse(
        request, "partials/uploads_list.html", {"files": files}
    )


@router.post("/config", response_class=HTMLResponse,
             dependencies=[Depends(require_auth)])
async def config_update(request: Request):
    """Validate and apply a config update from the form; return config form fragment."""
    if _manager.get_state()["status"] == "running":
        return _config_form(
            request,
            errors=["Cannot update config while a job is running."],
        )

    form = await request.form()
    errors = update_config(dict(form))
    return _config_form(request, errors=errors, save_ok=(not errors))
