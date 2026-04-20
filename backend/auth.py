"""
auth.py
-------
Minimal single-password session authentication for the DFL control panel.

Configuration (environment variables):
    DFL_AUTH_ENABLED   "true" to enable auth; any other value = disabled.
                       Default: "false" (auth off — safe for local-only use).
    DFL_AUTH_PASSWORD  Shared password. Required when DFL_AUTH_ENABLED=true.
    DFL_SESSION_SECRET Random string used to sign session cookies.
                       Required when DFL_AUTH_ENABLED=true.
                       Generate with: python -c "import secrets; print(secrets.token_hex(32))"
    DFL_API_TOKEN      Optional API bearer token.  When set, any request that
                       carries the header  X-DFL-Token: <value>  is treated as
                       authenticated without a session cookie.  Useful for curl,
                       CI scripts, or monitoring tools.  Ignored when
                       DFL_AUTH_ENABLED is not "true".

When DFL_AUTH_ENABLED is not "true":
    - require_auth is a no-op: all routes pass through.
    - Login/logout routes redirect to / immediately.
    - The main UI shows a visible warning banner.

Route protection:
    Protected (auth required):
        GET  /                    main UI page
        GET  /config/partial      config form (contains all path values)
        POST /config              update config
        POST /ui/run/{stage}      launch stage
        POST /ui/stop             stop stage
        POST /ui/backup           trigger backup
        GET  /backup/partial      backup card
        POST /run/{stage}         JSON API launch
        POST /stop                JSON API stop
        POST /backup              JSON API backup
        POST /logout              clear session

    Public (no auth, even when auth is enabled):
        GET  /health              liveness probe (monitoring tools need this)
        GET  /state               read-only JSON
        GET  /history             read-only JSON
        GET  /backup/last         read-only JSON
        GET  /logs/stream         read-only SSE
        GET  /state/partial       HTMX polling (read-only)
        GET  /history/partial     HTMX polling (read-only)
        GET  /login               login form
        POST /login               process login

CSRF:
    SameSite=lax on the session cookie prevents cross-origin form submissions
    from carrying the cookie. No additional CSRF token is needed for this
    single-origin, single-user use case.
    HTMX adds HX-Request: true on all its requests, which cannot be forged
    cross-origin, giving additional implicit same-origin verification.
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates


# ── Config readers ─────────────────────────────────────────────────────────────

def auth_enabled() -> bool:
    """Return True if DFL_AUTH_ENABLED=true is set."""
    return os.environ.get("DFL_AUTH_ENABLED", "").lower() == "true"


def _password() -> str:
    return os.environ.get("DFL_AUTH_PASSWORD", "")


# ── Exception for unauth'd access ─────────────────────────────────────────────

class NotAuthenticated(Exception):
    """Raised by require_auth when a request has no valid session."""
    def __init__(self, htmx: bool = False) -> None:
        self.htmx = htmx   # True → caller is an HTMX request


# ── Auth dependency ───────────────────────────────────────────────────────────

async def require_auth(request: Request) -> None:
    """
    FastAPI dependency. Raises NotAuthenticated if the request has no valid
    session. No-op when DFL_AUTH_ENABLED is not "true".

    Accepts either:
      - A valid session cookie (browser login), or
      - X-DFL-Token: <DFL_API_TOKEN> header (scripted / API access).

    Add to protected routes:
        @app.post("/example", dependencies=[Depends(require_auth)])
    """
    if not auth_enabled():
        return
    if request.session.get("authenticated"):
        return
    # API token — constant-time compare to prevent timing attacks.
    api_token = os.environ.get("DFL_API_TOKEN", "")
    if api_token:
        header_token = request.headers.get("X-DFL-Token", "")
        if header_token and hmac.compare_digest(
            header_token.encode(), api_token.encode()
        ):
            return
    raise NotAuthenticated(htmx=bool(request.headers.get("HX-Request")))


# ── Login / logout router ──────────────────────────────────────────────────────

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_templates     = Jinja2Templates(directory=str(_TEMPLATES_DIR))

router = APIRouter()


@router.get("/login", response_class=HTMLResponse, include_in_schema=False)
def login_page(request: Request) -> HTMLResponse:
    """Render the login form.  If auth is off or already authenticated → redirect."""
    if not auth_enabled():
        return RedirectResponse("/", status_code=302)
    if request.session.get("authenticated"):
        return RedirectResponse("/", status_code=302)
    return _templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login", response_class=HTMLResponse, include_in_schema=False)
async def do_login(request: Request) -> HTMLResponse:
    """Validate password and set session cookie."""
    if not auth_enabled():
        return RedirectResponse("/", status_code=302)

    form     = await request.form()
    password = str(form.get("password", ""))
    expected = _password()

    if not expected:
        # Auth is enabled but DFL_AUTH_PASSWORD is empty — refuse all logins
        # to force the operator to set the variable explicitly.
        return _templates.TemplateResponse(
            request,
            "login.html",
            {"error": "DFL_AUTH_PASSWORD is not set. Set it in the environment."},
            status_code=503,
        )

    if hmac.compare_digest(password.encode(), expected.encode()):
        request.session["authenticated"] = True
        return RedirectResponse("/", status_code=302)

    return _templates.TemplateResponse(
        request,
        "login.html",
        {"error": "Incorrect password."},
        status_code=401,
    )


@router.post("/logout", include_in_schema=False, dependencies=[Depends(require_auth)])
def do_logout(request: Request) -> RedirectResponse:
    """Clear the session and return to the login page."""
    request.session.clear()
    return RedirectResponse("/login", status_code=302)
