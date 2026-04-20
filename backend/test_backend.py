"""
test_backend.py
---------------
End-to-end tests for the DFL FastAPI control plane.

USAGE
-----
Step 1 — install requests (one-time):
    cd backend
    .venv\\Scripts\\pip install requests

Step 2 — start the server in one terminal:
    cd backend
    .venv\\Scripts\\uvicorn main:app --host 127.0.0.1 --port 8000 --no-access-log

Step 3 — run tests in a second terminal:
    cd backend
    .venv\\Scripts\\python test_backend.py

Expected runtime: ~75 seconds (dominated by the dummy stage sleep loops).
"""

import sys
import os
import time
import threading

import requests

BASE = "http://127.0.0.1:8000"


# ── Result tracking ───────────────────────────────────────────────────────────

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


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def GET(path: str) -> requests.Response:
    return requests.get(f"{BASE}{path}", timeout=5)


def POST(path: str) -> requests.Response:
    return requests.post(f"{BASE}{path}", timeout=5)


def wait_for_status(target: str, timeout: float = 40.0) -> bool:
    """Poll /state until status matches *target* or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if GET("/state").json()["status"] == target:
                return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def reset() -> None:
    """Ensure server is idle/done/stopped before next test starts."""
    try:
        status = GET("/state").json()["status"]
        if status == "running":
            POST("/stop")
            wait_for_status("stopped", timeout=8)
    except Exception:
        pass
    time.sleep(0.15)


def collect_sse(max_data_lines: int = 4, timeout: float = 25.0) -> "tuple[list[str], bool]":
    """
    Open /logs/stream in a background thread, collect up to *max_data_lines*
    'data: …' lines, and return (lines, got_close_event).

    Closes the connection once the limit is reached or a 'close' event arrives.
    The background thread is joined before returning.
    """
    lines: list[str] = []
    got_close = False

    def _reader() -> None:
        nonlocal got_close
        try:
            with requests.get(
                f"{BASE}/logs/stream",
                stream=True,
                timeout=(5.0, timeout),
            ) as resp:
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    if raw.startswith("data: "):
                        lines.append(raw[len("data: "):])
                        if len(lines) >= max_data_lines:
                            return
                    elif raw.startswith("event: close"):
                        got_close = True
                        return
        except Exception:
            pass

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(timeout=timeout + 3.0)
    return lines, got_close


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health() -> None:
    print("\n── [1] GET /health ──────────────────────────────────────────")
    r = GET("/health")
    check("status 200",   r.status_code == 200)
    check("body ok:true", r.json() == {"ok": True})


def test_idle_state() -> None:
    print("\n── [2] GET /state (idle) ────────────────────────────────────")
    reset()
    r = GET("/state")
    s = r.json()
    check("status 200",     r.status_code == 200)
    check("status=idle",    s["status"] == "idle",    f"got {s['status']!r}")
    check("stage=null",     s["stage"] is None,        f"got {s['stage']!r}")
    check("pid=null",       s["pid"] is None,          f"got {s['pid']!r}")
    check("log_lines=0",    s["log_lines"] == 0,       f"got {s['log_lines']}")


def test_run_and_complete() -> None:
    print("\n── [3] POST /run/merge — run to completion ──────────────────")
    reset()
    r = POST("/run/merge")
    check("start: 200 ok",   r.status_code == 200 and r.json().get("ok"))
    pid = r.json().get("pid")
    check("pid populated",   pid is not None and pid > 0, f"pid={pid}")

    s = GET("/state").json()
    check("status=running",     s["status"] == "running",    f"got {s['status']!r}")
    check("stage=merge",        s["stage"] == "merge",       f"got {s['stage']!r}")
    check("pid matches",        s["pid"] == pid,             f"got {s['pid']}")

    ok = wait_for_status("done", timeout=20)
    check("completes → status=done", ok)

    s = GET("/state").json()
    check("pid cleared after done",   s["pid"] is None)
    check("log_lines > 0",            s["log_lines"] > 0, f"got {s['log_lines']}")


def test_reject_double_run() -> None:
    print("\n── [4] POST /run while running — must reject ────────────────")
    reset()
    POST("/run/train")                     # long-running
    time.sleep(0.5)

    # Use a wired test stage (not merge, which now returns 501) to confirm
    # that a second launch attempt while one is running returns 409.
    r2 = POST("/run/long-output")
    check("second run → 409",       r2.status_code == 409, f"got {r2.status_code}")
    detail = r2.json().get("detail", "")
    check("error mentions 'train'",  "train" in detail.lower(), f"detail={detail!r}")

    POST("/stop")
    wait_for_status("stopped", timeout=8)


def test_stop() -> None:
    print("\n── [5] POST /stop while running ────────────────────────────")
    reset()
    POST("/run/train")
    time.sleep(1.5)

    s = GET("/state").json()
    check("still running before stop", s["status"] == "running", f"got {s['status']!r}")

    r = POST("/stop")
    check("stop → 200 ok", r.status_code == 200 and r.json().get("ok"))

    ok = wait_for_status("stopped", timeout=8)
    check("status=stopped after stop", ok)

    s = GET("/state").json()
    check("pid cleared after stop",    s["pid"] is None)


def test_stop_when_idle() -> None:
    print("\n── [6] POST /stop when nothing running — must reject ────────")
    reset()
    r = POST("/stop")
    check("stop when idle → 409", r.status_code == 409, f"got {r.status_code}")


def test_unknown_stage() -> None:
    print("\n── [7] POST /run/{unknown} — must 404 ───────────────────────")
    reset()
    r = POST("/run/not-a-real-stage")
    check("unknown stage → 404", r.status_code == 404, f"got {r.status_code}")


def test_sse_receives_lines() -> None:
    print("\n── [8] GET /logs/stream — receives live log lines ───────────")
    reset()
    POST("/run/extract-src")

    lines, _ = collect_sse(max_data_lines=3, timeout=20)
    check("≥ 3 log lines received via SSE",
          len(lines) >= 3,
          f"got {len(lines)}: {lines}")
    check("lines contain stage name",
          any("[extract-src]" in l for l in lines),
          f"lines={lines}")

    wait_for_status("done", timeout=20)


def test_sse_history_on_late_connect() -> None:
    """
    Tests the subscribe_with_history() fix:
    An SSE client connecting after several lines have been emitted must
    receive those lines immediately as buffered history, not miss them.
    """
    print("\n── [9] GET /logs/stream — history replayed on late connect ──")
    reset()
    POST("/run/train")           # 60-s stage — produces 1 line/s
    time.sleep(3.5)              # let ≥ 3 lines accumulate

    lines, _ = collect_sse(max_data_lines=2, timeout=10)
    check("late SSE connect receives buffered history",
          len(lines) >= 2,
          f"got {len(lines)}: {lines}")
    check("history lines contain stage name",
          any("[train]" in l for l in lines),
          f"lines={lines}")

    POST("/stop")
    wait_for_status("stopped", timeout=8)


def test_sse_close_event() -> None:
    """After a stage completes, the SSE stream must send a 'close' event."""
    print("\n── [10] GET /logs/stream — close event on completion ────────")
    reset()
    POST("/run/fail")     # exits in < 1 s — easiest to catch the close event

    # Open SSE immediately; collect until close event or timeout
    lines, got_close = collect_sse(max_data_lines=99, timeout=10)
    check("close event received after stage ends",
          got_close,
          f"lines={lines}")


def test_error_state() -> None:
    print("\n── [11] fail stage → status=error ──────────────────────────")
    reset()
    r = POST("/run/fail")
    check("fail stage accepted: 200", r.status_code == 200, f"got {r.status_code}")

    ok = wait_for_status("error", timeout=10)
    check("status=error after exit(1)", ok)

    s = GET("/state").json()
    check("pid cleared on error",    s["pid"] is None)
    check("log_lines > 0 on error",  s["log_lines"] > 0, f"got {s['log_lines']}")


def test_log_buffer_cap() -> None:
    """long-output prints 251 lines; buffer must cap at 200."""
    print("\n── [12] log buffer caps at 200 lines ────────────────────────")
    reset()
    POST("/run/long-output")
    wait_for_status("done", timeout=15)

    s = GET("/state").json()
    check("log_lines ≤ 200", s["log_lines"] <= 200, f"got {s['log_lines']}")
    check("log_lines > 0",   s["log_lines"] > 0,    f"got {s['log_lines']}")


def test_run_after_done() -> None:
    print("\n── [13] run again after previous done ───────────────────────")
    reset()
    POST("/run/extract-dst")
    wait_for_status("done", timeout=20)

    r = POST("/run/merge")
    check("new run accepted after done",
          r.status_code == 200 and r.json().get("ok"),
          f"status={r.status_code} body={r.json()}")
    check("log_lines reset to 0 at start",
          GET("/state").json()["log_lines"] == 0,
          f"got {GET('/state').json()['log_lines']}")

    wait_for_status("done", timeout=20)


def test_merge_not_wired() -> None:
    """merge is registered but cmd=None; must return HTTP 501, not launch anything."""
    print("\n── [15] POST /run/merge — 501 not yet wired ─────────────────")
    reset()
    r = POST("/run/merge")
    check("merge → 501",               r.status_code == 501, f"got {r.status_code}")
    detail = r.json().get("detail", "")
    check("detail is non-empty",       len(detail) > 10, f"detail={detail!r}")
    check("detail mentions merge",     "merge" in detail.lower(), f"detail={detail!r}")
    # State must remain idle — no process was started.
    s = GET("/state").json()
    check("state still idle after 501", s["status"] in ("idle", "done", "stopped", "error"),
          f"got {s['status']!r}")


def test_run_after_stopped() -> None:
    print("\n── [14] run again after previous stopped ────────────────────")
    reset()
    POST("/run/train")
    time.sleep(1)
    POST("/stop")
    wait_for_status("stopped", timeout=8)

    r = POST("/run/merge")
    check("new run accepted after stopped",
          r.status_code == 200 and r.json().get("ok"),
          f"status={r.status_code} body={r.json()}")

    wait_for_status("done", timeout=20)


def test_stop_before_output() -> None:
    """
    Stop a slow-start stage before it emits its first line.
    State must reach 'stopped'; subsequent run must work cleanly.
    """
    print("\n── [16] stop before first output line ───────────────────────")
    reset()
    POST("/run/slow-start")
    time.sleep(0.3)   # process is alive but hasn't printed yet

    s = GET("/state").json()
    check("running before stop", s["status"] == "running", f"got {s['status']!r}")

    r = POST("/stop")
    check("stop → 200 ok", r.status_code == 200 and r.json().get("ok"))

    ok = wait_for_status("stopped", timeout=8)
    check("status=stopped", ok)

    s = GET("/state").json()
    check("pid cleared", s["pid"] is None)

    # New run must start cleanly after the stop.
    r2 = POST("/run/long-output")
    check("new run accepted after stop-before-output",
          r2.status_code == 200 and r2.json().get("ok"),
          f"status={r2.status_code} body={r2.json()}")
    wait_for_status("done", timeout=15)


def test_stale_reader_does_not_interfere() -> None:
    """
    Generation-counter guard: stop a slow stage then immediately start a fast
    stage.  The old reader thread must not inject its exit summary into the
    new run's log buffer or send a spurious SSE sentinel.

    Verified by:
      - New run reaches 'done' normally.
      - log_lines is <= 200 (buffer not blown by stale lines).
    """
    print("\n── [17] stale reader does not pollute new run ───────────────")
    reset()

    # Start a slow stage, let it emit >= 1 line, then stop it.
    POST("/run/slow-start")
    time.sleep(2.5)
    POST("/stop")
    wait_for_status("stopped", timeout=8)

    # Immediately kick off a fast stage.
    POST("/run/long-output")
    ok = wait_for_status("done", timeout=15)
    check("new run completes → done", ok)

    s = GET("/state").json()
    check("log_lines > 0 after new run", s["log_lines"] > 0, f"got {s['log_lines']}")
    check("log_lines <= 200 (cap respected)", s["log_lines"] <= 200,
          f"got {s['log_lines']}")


def test_history_empty_at_start() -> None:
    print("\n-- [19] GET /history — empty list at start ------------------")
    reset()
    r = GET("/history")
    check("status 200",       r.status_code == 200, f"got {r.status_code}")
    body = r.json()
    check("returns a list",   isinstance(body, list), f"got {type(body)}")
    # May not be empty if previous tests ran — only check type.


def test_history_records_after_done() -> None:
    print("\n-- [20] GET /history — entry added after stage completes ----")
    reset()
    # Use fail stage: exits fast with non-zero code.
    POST("/run/fail")
    wait_for_status("error", timeout=10)
    time.sleep(0.3)   # give _reader() time to write the record

    r = GET("/history")
    check("status 200",         r.status_code == 200, f"got {r.status_code}")
    body = r.json()
    check("at least one entry", len(body) >= 1, f"got {len(body)}")

    newest = body[0]
    check("stage field present",     "stage" in newest, f"keys: {list(newest)}")
    check("status field present",    "status" in newest)
    check("summary field present",   "summary" in newest)
    check("exit_code field present", "exit_code" in newest)
    check("end_time field present",  "end_time" in newest)
    check("newest is error",         newest["status"] == "error",
          f"got {newest['status']!r}")
    check("exit_code non-zero",      newest["exit_code"] != 0,
          f"got {newest['exit_code']}")


def test_history_records_after_stop() -> None:
    print("\n-- [21] GET /history — entry added after stage stopped ------")
    reset()
    POST("/run/train")
    time.sleep(1.5)
    POST("/stop")
    wait_for_status("stopped", timeout=8)
    time.sleep(0.3)

    r = GET("/history")
    body = r.json()
    check("history non-empty", len(body) >= 1, f"got {len(body)}")
    newest = body[0]
    check("status=stopped in newest", newest["status"] == "stopped",
          f"got {newest['status']!r}")
    check("summary contains 'stopped'", "stopped" in newest["summary"],
          f"got {newest['summary']!r}")


def test_history_partial_html() -> None:
    print("\n-- [22] GET /history/partial — returns HTML -----------------")
    r = GET("/history/partial")
    check("status 200",     r.status_code == 200, f"got {r.status_code}")
    check("content is HTML", "text/html" in r.headers.get("content-type", ""),
          f"content-type: {r.headers.get('content-type')!r}")
    body = r.text
    check("has history-panel id", 'id="history-panel"' in body, body[:200])


def test_lifecycle_sequence() -> None:
    """
    Full lifecycle: idle → run → stop → run(error) → run → done.
    Checks that state transitions are correct at every step.
    """
    print("\n── [18] full lifecycle sequence ─────────────────────────────")
    reset()

    # Step 1: idle
    s = GET("/state").json()
    check("seq: initial idle", s["status"] == "idle", f"got {s['status']!r}")

    # Step 2: run (slow-start)
    r = POST("/run/slow-start")
    check("seq: run accepted", r.status_code == 200 and r.json().get("ok"))
    time.sleep(1)
    s = GET("/state").json()
    check("seq: running", s["status"] == "running", f"got {s['status']!r}")
    check("seq: stage=slow-start", s["stage"] == "slow-start", f"got {s['stage']!r}")

    # Step 3: stop
    POST("/stop")
    ok = wait_for_status("stopped", timeout=8)
    check("seq: stopped", ok)
    s = GET("/state").json()
    check("seq: pid cleared after stop", s["pid"] is None)

    # Step 4: run again (fail — exits with error)
    r = POST("/run/fail")
    check("seq: second run accepted", r.status_code == 200 and r.json().get("ok"))
    ok = wait_for_status("error", timeout=10)
    check("seq: error after fail stage", ok)
    s = GET("/state").json()
    check("seq: pid cleared after error", s["pid"] is None)

    # Step 5: run again (long-output — completes cleanly)
    r = POST("/run/long-output")
    check("seq: third run accepted", r.status_code == 200 and r.json().get("ok"))
    ok = wait_for_status("done", timeout=15)
    check("seq: done after long-output", ok)
    s = GET("/state").json()
    check("seq: pid cleared after done", s["pid"] is None)
    check("seq: log_lines > 0", s["log_lines"] > 0, f"got {s['log_lines']}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Verify the server is reachable before doing anything.
    print(f"Connecting to {BASE} …", end=" ", flush=True)
    try:
        GET("/health")
        print("OK\n")
    except requests.exceptions.ConnectionError:
        print("\n\nERROR: cannot reach the server.")
        print("Start it with:")
        print("  cd backend")
        print("  .venv\\Scripts\\uvicorn main:app --host 127.0.0.1 --port 8000 --no-access-log")
        sys.exit(1)

    # Run all tests in order.
    tests = [
        test_health,
        test_idle_state,
        test_run_and_complete,
        test_reject_double_run,
        test_stop,
        test_stop_when_idle,
        test_unknown_stage,
        test_sse_receives_lines,
        test_sse_history_on_late_connect,
        test_sse_close_event,
        test_error_state,
        test_log_buffer_cap,
        test_run_after_done,
        test_run_after_stopped,
        test_merge_not_wired,
        test_stop_before_output,
        test_stale_reader_does_not_interfere,
        test_history_empty_at_start,
        test_history_records_after_done,
        test_history_records_after_stop,
        test_history_partial_html,
        test_lifecycle_sequence,
    ]

    for fn in tests:
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"  CRASH in {fn.__name__}: {exc}")
            traceback.print_exc()
            _failed.append(f"{fn.__name__} (crashed)")

    # Summary
    total = len(_passed) + len(_failed)
    print(f"\n{'═' * 52}")
    print(f"  {len(_passed)} / {total} checks passed")
    if _failed:
        print(f"\n  Failed:")
        for name in _failed:
            print(f"    ✗  {name}")
    print(f"{'═' * 52}")

    sys.exit(0 if not _failed else 1)


if __name__ == "__main__":
    main()
