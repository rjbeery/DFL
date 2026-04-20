"""
test_history.py
---------------
Unit tests for JobHistory and make_summary.  No server required.

Usage:
    cd backend
    .venv\\Scripts\\python test_history.py
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))

from job_history import JobHistory, JobRecord, make_summary
from progress_parser import Progress

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
            msg += f"\n        {detail}"
        print(msg)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_record(stage="train", status="done", exit_code=0,
                start_time=None, duration=None, summary="train done",
                progress=None) -> JobRecord:
    now = time.time()
    return JobRecord(
        stage      = stage,
        status     = status,
        start_time = start_time if start_time is not None else now - 10.0,
        end_time   = now,
        duration   = duration if duration is not None else 10.0,
        exit_code  = exit_code,
        summary    = summary,
        progress   = progress or {},
    )


# -- 1. JobHistory storage ──────────────────────────────────────────────────────

def test_empty_history():
    print("\n-- [1] empty history ----------------------------------------")
    h = JobHistory()
    check("get_all empty",  h.get_all() == [])
    check("to_list empty",  h.to_list() == [])


def test_add_single():
    print("\n-- [2] add single record -----------------------------------")
    h = JobHistory()
    r = make_record()
    h.add(r)
    all_r = h.get_all()
    check("one record",     len(all_r) == 1)
    check("correct record", all_r[0] is r)


def test_newest_first():
    """Newest entry must be at index 0."""
    print("\n-- [3] newest first ordering --------------------------------")
    h = JobHistory()
    r1 = make_record(stage="extract-src", summary="first")
    r2 = make_record(stage="train",       summary="second")
    h.add(r1)
    h.add(r2)
    all_r = h.get_all()
    check("index 0 is newest", all_r[0] is r2, f"got {all_r[0].summary!r}")
    check("index 1 is older",  all_r[1] is r1)


def test_bounded_deque():
    """History must not grow beyond max_entries."""
    print("\n-- [4] bounded to max_entries ------------------------------")
    h = JobHistory(max_entries=3)
    for i in range(5):
        h.add(make_record(summary=f"job {i}"))
    all_r = h.get_all()
    check("capped at 3",     len(all_r) == 3,  f"got {len(all_r)}")
    # Oldest (0,1) are evicted; newest (4,3,2) remain in order.
    check("newest at [0]",   all_r[0].summary == "job 4", f"got {all_r[0].summary!r}")
    check("second at [1]",   all_r[1].summary == "job 3")
    check("oldest kept [2]", all_r[2].summary == "job 2")


def test_to_dict_keys():
    """to_dict() must include all required keys."""
    print("\n-- [5] to_dict keys ----------------------------------------")
    r = make_record()
    d = r.to_dict()
    for key in ("stage", "status", "start_time", "end_time", "duration",
                "exit_code", "summary", "progress"):
        check(f"key '{key}' present", key in d, f"keys: {list(d)}")


def test_to_list():
    """to_list() returns list of dicts."""
    print("\n-- [6] to_list ----------------------------------------------")
    h = JobHistory()
    h.add(make_record(summary="one"))
    h.add(make_record(summary="two"))
    lst = h.to_list()
    check("two dicts",          len(lst) == 2)
    check("newest summary",     lst[0]["summary"] == "two", f"got {lst[0]['summary']!r}")
    check("both are dicts",     all(isinstance(d, dict) for d in lst))


def test_get_all_returns_copy():
    """get_all() must return an independent copy — mutations must not affect history."""
    print("\n-- [7] get_all returns independent copy --------------------")
    h = JobHistory()
    h.add(make_record())
    copy = h.get_all()
    copy.clear()
    check("original unaffected", len(h.get_all()) == 1)


# -- 2. make_summary -----------------------------------------------------------

def test_summary_done_no_progress():
    print("\n-- [8] summary: done, no progress ---------------------------")
    s = make_summary("train", "done", 0, None)
    check("done no progress", s == "train done", f"got {s!r}")


def test_summary_done_iteration():
    print("\n-- [9] summary: done + iteration ----------------------------")
    p = Progress(iteration=12345)
    s = make_summary("train", "done", 0, p)
    check("done iteration", s == "train done \u2014 iteration 12,345", f"got {s!r}")


def test_summary_done_tqdm():
    print("\n-- [10] summary: done + tqdm current/total -----------------")
    p = Progress(current=450, total=1000)
    s = make_summary("extract-src", "done", 0, p)
    check("done tqdm", s == "extract-src done \u2014 450/1000 processed", f"got {s!r}")


def test_summary_error():
    print("\n-- [11] summary: error -------------------------------------")
    s = make_summary("train", "error", 1, None)
    check("error summary", s == "train error \u2014 exit code 1", f"got {s!r}")


def test_summary_error_nonzero():
    print("\n-- [12] summary: error non-zero exit code ------------------")
    s = make_summary("extract-dst", "error", 42, None)
    check("error exit 42", "42" in s, f"got {s!r}")


def test_summary_stopped_no_progress():
    print("\n-- [13] summary: stopped, no progress ----------------------")
    s = make_summary("train", "stopped", -1, None)
    check("stopped no progress", s == "train stopped", f"got {s!r}")


def test_summary_stopped_iteration():
    print("\n-- [14] summary: stopped + iteration -----------------------")
    p = Progress(iteration=5000)
    s = make_summary("train", "stopped", -1, p)
    check("stopped iteration", s == "train stopped \u2014 last iteration 5,000", f"got {s!r}")


def test_summary_stopped_tqdm():
    print("\n-- [15] summary: stopped + tqdm ----------------------------")
    p = Progress(current=30, total=100)
    s = make_summary("extract-src", "stopped", -1, p)
    check("stopped tqdm", s == "extract-src stopped \u2014 30/100 processed", f"got {s!r}")


def test_summary_done_prefers_tqdm_over_iteration():
    """When both pct and iteration are set, tqdm (current/total) takes priority for done."""
    print("\n-- [16] summary: done prefers tqdm over iteration ----------")
    p = Progress(iteration=100, current=50, total=200)
    s = make_summary("extract-src", "done", 0, p)
    check("tqdm preferred", "50/200" in s, f"got {s!r}")


def test_summary_stopped_prefers_iteration_over_tqdm():
    """For stopped, iteration is checked first (matches make_summary rule order)."""
    print("\n-- [17] summary: stopped iteration checked before tqdm -----")
    p = Progress(iteration=77, current=50, total=200)
    s = make_summary("train", "stopped", -1, p)
    check("iteration preferred for stopped", "77" in s and "last iteration" in s, f"got {s!r}")


def test_summary_fallback():
    print("\n-- [18] summary: unknown status fallback -------------------")
    s = make_summary("merge", "unknown-status", 0, None)
    check("fallback contains stage and status",
          "merge" in s and "unknown-status" in s, f"got {s!r}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    tests = [
        test_empty_history,
        test_add_single,
        test_newest_first,
        test_bounded_deque,
        test_to_dict_keys,
        test_to_list,
        test_get_all_returns_copy,
        test_summary_done_no_progress,
        test_summary_done_iteration,
        test_summary_done_tqdm,
        test_summary_error,
        test_summary_error_nonzero,
        test_summary_stopped_no_progress,
        test_summary_stopped_iteration,
        test_summary_stopped_tqdm,
        test_summary_done_prefers_tqdm_over_iteration,
        test_summary_stopped_prefers_iteration_over_tqdm,
        test_summary_fallback,
    ]
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"  CRASH in {fn.__name__}: {exc}")
            traceback.print_exc()
            _failed.append(f"{fn.__name__} (crashed)")

    total = len(_passed) + len(_failed)
    print(f"\n{'=' * 50}")
    print(f"  {len(_passed)} / {total} checks passed")
    if _failed:
        print(f"\n  Failed:")
        for name in _failed:
            print(f"    x  {name}")
    print(f"{'=' * 50}")
    sys.exit(0 if not _failed else 1)


if __name__ == "__main__":
    main()
