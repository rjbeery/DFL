"""
test_progress_parser.py
-----------------------
Unit tests for progress_parser.  No server required.

Usage:
    cd backend
    .venv\\Scripts\\python test_progress_parser.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from progress_parser import ProgressParser

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


# -- Helper --------------------------------------------------------------------

def feed(lines, stage="train"):
    p = ProgressParser()
    for line in lines:
        p.feed(line, stage=stage)
    return p.get()


# -- 1. Training iteration parsing ---------------------------------------------

def test_train_iter_ms():
    """Standard sub-10s iteration line."""
    print("\n-- [1] training iteration (ms format) ----------------------")
    line = "[12:34:56][#000100][0250ms][0.1234][0.5678]"
    p = feed([line])
    check("iteration parsed",  p.iteration == 100,   f"got {p.iteration}")
    check("status_text set",   "100" in p.status_text)
    check("pct is None",       p.pct is None,         f"got {p.pct}")
    check("no fabricated pct", p.pct is None)


def test_train_iter_sec():
    """Iteration line for slow iterations (>= 10 s, float-seconds format)."""
    print("\n-- [2] training iteration (seconds format) -----------------")
    line = "[09:15:03][#001000][3.1234s][0.0567][0.0432]"
    p = feed([line])
    check("iteration 1000",   p.iteration == 1000, f"got {p.iteration}")
    check("pct still None",   p.pct is None)


def test_train_iter_large():
    """Large iteration numbers (more than 6 digits are fine too)."""
    print("\n-- [3] training iteration (large number) -------------------")
    line = "[00:01:00][#010000][0100ms][0.0300][0.0280]"
    p = feed([line])
    check("iteration 10000", p.iteration == 10000, f"got {p.iteration}")


def test_train_iter_advances():
    """Progress should track the latest iteration, not the first."""
    print("\n-- [4] training iteration advances -------------------------")
    lines = [
        "[12:00:00][#000001][0100ms][0.5][0.5]",
        "[12:00:01][#000002][0100ms][0.4][0.4]",
        "[12:00:02][#000050][0100ms][0.3][0.3]",
    ]
    p = feed(lines)
    check("latest iteration", p.iteration == 50, f"got {p.iteration}")


# -- 2. Extract / tqdm progress parsing ---------------------------------------

def test_tqdm_standard():
    """Typical tqdm line from DFL extractor (ascii=True)."""
    print("\n-- [5] tqdm standard progress bar --------------------------")
    line = "Extracting faces:  45%|======>    | 450/1000 [00:12<00:15, 35.50it/s]"
    p = feed([line], stage="extract-src")
    check("pct 45.0",     p.pct == 45.0,  f"got {p.pct}")
    check("current 450",  p.current == 450)
    check("total 1000",   p.total == 1000)
    check("status_text",  "450" in p.status_text and "1000" in p.status_text)
    check("iteration is None", p.iteration is None)


def test_tqdm_loading():
    """Loading samples bar."""
    print("\n-- [6] tqdm loading bar -------------------------------------")
    line = "Loading:   5%|=         | 5/100 [00:01<00:19,  4.92it/s]"
    p = feed([line], stage="extract-dst")
    check("pct 5.0",   p.pct == 5.0,  f"got {p.pct}")
    check("cur 5",     p.current == 5)
    check("tot 100",   p.total == 100)


def test_tqdm_complete():
    """100% line."""
    print("\n-- [7] tqdm 100%% -------------------------------------------")
    line = "Extracting faces: 100%|==========| 100/100 [01:23<00:00,  1.20it/s]"
    p = feed([line], stage="extract-src")
    check("pct 100", p.pct == 100.0, f"got {p.pct}")
    check("cur==tot", p.current == p.total == 100)


# -- 3. No fabrication for training -------------------------------------------

def test_no_pct_for_training():
    """Training lines must never produce a pct value (no known total)."""
    print("\n-- [8] no fabricated percentage for training ----------------")
    lines = [
        "[12:34:56][#000001][0100ms][0.5][0.5]",
        "[12:34:56][#000010][0100ms][0.4][0.4]",
        "[12:34:56][#000100][0100ms][0.3][0.3]",
    ]
    p = feed(lines, stage="train")
    check("pct is None throughout", p.pct is None, f"got {p.pct}")
    check("iteration is set",       p.iteration is not None)


# -- 4. Extract summary lines --------------------------------------------------

def test_extract_summary():
    """Face detection summary lines at the end of extraction."""
    print("\n-- [9] extract summary lines --------------------------------")
    lines = [
        "Images found:        100",
        "Faces detected:      80",
    ]
    p = feed(lines, stage="extract-src")
    check("status_text has faces count", "80" in p.status_text, f"got {p.status_text!r}")
    check("no pct on summary",           p.pct is None)


def test_images_found_only():
    """Only Images found line (no Faces detected yet)."""
    print("\n-- [10] images-found without faces -------------------------")
    p = feed(["Images found:        50"], stage="extract-src")
    check("Images found in text", "50" in p.status_text, f"got {p.status_text!r}")


# -- 5. System and irrelevant lines are ignored --------------------------------

def test_system_lines_ignored():
    """[system] wrapper lines must not change progress."""
    print("\n-- [11] system lines ignored --------------------------------")
    parser = ProgressParser()
    parser.feed("[12:34:56][#000005][0100ms][0.5]", stage="train")
    before = parser.get().iteration

    parser.feed("[system] Stage 'train' finished (exit code 0, status=done).")
    after = parser.get().iteration

    check("iteration unchanged after system line", before == after == 5,
          f"before={before} after={after}")


def test_tensorflow_banner_ignored():
    """TensorFlow startup banner lines must not crash or change progress."""
    print("\n-- [12] TensorFlow banner ignored ---------------------------")
    parser = ProgressParser()
    noisy_lines = [
        "2024-01-01 00:00:00.000000: I tensorflow/core/platform/cpu_feature_guard.cc:182] ...",
        "W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: ...",
        "Using TensorFlow backend.",
        "",
        "   ",
    ]
    for line in noisy_lines:
        parser.feed(line, stage="train")
    p = parser.get()
    check("status_text empty after banners", p.status_text == "",
          f"got {p.status_text!r}")
    check("iteration None after banners", p.iteration is None)


def test_setup_lines_ignored():
    """debug_train.py [setup] lines must not change progress."""
    print("\n-- [13] [setup] lines ignored -------------------------------")
    parser = ProgressParser()
    parser.feed("[setup] Wrote model seed -> C:\\DFL\\workspace\\model\\poc_SAEHD_data.dat")
    parser.feed("[setup] Wrote default options -> C:\\DFL\\workspace\\model\\SAEHD_default_options.dat")
    p = parser.get()
    check("no progress from setup lines", p.status_text == "" and p.iteration is None)


# -- 6. ANSI escape stripping --------------------------------------------------

def test_ansi_stripped():
    """ANSI color codes must be stripped before matching."""
    print("\n-- [14] ANSI escape codes stripped -------------------------")
    # tqdm sometimes emits SGR color codes around the bar
    line = "\x1b[32mLoading\x1b[0m:  50%|=====     | 50/100 [00:01<00:01, 36.33it/s]"
    p = feed([line], stage="extract-src")
    check("pct parsed through ANSI", p.pct == 50.0, f"got {p.pct}")
    check("current parsed",          p.current == 50)


def test_carriage_return_stripped():
    """Bare \\r from tqdm / trainer must not break parsing."""
    print("\n-- [15] carriage return stripped ----------------------------")
    # Trainer emits end='\r'; Python universal-newlines translates to '\n'
    # but we test the raw \\r case too.
    line = "\r[12:34:56][#000042][0100ms][0.3][0.3]"
    p = feed([line], stage="train")
    check("iteration parsed through \\r", p.iteration == 42, f"got {p.iteration}")


# -- 7. Reset behavior ---------------------------------------------------------

def test_reset_clears_progress():
    """reset() must wipe all progress fields."""
    print("\n-- [16] reset clears progress -------------------------------")
    parser = ProgressParser()
    parser.feed("[12:34:56][#000100][0100ms][0.3]", stage="train")
    check("iteration set before reset", parser.get().iteration == 100)

    parser.reset(stage="extract-src")
    p = parser.get()
    check("iteration cleared", p.iteration is None, f"got {p.iteration}")
    check("pct cleared",       p.pct is None,       f"got {p.pct}")
    check("status_text empty", p.status_text == "",  f"got {p.status_text!r}")
    check("stage updated",     p.stage == "extract-src")


def test_reset_then_new_stage():
    """After reset, new stage lines should parse correctly."""
    print("\n-- [17] reset then new stage --------------------------------")
    parser = ProgressParser()
    parser.feed("[12:34:56][#000500][0100ms][0.1]", stage="train")
    parser.reset(stage="extract-src")
    parser.feed("Extracting faces:  20%|==>    | 20/100 [00:02<00:08, 10.0it/s]",
                stage="extract-src")
    p = parser.get()
    check("pct from new stage",  p.pct == 20.0,  f"got {p.pct}")
    check("no train iteration",  p.iteration is None)


# -- Entry point ---------------------------------------------------------------

def main() -> None:
    tests = [
        test_train_iter_ms,
        test_train_iter_sec,
        test_train_iter_large,
        test_train_iter_advances,
        test_tqdm_standard,
        test_tqdm_loading,
        test_tqdm_complete,
        test_no_pct_for_training,
        test_extract_summary,
        test_images_found_only,
        test_system_lines_ignored,
        test_tensorflow_banner_ignored,
        test_setup_lines_ignored,
        test_ansi_stripped,
        test_carriage_return_stripped,
        test_reset_clears_progress,
        test_reset_then_new_stage,
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
            print(f"    ✗  {name}")
    print(f"{'=' * 50}")
    sys.exit(0 if not _failed else 1)


if __name__ == "__main__":
    main()
