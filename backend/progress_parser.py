"""
progress_parser.py
------------------
Parses stage log lines to extract structured progress information.

Public interface:
    ProgressParser   -- stateful parser; call feed() per line, get() to read progress
    Progress         -- dataclass returned by ProgressParser.get()

Parsing rules implemented
──────────────────────────
1. Training iteration  [HH:MM:SS][#NNNNNN][NNNNms][loss...]
       → sets iteration (int), no pct (DFL training has no known total)

2. tqdm progress bar   Desc:  45%|====>  | 450/1000 [00:12<00:15, 35.50it/s]
       → sets pct (float 0-100), current (int), total (int)
       → only extracted when both n and total are present — no fabrication

3. Extract summary     'Faces detected:  80' / 'Images found:  100'
       → sets status_text only

System wrapper lines ([system] ...) are silently ignored — they carry no stage progress.
TensorFlow/CUDA startup banners and debug_train.py [setup] lines are also ignored.
ANSI escape codes and carriage returns are stripped before matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ── Cleaning ──────────────────────────────────────────────────────────────────

# Covers SGR colours, cursor positioning, and clear-to-EOL codes used by tqdm.
_RE_ANSI = re.compile(r'\x1b\[[0-9;]*[mGKHFJA-D]')


def _clean(raw: str) -> str:
    """Strip ANSI escape codes and bare carriage returns; trim whitespace."""
    return _RE_ANSI.sub('', raw).replace('\r', '').strip()


# ── Regex patterns ────────────────────────────────────────────────────────────

# Training iteration line: [HH:MM:SS][#000123][0250ms][0.1234][0.5678]
# Anchoring on the time + iter prefix is sufficient; we don't need loss values.
_RE_TRAIN = re.compile(
    r'\[\d{2}:\d{2}:\d{2}\]\[#(?P<iter>\d+)\]'
)

# tqdm progress bar: "Desc:  45%|==>  | 450/1000 [elapsed, rate]"
# desc must start with a letter (rejects bare paths like /some/file.jpg: N%)
_RE_TQDM = re.compile(
    r'(?P<desc>[A-Za-z][^:\n]*):\s+(?P<pct>\d+)%\|[^|]*\|\s*(?P<cur>\d+)/(?P<tot>\d+)'
)

# Extract summary lines
_RE_FACES  = re.compile(r'Faces detected:\s+(?P<n>\d+)')
_RE_IMAGES = re.compile(r'Images found:\s+(?P<n>\d+)')


# ── Progress dataclass ────────────────────────────────────────────────────────

@dataclass
class Progress:
    stage:       Optional[str]   = None
    status_text: str             = ""
    iteration:   Optional[int]   = None   # training: current iter number
    pct:         Optional[float] = None   # extract: tqdm percent (0-100)
    current:     Optional[int]   = None   # extract: tqdm n
    total:       Optional[int]   = None   # extract: tqdm total

    def to_dict(self) -> dict:
        return {
            "status_text": self.status_text,
            "iteration":   self.iteration,
            "pct":         self.pct,
            "current":     self.current,
            "total":       self.total,
        }


# ── Parser ────────────────────────────────────────────────────────────────────

class ProgressParser:
    """
    Stateful single-stage progress parser.

    Thread-safety: feed() and reset() must be called under the ProcessManager
    lock (they are). get() returns the current Progress dataclass; callers must
    not mutate it.
    """

    def __init__(self) -> None:
        self._p = Progress()

    def reset(self, stage: Optional[str] = None) -> None:
        """Replace progress with a blank slate. Call at the start of each new run."""
        self._p = Progress(stage=stage)

    def get(self) -> Progress:
        """Return the most recently parsed progress (read-only)."""
        return self._p

    def feed(self, line: str, stage: Optional[str] = None) -> None:
        """
        Parse one raw log line and update internal progress if the line is
        meaningful.  Lines that don't match any pattern are silently ignored.
        """
        text = _clean(line)
        if not text:
            return

        # Ignore our own wrapper messages — they don't carry stage progress.
        if text.startswith('[system]') or text.startswith('[setup]'):
            return

        # ── Training iteration ─────────────────────────────────────────────
        m = _RE_TRAIN.search(text)
        if m:
            it = int(m.group('iter'))
            self._p = Progress(
                stage       = stage,
                status_text = f"Iteration {it:,}",
                iteration   = it,
                # pct intentionally omitted — DFL training has no known total
            )
            return

        # ── tqdm progress bar ──────────────────────────────────────────────
        m = _RE_TQDM.search(text)
        if m:
            pct  = float(m.group('pct'))
            cur  = int(m.group('cur'))
            tot  = int(m.group('tot'))
            desc = m.group('desc').strip()
            self._p = Progress(
                stage       = stage,
                status_text = f"{desc}: {cur} / {tot}",
                pct         = pct,
                current     = cur,
                total       = tot,
            )
            return

        # ── Extract summary ────────────────────────────────────────────────
        m = _RE_FACES.search(text)
        if m:
            self._p = Progress(
                stage       = stage,
                status_text = f"Faces detected: {m.group('n')}",
            )
            return

        m = _RE_IMAGES.search(text)
        if m:
            # Don't overwrite 'Faces detected' if we already have it.
            if 'Faces' not in self._p.status_text:
                self._p = Progress(
                    stage       = stage,
                    status_text = f"Images found: {m.group('n')}",
                )
            return
