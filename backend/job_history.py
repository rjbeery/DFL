"""
job_history.py
--------------
In-memory job history: records, bounded storage, and summary generation.

Public interface:
    JobRecord   -- dataclass for one completed job
    JobHistory  -- thread-safe bounded deque (newest first)
    make_summary(stage, status, exit_code, progress) -> str
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from progress_parser import Progress


# ── JobRecord ─────────────────────────────────────────────────────────────────

@dataclass
class JobRecord:
    stage:      str
    status:     str             # "done" | "error" | "stopped" | "interrupted"
    start_time: Optional[float] # unix timestamp, may be None if never set
    end_time:   float           # unix timestamp
    duration:   Optional[float] # seconds, None if start_time unavailable
    exit_code:  int
    summary:    str             # human-readable one-liner
    progress:   dict            # snapshot of Progress.to_dict() at exit
    # Task 32 enrichment — default values preserve backward compat with persisted records
    job_id:               str          = ""
    preset:               Optional[str] = None
    config_snapshot:      dict         = field(default_factory=dict)
    artifact_paths:       list         = field(default_factory=list)
    pretrained_bootstrap: bool         = False

    def to_dict(self) -> dict:
        return {
            "stage":                self.stage,
            "status":               self.status,
            "start_time":           self.start_time,
            "end_time":             self.end_time,
            "duration":             self.duration,
            "exit_code":            self.exit_code,
            "summary":              self.summary,
            "progress":             self.progress,
            "job_id":               self.job_id,
            "preset":               self.preset,
            "config_snapshot":      self.config_snapshot,
            "artifact_paths":       self.artifact_paths,
            "pretrained_bootstrap": self.pretrained_bootstrap,
        }


# ── JobHistory ────────────────────────────────────────────────────────────────

class JobHistory:
    """
    Thread-safe bounded store for completed job records.
    Newest entry is always at index 0 of get_all().
    """

    def __init__(self, max_entries: int = 20) -> None:
        self._records: deque[JobRecord] = deque(maxlen=max_entries)
        self._lock = threading.Lock()

    def add(self, record: JobRecord) -> None:
        with self._lock:
            self._records.appendleft(record)   # newest at front

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def get_all(self) -> list[JobRecord]:
        with self._lock:
            return list(self._records)

    def to_list(self) -> list[dict]:
        return [r.to_dict() for r in self.get_all()]


# ── Summary generation ────────────────────────────────────────────────────────

def make_summary(
    stage:     str,
    status:    str,
    exit_code: int,
    progress:  Optional[Progress],
) -> str:
    """
    Derive a concise one-line summary from the final job state.

    Rules:
      done  + tqdm current/total  → "{stage} done — {n}/{total} processed"
      done  + training iteration  → "{stage} done — iteration {iter:,}"
      done  (no other info)       → "{stage} done"
      error                       → "{stage} error — exit code {code}"
      stopped + iteration         → "{stage} stopped — last iteration {iter:,}"
      stopped + tqdm current      → "{stage} stopped — {n}/{total} processed"
      stopped (no other info)     → "{stage} stopped"
    """
    p = progress  # shorthand

    if status == "done":
        if p and p.current is not None and p.total is not None:
            return f"{stage} done \u2014 {p.current}/{p.total} processed"
        if p and p.iteration is not None:
            return f"{stage} done \u2014 iteration {p.iteration:,}"
        return f"{stage} done"

    if status == "error":
        return f"{stage} error \u2014 exit code {exit_code}"

    if status == "stopped":
        if p and p.iteration is not None:
            return f"{stage} stopped \u2014 last iteration {p.iteration:,}"
        if p and p.current is not None and p.total is not None:
            return f"{stage} stopped \u2014 {p.current}/{p.total} processed"
        return f"{stage} stopped"

    # Fallback for any unexpected status
    return f"{stage} {status}"
