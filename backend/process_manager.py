"""
process_manager.py
------------------
Job controller: state, queue, history, and SSE broadcast.

Subprocess execution is fully delegated to worker.Worker so this module
never calls Popen or taskkill directly.  That separation keeps the uvicorn
event-loop threads free during long DFL jobs.

Design:
- One active job at a time (enforced by ProcessState + _run_gen).
- FIFO in-memory queue for requests that arrive while a job is running.
- When a job ends the worker calls on_done(), which calls _start_next()
  to auto-start the next queued entry.
- SSE clients each hold a private asyncio.Queue fed via call_soon_threadsafe
  so the worker thread never touches the event loop directly.
- Windows process-tree kill is handled by worker.Worker.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import history_store
import state_store
from artifact_scanner import scan_dir_shallow
from job_history import JobHistory, JobRecord, make_summary
from progress_parser import ProgressParser
from worker import Worker, WorkerJob


# ── State types ───────────────────────────────────────────────────────────────

@dataclass
class ProcessState:
    stage:          Optional[str]   = None
    status:         str             = "idle"   # idle | running | done | error | stopped
    pid:            Optional[int]   = None
    start_time:     Optional[float] = None
    container_name: Optional[str]   = None


@dataclass
class QueueEntry:
    """One pending job waiting to run."""
    stage:               str
    command:             list
    cwd:                 Optional[str]
    container_name:      Optional[str]
    preset:              Optional[str]
    config_snapshot:     dict
    artifact_dirs:       list
    pretrained_bootstrap: bool
    queued_at:           float = field(default_factory=time.time)


# ── Manager ───────────────────────────────────────────────────────────────────

class ProcessManager:
    """
    Thread-safe job controller.

    Public API:
        run(stage, command, ...)  -> dict   (starts or enqueues)
        stop()                    -> dict   (stops active job; queue preserved)
        shutdown()                          (server exit: stop job + worker thread)
        get_state()               -> dict
        get_history()             -> list[dict]
        get_logs()                -> list[str]
        subscribe_with_history()  -> (list[str], asyncio.Queue)
        unsubscribe(queue)
        set_loop(loop)            (called once at startup)
    """

    def __init__(self, max_log_lines: int = 200) -> None:
        self._worker   = Worker()
        self._state    = ProcessState()
        self._lock     = threading.Lock()
        self._log_buf: deque[str]         = deque(maxlen=max_log_lines)
        self._clients: list[asyncio.Queue] = []
        self._loop:    Optional[asyncio.AbstractEventLoop] = None
        self._run_gen: int  = 0
        self._parser   = ProgressParser()
        self._history  = JobHistory(max_entries=50)
        self._recovered: bool = False
        self._queue:   deque[QueueEntry]  = deque()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # ── Public read ───────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        with self._lock:
            s = self._state
            queue_summary = [
                {"stage": e.stage, "queued_at": e.queued_at}
                for e in self._queue
            ]
            return {
                "stage":        s.stage,
                "status":       s.status,
                "pid":          s.pid,
                "start_time":   s.start_time,
                "log_lines":    len(self._log_buf),
                "progress":     self._parser.get().to_dict(),
                "recovered":    self._recovered,
                "queue_length": len(self._queue),
                "queue":        queue_summary,
            }

    def mark_recovered(self) -> None:
        self._recovered = True

    def load_history(self, records: list[dict]) -> None:
        for r in reversed(records):
            try:
                self._history.add(JobRecord(
                    stage                = r["stage"],
                    status               = r["status"],
                    start_time           = r.get("start_time"),
                    end_time             = r.get("end_time", 0.0),
                    duration             = r.get("duration"),
                    exit_code            = r.get("exit_code", -1),
                    summary              = r.get("summary", ""),
                    progress             = r.get("progress", {}),
                    job_id               = r.get("job_id", ""),
                    preset               = r.get("preset"),
                    config_snapshot      = r.get("config_snapshot", {}),
                    artifact_paths       = r.get("artifact_paths", []),
                    pretrained_bootstrap = r.get("pretrained_bootstrap", False),
                ))
            except Exception:
                pass

    def replace_history(self, records: list[dict]) -> None:
        self._history.clear()
        self.load_history(records)

    def get_history(self) -> list[dict]:
        return self._history.to_list()

    def get_logs(self) -> list[str]:
        with self._lock:
            return list(self._log_buf)

    # ── SSE subscription ──────────────────────────────────────────────────────

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._clients.append(q)
        return q

    def subscribe_with_history(self) -> "tuple[list[str], asyncio.Queue]":
        """
        Atomically snapshot the log buffer and register the client queue.
        No line can fall between the snapshot and first live push.
        """
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            history = list(self._log_buf)
            self._clients.append(q)
        return history, q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            try:
                self._clients.remove(q)
            except ValueError:
                pass

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(
        self,
        stage: str,
        command: list[str],
        cwd: Optional[str] = None,
        container_name: Optional[str] = None,
        preset: Optional[str] = None,
        config_snapshot: Optional[dict] = None,
        artifact_dirs: Optional[list] = None,
        pretrained_bootstrap: bool = False,
    ) -> dict:
        """
        Start a subprocess for *stage*, or enqueue if one is already running.

        Returns:
            {"ok": True,  "queued": False, "pid": None, "stage": ...}  — started
            {"ok": True,  "queued": True,  "stage": ..., "queue_position": n} — enqueued
            {"ok": False, "error": ...}  — worker failed to launch (Popen error)

        Note: pid is None in the immediate response; it becomes available in
        GET /state within milliseconds once the worker starts the subprocess.
        """
        entry = QueueEntry(
            stage=stage,
            command=command,
            cwd=cwd,
            container_name=container_name,
            preset=preset,
            config_snapshot=config_snapshot or {},
            artifact_dirs=artifact_dirs or [],
            pretrained_bootstrap=pretrained_bootstrap,
        )

        with self._lock:
            if self._state.status == "running":
                self._queue.append(entry)
                pos = len(self._queue)
                msg = f"[system] Stage '{stage}' queued (position {pos})."
                self._log_buf.append(msg)
                self._broadcast(msg)
                return {"ok": True, "queued": True, "stage": stage, "queue_position": pos}

            # Idle: claim running status atomically before releasing the lock.
            current_gen, job_id = self._claim_run(entry)

        self._submit_to_worker(entry, current_gen, job_id)
        return {"ok": True, "queued": False, "pid": None, "stage": stage}

    # ── Stop ──────────────────────────────────────────────────────────────────

    def stop(self) -> dict:
        """
        Signal the worker to kill the active subprocess.
        Queued jobs are preserved; the next one starts after the active process exits.
        """
        with self._lock:
            if self._state.status != "running":
                return {"ok": False, "error": "No process is currently running."}
            self._state.status = "stopped"
            self._state.pid    = None

        self._worker.stop_current()
        # on_done() will fire from the worker thread once the proc exits,
        # update final state, persist the record, and call _start_next().
        return {"ok": True}

    def shutdown(self) -> None:
        """Stop any active job and exit the worker thread (called at server exit)."""
        self._worker.shutdown(timeout=5.0)

    # ── Internal: launch helpers ───────────────────────────────────────────────

    def _claim_run(self, entry: QueueEntry) -> "tuple[int, str]":
        """
        Transition state to 'running' and return (gen, job_id).
        MUST be called while self._lock is held.
        """
        self._run_gen += 1
        current_gen = self._run_gen
        start_time  = time.time()
        job_id      = str(int(start_time * 1000))
        self._state = ProcessState(
            stage=entry.stage,
            status="running",
            start_time=start_time,
            container_name=entry.container_name,
        )
        self._log_buf.clear()
        self._parser.reset(entry.stage)
        return current_gen, job_id

    def _submit_to_worker(self, entry: QueueEntry, current_gen: int, job_id: str) -> None:
        """Build WorkerJob with closured callbacks and hand it to the worker."""
        stage = entry.stage

        def on_started(pid: int) -> None:
            with self._lock:
                if self._run_gen == current_gen:
                    self._state.pid = pid

        def on_line(line: str) -> None:
            with self._lock:
                if self._run_gen != current_gen:
                    return          # orphaned job — discard
                self._log_buf.append(line)
                self._parser.feed(line, stage)
                self._broadcast(line)

        def on_done(exit_code: int) -> None:
            self._on_done(exit_code, current_gen, entry, job_id)

        self._worker.submit(WorkerJob(
            stage=stage,
            command=entry.command,
            cwd=entry.cwd,
            container_name=entry.container_name,
            on_started=on_started,
            on_line=on_line,
            on_done=on_done,
        ))

    def _on_done(
        self,
        exit_code: int,
        gen: int,
        entry: QueueEntry,
        job_id: str,
    ) -> None:
        """
        Called by the worker thread when a subprocess exits.
        Updates state, persists the record, broadcasts the summary, then
        starts the next queued job (if any).
        """
        with self._lock:
            is_current = (self._run_gen == gen)
            if is_current and self._state.status == "running":
                self._state.status = "done" if exit_code == 0 else "error"
                self._state.pid    = None
            final_status    = self._state.status if is_current else "orphaned"
            start_time_snap = self._state.start_time if is_current else None
            progress_snap   = self._parser.get() if is_current else None

        if not is_current:
            return

        artifact_paths = [scan_dir_shallow(d) for d in entry.artifact_dirs]

        end_time = time.time()
        record = JobRecord(
            stage                = entry.stage,
            status               = final_status,
            start_time           = start_time_snap,
            end_time             = end_time,
            duration             = (end_time - start_time_snap) if start_time_snap else None,
            exit_code            = exit_code,
            summary              = make_summary(entry.stage, final_status, exit_code, progress_snap),
            progress             = progress_snap.to_dict() if progress_snap else {},
            job_id               = job_id,
            preset               = entry.preset,
            config_snapshot      = entry.config_snapshot,
            artifact_paths       = artifact_paths,
            pretrained_bootstrap = entry.pretrained_bootstrap,
        )
        self._history.add(record)
        state_store.write_job(record.to_dict())
        history_store.append(record.to_dict())

        summary_line = (
            f"[system] Stage '{entry.stage}' finished "
            f"(exit code {exit_code}, status={final_status})."
        )
        self._append_and_broadcast(summary_line)
        self._broadcast_sentinel()

        self._start_next()

    def _start_next(self) -> None:
        """Dequeue and launch the next pending job. No-op if queue is empty."""
        with self._lock:
            if not self._queue or self._state.status == "running":
                return
            entry = self._queue.popleft()
            current_gen, job_id = self._claim_run(entry)

        msg = f"[system] Starting queued stage '{entry.stage}'."
        self._append_and_broadcast(msg)
        self._submit_to_worker(entry, current_gen, job_id)

    # ── Broadcast helpers ─────────────────────────────────────────────────────

    def _append_and_broadcast(self, line: str) -> None:
        with self._lock:
            self._log_buf.append(line)
            self._broadcast(line)

    def _broadcast(self, line: str) -> None:
        """Push *line* to every SSE client. MUST be called while self._lock is held."""
        if self._loop is None:
            return
        for q in list(self._clients):
            self._loop.call_soon_threadsafe(_safe_put, q, line)

    def _broadcast_sentinel(self) -> None:
        """Push None to signal SSE consumers the stream has closed."""
        if self._loop is None:
            return
        for q in list(self._clients):
            self._loop.call_soon_threadsafe(_safe_put, q, None)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_put(q: asyncio.Queue, item) -> None:
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        pass
