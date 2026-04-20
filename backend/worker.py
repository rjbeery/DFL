"""
worker.py
---------
Background thread that executes DFL subprocesses one at a time.

ProcessManager submits WorkerJob instances via Worker.submit().
The worker thread pulls jobs from an internal queue.Queue and executes
them sequentially, calling per-job callbacks for lifecycle events.

Keeping subprocess execution off the uvicorn event-loop threads means
the API stays responsive during long-running DFL stages.

Lifecycle callbacks (all called from the worker thread):
    on_started(pid)       — subprocess is running; pid is the OS PID
    on_line(line)         — one decoded stdout line (newline stripped)
    on_done(exit_code)    — subprocess has exited; exit_code from waitpid
"""
from __future__ import annotations

import os
import queue
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class WorkerJob:
    """Everything the worker needs to run one DFL stage."""
    stage:          str
    command:        list
    cwd:            Optional[str]
    container_name: Optional[str]
    on_started:     Callable[[int], None]   # pid
    on_line:        Callable[[str], None]   # stdout line
    on_done:        Callable[[int], None]   # exit code


class Worker:
    """
    Single daemon thread that executes WorkerJobs one at a time.

    Public methods are thread-safe and may be called from any thread.
    """

    def __init__(self) -> None:
        self._job_queue: queue.Queue[Optional[WorkerJob]] = queue.Queue()
        self._lock:            threading.Lock             = threading.Lock()
        self._proc:            Optional[subprocess.Popen] = None
        self._container_name:  Optional[str]              = None
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="dfl-worker"
        )
        self._thread.start()

    # ── Public API ────────────────────────────────────────────────────────────

    def submit(self, job: WorkerJob) -> None:
        """Queue *job* for execution. Returns immediately."""
        self._job_queue.put(job)

    def stop_current(self) -> None:
        """Kill the active subprocess. No-op if idle."""
        with self._lock:
            proc  = self._proc
            cname = self._container_name
        if proc is None:
            return
        try:
            if cname:
                subprocess.run(
                    ["docker", "stop", cname],
                    capture_output=True,
                    timeout=15,
                )
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                )
            else:
                proc.terminate()
        except Exception:
            pass

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop the active job and exit the worker thread (called at server shutdown)."""
        self.stop_current()
        self._job_queue.put(None)       # None sentinel → loop exits
        self._thread.join(timeout=timeout)

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while True:
            job = self._job_queue.get()
            if job is None:
                break               # shutdown sentinel
            self._execute(job)

    def _execute(self, job: WorkerJob) -> None:
        """Run *job*'s subprocess and relay callbacks. Runs inside the worker thread."""
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            proc = subprocess.Popen(
                job.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=job.cwd,
                creationflags=creationflags,
            )
        except Exception:
            job.on_done(-1)
            return

        with self._lock:
            self._proc           = proc
            self._container_name = job.container_name

        job.on_started(proc.pid)

        exit_code = -1
        try:
            for raw_line in proc.stdout:
                job.on_line(raw_line.rstrip("\n"))
            proc.wait()
            exit_code = proc.returncode
        except Exception:
            pass
        finally:
            with self._lock:
                self._proc           = None
                self._container_name = None

        job.on_done(exit_code)
