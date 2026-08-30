"""Run a stage as a subprocess (``python -m <module> …``) and tail its output.

One job at a time — the stages share the GPU. The server process never imports
torch; all model loading happens in the child.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Job:
    id: str
    stage: str
    argv: list[str]
    home: Path
    started: float = field(default_factory=time.time)
    finished: float | None = None
    exit_code: int | None = None
    lines: list[str] = field(default_factory=list)
    report_path: str | None = None
    values: dict[str, Any] = field(default_factory=dict)
    apply: bool = False
    _proc: subprocess.Popen | None = field(default=None, repr=False)
    _cond: threading.Condition = field(default_factory=threading.Condition, repr=False)

    @property
    def state(self) -> str:
        if self.exit_code is None:
            return "running"
        if self.exit_code == 0:
            return "done"
        if self.exit_code < 0:
            return "cancelled"
        return "failed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stage": self.stage,
            "argv": self.argv,
            "state": self.state,
            "started": self.started,
            "finished": self.finished,
            "exit_code": self.exit_code,
            "lines": len(self.lines),
            "report_path": self.report_path,
            "apply": self.apply,
            "values": self.values,
        }

    def wait_lines(self, index: int, timeout: float = 1.0) -> list[str]:
        """Lines from ``index`` on, blocking up to ``timeout`` for new ones."""
        with self._cond:
            if len(self.lines) <= index and self.exit_code is None:
                self._cond.wait(timeout)
            return self.lines[index:]


class JobManager:
    def __init__(self, *, log_dir: Path | None = None, max_lines: int = 20000):
        self.jobs: dict[str, Job] = {}
        self.log_dir = log_dir
        self.max_lines = max_lines
        self._lock = threading.Lock()

    @property
    def running(self) -> Job | None:
        return next((j for j in self.jobs.values() if j.exit_code is None), None)

    def start(
        self,
        stage: str,
        module: str,
        argv: list[str],
        *,
        home: Path,
        report_path: str | None = None,
        values: dict[str, Any] | None = None,
        apply: bool = False,
        env: dict[str, str] | None = None,
    ) -> Job:
        with self._lock:
            if self.running is not None:
                raise RuntimeError(f"job {self.running.id} is still running")
            job = Job(
                id=uuid.uuid4().hex[:12],
                stage=stage,
                argv=[sys.executable, "-m", module, *argv],
                home=home,
                report_path=report_path,
                values=values or {},
                apply=apply,
            )
            child_env = {
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "ANIME_TOOLS_HOME": str(home),
                **(env or {}),
            }
            job._proc = subprocess.Popen(
                job.argv,
                cwd=str(home),
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=(os.name != "nt"),
            )
            self.jobs[job.id] = job
        threading.Thread(target=self._pump, args=(job,), daemon=True).start()
        return job

    def _pump(self, job: Job) -> None:
        proc = job._proc
        assert proc is not None and proc.stdout is not None
        log = None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log = (self.log_dir / f"{job.id}.log").open("w", encoding="utf-8")
        try:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                if log is not None:
                    log.write(line + "\n")
                with job._cond:
                    job.lines.append(line)
                    if len(job.lines) > self.max_lines:
                        del job.lines[: len(job.lines) - self.max_lines]
                    job._cond.notify_all()
        finally:
            code = proc.wait()
            if log is not None:
                log.close()
            with job._cond:
                job.exit_code = code
                job.finished = time.time()
                job._cond.notify_all()

    def cancel(self, job_id: str) -> bool:
        job = self.jobs[job_id]
        proc = job._proc
        if proc is None or job.exit_code is not None:
            return False
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    check=False,
                )
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        return True

    def shutdown(self) -> None:
        for job in list(self.jobs.values()):
            if job.exit_code is None:
                self.cancel(job.id)
