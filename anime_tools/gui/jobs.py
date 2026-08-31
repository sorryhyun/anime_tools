"""Run a stage as a subprocess (``python -m <module> …``) and tail its output.

One job at a time — the stages share the GPU. The server process never imports
torch; all model loading happens in the child.

A job is a *sequence* of steps sharing one slot, one log and one SSE stream,
because a stage that reads the resized tree has to be preceded by the resize
preflight (:data:`anime_tools.gui.stages.PREPROCESS_STAGE`). A failing step
stops the chain — there is no point tagging images that were never resized.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Step:
    """One ``python -m <module> <argv>`` invocation inside a job."""

    module: str
    argv: list[str]
    label: str = ""
    """Shown in the step header the log stream prints; defaults to ``module``."""

    def command(self) -> list[str]:
        return [sys.executable, "-m", self.module, *self.argv]


@dataclass
class Job:
    id: str
    stage: str
    steps: list[Step]
    home: Path
    started: float = field(default_factory=time.time)
    finished: float | None = None
    exit_code: int | None = None
    lines: list[str] = field(default_factory=list)
    report_path: str | None = None
    values: dict[str, Any] = field(default_factory=dict)
    apply: bool = False
    cancelled: bool = False
    _proc: subprocess.Popen | None = field(default=None, repr=False)
    _cond: threading.Condition = field(default_factory=threading.Condition, repr=False)

    @property
    def argv(self) -> list[str]:
        """The stage's own command — the last step; the earlier ones are
        preflight. This is what the UI labels the job with."""
        return self.steps[-1].command() if self.steps else []

    @property
    def state(self) -> str:
        if self.exit_code is None:
            return "running"
        # `cancelled` is set by JobManager.cancel before it kills: on Windows
        # taskkill leaves a plain non-zero exit code, indistinguishable from a
        # stage that failed on its own. A negative code still means a signal.
        if self.cancelled or self.exit_code < 0:
            return "cancelled"
        if self.exit_code == 0:
            return "done"
        return "failed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stage": self.stage,
            "argv": self.argv,
            "steps": [
                {"module": st.module, "label": st.label or st.module, "argv": st.argv}
                for st in self.steps
            ],
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
        steps: Sequence[Step],
        *,
        home: Path,
        report_path: str | None = None,
        values: dict[str, Any] | None = None,
        apply: bool = False,
        env: dict[str, str] | None = None,
    ) -> Job:
        """Queue a job's steps into the single slot and start pumping them.

        ``steps`` runs in order, the stage itself last; a non-zero step ends the
        job with that code and the rest never start.
        """
        if not steps:
            raise ValueError("a job needs at least one step")
        with self._lock:
            if self.running is not None:
                raise RuntimeError(f"job {self.running.id} is still running")
            job = Job(
                id=uuid.uuid4().hex[:12],
                stage=stage,
                steps=list(steps),
                home=home,
                report_path=report_path,
                values=values or {},
                apply=apply,
            )
            self.jobs[job.id] = job
        child_env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            # We decode the pipe as UTF-8 below; a child left on the
            # locale encoding (cp949/cp1252 on Windows) would arrive
            # mojibake, or die encoding the arrows the stages print.
            "PYTHONIOENCODING": "utf-8",
            "ANIME_TOOLS_HOME": str(home),
            **(env or {}),
        }
        threading.Thread(target=self._run, args=(job, child_env), daemon=True).start()
        return job

    def _run(self, job: Job, child_env: dict[str, str]) -> None:
        """Run every step in turn, appending all of their output to one log."""
        log = None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log = (self.log_dir / f"{job.id}.log").open("w", encoding="utf-8")
        code = 0
        try:
            for index, step in enumerate(job.steps, 1):
                if job.cancelled:
                    break
                if len(job.steps) > 1:
                    self._emit(
                        job,
                        log,
                        f"── step {index}/{len(job.steps)}: "
                        f"{step.label or step.module} ──",
                    )
                code = self._run_step(job, step, child_env, log)
                if code != 0:
                    break
        finally:
            if log is not None:
                log.close()
            with job._cond:
                job.exit_code = code
                job.finished = time.time()
                job._proc = None
                job._cond.notify_all()

    def _run_step(self, job: Job, step: Step, child_env, log) -> int:
        proc = subprocess.Popen(
            step.command(),
            cwd=str(job.home),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=(os.name != "nt"),
        )
        # Published before the first read so ``cancel`` can reach this step; a
        # cancel that landed while we were spawning is honoured right after.
        job._proc = proc
        if job.cancelled:
            self._terminate(proc)
        assert proc.stdout is not None
        for raw in proc.stdout:
            self._emit(job, log, raw.rstrip("\n"))
        return proc.wait()

    def _emit(self, job: Job, log, line: str) -> None:
        if log is not None:
            log.write(line + "\n")
        with job._cond:
            job.lines.append(line)
            if len(job.lines) > self.max_lines:
                del job.lines[: len(job.lines) - self.max_lines]
            job._cond.notify_all()

    def cancel(self, job_id: str) -> bool:
        """Stop a running job: kill the current step and skip the rest.

        The flag is set before the kill and read by :meth:`_run` between steps,
        so a cancel that lands in the gap between two steps still stops the
        chain instead of letting the next one start.
        """
        job = self.jobs[job_id]
        if job.exit_code is not None:
            return False
        job.cancelled = True
        proc = job._proc
        if proc is not None:
            self._terminate(proc)
        return True

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
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

    def shutdown(self) -> None:
        for job in list(self.jobs.values()):
            if job.exit_code is None:
                self.cancel(job.id)
