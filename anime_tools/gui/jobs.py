"""Run a stage as a subprocess (``python -m <module> …``) and tail its output.

One job at a time — the stages share the GPU — and all model loading happens in
the child, keeping this process torch-free. A job is a *sequence* of steps sharing
one slot, log and SSE stream, so a stage reading the resized tree can be preceded
by the resize preflight (:data:`anime_tools.gui.stages.PREPROCESS_STAGE`). A
failing step stops the chain.
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

from anime_tools._env import resolve_path
from anime_tools._json import read_json

STAGE_JOB = "stage"
DOWNLOAD_JOB = "download"
UPDATE_JOB = "update"
JOB_KINDS = (STAGE_JOB, DOWNLOAD_JOB, UPDATE_JOB)
"""What a job *is*, carried on it as :attr:`Job.kind` — a stage run (the dock
follows it), a weights download (the Models dialog), or the self-update (the
Update dialog). The browser adopts a job it did not start by this, never by
parsing its name."""


class JobBusy(RuntimeError):
    """The single job slot is taken. The app answers it as a 409 wherever a
    route starts a job, so no route catches it itself."""


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
    dropped: int = 0
    """How many lines have been trimmed off the front of :attr:`lines`.

    The buffer is capped at ``JobManager.max_lines``, so on a long run the list
    is a sliding window and a reader's position has to be an *absolute* line
    number — ``dropped`` is what converts one into an index into the window.
    Monotonic for the life of the job."""
    report_path: str | None = None
    values: dict[str, Any] = field(default_factory=dict)
    apply: bool = False
    space: str = "dataset"
    """Which trees the run was over (``gui.dataset.SPACES``) — what its report is
    read back against, since a drop-area run names rels the dataset may not have."""
    cancelled: bool = False
    kind: str = STAGE_JOB
    """One of :data:`JOB_KINDS`."""
    ids: list[str] = field(default_factory=list)
    """What the job was asked for, by its own ids: a download's model rows
    (``[]`` = every missing one), an update's tag (``[]`` = the latest), nothing
    for a stage."""
    _written: list[str] | None = field(default=None, repr=False)
    _written_read: bool = field(default=False, repr=False)
    _proc: subprocess.Popen | None = field(default=None, repr=False)
    _cond: threading.Condition = field(default_factory=threading.Condition, repr=False)

    @property
    def argv(self) -> list[str]:
        """The stage's own command — the step labelled with :attr:`stage`, else the
        last; the others are a preflight before it or a follow-up after it."""
        own = [st for st in self.steps if st.label == self.stage]
        return (own or self.steps[-1:])[0].command() if self.steps else []

    @property
    def state(self) -> str:
        if self.exit_code is None:
            return "running"
        # Windows taskkill leaves a plain non-zero code, indistinguishable from a
        # stage that failed on its own, so `cancelled` is set before the kill. A
        # negative code still means a signal.
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
            "lines": self.total_lines,
            "report_path": self.report_path,
            "apply": self.apply,
            "space": self.space,
            "values": self.values,
            "kind": self.kind,
            "ids": self.ids,
            "written": self.written,
        }

    @property
    def written(self) -> list[str] | None:
        """The images a finished run's report says it wrote (its top-level
        ``written`` list), or ``None`` — still running, no report, or a report
        that names no such list. What the page re-stats after a run instead of
        re-walking the tree, carried here so it never fetches the report itself.

        Read once, after the job ends: the report is not rewritten after that.
        """
        if self.exit_code is None or not self.report_path:
            return None
        if not self._written_read:
            self._written_read = True
            try:
                got = read_json(resolve_path(self.report_path))
            except (OSError, ValueError):
                got = None
            w = got.get("written") if isinstance(got, dict) else None
            if isinstance(w, list) and all(isinstance(x, str) for x in w):
                self._written = w
        return self._written

    @property
    def total_lines(self) -> int:
        """Lines this job has emitted, trimmed ones counted — the absolute
        numbering :meth:`wait_lines` reads and returns."""
        return self.dropped + len(self.lines)

    def wait_lines(self, index: int, timeout: float = 1.0) -> tuple[list[str], int]:
        """Lines from absolute ``index`` on, plus the next absolute index.

        Blocks up to ``timeout`` for new ones. ``index`` counts from the start
        of the job, not into :attr:`lines`: once the buffer has been trimmed the
        two differ by :attr:`dropped`, and an index that has fallen off the
        front resumes at the oldest line still held rather than skipping past
        it. The returned index is what to pass back — never ``index + len(…)``,
        which a trim between two calls would leave short.
        """
        with self._cond:
            if self.total_lines <= index and self.exit_code is None:
                self._cond.wait(timeout)
            start = min(max(index - self.dropped, 0), len(self.lines))
            return self.lines[start:], self.total_lines


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
        space: str = "dataset",
        env: dict[str, str] | None = None,
        kind: str = STAGE_JOB,
        ids: Sequence[str] = (),
    ) -> Job:
        """Queue a job's steps into the single slot and start pumping them.

        ``steps`` runs in order, the stage itself last; a non-zero step ends the
        job with that code and the rest never start. :class:`JobBusy` while
        another job holds the slot.
        """
        if not steps:
            raise ValueError("a job needs at least one step")
        if kind not in JOB_KINDS:
            raise ValueError(f"unknown job kind: {kind!r}")
        with self._lock:
            if self.running is not None:
                raise JobBusy(f"job {self.running.id} is still running")
            job = Job(
                id=uuid.uuid4().hex[:12],
                stage=stage,
                steps=list(steps),
                home=home,
                report_path=report_path,
                values=values or {},
                apply=apply,
                space=space,
                kind=kind,
                ids=list(ids),
            )
            self.jobs[job.id] = job
        child_env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            # The pipe is decoded as UTF-8 below; a child on the locale
            # encoding (cp949/cp1252 on Windows) would arrive mojibake.
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
        # cancel that landed during the spawn is honoured right after.
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
                trim = len(job.lines) - self.max_lines
                del job.lines[:trim]
                job.dropped += trim
            job._cond.notify_all()

    def cancel(self, job_id: str) -> bool:
        """Stop a running job: kill the current step and skip the rest.

        The flag is set before the kill and read by :meth:`_run` between steps,
        so a cancel landing between two steps still stops the chain.
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
