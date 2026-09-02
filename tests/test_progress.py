"""``anime_tools._progress``: the stage's side of the daemon's ``progress.jsonl``."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from anime_tools import _progress
from anime_tools.stages.cli._args import make_progress


def lines(job_dir: Path) -> list[dict]:
    path = job_dir / _progress.PROGRESS_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


@pytest.fixture
def job_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv(_progress.JOB_DIR_ENV, str(tmp_path))
    return tmp_path


def test_outside_a_daemon_job_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.delenv(_progress.JOB_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert _progress.progress_path() is None
    assert _progress.emit("step") is False
    _progress.step(1, 2, "x")
    with _progress.phase("load"):
        pass
    assert list(tmp_path.iterdir()) == []


def test_a_step_is_spelled_the_way_the_daemon_reads_it(job_dir):
    """``ev`` and ``global_step`` are the keys ``anima_daemon/tail.py`` filters
    and thins on, so ``get_progress`` works on a curation job unchanged."""
    _progress.step(3, 10, "a/1.png")
    (rec,) = lines(job_dir)
    assert rec["ev"] == "step"
    assert rec["global_step"] == 3 and rec["total_steps"] == 10
    assert rec["detail"] == "a/1.png"
    assert isinstance(rec["ts"], float)


def test_a_phase_keeps_a_heartbeat_going_while_it_lasts(job_dir, monkeypatch):
    monkeypatch.setattr(_progress, "HEARTBEAT_S", 0.02)
    with _progress.phase("load sam3"):
        time.sleep(0.15)
    events = [rec["ev"] for rec in lines(job_dir)]
    assert events[0] == "phase" and events[-1] == "phase"
    assert events.count("heartbeat") >= 2
    start, end = lines(job_dir)[0], lines(job_dir)[-1]
    assert (start["name"], start["state"]) == ("load sam3", "start")
    assert (end["name"], end["state"]) == ("load sam3", "end")
    assert end["seconds"] >= 0.1


def test_a_phase_that_raises_still_closes(job_dir, monkeypatch):
    monkeypatch.setattr(_progress, "HEARTBEAT_S", 0.02)
    with pytest.raises(RuntimeError), _progress.phase("load"):
        time.sleep(0.05)
        raise RuntimeError("boom")
    assert lines(job_dir)[-1]["state"] == "end"
    # The heartbeat thread is gone: nothing more arrives.
    n = len(lines(job_dir))
    time.sleep(0.06)
    assert len(lines(job_dir)) == n


def test_the_stage_callback_prints_thinned_and_streams_every_call(job_dir, capsys):
    progress = make_progress(2)
    for i in range(1, 6):
        progress(i, 5, f"img{i}")
    out = capsys.readouterr().out.splitlines()
    assert out == ["  [2/5] img2", "  [4/5] img4", "  [5/5] img5"]
    assert [rec["global_step"] for rec in lines(job_dir)] == [1, 2, 3, 4, 5]


def test_a_write_that_fails_is_swallowed(job_dir, monkeypatch):
    """A progress line must never fail the stage it reports on."""
    monkeypatch.setenv(_progress.JOB_DIR_ENV, str(job_dir / "gone"))
    assert _progress.emit("step") is False
    monkeypatch.setenv(_progress.JOB_DIR_ENV, str(job_dir))
    assert _progress.emit("step", detail=object()) is False
    assert lines(job_dir) == []


def test_the_module_is_stdlib_only():
    code = (
        "import sys, anime_tools._progress; "
        "heavy = {'torch', 'numpy', 'cv2', 'sam3', 'onnxruntime', 'timm'} & set(sys.modules); "
        "assert not heavy, heavy"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr
