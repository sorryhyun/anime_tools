"""Web GUI: stage registry → form schema → argv, and the FastAPI surface.

Runs on the base install + ``gui`` extra; stages whose extra is missing are
reported *unavailable*, never as an import error.
"""

from __future__ import annotations

import sys
import time

import pytest

from anime_tools.gui import stages as S

pytest.importorskip("fastapi")


def _await_job(c, response, tries: int = 200) -> dict:
    """Block until a started job leaves ``running`` (or give up and say so)."""
    assert response.status_code == 200, response.text
    job_id = response.json()["id"]
    for _ in range(tries):
        job = c.get(f"/api/jobs/{job_id}").json()
        if job["state"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} still running")


def _stage(sid: str) -> tuple[S.Stage, list[dict]]:
    st = S.BY_ID[sid]
    sc = S.schema(st)
    if not sc["available"]:
        pytest.skip(f"{sid} unavailable: {sc['error']}")
    return st, sc["fields"]


def test_every_stage_has_a_schema():
    for st in S.STAGES:
        sc = S.schema(st)
        assert sc["id"] == st.id and "available" in sc
        if sc["available"]:
            assert sc["fields"], st.id
            assert all(
                f["kind"] in ("bool", "int", "float", "str", "enum", "list")
                for f in sc["fields"]
            )


def test_defaults_produce_empty_argv():
    _, fs = _stage("position")
    assert S.build_argv(fs, {}) == []
    assert S.build_argv(fs, {}, apply=True) == ["--apply"]


def test_argv_round_trips_through_the_real_parser():
    st, fs = _stage("position")
    values = {
        "src": "img",
        "path_pattern": "a/*|b/*",
        "score_threshold": "0.4",
        "blank_crops": False,
        "max_instances": 3,
        "flatten": True,
    }
    argv = S.build_argv(fs, values, apply=True)
    ns = S.load_parser(st).parse_args(argv)
    assert ns.src == "img" and ns.path_pattern == "a/*|b/*"
    assert ns.score_threshold == 0.4 and ns.blank_crops is False
    assert ns.max_instances == 3 and ns.flatten and ns.apply


def test_enum_and_float_kinds():
    _, fs = _stage("autotag")
    fields = {f["dest"]: f for f in fs}
    assert fields["mode"]["kind"] == "enum" and "missing" in fields["mode"]["choices"]
    assert fields["min_confidence"]["kind"] == "float"
    assert fields["src"]["path"] and fields["report_dir"]["path"]
    assert S.build_argv(fs, {"mode": "merge", "min_confidence": 0}) == [
        "--mode",
        "merge",
    ]


def test_dataset_roots_fill_the_bound_fields():
    """--src/--dst come from the Settings roots, not from the stage form."""
    _, fs = _stage("position")
    assert {f["dest"]: f["root"] for f in fs if f["root"]} == {
        "src": "src",
        "dst": "dst",
    }
    argv = S.build_argv(
        fs, {"src": "ignored", "dst": "ignored"}, roots={"src": "a", "dst": "b"}
    )
    assert argv == ["--src", "a", "--dst", "b"]
    # A root left at its default still drops out of the argv.
    assert S.build_argv(fs, {}, roots={"src": "image_dataset"}) == []


def test_required_field_is_enforced():
    _, fs = _stage("correct")
    with pytest.raises(ValueError, match="--src"):
        S.build_argv(fs, {"dst": "x"})


def test_boolean_optional_action_and_positional_list():
    _, fs = _stage("masks_mit")
    argv = S.build_argv(fs, {"image_dir": "i", "mask_dir": "m", "ctd_gate": False})
    assert argv == ["--image-dir", "i", "--mask-dir", "m", "--no-ctd-gate"]
    _, fs = _stage("masks_merge")
    argv = S.build_argv(fs, {"mask_dirs": "a\nb\n", "output_dir": "o"})
    assert argv == ["--output-dir", "o", "a", "b"]


def test_report_path_follows_form_value():
    st, fs = _stage("autotag")
    assert (
        S.report_path(st, fs, {}) == "post_image_dataset/captions/autotag/report.json"
    )
    assert S.report_path(st, fs, {"report_dir": "r"}) == "r/report.json"
    st, fs = _stage("groups")
    assert S.report_path(st, fs, {"out": "g.json"}) == "g.json"


# -- server ---------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from anime_tools.gui.jobs import JobManager
    from anime_tools.gui.server import create_app

    monkeypatch.setenv("ANIME_TOOLS_HOME", str(tmp_path))
    (tmp_path / "image_dataset").mkdir()
    (tmp_path / "image_dataset" / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    schemas = S.load_schemas()
    stub = tmp_path / "stub_stage.py"
    stub.write_text(
        "import argparse, json, os, sys, pathlib, time\n"
        "def build_parser():\n"
        "    p = argparse.ArgumentParser(); p.add_argument('--n', type=int, default=1)\n"
        "    p.add_argument('--apply', action='store_true')\n"
        "    p.add_argument('--sleep', type=float, default=0)\n"
        "    p.add_argument('--report_dir', default='out'); return p\n"
        "if __name__ == '__main__':\n"
        "    a = build_parser().parse_args()\n"
        "    for i in range(a.n): print('line', i, flush=True)\n"
        "    time.sleep(a.sleep)\n"
        "    d = pathlib.Path(os.environ['ANIME_TOOLS_HOME'], a.report_dir); d.mkdir(exist_ok=True)\n"
        "    (d/'report.json').write_text(json.dumps({'apply': a.apply, 'rows': [{'k': 1}]}))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    fake = S.Stage(
        "stub", "Stub", "stub_stage", "test", "", report=("report_dir", "report.json")
    )
    monkeypatch.setitem(S.BY_ID, "stub", fake)
    schemas["stub"] = S.schema(fake)
    app = create_app(jobs=JobManager(log_dir=tmp_path / "logs"), schemas=schemas)
    with TestClient(app) as c:
        yield c, tmp_path


def test_index_and_info(client):
    c, home = client
    assert "<title>anime_tools</title>" in c.get("/").text
    assert c.get("/api/info").json()["home"] == str(home)
    ids = [s["id"] for s in c.get("/api/stages").json()]
    assert ids == [s.id for s in S.STAGES]


def test_files_and_ls_are_confined_to_home(client):
    c, _ = client
    assert (
        c.get("/api/files", params={"path": "image_dataset/a.png"}).status_code == 200
    )
    assert c.get("/api/files", params={"path": "/etc/hostname"}).status_code == 404
    entries = c.get("/api/ls").json()["entries"]
    assert {"name": "image_dataset", "dir": True} in entries
    assert c.get("/api/ls", params={"path": "image_dataset"}).json()["entries"] == [
        {"name": "a.png", "dir": False}
    ]
    assert c.get("/api/ls", params={"path": "/"}).status_code == 404


def test_job_runs_streams_and_persists_values(client):
    c, _home = client
    r = c.post("/api/jobs", json={"stage": "stub", "values": {"n": 3}, "apply": True})
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["argv"][:3] == [sys.executable, "-m", "stub_stage"]
    assert job["argv"][3:] == ["--n", "3", "--apply"]

    with c.stream("GET", f"/api/jobs/{job['id']}/log") as s:
        body = "".join(s.iter_text())
    assert body.count('data: "line ') == 3 and "event: done" in body

    for _ in range(50):
        if c.get(f"/api/jobs/{job['id']}").json()["state"] != "running":
            break
        time.sleep(0.05)
    assert c.get(f"/api/jobs/{job['id']}").json()["state"] == "done"
    rep = c.get(f"/api/jobs/{job['id']}/report").json()
    assert rep["report"] == {"apply": True, "rows": [{"k": 1}]}
    assert c.get("/api/settings").json()["values"]["stub"] == {"n": 3}
    assert c.post("/api/jobs", json={"stage": "nope"}).status_code == 404


def test_second_concurrent_job_is_refused(client):
    c, _home = client
    j = c.post("/api/jobs", json={"stage": "stub", "values": {"sleep": 30}}).json()
    assert c.post("/api/jobs", json={"stage": "stub"}).status_code == 409
    assert c.post(f"/api/jobs/{j['id']}/cancel").json()["cancelled"] is True
    for _ in range(100):
        if c.get(f"/api/jobs/{j['id']}").json()["state"] == "cancelled":
            break
        time.sleep(0.05)
    assert c.get(f"/api/jobs/{j['id']}").json()["state"] == "cancelled"


def test_model_catalog_and_download_job(client, monkeypatch):
    """The Settings rows come from the download catalog, and Download starts a
    normal job -- so it shares the one slot with the stages."""
    c, home = client
    body = c.get("/api/models").json()
    assert body["models_dir"] == str(home / "models")
    rows = {m["id"]: m for m in body["models"]}
    assert not rows["sam3"]["installed"]
    assert rows["sam3"]["location"] == str(home / "models" / "sam3")
    assert rows["tagger_backbone"]["gated"].startswith("https://huggingface.co/")

    assert c.post("/api/models/download", json={"ids": ["nope"]}).status_code == 404

    # Downloads and stages contend for the same slot on purpose.
    j = c.post("/api/jobs", json={"stage": "stub", "values": {"sleep": 30}}).json()
    assert c.post("/api/models/download", json={"ids": ["sam3"]}).status_code == 409
    c.post(f"/api/jobs/{j['id']}/cancel")

    # The child is the downloads CLI; HF_HUB_OFFLINE keeps the test off the wire
    # (it fails fast, which is all we need to see it ran).
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    job = _await_job(c, c.post("/api/models/download", json={"ids": ["mit_text"]}))
    assert job["argv"][1:] == ["-m", "anime_tools.downloads", "mit_text"]
    assert job["state"] in ("done", "failed")


def test_pick_port_skips_busy_port():
    import socket

    from anime_tools.gui.server import pick_port

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        taken = busy.getsockname()[1]
        got = pick_port("127.0.0.1", taken)
    assert got != taken and got > taken
    assert pick_port("127.0.0.1", 0) > 0
