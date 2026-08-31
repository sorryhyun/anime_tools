"""Web GUI: stage registry → form schema → argv, and the FastAPI surface.

Runs on the base install + ``gui`` extra; stages whose extra is missing are
reported *unavailable*, never as an import error.
"""

from __future__ import annotations

import json
import sys
import threading
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
        "score_threshold": "0.4",
        "blank_crops": False,
        "max_instances": 3,
        "flatten": True,
    }
    argv = S.build_argv(
        fs,
        values,
        apply=True,
        roots={"src": "img"},
        settings={"path_pattern": "a/*|b/*"},
    )
    ns = S.load_parser(st).parse_args(argv)
    assert ns.src == "img" and ns.path_pattern == "a/*|b/*"
    assert ns.score_threshold == 0.4 and ns.blank_crops is False
    assert ns.max_instances == 3 and ns.flatten and ns.apply
    # --device never reaches the argv: the stage auto-detects it.
    assert "--device" not in argv and ns.device is None


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


def test_settings_fill_the_bound_stage_defaults():
    """--path_pattern / --tagger_dir are set once in Settings, not per form."""
    _, fs = _stage("autotag")
    assert {f["dest"]: f["setting"] for f in fs if f["setting"]} == {
        "path_pattern": "path_pattern",
        "tagger_dir": "tagger_dir",
    }
    argv = S.build_argv(
        fs, {}, settings={"path_pattern": "char/*", "tagger_dir": "ckpt"}
    )
    assert argv == ["--path_pattern", "char/*", "--tagger_dir", "ckpt"]
    # A value stranded in a saved form never beats (or fills in for) Settings.
    assert S.build_argv(fs, {"path_pattern": "stale/*", "tagger_dir": "stale"}) == []
    # ...and it is not written back into the settings file either.
    assert S.form_values(fs, {"path_pattern": "stale/*", "mode": "merge"}) == {
        "mode": "merge"
    }


def test_device_is_never_on_the_form_or_the_argv():
    """Auto-detected in the child (``_device.resolve_device``), because this
    process is torch-free and cannot see the child's hardware."""
    required = {"config": "c.yaml", "mask_dir": "m", "out": "g.json"}
    for stage_id in ("autotag", "position", "masks_sam", "masks_mit", "groups"):
        _, fs = _stage(stage_id)
        device = next(f for f in fs if f["dest"] == "device")
        assert device["auto"] is True
        argv = S.build_argv(fs, {**required, "device": "cuda"}, roots={"src": "i"})
        assert "--device" not in argv and "cuda" not in argv


def test_scoped_stages_are_the_ones_taking_a_pattern():
    """The run bar's per-image button exists exactly where the stage has a
    ``--path_pattern`` to narrow."""
    scoped = {s.id for s in S.STAGES if S.schema(s).get("scoped")}
    assert scoped == {
        "resize",
        "autotag",
        "position",
        "correct",
        "audit",
        "masks_sam",
        "masks_mit",
    }


def test_required_field_is_enforced():
    _, fs = _stage("correct")
    with pytest.raises(ValueError, match="--src"):
        S.build_argv(fs, {"dst": "x"})


def test_boolean_optional_action_and_positional_list():
    _, fs = _stage("masks_mit")
    argv = S.build_argv(fs, {"mask_dir": "m", "ctd_gate": False}, roots={"src": "i"})
    assert argv == ["--image-dir", "i", "--mask-dir", "m", "--no-ctd-gate"]
    _, fs = _stage("masks_merge")
    argv = S.build_argv(fs, {"mask_dirs": "a\nb\n"}, roots={"masks": "o"})
    assert argv == ["--output-dir", "o", "a", "b"]


def test_report_path_follows_form_value():
    st, fs = _stage("autotag")
    assert (
        S.report_path(st, fs, {}) == "post_image_dataset/captions/autotag/report.json"
    )
    assert S.report_path(st, fs, {"report_dir": "r"}) == "r/report.json"
    st, fs = _stage("groups")
    assert S.report_path(st, fs, {"out": "g.json"}) == "g.json"


# -- schema cache ---------------------------------------------------------


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """An empty, private schema cache dir + a counter for the child dumps."""
    monkeypatch.setenv(S.CACHE_ENV, str(tmp_path / "cache"))
    calls: list[int] = []
    real = S.dump_schemas_in_child

    def counted():
        calls.append(1)
        return {"fake": {"id": "fake", "available": True, "fields": []}}

    monkeypatch.setattr(S, "dump_schemas_in_child", counted)
    return calls, tmp_path, real


def test_cache_dir_is_outside_the_curation_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIME_TOOLS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(S.CACHE_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert S.cache_dir() == tmp_path / "xdg" / "anime_tools" / "gui"
    assert S.schema_cache_path().name == "schemas.json"
    monkeypatch.setenv(S.CACHE_ENV, str(tmp_path / "c"))
    assert S.cache_dir() == tmp_path / "c"


def test_second_load_hits_the_cache_and_skips_the_child(cache):
    calls, _, _ = cache
    first = S.load_schemas()
    assert calls == [1]
    assert S.schema_cache_path().is_file()
    assert S.load_schemas() == first
    assert calls == [1]  # no second interpreter
    # ...and opting out still shells out.
    S.load_schemas(cache=False)
    assert calls == [1, 1]


def test_touching_a_stage_module_invalidates_the_cache(cache, monkeypatch):
    """The key is keyed on the module files, so an edited parser is never stale."""
    calls, tmp_path, _ = cache
    mod = tmp_path / "cachestub_stage.py"
    mod.write_text("def build_parser():\n    pass\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    extra = S.Stage("cachestub", "Stub", "cachestub_stage", "test", "")
    monkeypatch.setattr(S, "STAGES", (*S.STAGES, extra))

    before = S.schema_cache_key()
    S.load_schemas()
    S.load_schemas()
    assert calls == [1]  # second call came off disk

    mod.write_text("def build_parser():\n    return None  # edited\n", encoding="utf-8")
    assert S.schema_cache_key() != before
    S.load_schemas()
    assert calls == [1, 1]  # rebuilt


def test_a_corrupt_cache_is_a_rebuild_not_a_crash(cache):
    calls, _, _ = cache
    S.load_schemas()
    p = S.schema_cache_path()
    for junk in ("{not json", "[]", '{"key": "x"}', ""):
        p.write_text(junk, encoding="utf-8")
        assert S.load_schemas() == {
            "fake": {"id": "fake", "available": True, "fields": []}
        }
    assert len(calls) == 5


def test_an_unwritable_cache_dir_still_serves_schemas(cache, monkeypatch):
    calls, tmp_path, _ = cache
    monkeypatch.setenv(S.CACHE_ENV, str(tmp_path / "blocked" / "sub"))
    (tmp_path / "blocked").write_text("not a directory", encoding="utf-8")
    assert S.load_schemas()["fake"]["id"] == "fake"
    assert calls == [1]


def test_the_real_dump_round_trips_through_the_cache(tmp_path, monkeypatch):
    """End to end, with the actual child interpreter, once."""
    monkeypatch.setenv(S.CACHE_ENV, str(tmp_path / "cache"))
    fresh = S.load_schemas()
    assert set(fresh) == {s.id for s in S.STAGES}
    assert S.load_schemas() == fresh


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
        # The two Settings-bound dests and the auto-detected one, so the stub
        # exercises the same binding the real stages get.
        "    p.add_argument('--path_pattern', default='*')\n"
        "    p.add_argument('--device', default=None)\n"
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


def test_files_and_ls_reject_dotdot_traversal(client):
    """`..` must not escape the home: `is_relative_to` is purely textual, so
    `<home>/image_dataset/../../x` would sail through without the normpath
    collapse `under_home` does."""
    c, home = client
    outside = home.parent / "gui_traversal_target.txt"
    outside.write_text("secret")
    try:
        traversal = f"image_dataset/../../{outside.name}"
        assert outside.is_file()  # the target really exists — 404 is the guard
        assert c.get("/api/files", params={"path": traversal}).status_code == 404
        assert (
            c.get("/api/ls", params={"path": "image_dataset/../.."}).status_code == 404
        )
        # `..` that stays inside the home keeps working, as do plain paths.
        assert (
            c.get(
                "/api/files", params={"path": "image_dataset/../image_dataset/a.png"}
            ).status_code
            == 200
        )
        assert c.get("/api/ls", params={"path": "image_dataset"}).status_code == 200
    finally:
        outside.unlink()


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


def test_job_start_creates_the_report_directory(client):
    """The stub's own mkdir has no ``parents=True``, so a nested report dir only
    works because ``POST /api/jobs`` made it first."""
    c, home = client
    job = _await_job(
        c,
        c.post(
            "/api/jobs",
            json={"stage": "stub", "values": {"report_dir": "deep/nested/reports"}},
        ),
    )
    assert job["state"] == "done", job
    assert (home / "deep/nested/reports/report.json").is_file()


def test_settings_pattern_and_rel_pick_the_run_scope(client):
    """The batch button sends no ``rel`` and gets the Settings pattern; the
    per-image button sends one and gets a pattern naming just that file."""
    c, _home = client
    c.put("/api/settings", json={"stage_defaults": {"path_pattern": "sub/*"}})

    batch = c.post(
        "/api/jobs", json={"stage": "stub", "values": {"n": 1, "path_pattern": "old/*"}}
    ).json()
    assert batch["argv"][-2:] == ["--path_pattern", "sub/*"]
    _await_job(c, c.get(f"/api/jobs/{batch['id']}"))
    # A bound dest is not the form's to remember, so it is not written back.
    assert c.get("/api/settings").json()["values"]["stub"] == {"n": 1}

    one = c.post("/api/jobs", json={"stage": "stub", "rel": "sub/b.jpg"}).json()
    # Stem, not filename: the resize step may have re-encoded it.
    assert one["argv"][-2:] == ["--path_pattern", "sub/b.*"]
    _await_job(c, c.get(f"/api/jobs/{one['id']}"))

    # --device is auto-detected in the child and never sent.
    assert "--device" not in batch["argv"] + one["argv"]


# -- the resize preflight: an implicit first step, not a panel -------------


def test_resize_is_not_a_dock_panel():
    """It has a schema and an argv, but nothing to click: it runs itself."""
    assert S.BY_ID["resize"].hidden is True
    assert "Resize" not in S.PANELS
    assert [s.id for s in S.STAGES if s.hidden] == ["resize"]


def test_only_resized_tree_stages_get_the_preflight():
    """A stage bound to ``dst`` reads the resized tree, so it needs resize in
    front of it; the ``src``-only ones (masks, groups) read the originals."""
    got = {s.id: S.preprocess_for(s.id) for s in S.STAGES}
    assert {k for k, v in got.items() if v == "resize"} == {
        "autotag",
        "position",
        "correct",
        "audit",
    }
    assert got["resize"] is None  # never its own preflight
    assert got["masks_sam"] is None and got["groups"] is None


def test_a_dst_bound_stage_runs_resize_first(client, monkeypatch):
    c, _home = client
    monkeypatch.setitem(S.ROOT_FIELDS, "stub", {"src": "src", "dst": "dst"})

    job = _await_job(c, c.post("/api/jobs", json={"stage": "stub", "values": {"n": 1}}))

    assert [st["label"] for st in job["steps"]] == ["resize", "stub"]
    assert job["steps"][0]["module"] == "anime_tools.stages.cli.resize_images"
    # `argv` stays the *stage's* command, so the UI still labels the job by it.
    assert job["argv"][:3] == [sys.executable, "-m", "stub_stage"]
    assert job["state"] == "done", job
    # Both steps' output lands in the one stream, under a step header.
    body = "".join(c.get(f"/api/jobs/{job['id']}/log").iter_text())
    assert "step 1/2" in body and "step 2/2" in body


def test_a_stage_without_a_preflight_is_a_single_step(client):
    c, _home = client
    job = _await_job(c, c.post("/api/jobs", json={"stage": "stub", "values": {"n": 1}}))
    assert [st["label"] for st in job["steps"]] == ["stub"]
    # A lone step prints no header -- the chain is invisible when there is none.
    body = "".join(c.get(f"/api/jobs/{job['id']}/log").iter_text())
    assert "step 1/1" not in body


def test_the_preflight_is_scoped_exactly_like_the_job(client, monkeypatch):
    """Per-image Apply must resize that one image, not the whole dataset."""
    c, _home = client
    monkeypatch.setitem(S.ROOT_FIELDS, "stub", {"src": "src", "dst": "dst"})
    c.put("/api/settings", json={"stage_defaults": {"path_pattern": "sub/*"}})

    one = c.post("/api/jobs", json={"stage": "stub", "rel": "a.png"}).json()
    assert one["steps"][0]["argv"][-2:] == ["--path_pattern", "a.*"]
    assert one["steps"][1]["argv"][-2:] == ["--path_pattern", "a.*"]
    _await_job(c, c.get(f"/api/jobs/{one['id']}"))

    batch = c.post("/api/jobs", json={"stage": "stub"}).json()
    assert batch["steps"][0]["argv"][-2:] == ["--path_pattern", "sub/*"]
    _await_job(c, c.get(f"/api/jobs/{batch['id']}"))


def test_settings_preprocess_block_configures_the_preflight(client, monkeypatch):
    """Resize has no form of its own, so its knobs come from Settings."""
    c, _home = client
    monkeypatch.setitem(S.ROOT_FIELDS, "stub", {"src": "src", "dst": "dst"})
    c.put(
        "/api/settings",
        json={"preprocess": {"min_pixels": 0, "target_res": [1024, 1536]}},
    )

    job = _await_job(c, c.post("/api/jobs", json={"stage": "stub"}))
    argv = job["steps"][0]["argv"]

    assert argv[argv.index("--min_pixels") + 1] == "0"
    assert argv[argv.index("--target_res") + 1 : argv.index("--target_res") + 3] == [
        "1024",
        "1536",
    ]
    # Untouched knobs stay off the argv so the CLI's own defaults hold.
    assert "--freefit_max_ratio" not in argv


def test_a_failing_step_stops_the_chain(tmp_path):
    """No point tagging images the resize step never produced."""
    from anime_tools.gui.jobs import JobManager, Step

    (tmp_path / "boom_step.py").write_text("import sys; print('first'); sys.exit(3)")
    (tmp_path / "ok_step.py").write_text("print('second')")
    mgr = JobManager(log_dir=tmp_path / "logs")

    job = mgr.start(
        "chain",
        [Step("boom_step", [], "pre"), Step("ok_step", [], "stage")],
        home=tmp_path,
        env={"PYTHONPATH": str(tmp_path)},
    )
    for _ in range(200):
        if job.exit_code is not None:
            break
        time.sleep(0.05)

    assert job.exit_code == 3 and job.state == "failed"
    assert "first" in job.lines and "second" not in job.lines


def test_steps_run_in_order_in_one_stream(tmp_path):
    from anime_tools.gui.jobs import JobManager, Step

    (tmp_path / "one_step.py").write_text("print('first')")
    (tmp_path / "two_step.py").write_text("print('second')")
    mgr = JobManager(log_dir=tmp_path / "logs")

    job = mgr.start(
        "chain",
        [Step("one_step", [], "pre"), Step("two_step", [], "stage")],
        home=tmp_path,
        env={"PYTHONPATH": str(tmp_path)},
    )
    for _ in range(200):
        if job.exit_code is not None:
            break
        time.sleep(0.05)

    assert job.exit_code == 0
    assert [ln for ln in job.lines if not ln.startswith("──")] == ["first", "second"]
    # One log file for the whole chain, not one per step.
    assert (tmp_path / "logs" / f"{job.id}.log").read_text().count("second") == 1


def test_scoping_an_unscopable_stage_is_refused(client):
    """No ``--path_pattern`` means nothing to narrow -- and "this image"
    quietly meaning "everything" is the one outcome worth a 400."""
    c, _home = client
    r = c.post("/api/jobs", json={"stage": "audit_apply", "rel": "a.png"})
    assert r.status_code == 400 and "scoped" in r.json()["detail"]
    r = c.post("/api/jobs", json={"stage": "stub", "rel": "../escape.png"})
    assert r.status_code == 400


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


ROW = "name,category,post_count,description\n"


def test_tag_descriptions_are_the_caption_panel_s_kb(client, monkeypatch):
    """Click a tag -> what the Danbooru KB says. The panel has to answer even
    with no KB on disk (that answer is the download prompt), and the English
    sibling must win for the *description* only -- the taxonomy is the base
    table's, or half the tags would lose their category with their blurb."""
    from anime_tools.captions import correction
    from anime_tools.gui import tags as T

    c, home = client
    # The source tree's own models/ is a fallback candidate; a test must not
    # find the developer's copy of a 16 MB CSV.
    monkeypatch.setattr(correction, "_REPO_ROOT", home / "elsewhere")
    monkeypatch.setattr(T, "_CACHE", None)

    body = c.get("/api/tags/describe", params={"tag": "1girl"}).json()
    assert body["installed"] is False and body["known"] is False
    assert body["download_id"] == "danbooru_tags"
    assert c.get("/api/tags/describe", params={"tag": " "}).status_code == 400

    models = home / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / correction.TAG_CSV_NAME).write_text(
        ROW
        + '1girl,0,7598073,"[people > count] one female character, in Korean"\n'
        + "fumihiko,1,4200,\n",
        encoding="utf-8",
    )
    body = c.get("/api/tags/describe", params={"tag": "1girl"}).json()
    assert body["installed"] and body["known"] and body["exact"] is True
    assert body["kind"] == "general" and body["post_count"] == 7598073
    assert body["category_path"] == "people > count"
    assert body["description"] == "one female character, in Korean"
    assert body["source"] == correction.TAG_CSV_NAME

    # The English sibling, built later: picked up without a restart (the cache
    # is keyed on both files), and it replaces the blurb, nothing else.
    (models / correction.TAG_CSV_EN_NAME).write_text(
        ROW + "1girl,0,7598073,One female character.\n", encoding="utf-8"
    )
    body = c.get("/api/tags/describe", params={"tag": "1girl"}).json()
    assert body["description"] == "One female character."
    assert body["category_path"] == "people > count" and body["post_count"] == 7598073
    assert body["source"] == correction.TAG_CSV_EN_NAME

    # An Anima ``@artist`` is looked up bare, and says so...
    body = c.get("/api/tags/describe", params={"tag": "@fumihiko"}).json()
    assert body["known"] and body["name"] == "fumihiko" and body["exact"] is False
    assert body["kind"] == "artist"
    # ...and an injected quality tag is simply not a Danbooru tag.
    assert c.get("/api/tags/describe", params={"tag": "masterpiece"}).json() == {
        "tag": "masterpiece",
        "installed": True,
        "known": False,
        "source": correction.TAG_CSV_EN_NAME,
        "download_id": "danbooru_tags",
    }


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


# -- startup: the schema dump is off the critical path ---------------------


def _app(tmp_path, monkeypatch, loader):
    from anime_tools.gui.jobs import JobManager
    from anime_tools.gui.server import create_app

    monkeypatch.setenv("ANIME_TOOLS_HOME", str(tmp_path))
    monkeypatch.setattr(S, "load_schemas", loader)
    return create_app(jobs=JobManager(log_dir=tmp_path / "logs"))


def test_startup_does_not_wait_for_the_schema_dump(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    gate = threading.Event()

    def slow():
        assert gate.wait(30)
        return {s.id: S.schema(s) for s in S.STAGES}

    t0 = time.perf_counter()
    app = _app(tmp_path, monkeypatch, slow)
    assert time.perf_counter() - t0 < 1.0  # bound the port, don't dump schemas

    with TestClient(app) as c:
        assert c.get("/api/info").json()["schemas_ready"] is False
        assert c.get("/").status_code == 200  # the page loads meanwhile
        gate.set()
        ids = [s["id"] for s in c.get("/api/stages").json()]
    assert ids == [s.id for s in S.STAGES]


def test_stages_time_out_rather_than_hang_forever(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from anime_tools.gui.server import Schemas

    monkeypatch.setattr(Schemas, "TIMEOUT", 0.05)
    stuck = threading.Event()
    app = _app(tmp_path, monkeypatch, lambda: (stuck.wait(30), {})[1])
    try:
        with TestClient(app) as c:
            assert c.get("/api/stages").status_code == 503
    finally:
        stuck.set()


def test_a_failed_schema_dump_is_reported_not_fatal(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    def boom():
        raise RuntimeError("stage schema dump failed: boom")

    with TestClient(_app(tmp_path, monkeypatch, boom)) as c:
        assert c.get("/api/info").status_code == 200  # the process is alive
        r = c.get("/api/stages")
        assert r.status_code == 500 and "boom" in r.json()["detail"]
        assert c.post("/api/jobs", json={"stage": "position"}).status_code == 500


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


# -- replay: Apply writes the dry run's proposals -------------------------


def test_replay_capable_stages_advertise_it():
    """The GUI's Apply offers a replay off ``schema()["replay"]``, so the two
    caption stages that grew ``--from_report`` have to carry the flag."""
    for stage_id in ("autotag", "position"):
        st, fs = _stage(stage_id)
        assert S.schema(st)["replay"] is True
        assert any(f["dest"] == S.REPLAY_FIELD for f in fs)
    assert S.schema(S.BY_ID["groups"])["replay"] is False


def test_replay_report_name_matches_the_stages():
    """``report_path`` hard-codes the replay's filename to stay torch-free;
    this is the assertion that keeps the copy honest."""
    from anime_tools.stages.replay import REPLAY_REPORT_NAME

    assert S.REPLAY_REPORT_NAME == REPLAY_REPORT_NAME


def test_a_replay_reports_beside_the_run_it_replays():
    """``--from_report`` and ``--report_dir`` normally name the same directory:
    the replay must not clobber the dry run it is reading."""
    st, fs = _stage("autotag")
    dry = S.report_path(st, fs, {"report_dir": "r"})
    assert dry == "r/report.json"
    replay = S.report_path(st, fs, {"report_dir": "r", S.REPLAY_FIELD: dry})
    assert replay == f"r/{S.REPLAY_REPORT_NAME}" != dry


def test_from_report_reaches_the_argv():
    _, fs = _stage("autotag")
    argv = S.build_argv(fs, {S.REPLAY_FIELD: "r/report.json"}, apply=True)
    assert "--from_report" in argv
    assert argv[argv.index("--from_report") + 1] == "r/report.json"
    assert "--apply" in argv


def test_dataset_items_refreshes_only_what_it_is_asked_for(client):
    """The sidebar patch path: a job's ``written`` list in, those rows out."""
    c, tmp_path = client
    src = tmp_path / "image_dataset"
    (src / "sub").mkdir()
    (src / "sub" / "b.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    r = c.post("/api/dataset/items", json={"rels": ["sub/b.png", "a.png"]})
    assert r.status_code == 200
    rows = r.json()["items"]
    assert [x["rel"] for x in rows] == ["sub/b.png", "a.png"]
    assert rows[0]["dir"] == "sub" and rows[0]["derived"] is False
    # A caption written since the listing shows up on the refreshed row.
    dst = tmp_path / "post_image_dataset" / "resized" / "sub"
    dst.mkdir(parents=True)
    (dst / "b.txt").write_text("1girl.")
    assert c.post("/api/dataset/items", json={"rels": ["sub/b.png"]}).json()["items"][
        0
    ]["derived"]


def test_dataset_items_drops_what_it_cannot_refresh(client):
    """Traversal and vanished rows are dropped, not raised: the caller is
    patching a listing, and a row it cannot refresh is one to leave alone."""
    c, _ = client
    r = c.post(
        "/api/dataset/items", json={"rels": ["../escape.png", "/abs.png", "gone.png"]}
    )
    assert r.status_code == 200 and r.json()["items"] == []


def test_apply_replays_a_dry_run_end_to_end(tmp_path, monkeypatch):
    """The whole Apply path, over the real HTTP API and the real autotag CLI:
    a dry run's report goes in, captions come out, and no model is loaded."""
    from fastapi.testclient import TestClient

    from anime_tools.gui.server import create_app

    monkeypatch.setenv("ANIME_TOOLS_HOME", str(tmp_path))
    c = TestClient(create_app())
    # Saving Settings makes the roots real (nothing existed a moment ago).
    assert c.put("/api/dataset/roots", json={}).json()["created"] == [
        "src",
        "dst",
        "masks",
    ]
    src, dst = tmp_path / "image_dataset", tmp_path / "post_image_dataset" / "resized"
    for n in ("a", "b", "c"):
        (src / f"{n}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (src / f"{n}.txt").write_text("1girl, solo.")
        (dst / f"{n}.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    # What the tagger pass left behind, which is exactly what Apply now skips.
    rdir = tmp_path / "post_image_dataset" / "captions" / "autotag"
    rdir.mkdir(parents=True)
    (rdir / "report.json").write_text(
        json.dumps(
            {
                "apply": False,
                "src": str(src),
                "dst": str(dst),
                "stats": {"seen": 3, "proposed": 3, "written": 0},
                "rows": [
                    {
                        "image": f"{n}.png",
                        "caption_path": str(src / f"{n}.txt"),
                        "existing": "1girl, solo.",
                        "proposed": f"1girl, solo, {n}_tag.",
                        "status": "ok",
                    }
                    for n in ("a", "b", "c")
                ],
            }
        )
    )
    # Hand-edited after the dry run: its proposal is stale, so it is skipped.
    (src / "c.txt").write_text("1girl, solo, hand edited.")

    job = c.post(
        "/api/jobs",
        json={
            "stage": "autotag",
            "apply": True,
            "values": {
                S.REPLAY_FIELD: "post_image_dataset/captions/autotag/report.json"
            },
        },
    ).json()
    assert "--from_report" in job["argv"] and "--apply" in job["argv"]
    deadline = time.monotonic() + 120
    while (
        c.get(f"/api/jobs/{job['id']}").json()["state"] == "running"
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    final = c.get(f"/api/jobs/{job['id']}").json()
    assert final["state"] == "done", final
    # Beside the report it replayed, never over it.
    assert final["report_path"].endswith(S.REPLAY_REPORT_NAME)
    assert (rdir / "report.json").exists()

    report = c.get(f"/api/jobs/{job['id']}/report").json()["report"]
    assert report["written"] == ["a.png", "b.png"]
    assert (src / "a.txt").read_text() == "1girl, solo, a_tag."
    assert (src / "c.txt").read_text() == "1girl, solo, hand edited."
    # …and that list is all the sidebar has to re-stat.
    rows = c.post("/api/dataset/items", json={"rels": report["written"]}).json()[
        "items"
    ]
    assert [r["rel"] for r in rows] == ["a.png", "b.png"]


# ---- one refusal, one status code -------------------------------------


def test_a_refused_root_is_a_bad_request_not_a_crash(client):
    """`DatasetError` and `ProposalError` are registered app-wide rather than
    caught at nine call sites. Anything outside the curation home is refused."""
    c, *_ = client
    r = c.get("/api/dataset", params={"src": "/etc"})
    assert r.status_code == 400
    assert "curation home" in r.json()["detail"]


def test_an_empty_caption_write_is_still_a_bad_request(client):
    c, *_ = client
    r = c.put(
        "/api/dataset/item", json={"rel": "a.png", "kind": "master", "text": "  "}
    )
    assert r.status_code == 400
    assert "empty caption" in r.json()["detail"]


def test_an_image_that_is_not_in_the_dataset_is_a_404(client):
    """The roots resolved — only the image is missing, so this one keeps its own
    status rather than taking the app-wide 400."""
    c, *_ = client
    r = c.get("/api/dataset/item", params={"rel": "nope.png"})
    assert r.status_code == 404
    assert "not in the dataset" in r.json()["detail"]
