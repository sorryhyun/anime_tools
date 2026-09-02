"""Web GUI: stage registry → form schema → argv, and the FastAPI surface.

Runs on the base install + ``gui`` extra; stages whose extra is missing are
reported *unavailable*, never as an import error.
"""

from __future__ import annotations

import json
import re
import subprocess
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


def _opt(argv: list[str], flag: str) -> str:
    """The value ``flag`` carries in ``argv``, read position-free."""
    return argv[argv.index(flag) + 1]


def _stage(sid: str) -> tuple[S.Stage, dict]:
    st = S.BY_ID[sid]
    sc = S.schema(st)
    if not sc["available"]:
        pytest.skip(f"{sid} unavailable: {sc['error']}")
    return st, sc


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
    _, sc = _stage("position")
    assert S.build_argv(sc, {}) == []
    assert S.build_argv(sc, {}, apply=True) == ["--apply"]


def test_argv_round_trips_through_the_real_parser():
    st, sc = _stage("position")
    values = {
        "score_threshold": "0.4",
        "blank_crops": False,
        "max_instances": 3,
        "flatten": True,
    }
    argv = S.build_argv(
        sc,
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
    _, sc = _stage("autotag")
    fields = {f["dest"]: f for f in sc["fields"]}
    assert fields["mode"]["kind"] == "enum" and "missing" in fields["mode"]["choices"]
    assert fields["min_confidence"]["kind"] == "float"
    assert fields["src"]["path"] and fields["report_dir"]["path"]
    assert S.build_argv(sc, {"mode": "merge", "min_confidence": 0}) == [
        "--mode",
        "merge",
    ]


def test_dataset_roots_fill_the_bound_fields():
    """--src/--dst come from the Settings roots, not from the stage form."""
    _, sc = _stage("position")
    assert {f["dest"]: f["root"] for f in sc["fields"] if f["root"]} == {
        "src": "src",
        "dst": "dst",
    }
    argv = S.build_argv(
        sc, {"src": "ignored", "dst": "ignored"}, roots={"src": "a", "dst": "b"}
    )
    assert argv == ["--src", "a", "--dst", "b"]
    # A root left at its default still drops out of the argv.
    assert S.build_argv(sc, {}, roots={"src": "image_dataset"}) == []


def test_settings_fill_the_bound_stage_defaults():
    """--path_pattern / --tagger_dir are set once in Settings, not per form."""
    _, sc = _stage("autotag")
    assert {f["dest"]: f["setting"] for f in sc["fields"] if f["setting"]} == {
        "path_pattern": "path_pattern",
        "tagger_dir": "tagger_dir",
    }
    argv = S.build_argv(
        sc, {}, settings={"path_pattern": "char/*", "tagger_dir": "ckpt"}
    )
    assert argv == ["--path_pattern", "char/*", "--tagger_dir", "ckpt"]
    # A value stranded in a saved form never beats Settings.
    assert S.build_argv(sc, {"path_pattern": "stale/*", "tagger_dir": "stale"}) == []
    # ...and it is not written back into the settings file either.
    assert S.form_values(
        sc["fields"], {"path_pattern": "stale/*", "mode": "merge"}
    ) == {"mode": "merge"}


def test_device_is_never_on_the_form_or_the_argv():
    """``--device`` is resolved in the child, since this process is torch-free."""
    required = {"mask_dir": "m", "out": "g.json"}
    for stage_id in ("autotag", "position", "masks_sam", "masks_mit", "groups"):
        _, sc = _stage(stage_id)
        device = next(f for f in sc["fields"] if f["dest"] == "device")
        assert device["auto"] is True
        argv = S.build_argv(
            sc, {**required, "device": "cuda"}, roots={"src": "i", "dst": "d"}
        )
        assert "--device" not in argv and "cuda" not in argv


def test_scoped_stages_are_the_ones_taking_a_pattern():
    """The per-image button exists exactly where a ``--path_pattern`` can narrow."""
    scoped = {s.id for s in S.STAGES if S.schema(s).get("scoped")}
    assert scoped == {
        "resize",
        "autotag",
        "position",
        "correct",
        "audit",
        "ocr",
        "masks_sam",
        "masks_mit",
        # Export narrows the same way.
        "export",
    }


def test_required_field_is_enforced():
    _, sc = _stage("correct")
    with pytest.raises(ValueError, match="--src"):
        S.build_argv(sc, {"dst": "x"})


def test_boolean_optional_action_and_positional_list():
    _, sc = _stage("masks_mit")
    # --image-dir is bound to `dst` (the mask is cut from the pixels the loader
    # rescales it onto); --mask-dir to the mask root plus this generator's tail.
    argv = S.build_argv(sc, {"ctd_gate": False}, roots={"dst": "d"}, mask_root="ws")
    assert argv == ["--image-dir", "d", "--mask-dir", "ws/masks_mit", "--no-ctd-gate"]
    # A positional list binds the same way, one joined tail per input.
    _, sc = _stage("masks_merge")
    argv = S.build_argv(sc, {}, roots={"masks": "o"}, mask_root="ws")
    assert argv == ["--output-dir", "o", "ws/masks_sam", "ws/masks_mit"]


def test_a_shut_drawer_sends_none_of_its_knobs():
    """Two detectors behind two checkboxes: a knob under a shut switch never
    reaches the argv."""
    _, sc = _stage("masks_mit")
    roots, settings = {"dst": "d"}, {"checkpoint": "sam3.pt"}
    bound = ["--image-dir", "d", "--mask-dir", "ws/masks_mit"]

    # SAM3 is opt-in, so its prompt and its checkpoint stay off the argv...
    argv = S.build_argv(
        sc, {"sam_prompts": "text"}, roots=roots, settings=settings, mask_root="ws"
    )
    assert argv == bound
    # ...and both arrive the moment the drawer opens.
    argv = S.build_argv(
        sc,
        {"use_sam": True, "sam_prompts": "text"},
        roots=roots,
        settings=settings,
        mask_root="ws",
    )
    assert argv == [
        *bound,
        "--use-sam",
        "--sam-prompts",
        "text",
        "--checkpoint",
        "sam3.pt",
    ]
    # The other switch folds away the gate its own drawer holds.
    argv = S.build_argv(
        sc,
        {"use_mit": False, "ctd_gate": False, "use_sam": True},
        roots=roots,
        mask_root="ws",
    )
    assert argv == [*bound, "--use-sam", "--no-use-mit"]


def test_export_destinations_are_bound_and_on_the_panel():
    """Export keeps ``--out`` / ``--index`` on its form as ``overridable`` paths
    (a per-run choice); every other bound field stays hidden."""
    _, sc = _stage("export")
    over = {f["dest"] for f in sc["fields"] if f["overridable"]}
    assert over == S.PANEL_FIELDS["export"] == {"out", "index"}
    by = {f["dest"]: f for f in sc["fields"]}
    # Bound as before: the destination to a root, the index to the report root.
    assert by["out"]["root"] == "out"
    assert by["index"]["report"] == "captions/caption_index.json"
    assert (by["out"]["path"], by["out"]["path_kind"]) == (True, "dir")
    assert (by["index"]["path"], by["index"]["path_kind"]) == (True, "file")
    # Everything Export reads stays bound and hidden.
    assert not any(
        f["overridable"] for f in sc["fields"] if f["dest"] in ("src", "dst", "masks")
    )


def test_every_basic_field_names_a_flag_the_stage_actually_has():
    """Every dest in :data:`BASIC_FIELDS` names a flag the stage has; a typo
    would silently fold the knob it meant to keep."""
    for sid, basic in S.BASIC_FIELDS.items():
        _, sc = _stage(sid)
        assert basic <= {f["dest"] for f in sc["fields"]}, sid


def test_advanced_folds_the_research_parameters_and_never_the_form_itself():
    """A stage with no :data:`BASIC_FIELDS` row has no advanced fields; one with a
    row still never folds a drawer's gate, a required field, or an already-hidden
    one.
    """
    _, sc = _stage("autotag")
    assert not any(f["advanced"] for f in sc["fields"])

    _, sc = _stage("position")
    by = {f["dest"]: f for f in sc["fields"]}
    shown = [
        f
        for f in sc["fields"]
        if not any(f[k] for k in ("root", "setting", "report", "mask", "auto"))
    ]
    kept = {f["dest"] for f in shown if not f["advanced"]} - {"apply", S.REPLAY_FIELD}
    assert kept == S.BASIC_FIELDS["position"]
    assert by["prompt"]["advanced"] is False and by["iou_threshold"]["advanced"] is True
    # Bound fields are hidden already.
    assert not any(
        f["advanced"] for f in sc["fields"] if f["setting"] or f["root"] or f["report"]
    )

    # The two detector switches are gates: a folded gate is a drawer you cannot open.
    _, sc = _stage("masks_mit")
    for f in sc["fields"]:
        if f["gate"] == f["dest"] or f["required"]:
            assert f["advanced"] is False, f["dest"]


def test_an_overridable_field_opens_on_settings_and_yields_to_the_form():
    """Blank means "whatever Settings says"; a typed value wins for that run.

    The schema knows nothing about a settings file, so the Settings value
    reaches the form as the field's *default*, through ``resolved_schema``.
    """
    sc = S.schema(S.BY_ID["export"])
    roots, reports = {"out": "/data/export"}, "ws/captions"
    got = S.resolved_schema(sc, roots=roots, report_root=reports)
    by = {f["dest"]: f for f in got["fields"]}
    assert by["out"]["default"] == "/data/export"
    assert by["index"]["default"] == "ws/captions/captions/caption_index.json"
    # Untouched for every other stage, and the stored schema is left alone.
    assert S.resolved_schema(S.schema(S.BY_ID["autotag"]), roots=roots) == S.schema(
        S.BY_ID["autotag"]
    )
    raw = {f["dest"]: f["default"] for f in sc["fields"]}
    assert raw["out"] == "post_image_dataset"

    def _out(schema, values):
        argv = S.build_argv(schema, values, roots=roots, report_root=reports)
        return argv[argv.index("--out") + 1]

    # Resolved and raw build the same argv: the bound value is what goes out,
    # whichever default the form was shown.
    assert _out(got, {}) == _out(sc, {}) == "/data/export"
    assert _out(got, {"out": "/tmp/scratch"}) == "/tmp/scratch"
    assert _out(got, {"out": "  "}) == "/data/export"
    # ...and unlike every other bound dest, it is the form's to remember.
    assert S.form_values(sc["fields"], {"out": "/tmp/scratch", "src": "stale"}) == {
        "out": "/tmp/scratch"
    }


def test_export_has_no_undo_flag():
    """Taking an export back is the GUI's Undo over the run's report, not a flag."""
    _, sc = _stage("export")
    assert "undo" not in {f["dest"] for f in sc["fields"]}


def test_a_stale_mask_dir_in_a_saved_form_never_wins():
    """``mask_dir`` lives in ⚙ Settings, so a value left in a saved payload is
    dead: two generators sharing one directory overwrite each other."""
    _, sc = _stage("masks_sam")
    argv = S.build_argv(sc, {"mask_dir": "typo"}, roots={"dst": "d"}, mask_root="ws")
    assert "typo" not in argv
    assert argv == ["--image-dir", "d", "--mask-dir", "ws/masks_sam"]
    # With no root to bind against, the flag falls away and the CLI default stands.
    assert S.build_argv(sc, {"mask_dir": "typo"}, roots={"dst": "d"}) == [
        "--image-dir",
        "d",
    ]


def test_the_sam3_checkpoint_is_one_setting_for_three_stages():
    """position / audit / masks_sam build the same SAM3 from one Settings value."""
    required = {"mask_dir": "m"}
    for stage_id in ("position", "audit", "masks_sam"):
        _, sc = _stage(stage_id)
        ckpt = next(f for f in sc["fields"] if f["dest"] == "checkpoint")
        assert ckpt["setting"] == "checkpoint", stage_id
        assert ckpt["default"] == "models/sam3/sam3.pt", stage_id
        argv = S.build_argv(
            sc,
            required,
            settings={"checkpoint": "w.pt"},
            roots={"src": "i", "dst": "d"},
        )
        assert _opt(argv, "--checkpoint") == "w.pt", stage_id
        # A path stranded in a saved form never stands in for the setting.
        assert "--checkpoint" not in S.build_argv(
            sc,
            {**required, "checkpoint": "stale.pt"},
            roots={"src": "i", "dst": "d"},
        )


def test_the_soft_prompt_is_one_setting_for_both_detector_stages():
    """Both detector stages take the soft prompt from one Settings value."""
    for stage_id in ("position", "audit"):
        _, sc = _stage(stage_id)
        embed = next(f for f in sc["fields"] if f["dest"] == "prompt_embed")
        assert embed["setting"] == "prompt_embed", stage_id
        argv = S.build_argv(sc, {}, settings={"prompt_embed": "none"})
        assert _opt(argv, "--prompt_embed") == "none", stage_id
        # Blank in Settings = the shipped default, which the CLI already holds.
        assert "--prompt_embed" not in S.build_argv(
            sc, {"prompt_embed": "stale.safetensors"}
        )


def test_the_report_root_moves_every_report_and_splits_none():
    """One Settings knob, one directory per stage: the tail comes off each stage's
    own CLI default, so no two stages share a report."""
    tails = {}
    for stage_id in ("resize", "autotag", "position", "audit", "groups"):
        st, sc = _stage(stage_id)
        dest = st.report[0]
        f = next(x for x in sc["fields"] if x["dest"] == dest)
        assert f["report"] and f["default"].endswith(f["report"])
        tails[stage_id] = f["report"]
        assert _opt(S.build_argv(sc, {}, report_root="d"), f["flags"][0]) == (
            f"d/{f['report']}"
        )
    assert len(set(tails.values())) == len(tails), tails


def test_report_path_follows_the_settings_report_root():
    st, sc = _stage("autotag")
    # No root: the CLI's own default, which is what a hand-run stage writes.
    assert (
        S.report_path(st, sc["fields"], {}) == "workspace/captions/autotag/report.json"
    )
    assert (
        S.report_path(st, sc["fields"], {}, "moved")
        == "moved/captions/autotag/report.json"
    )
    # A value stranded in a saved form is not consulted at all.
    assert S.report_path(st, sc["fields"], {"report_dir": "r"}, "moved").startswith(
        "moved/"
    )
    assert S.form_values(sc["fields"], {"report_dir": "r", "mode": "merge"}) == {
        "mode": "merge"
    }
    st, sc = _stage("groups")
    assert S.report_path(st, sc["fields"], {}, "moved") == "moved/groups/groups.json"


# -- the schemas are built in-process, off the request classes --------------


def test_schemas_build_in_process_without_a_model_library():
    """No child interpreter and no cache: the request modules are torch-free,
    so the server describes every stage itself. Pinned in a fresh interpreter,
    since a model library already imported by another test would hide it."""
    code = (
        "import sys; from anime_tools.gui import stages as S; "
        "schemas = S.load_schemas(); "
        "assert set(schemas) == {s.id for s in S.STAGES}, set(schemas); "
        "assert all(sc['available'] for sc in schemas.values()); "
        "heavy = {'torch', 'cv2', 'sam3', 'onnxruntime', 'timm'} & set(sys.modules); "
        "assert not heavy, heavy"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr


def test_the_form_and_the_cli_are_one_field_list():
    """Every form field is a flag of the stage's generated parser with the same
    default, and the request's docstring is the stage's doc."""
    import inspect

    for st in S.STAGES:
        sc = S.schema(st)
        actions = {a.dest: a for a in S.load_parser(st)._actions if a.dest != "help"}
        assert {f["dest"] for f in sc["fields"]} == set(actions), st.id
        for f in sc["fields"]:
            a = actions[f["dest"]]
            # A bool's ``--no-`` spellings are the parser's; the form carries
            # one ``negate`` instead.
            assert set(f["flags"]) <= set(a.option_strings), (st.id, f["dest"])
            assert (a.option_strings[:1] or [None])[0] == (f["flags"] or [None])[0]
            if not f["required"]:
                got = list(a.default) if isinstance(a.default, tuple) else a.default
                assert got == f["default"], (st.id, f["dest"])
        assert sc["doc"] == inspect.getdoc(st.request_class())


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
        "import json, os, pathlib, time\n"
        "from dataclasses import dataclass\n"
        "from anime_tools._request import Request, arg\n"
        "@dataclass(frozen=True, kw_only=True)\n"
        "class StubRequest(Request):\n"
        "    FLAG_SEP = '_'\n"
        "    n: int = arg(1)\n"
        "    apply: bool = arg(False)\n"
        "    sleep: float = arg(0.0)\n"
        # The two Settings-bound dests and the auto-detected one, so the stub
        # exercises the same binding the real stages get.
        "    path_pattern: str = arg('*')\n"
        "    device: str | None = arg(None)\n"
        "    report_dir: str = arg('out')\n"
        "def build_parser():\n"
        "    return StubRequest.parser()\n"
        "if __name__ == '__main__':\n"
        "    a = StubRequest.from_argv(build_parser())\n"
        "    for i in range(a.n): print('line', i, flush=True)\n"
        "    time.sleep(a.sleep)\n"
        "    d = pathlib.Path(os.environ['ANIME_TOOLS_HOME'], a.report_dir); d.mkdir(exist_ok=True)\n"
        "    (d/'report.json').write_text(json.dumps({'apply': a.apply, 'rows': [{'k': 1}]}))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    fake = S.Stage(
        id="stub",
        title="Stub",
        request="stub_stage:StubRequest",
        module="stub_stage",
        panel="test",
        report=("report_dir", "report.json"),
    )
    monkeypatch.setitem(S.BY_ID, "stub", fake)
    schemas["stub"] = S.schema(fake)
    app = create_app(jobs=JobManager(log_dir=tmp_path / "logs"), schemas=schemas)
    # A local browser; `/api/pick` and `/api/ls` answer a remote one differently
    # (pinned in `tests/test_gui_nativepick.py`).
    with TestClient(app, client=("127.0.0.1", 4242)) as c:
        yield c, tmp_path


def test_index_and_info(client):
    c, home = client
    assert "<title>anime_tools</title>" in c.get("/").text
    assert c.get("/api/info").json()["home"] == str(home)
    ids = [s["id"] for s in c.get("/api/stages").json()]
    assert ids == [s.id for s in S.STAGES]


def test_bundle_assets_resolve(client):
    """Every ``url()`` the built CSS points at must be a name ``/assets`` serves."""
    c, _ = client
    refs = re.findall(r"url\((/assets/[^)]+)\)", c.get("/").text)
    assert refs, "the bundle references no assets -- did build.ts start inlining again?"
    for ref in refs:
        r = c.get(ref)
        assert r.status_code == 200, ref
        assert r.headers["content-type"] == "font/woff2"
    # The route serves the page's siblings, not the page itself.
    assert c.get("/assets/index.html").status_code == 404
    assert c.get("/assets/nope.woff2").status_code == 404


def test_files_are_confined_to_the_dataset(client):
    """Serving a file is bounded by the trees the panel is showing; browsing
    (``/api/ls``) is the looser half."""
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


def test_the_browser_walks_up_by_the_parent_the_server_names(client):
    """The picker never takes a path apart, so ``parent`` is a field in the answer
    — and it goes past the home."""
    c, home = client
    at_home = c.get("/api/ls").json()
    assert at_home["path"] == "" and at_home["parent"] == home.parent.as_posix()
    up = c.get("/api/ls", params={"path": at_home["parent"]}).json()
    assert up["path"] == home.parent.as_posix()
    assert {"name": home.name, "dir": True} in up["entries"]
    assert c.get("/api/ls", params={"path": "/"}).json()["parent"] is None


def test_files_reject_dotdot_traversal(client):
    """`..` must not escape the dataset; `reachable` collapses it with normpath
    before the textual `is_relative_to`."""
    c, home = client
    outside = home.parent / "gui_traversal_target.txt"
    outside.write_text("secret")
    try:
        traversal = f"image_dataset/../../{outside.name}"
        assert outside.is_file()  # the target really exists — 404 is the guard
        assert c.get("/api/files", params={"path": traversal}).status_code == 404
        # The browser may walk out there; reading what it finds is still refused.
        assert (
            c.get("/api/ls", params={"path": "image_dataset/../.."}).status_code == 200
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
    # --report_dir binds to the Settings report root, which defaults to the parent
    # of the `dst` root rather than to the CLI's literal.
    assert job["argv"][3:] == [
        "--n",
        "3",
        "--apply",
        "--report_dir",
        "workspace/out",
    ]

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


def test_stage_schemas_carry_the_settings_value_for_a_panel_field(client):
    """``/api/stages`` fills a panel field's default from Settings; every other
    stage's schema is served verbatim."""
    c, _home = client
    c.put("/api/settings", json={"dataset": {"out": "elsewhere/export"}})
    export = next(s for s in c.get("/api/stages").json() if s["id"] == "export")
    by = {f["dest"]: f for f in export["fields"]}
    assert by["out"]["default"] == "elsewhere/export"
    assert by["out"]["overridable"] is True
    # Every other stage's schema is served verbatim.
    autotag = next(s for s in c.get("/api/stages").json() if s["id"] == "autotag")
    assert autotag == S.schema(S.BY_ID["autotag"])
    c.put("/api/settings", json={"dataset": {}})


def test_job_start_creates_the_report_directory(client):
    """``POST /api/jobs`` creates the report directory before the child runs."""
    c, home = client
    c.put("/api/settings", json={"stage_defaults": {S.REPORT_SETTING: "deep/nested"}})
    job = _await_job(c, c.post("/api/jobs", json={"stage": "stub"}))
    assert job["state"] == "done", job
    assert _opt(job["argv"], "--report_dir") == "deep/nested/out"
    assert (home / "deep/nested/out/report.json").is_file()
    assert c.get(f"/api/jobs/{job['id']}/report").json()["report"]["rows"] == [{"k": 1}]
    c.put("/api/settings", json={"stage_defaults": {}})


def test_settings_pattern_and_rel_pick_the_run_scope(client):
    """The batch button sends no ``rel`` and gets the Settings pattern; the
    per-image button sends one and gets a pattern naming just that file."""
    c, _home = client
    c.put("/api/settings", json={"stage_defaults": {"path_pattern": "sub/*"}})

    batch = c.post(
        "/api/jobs", json={"stage": "stub", "values": {"n": 1, "path_pattern": "old/*"}}
    ).json()
    assert _opt(batch["argv"], "--path_pattern") == "sub/*"
    _await_job(c, c.get(f"/api/jobs/{batch['id']}"))
    # A bound dest is not the form's to remember, so it is not written back.
    assert c.get("/api/settings").json()["values"]["stub"] == {"n": 1}

    one = c.post("/api/jobs", json={"stage": "stub", "rel": "sub/b.jpg"}).json()
    # Stem, not filename: the resize step may have re-encoded it.
    assert _opt(one["argv"], "--path_pattern") == "sub/b.*"
    _await_job(c, c.get(f"/api/jobs/{one['id']}"))

    # --device is auto-detected in the child and never sent.
    assert "--device" not in batch["argv"] + one["argv"]


# -- the resize preflight: an implicit first step, not a panel -------------


def test_resize_is_not_a_dock_panel():
    """Resize has a schema and an argv but no panel: it runs itself."""
    assert S.BY_ID["resize"].hidden is True
    assert "Resize" not in S.PANELS
    assert [s.id for s in S.STAGES if s.hidden] == ["resize"]


def test_only_resized_tree_stages_get_the_preflight():
    """A stage bound to ``dst`` reads the resized tree and needs resize in front of
    it. ``export`` publishes ``dst`` rather than consuming it, and ``masks_merge``
    opens no image."""
    got = {s.id: S.preprocess_for(s.id) for s in S.STAGES}
    assert got["export"] is None and "dst" in S.ROOT_FIELDS["export"].values()
    assert {k for k, v in got.items() if v == "resize"} == {
        "autotag",
        "position",
        "correct",
        "audit",
        "ocr",
        "masks_sam",
        "masks_mit",
        "groups",
    }
    assert got["resize"] is None  # never its own preflight
    # Unions two mask trees; never opens an image.
    assert got["masks_merge"] is None


def test_a_dst_bound_stage_runs_resize_first(client, monkeypatch):
    c, _home = client
    monkeypatch.setitem(S.ROOT_FIELDS, "stub", {"src": "src", "dst": "dst"})

    job = _await_job(c, c.post("/api/jobs", json={"stage": "stub", "values": {"n": 1}}))

    assert [st["label"] for st in job["steps"]] == ["resize", "stub"]
    assert job["steps"][0]["module"] == "anime_tools.stages.cli.resize_images"
    # `argv` stays the stage's own command, so the UI labels the job by it.
    assert job["argv"][:3] == [sys.executable, "-m", "stub_stage"]
    assert job["state"] == "done", job
    # Both steps' output lands in the one stream, under a step header.
    body = "".join(c.get(f"/api/jobs/{job['id']}/log").iter_text())
    assert "step 1/2" in body and "step 2/2" in body


def test_a_stage_without_a_preflight_is_a_single_step(client):
    c, _home = client
    job = _await_job(c, c.post("/api/jobs", json={"stage": "stub", "values": {"n": 1}}))
    assert [st["label"] for st in job["steps"]] == ["stub"]
    # A lone step prints no header.
    body = "".join(c.get(f"/api/jobs/{job['id']}/log").iter_text())
    assert "step 1/1" not in body


def test_the_preflight_is_scoped_exactly_like_the_job(client, monkeypatch):
    """Per-image Apply must resize that one image, not the whole dataset."""
    c, _home = client
    monkeypatch.setitem(S.ROOT_FIELDS, "stub", {"src": "src", "dst": "dst"})
    c.put("/api/settings", json={"stage_defaults": {"path_pattern": "sub/*"}})

    one = c.post("/api/jobs", json={"stage": "stub", "rel": "a.png"}).json()
    assert _opt(one["steps"][0]["argv"], "--path_pattern") == "a.*"
    assert _opt(one["steps"][1]["argv"], "--path_pattern") == "a.*"
    _await_job(c, c.get(f"/api/jobs/{one['id']}"))

    batch = c.post("/api/jobs", json={"stage": "stub"}).json()
    assert _opt(batch["steps"][0]["argv"], "--path_pattern") == "sub/*"
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
    """A failing step stops the chain."""
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


def test_scoping_an_unscopable_stage_is_refused(client, monkeypatch):
    """No ``--path_pattern`` means nothing to narrow, so scoping is a 400 rather
    than a run over everything."""
    c, _home = client
    r = c.post("/api/jobs", json={"stage": "stub", "rel": "../escape.png"})
    assert r.status_code == 400
    # The same stub with its pattern taken away.
    schemas = c.app.state.schemas.get()
    monkeypatch.setitem(schemas, "stub", {**schemas["stub"], "scoped": False})
    r = c.post("/api/jobs", json={"stage": "stub", "rel": "a.png"})
    assert r.status_code == 400 and "scoped" in r.json()["detail"]


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
    """The panel answers even with no KB on disk (the download prompt), and the
    English sibling replaces the description only — the taxonomy stays the base
    table's."""
    from anime_tools.captions import correction
    from anime_tools.gui import tags as T

    c, home = client
    # The source tree's own models/ is a fallback candidate; keep the test off it.
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

    # The English sibling, built later: the cache is keyed on both files, and it
    # replaces the blurb only.
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
    normal job sharing the one slot."""
    c, home = client
    body = c.get("/api/models").json()
    assert body["models_dir"] == str(home / "models")
    rows = {m["id"]: m for m in body["models"]}
    assert not rows["sam3"]["installed"]
    assert rows["sam3"]["location"] == str(home / "models" / "sam3")
    assert rows["tagger_backbone"]["gated"].startswith("https://huggingface.co/")

    assert c.post("/api/models/download", json={"ids": ["nope"]}).status_code == 404

    # Downloads and stages contend for the same slot.
    j = c.post("/api/jobs", json={"stage": "stub", "values": {"sleep": 30}}).json()
    assert c.post("/api/models/download", json={"ids": ["sam3"]}).status_code == 409
    c.post(f"/api/jobs/{j['id']}/cancel")

    # HF_HUB_OFFLINE keeps the test off the wire; it fails fast, which is enough
    # to see the child ran.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    job = _await_job(c, c.post("/api/models/download", json={"ids": ["mit_text"]}))
    assert job["argv"][1:] == ["-m", "anime_tools.downloads", "mit_text"]
    assert job["state"] in ("done", "failed")


# -- startup: the schema build is off the critical path --------------------


def _app(tmp_path, monkeypatch, loader):
    from anime_tools.gui.jobs import JobManager
    from anime_tools.gui.server import create_app

    monkeypatch.setenv("ANIME_TOOLS_HOME", str(tmp_path))
    monkeypatch.setattr(S, "load_schemas", loader)
    return create_app(jobs=JobManager(log_dir=tmp_path / "logs"))


def test_startup_does_not_wait_for_the_schemas(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    gate = threading.Event()

    def slow():
        assert gate.wait(30)
        return {s.id: S.schema(s) for s in S.STAGES}

    t0 = time.perf_counter()
    app = _app(tmp_path, monkeypatch, slow)
    assert time.perf_counter() - t0 < 1.0  # bind the port, don't build schemas

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


def test_a_failed_schema_build_is_reported_not_fatal(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    def boom():
        raise RuntimeError("stage schema build failed: boom")

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
    """Apply reads ``schema()["replay"]``, so a ``--from_report`` stage carries it."""
    for stage_id in ("autotag", "position"):
        st, sc = _stage(stage_id)
        assert S.schema(st)["replay"] is True
        assert any(f["dest"] == S.REPLAY_FIELD for f in sc["fields"])
    assert S.schema(S.BY_ID["groups"])["replay"] is False


def test_replay_report_name_matches_the_stages():
    """``report_path`` hard-codes the replay filename to stay torch-free."""
    from anime_tools.stages.replay import REPLAY_REPORT_NAME

    assert S.REPLAY_REPORT_NAME == REPLAY_REPORT_NAME


def test_a_replay_reports_beside_the_run_it_replays():
    """A replay reports beside the dry run it reads, never over it."""
    st, sc = _stage("autotag")
    dry = S.report_path(st, sc["fields"], {}, "r")
    assert dry == "r/captions/autotag/report.json"
    replay = S.report_path(st, sc["fields"], {S.REPLAY_FIELD: dry}, "r")
    assert replay == f"r/captions/autotag/{S.REPLAY_REPORT_NAME}" != dry


def test_from_report_reaches_the_argv():
    _, sc = _stage("autotag")
    argv = S.build_argv(sc, {S.REPLAY_FIELD: "r/report.json"}, apply=True)
    assert "--from_report" in argv
    assert argv[argv.index("--from_report") + 1] == "r/report.json"
    assert "--apply" in argv


def test_dataset_items_refreshes_only_what_it_is_asked_for(client):
    """A job's ``written`` list in, those rows out."""
    c, tmp_path = client
    src = tmp_path / "image_dataset"
    (src / "sub").mkdir()
    (src / "sub" / "b.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    r = c.post("/api/dataset/items", json={"rels": ["sub/b.png", "a.png"]})
    assert r.status_code == 200
    rows = r.json()["items"]
    assert [x["rel"] for x in rows] == ["sub/b.png", "a.png"]
    assert rows[0]["dir"] == "sub" and rows[0]["captions"]["revised"] is False
    # A caption written since the listing shows up on the refreshed row.
    dst = tmp_path / "workspace" / "resized" / "sub"
    dst.mkdir(parents=True)
    (dst / "b.txt").write_text("1girl.")
    assert c.post("/api/dataset/items", json={"rels": ["sub/b.png"]}).json()["items"][
        0
    ]["captions"]["revised"]


def test_dataset_items_drops_what_it_cannot_refresh(client):
    """Traversal and vanished rows are dropped, not raised."""
    c, _ = client
    r = c.post(
        "/api/dataset/items", json={"rels": ["../escape.png", "/abs.png", "gone.png"]}
    )
    assert r.status_code == 200 and r.json()["items"] == []


def test_apply_replays_a_dry_run_end_to_end(tmp_path, monkeypatch):
    """The whole Apply path over the real HTTP API and autotag CLI: a dry run's
    report in, captions out, no model loaded."""
    from fastapi.testclient import TestClient

    from anime_tools.gui.server import create_app

    monkeypatch.setenv("ANIME_TOOLS_HOME", str(tmp_path))
    c = TestClient(create_app())
    # Saving Settings makes the roots real, in registry order; `out` is Export's
    # to create, so it is not here.
    assert c.put("/api/dataset/roots", json={}).json()["created"] == [
        "src",
        "master",
        "dst",
        "masks",
    ]
    src, dst = tmp_path / "image_dataset", tmp_path / "workspace" / "resized"
    for n in ("a", "b", "c"):
        (src / f"{n}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (src / f"{n}.txt").write_text("1girl, solo.")
        (dst / f"{n}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        # Autotag writes the revised caption, so that is what a replay of it
        # gates on and what comes out the other end.
        (dst / f"{n}.txt").write_text("1girl, solo.")

    # What the tagger pass left behind.
    rdir = tmp_path / "workspace" / "captions" / "autotag"
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
                        "caption_path": f"{n}.txt",
                        "existing": "1girl, solo.",
                        "target_before": "1girl, solo.",
                        "proposed": f"1girl, solo, {n}_tag.",
                        "status": "ok",
                    }
                    for n in ("a", "b", "c")
                ],
            }
        )
    )
    # Hand-edited after the dry run: its proposal is stale, so it is skipped.
    (dst / "c.txt").write_text("1girl, solo, hand edited.")

    job = c.post(
        "/api/jobs",
        json={
            "stage": "autotag",
            "apply": True,
            "values": {S.REPLAY_FIELD: "workspace/captions/autotag/report.json"},
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
    assert (dst / "a.txt").read_text() == "1girl, solo, a_tag."
    assert (dst / "c.txt").read_text() == "1girl, solo, hand edited."
    # The master is not autotag's to write.
    assert (src / "a.txt").read_text() == "1girl, solo."
    # …and that list is all the sidebar has to re-stat.
    rows = c.post("/api/dataset/items", json={"rels": report["written"]}).json()[
        "items"
    ]
    assert [r["rel"] for r in rows] == ["a.png", "b.png"]


# ---- one refusal, one status code -------------------------------------


def test_a_refused_root_is_a_bad_request_not_a_crash(client):
    """`DatasetError` / `ProposalError` are registered app-wide as 400s."""
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
    """The roots resolved, so a missing image is a 404, not the app-wide 400."""
    c, *_ = client
    r = c.get("/api/dataset/item", params={"rel": "nope.png"})
    assert r.status_code == 404
    assert "not in the dataset" in r.json()["detail"]
