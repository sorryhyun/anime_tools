"""The Run → Apply → Undo loop, read off the stage reports.

``Run`` writes a report; the caption panel shows it as a per-image diff; ``Apply``
replays it; ``Undo`` reads the *apply* report back and puts the captions where
they were. All three go through :mod:`anime_tools.gui.proposals`, which is what
this file exercises — plus the one thing that can silently rot: its copy of each
stage's report shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

# Imported after the skip so a fastapi-less env skips rather than errors.
from anime_tools.gui import proposals as P


def _png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (128, 128, 128)).save(path)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Two images: one captioned in both trees, one with no master at all."""
    monkeypatch.setenv("ANIME_TOOLS_HOME", str(tmp_path))
    src = tmp_path / "image_dataset"
    dst = tmp_path / "workspace" / "resized"
    _png(src / "a.png")
    (src / "a.txt").write_text("1girl, solo", encoding="utf-8")
    _png(dst / "a.png")
    (dst / "a.txt").write_text("1girl, solo", encoding="utf-8")
    # A jpg master re-encoded to png on the way into the resized tree: the join
    # back from a report's ``image`` has to survive that.
    _png(src / "sub" / "b.jpg")
    _png(dst / "sub" / "b.png")
    return tmp_path


@pytest.fixture
def roots(home):
    from anime_tools.gui import dataset as D

    return D.resolve_roots()


def _autotag_report(home: Path, *, apply: bool = False) -> Path:
    """An ``autotag_captions`` dry report over both images."""
    report = {
        "stage": "autotag_captions",
        "apply": apply,
        "src": str(home / "image_dataset"),
        "dst": str(home / "workspace" / "resized"),
        "rows": [
            {
                "image": "a.png",
                "caption_path": "a.txt",
                "existing": "1girl, solo",
                "proposed": "safe, 1girl, solo, long hair",
                "status": "ok",
            },
            {
                "image": "sub/b.png",
                "caption_path": "sub/b.txt",
                "existing": "",
                "proposed": "safe, 1boy",
                "status": "ok",
            },
            {
                "image": "a.png",
                "caption_path": "a.txt",
                "existing": "x",
                "proposed": "x",
                "status": "skip:unchanged",
            },
        ],
    }
    p = home / "workspace" / "captions" / "autotag" / "report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report), encoding="utf-8")
    return p


# ---- reading a report as proposals ------------------------------------


def test_proposals_are_keyed_by_dataset_rel(home, roots):
    """Reports name images in the *resized* tree; the sidebar selects source
    rels. A re-encoded image (``b.jpg`` → ``b.png``) has to join on the stem."""
    found = P.read(_autotag_report(home), roots, "autotag")
    assert sorted(found) == ["a.png", "sub/b.jpg"]
    assert found["a.png"].after == "safe, 1girl, solo, long hair"
    assert found["a.png"].kind == "master"
    assert found["a.png"].path == "image_dataset/a.txt"


def test_a_row_that_changes_nothing_is_not_a_proposal(home, roots):
    """``skip:unchanged`` rows carry a before and an after that are the same
    text — showing them as a diff would be a diff of nothing."""
    found = P.read(_autotag_report(home), roots, "autotag")
    assert all(p.before != p.after for p in found.values())


def test_the_revised_caption_is_the_position_stage_s_target(home, roots):
    """``position_captions`` only ever rewrites ``workspace/resized``,
    so its diff belongs on the *derived* card, not the master's."""
    report = {
        "summary": {"apply": False},
        "images": [
            {
                "image": "a.png",
                "caption_path": "a.txt",
                "original": "1girl, solo",
                "proposed": "1girl. On the left, solo.",
                "status": "proposed",
            }
        ],
    }
    p = home / "pos.json"
    p.write_text(json.dumps(report), encoding="utf-8")
    got = P.read(p, roots, "position")["a.png"]
    assert got.kind == "revised"
    assert got.path == "workspace/resized/a.txt"


def test_a_stage_that_proposes_nothing_is_refused(home, roots):
    with pytest.raises(P.ProposalError):
        P.read(_autotag_report(home), roots, "groups")


def test_the_parsed_pair_never_needs_a_client_side_split(home, roots):
    """Both texts go out already parsed: the caption grammar has exactly one
    implementation, and a browser-side comma split would not be it."""
    p = P.read(_autotag_report(home), roots, "autotag")["a.png"].to_dict()
    assert p["after_parsed"]["flat_tags"] == ["safe", "1girl", "solo", "long hair"]
    assert p["before_parsed"]["flat_tags"] == ["1girl", "solo"]


def test_a_reread_after_a_rerun_is_not_the_cached_answer(home, roots):
    """The read is cached per (path, mtime) — a second Run over the same report
    path must not serve the first one's proposals."""
    path = _autotag_report(home)
    first = P.read(path, roots, "autotag")["a.png"].after
    report = json.loads(path.read_text(encoding="utf-8"))
    report["rows"][0]["proposed"] = "safe, 1girl, solo, twintails"
    path.write_text(json.dumps(report), encoding="utf-8")
    import os

    os.utime(path, (0, 0))  # a distinct mtime, whatever the filesystem rounds to
    assert P.read(path, roots, "autotag")["a.png"].after != first


# ---- undo --------------------------------------------------------------


def _apply_report(home: Path) -> Path:
    """What a replayed ``--apply`` run writes (``stages.replay.ReplayRow``):
    ``before``/``after``, whatever the source stage called them."""
    report = {
        "stage": "autotag_captions",
        "apply": True,
        "rows": [
            {
                "image": "a.png",
                "caption_path": "a.txt",
                "before": "1girl, solo",
                "after": "safe, 1girl, solo, long hair",
                "status": "written",
            },
            {
                "image": "sub/b.png",
                "caption_path": "sub/b.txt",
                "before": "",
                "after": "safe, 1boy",
                "status": "written",
            },
        ],
        "written": ["a.png", "sub/b.png"],
    }
    p = home / "apply_report.json"
    p.write_text(json.dumps(report), encoding="utf-8")
    return p


def test_undo_restores_what_the_apply_overwrote(home, roots):
    src = home / "image_dataset"
    (src / "a.txt").write_text("safe, 1girl, solo, long hair", encoding="utf-8")
    (src / "sub" / "b.txt").write_text("safe, 1boy", encoding="utf-8")

    out = P.undo(_apply_report(home), roots, "autotag")
    assert (src / "a.txt").read_text(encoding="utf-8") == "1girl, solo"
    # A row whose before-text was empty was a file the run *created*: its
    # inverse is a delete, not a write of nothing.
    assert not (src / "sub" / "b.txt").exists()
    assert out["restored"] == 1 and out["removed"] == 1
    assert sorted(out["written"]) == ["a.png", "sub/b.jpg"]


def test_undo_leaves_a_caption_edited_since_alone(home, roots):
    src = home / "image_dataset"
    (src / "a.txt").write_text("hand-written since the apply", encoding="utf-8")
    (src / "sub" / "b.txt").write_text("safe, 1boy", encoding="utf-8")

    out = P.undo(_apply_report(home), roots, "autotag")
    assert (src / "a.txt").read_text(encoding="utf-8") == "hand-written since the apply"
    assert out["skipped"]["drifted"] == 1 and out["restored"] == 0


def test_undo_twice_is_a_no_op_not_a_second_revert(home, roots):
    src = home / "image_dataset"
    (src / "a.txt").write_text("safe, 1girl, solo, long hair", encoding="utf-8")
    (src / "sub" / "b.txt").write_text("safe, 1boy", encoding="utf-8")
    P.undo(_apply_report(home), roots, "autotag")
    out = P.undo(_apply_report(home), roots, "autotag")
    assert out["restored"] == 0 and out["removed"] == 0
    assert out["skipped"]["already-undone"] == 2
    assert (src / "a.txt").read_text(encoding="utf-8") == "1girl, solo"


def test_undoing_the_clause_rewrite_drops_the_stale_sidecar(home, roots):
    """``.variants.txt`` wins over the caption at encode time, so a sidecar left
    beside a restored caption would keep training the applied text."""
    dst = home / "workspace" / "resized"
    (dst / "a.txt").write_text("1girl. On the left, solo.", encoding="utf-8")
    (dst / "a.variants.txt").write_text(
        "v0\t1girl. On the left, solo.\n", encoding="utf-8"
    )
    report = {
        "summary": {"apply": True},
        "images": [
            {
                "image": "a.png",
                "caption_path": "a.txt",
                "before": "1girl, solo",
                "after": "1girl. On the left, solo.",
                "status": "written",
            }
        ],
    }
    p = home / "pos_apply.json"
    p.write_text(json.dumps(report), encoding="utf-8")

    P.undo(p, roots, "position")
    assert (dst / "a.txt").read_text(encoding="utf-8") == "1girl, solo"
    assert not (dst / "a.variants.txt").exists()


# ---- the shapes stay in step with the stages they mirror ---------------


def test_the_report_shapes_match_the_stages_own_replay_specs():
    """:data:`P.SHAPES` holds a copy of each CLI's ``REPLAY_SPEC``, kept out of
    the server process because those modules import torch. The ``ReplaySpec``
    *class* is imported, so this is now a whole-object comparison — the whole
    diff is wrong if any field drifts, including one added later.
    """
    from anime_tools.stages.cli.audit_multiview import REPLAY_SPEC as audit
    from anime_tools.stages.cli.autotag_captions import REPLAY_SPEC as autotag
    from anime_tools.stages.cli.position_captions import REPLAY_SPEC as position

    assert P.SHAPES == {"autotag": autotag, "position": position, "audit": audit}


def test_the_audit_gate_is_the_only_thing_its_replay_closes_over():
    """The audit's writable set is a verdict/confidence gate rather than a row
    status, so its module-level spec leaves ``row_filter`` open and the replay
    path fills it in. Everything else about the spec — the half
    :data:`P.SHAPES` mirrors — is pinned by the test above; this pins that the
    gate is the *only* thing the closure changes, so the mirror stays complete.
    """
    from dataclasses import replace

    from anime_tools.stages.cli.audit_multiview import REPLAY_SPEC as audit

    assert audit.ok_status is None and audit.row_filter is None
    gated = replace(audit, row_filter=lambda row: True)
    assert replace(gated, row_filter=None) == audit


# ---- the HTTP surface the run bar and the caption panel talk to --------


@pytest.fixture
def client(home):
    from fastapi.testclient import TestClient

    from anime_tools.gui.jobs import JobManager
    from anime_tools.gui.server import create_app

    app = create_app(jobs=JobManager(log_dir=home / "logs"), schemas={})
    with TestClient(app) as c:
        yield c, app.state.jobs, home


def _finished_job(mgr, home, *, apply: bool, report: Path):
    """A finished job record pointing at ``report``.

    Built rather than run: these endpoints only ever read ``stage``, ``apply``
    and ``report_path``, and spawning a real child would only test the runner.
    """
    from anime_tools.gui.jobs import Job

    job = Job(
        id=f"j{len(mgr.jobs)}",
        stage="autotag",
        steps=[],
        home=home,
        report_path=str(report.relative_to(home)),
        apply=apply,
        exit_code=0,
    )
    mgr.jobs[job.id] = job
    return job


def test_the_index_is_rels_only_and_one_proposal_carries_the_text(client):
    c, mgr, home = client
    job = _finished_job(mgr, home, apply=False, report=_autotag_report(home))

    index = c.get(f"/api/jobs/{job.id}/proposals").json()
    assert index["kind"] == "master" and index["total"] == 2
    assert index["rels"] == ["a.png", "sub/b.jpg"]
    assert "before" not in json.dumps(index)  # the index carries no caption text

    one = c.get(f"/api/jobs/{job.id}/proposal", params={"rel": "a.png"}).json()
    assert one["after"] == "safe, 1girl, solo, long hair"
    assert one["after_parsed"]["flat_tags"][0] == "safe"
    assert (
        c.get(f"/api/jobs/{job.id}/proposal", params={"rel": "nope.png"}).status_code
        == 404
    )


def test_undo_refuses_a_run_that_wrote_nothing(client):
    """A dry run has nothing to put back; "undo" could then only mean undoing
    some other run that happens to share the report shape."""
    c, mgr, home = client
    job = _finished_job(mgr, home, apply=False, report=_autotag_report(home))
    r = c.post(f"/api/jobs/{job.id}/undo")
    assert r.status_code == 400 and "nothing to undo" in r.json()["detail"]


def test_undo_over_http_restores_and_names_what_to_reload(client):
    c, mgr, home = client
    (home / "image_dataset" / "a.txt").write_text(
        "safe, 1girl, solo, long hair", encoding="utf-8"
    )
    (home / "image_dataset" / "sub" / "b.txt").write_text(
        "safe, 1boy", encoding="utf-8"
    )
    job = _finished_job(mgr, home, apply=True, report=_apply_report(home))

    out = c.post(f"/api/jobs/{job.id}/undo").json()
    assert out["restored"] == 1 and out["removed"] == 1
    # The rels the sidebar re-stats, same contract as a job's ``written``.
    assert sorted(out["written"]) == ["a.png", "sub/b.jpg"]
    assert (home / "image_dataset" / "a.txt").read_text(
        encoding="utf-8"
    ) == "1girl, solo"


# ---- export: the one report that is not a caption diff ------------------


def test_undo_dispatches_an_export_report_to_the_reverter(home, roots, tmp_path):
    """`P.undo` takes the stage id, and `export`'s rows are file copies rather
    than before/after captions — so it branches instead of trying to read them
    through a `ReplaySpec` it has no shape for."""
    from anime_tools._json import write_json
    from anime_tools.gui import proposals as P
    from anime_tools.stages.export_workspace import ExportPaths, run_export

    out = home / "post_image_dataset"
    paths = ExportPaths(
        resized=home / "workspace" / "resized",
        masks=home / "workspace" / "masks",
        master=home / "workspace" / "master",
        index=home / "workspace" / "captions" / "caption_index.json",
        src=home / "image_dataset",
        out=out,
    )
    rows, _ = run_export(paths, apply=True)
    assert (out / "resized" / "a.png").is_file()

    report = home / "export_apply.json"
    write_json(report, {"rows": [r.to_dict() for r in rows]})

    got = P.undo(report, roots, "export")
    assert got["stage"] == "export"
    assert got["removed"] == len(rows) and got["restored"] == 0
    assert not (out / "resized" / "a.png").exists()
    # The rels it reports back are dataset rels the sidebar can re-stat: the
    # resized tree calls the second image `sub/b.png`, the master `sub/b.jpg`.
    assert set(got["written"]) == {"a.png", "sub/b.jpg"}


def test_an_unknown_stage_is_still_refused(home, roots):
    from anime_tools.gui import proposals as P

    with pytest.raises(P.ProposalError):
        P.read(home / "nope.json", roots, "masks_merge")
