"""The caption history sidecar: what a write replaces, kept as a version.

A stage run writes for real now -- the run bar has no Apply gate -- so the text
it overwrites has to survive somewhere or it is simply gone. These pin the three
halves of that: the sidecar itself, the one caption write that pushes into it,
and the ladder that turns it into badges.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anime_tools.captions.history import (
    HISTORY_SIDECAR_SUFFIX,
    drop_history,
    history_sidecar_path,
    push_history,
    read_history,
)
from anime_tools.stages._caption_io import write_caption


def test_the_sidecar_sits_beside_the_caption_and_keeps_a_multi_dot_stem():
    assert history_sidecar_path(Path("a/b.txt")).name == "b" + HISTORY_SIDECAR_SUFFIX
    assert (
        history_sidecar_path(Path("a/b.c.txt")).name == "b.c" + HISTORY_SIDECAR_SUFFIX
    )
    # Given an image, the same answer: both sidecars key off the stem.
    assert history_sidecar_path(Path("a/b.png")).name == "b" + HISTORY_SIDECAR_SUFFIX


def test_versions_come_back_oldest_first_with_who_replaced_them(tmp_path):
    cap = tmp_path / "a.txt"
    push_history(cap, "1girl", by="position")
    push_history(cap, "1girl, solo", by="edit")

    got = read_history(history_sidecar_path(cap))
    assert [(e.seq, e.by, e.text) for e in got] == [
        (1, "position", "1girl"),
        (2, "edit", "1girl, solo"),
    ]
    assert got[1].label("revised") == "revised@2"
    assert got[1].note().startswith("edit · ")


def test_nothing_worth_recording_is_not_recorded(tmp_path):
    cap = tmp_path / "a.txt"
    # A caption being *created* replaces nothing.
    assert push_history(cap, "   ", by="edit") is None
    assert not history_sidecar_path(cap).exists()
    # A re-run that rewrites the same caption twice pushes one version, not two.
    assert push_history(cap, "1girl", by="position") is not None
    assert push_history(cap, "1girl", by="position") is None
    assert len(read_history(history_sidecar_path(cap))) == 1


def test_the_cap_drops_the_oldest_and_never_renumbers(tmp_path):
    """A badge you are looking at must not become a different version.

    Sequence numbers continue from the highest ever recorded rather than from
    the list length, so trimming the front cannot make ``revised@2`` mean two
    different captions at two different times.
    """
    cap = tmp_path / "a.txt"
    for i in range(6):
        push_history(cap, f"tag{i}", by="position", limit=3)

    got = read_history(history_sidecar_path(cap))
    assert [(e.seq, e.text) for e in got] == [(4, "tag3"), (5, "tag4"), (6, "tag5")]


def test_a_damaged_sidecar_costs_history_not_a_run(tmp_path):
    sidecar = tmp_path / ("a" + HISTORY_SIDECAR_SUFFIX)
    sidecar.write_text(
        "# header\n\nnot a record\nx\tnow\tedit\tbad seq\n7\tnow\tedit\t1girl\n",
        encoding="utf-8",
    )
    assert [e.text for e in read_history(sidecar)] == ["1girl"]


def test_drop_history_removes_it(tmp_path):
    cap = tmp_path / "a.txt"
    push_history(cap, "1girl", by="edit")
    drop_history(cap)
    assert not history_sidecar_path(cap).exists()
    drop_history(cap)  # idempotent: undoing an undone row is not an error


# ---- the write seam ---------------------------------------------------------


def test_the_caption_write_pushes_only_when_asked(tmp_path):
    """``history_by`` is per call site, not automatic: a rung the ladder gives
    no history rung would only accumulate a file nothing reads."""
    cap = tmp_path / "a.txt"
    write_caption(cap, "1girl")
    write_caption(cap, "1girl, solo")
    assert not history_sidecar_path(cap).exists()

    write_caption(cap, "1girl, solo, smile", history_by="position")
    assert [(e.by, e.text) for e in read_history(history_sidecar_path(cap))] == [
        ("position", "1girl, solo")
    ]
    assert cap.read_text(encoding="utf-8") == "1girl, solo, smile"


def test_creating_a_caption_records_no_version(tmp_path):
    cap = tmp_path / "sub" / "a.txt"
    write_caption(cap, "1girl", history_by="position")
    assert cap.read_text(encoding="utf-8") == "1girl"
    assert not history_sidecar_path(cap).exists()


def test_a_replay_files_the_same_version_the_live_pass_would(tmp_path):
    """``ReplaySpec.history_by`` is what keeps ``--from_report --apply`` and the
    stage's own apply writing the same two files."""
    from anime_tools.stages.replay import apply_one

    cap = tmp_path / "a.txt"
    cap.write_text("1girl, solo", encoding="utf-8")
    assert (
        apply_one(
            cap,
            "1girl, solo",
            "1girl, solo. On the left, cat.",
            apply=True,
            history_by="position",
        )
        == "written"
    )
    assert [e.text for e in read_history(history_sidecar_path(cap))] == ["1girl, solo"]
    # A dry pass writes nothing, so it supersedes nothing.
    apply_one(
        cap,
        "1girl, solo. On the left, cat.",
        "1girl",
        apply=False,
        history_by="position",
    )
    assert len(read_history(history_sidecar_path(cap))) == 1


# ---- the ladder -------------------------------------------------------------


pytest.importorskip("fastapi")


def _png(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (128, 128, 128)).save(path)


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from anime_tools.gui.jobs import JobManager
    from anime_tools.gui.server import create_app

    monkeypatch.setenv("ANIME_TOOLS_HOME", str(tmp_path))
    _png(tmp_path / "image_dataset" / "a.png")
    (tmp_path / "image_dataset" / "a.txt").write_text("1girl", encoding="utf-8")
    dst = tmp_path / "workspace" / "resized"
    _png(dst / "a.png")
    (dst / "a.txt").write_text("1girl, solo", encoding="utf-8")

    app = create_app(jobs=JobManager(log_dir=tmp_path / "logs"), schemas={})
    with TestClient(app) as c:
        yield c, tmp_path


def test_an_edit_is_a_version_and_the_badge_row_says_what_it_was(client):
    """The panel's promise: an edit leaves what it replaced one badge away."""
    c, _home = client
    r = c.put(
        "/api/dataset/item",
        json={"rel": "a.png", "kind": "revised", "text": "1girl, solo, smile"},
    )
    assert r.status_code == 200

    it = c.get("/api/dataset/item", params={"rel": "a.png"}).json()
    assert [v["kind"] for v in it["versions"]] == [
        "master",
        "revised@1",
        "revised",
        "variants",
    ]
    was = it["versions"][1]
    assert was["text"] == "1girl, solo" and not was["editable"]
    # It wears its rung's colour and its own label, and says who replaced it.
    assert was["rung"] == "history" and was["note"].startswith("edit · ")
    # Already parsed, like every other version: the browser splits no caption.
    assert was["parsed"]["flat_tags"] == ["1girl", "solo"]
    assert it["versions"][2]["text"] == "1girl, solo, smile"


def test_the_master_rung_keeps_no_history(client):
    """``image_dataset/`` is the input tree, so nothing of ours lands in it.

    The ladder is the one declaration of which rungs keep versions
    (``HISTORY_OF``), so this is the ladder's answer, not a second rule.
    """
    c, home = client
    c.put("/api/dataset/item", json={"rel": "a.png", "kind": "master", "text": "1boy"})
    assert not (home / "image_dataset" / ("a" + HISTORY_SIDECAR_SUFFIX)).exists()


def test_the_history_dot_is_a_row_flag_like_every_other_rung(client):
    c, _home = client
    assert c.get("/api/dataset").json()["items"][0]["captions"]["history"] is False
    c.put(
        "/api/dataset/item",
        json={"rel": "a.png", "kind": "revised", "text": "1girl, solo, smile"},
    )
    assert c.get("/api/dataset").json()["items"][0]["captions"]["history"] is True
