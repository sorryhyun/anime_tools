"""``--from_report``: re-applying a stage's dry run without re-running its models.

1. Round trip: a replayed report writes byte-for-byte what the live apply would.
2. Drift is skipped, not clobbered — a caption edited between the passes is
   counted and left alone.
3. No model: the replay path never imports ``torch`` (pinned in a subprocess).
4. Roots must agree, and an already-applied report is refused.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anime_tools.stages.autotag import AutotagOptions, run_autotag_captions
from anime_tools.stages.cli.autotag_captions import REPLAY_SPEC as AUTOTAG_SPEC
from anime_tools.stages.cli.position_captions import REPLAY_SPEC as POSITION_SPEC
from anime_tools.stages.position_captions import ImageProposal
from anime_tools.stages.replay import (
    REPLAY_REPORT_NAME,
    StaleReportError,
    load_report,
    replay_rows,
    run_replay,
    validate_report,
)

TAGGED = "safe, 1girl, blue hair, smile"


# ---------------------------------------------------------------------------
# fixtures / builders
# ---------------------------------------------------------------------------


def _dataset(tmp_path: Path, images: dict[str, str | None]) -> tuple[Path, Path]:
    """``(resized_dir, source_dir)``; a ``None`` caption means no sidecar."""
    resized = tmp_path / "resized"
    source = tmp_path / "master"
    resized.mkdir(parents=True)
    source.mkdir(parents=True)
    for stem, caption in images.items():
        Image.new("RGB", (8, 8), (10, 20, 30)).save(resized / f"{stem}.png")
        if caption is not None:
            (source / f"{stem}.txt").write_text(caption, encoding="utf-8")
    return resized, source


def _autotag_dry_run(resized: Path, source: Path, *, mode: str = "missing") -> dict:
    """A real dry run, packaged into the report the CLI writes."""
    rows, stats = run_autotag_captions(
        resized_dir=resized,
        source_dir=source,
        tag_fn=lambda _img: TAGGED,
        options=AutotagOptions(mode=mode),
        apply=False,
    )
    assert stats.written == 0, "a dry run must not write"
    return {
        "mode": mode,
        "apply": False,
        "src": str(source),
        "dst": str(resized),
        "path_pattern": "*",
        "stats": {"proposed": stats.proposed, "written": stats.written},
        "rows": [asdict(r) for r in rows],
    }


def _position_report(
    resized: Path, source: Path, proposals: list[ImageProposal]
) -> dict:
    """A position-shaped report built from the real dataclass."""
    return {
        "summary": {
            "applied": False,
            "src": str(source),
            "dst": str(resized),
            "path_pattern": "*",
        },
        "images": [asdict(p) for p in proposals],
    }


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------


def test_autotag_round_trip_writes_exactly_the_proposed_text(tmp_path: Path):
    resized, source = _dataset(tmp_path, {"a": None, "b": None})
    report = _autotag_dry_run(resized, source)
    assert not (source / "a.txt").exists()

    rows, stats = replay_rows(
        report, spec=AUTOTAG_SPEC, src=source, dst=resized, apply=True
    )

    assert stats.written == 2
    assert not stats.skipped
    assert {r.status for r in rows} == {"written"}
    for stem in ("a", "b"):
        assert (source / f"{stem}.txt").read_text(encoding="utf-8") == TAGGED


def test_replay_matches_a_live_apply_byte_for_byte(tmp_path: Path):
    """The replay writes the bytes the live apply would have written."""
    live_resized, live_source = _dataset(tmp_path / "live", {"a": None})
    run_autotag_captions(
        resized_dir=live_resized,
        source_dir=live_source,
        tag_fn=lambda _img: TAGGED,
        options=AutotagOptions(mode="missing"),
        apply=True,
    )

    replay_resized, replay_source = _dataset(tmp_path / "replay", {"a": None})
    report = _autotag_dry_run(replay_resized, replay_source)
    replay_rows(
        report, spec=AUTOTAG_SPEC, src=replay_source, dst=replay_resized, apply=True
    )

    assert (replay_source / "a.txt").read_bytes() == (
        live_source / "a.txt"
    ).read_bytes()


def test_merge_mode_round_trips_a_caption_with_clauses(tmp_path: Path):
    """The replay carries composed text, so clause structure survives untouched."""
    existing = "safe, 2girls. On the left, akita neru. On the right, kasane teto."
    resized, source = _dataset(tmp_path, {"a": existing})
    report = _autotag_dry_run(resized, source, mode="merge")

    replay_rows(report, spec=AUTOTAG_SPEC, src=source, dst=resized, apply=True)

    written = (source / "a.txt").read_text(encoding="utf-8")
    assert written == report["rows"][0]["proposed"]
    assert written.endswith("On the left, akita neru. On the right, kasane teto.")


# ---------------------------------------------------------------------------
# staleness
# ---------------------------------------------------------------------------


def test_drifted_caption_is_skipped_and_counted(tmp_path: Path):
    resized, source = _dataset(tmp_path, {"a": "safe, 1girl", "b": "safe, 1girl"})
    report = _autotag_dry_run(resized, source, mode="merge")

    # A hand edit lands between the dry run and the apply.
    (source / "a.txt").write_text("safe, 1girl, hand edited", encoding="utf-8")

    rows, stats = replay_rows(
        report, spec=AUTOTAG_SPEC, src=source, dst=resized, apply=True
    )

    assert stats.written == 1
    assert stats.skipped["drifted"] == 1
    drifted = next(r for r in rows if r.image == "a.png")
    assert drifted.status == "skip:drifted"
    # Untouched — the hand edit survives.
    assert (source / "a.txt").read_text(encoding="utf-8") == "safe, 1girl, hand edited"
    assert (source / "b.txt").read_text(encoding="utf-8") == report["rows"][1][
        "proposed"
    ]


def test_replay_is_idempotent(tmp_path: Path):
    """Re-running a replay reports already-applied rather than rewriting."""
    resized, source = _dataset(tmp_path, {"a": None})
    report = _autotag_dry_run(resized, source)

    replay_rows(report, spec=AUTOTAG_SPEC, src=source, dst=resized, apply=True)
    rows, stats = replay_rows(
        report, spec=AUTOTAG_SPEC, src=source, dst=resized, apply=True
    )

    assert stats.written == 0
    assert stats.skipped["already-applied"] == 1
    assert rows[0].status == "skip:already-applied"


def test_deleted_caption_that_the_dry_run_saw_is_skipped(tmp_path: Path):
    resized, source = _dataset(tmp_path, {"a": "safe, 1girl"})
    report = _autotag_dry_run(resized, source, mode="merge")
    (source / "a.txt").unlink()

    rows, stats = replay_rows(
        report, spec=AUTOTAG_SPEC, src=source, dst=resized, apply=True
    )

    assert stats.written == 0
    assert stats.skipped["missing-caption"] == 1
    assert rows[0].status == "skip:missing-caption"
    assert not (source / "a.txt").exists()


def test_applied_report_is_refused(tmp_path: Path):
    resized, source = _dataset(tmp_path, {"a": None})
    report = _autotag_dry_run(resized, source)
    report["apply"] = True

    with pytest.raises(StaleReportError, match="--apply"):
        validate_report(report, spec=AUTOTAG_SPEC, src=source, dst=resized)


def test_root_mismatch_is_refused(tmp_path: Path):
    resized, source = _dataset(tmp_path, {"a": None})
    report = _autotag_dry_run(resized, source)
    other = tmp_path / "elsewhere"
    other.mkdir()

    with pytest.raises(StaleReportError, match="refusing to replay across trees"):
        validate_report(report, spec=AUTOTAG_SPEC, src=other, dst=resized)


def test_report_without_recorded_roots_is_refused(tmp_path: Path):
    resized, source = _dataset(tmp_path, {"a": None})
    report = _autotag_dry_run(resized, source)
    del report["dst"]

    with pytest.raises(StaleReportError, match="predates"):
        validate_report(report, spec=AUTOTAG_SPEC, src=source, dst=resized)


def test_wrong_shape_report_is_refused(tmp_path: Path):
    resized, source = _dataset(tmp_path, {"a": None})
    position_shaped = _position_report(resized, source, [])

    with pytest.raises(StaleReportError, match="autotag_captions report"):
        replay_rows(position_shaped, spec=AUTOTAG_SPEC, src=source, dst=resized)


# ---------------------------------------------------------------------------
# dry replay, filtering, report shape
# ---------------------------------------------------------------------------


def test_replay_without_apply_writes_nothing(tmp_path: Path):
    resized, source = _dataset(tmp_path, {"a": None})
    report = _autotag_dry_run(resized, source)

    rows, stats = replay_rows(
        report, spec=AUTOTAG_SPEC, src=source, dst=resized, apply=False
    )

    assert stats.would_write == 1 and stats.written == 0
    assert rows[0].status == "would-write"
    assert not (source / "a.txt").exists()


def test_path_pattern_filters_a_replay(tmp_path: Path):
    resized, source = _dataset(tmp_path, {"a": None, "b": None})
    (resized / "sub").mkdir()
    Image.new("RGB", (8, 8)).save(resized / "sub" / "c.png")
    report = _autotag_dry_run(resized, source)
    assert len(report["rows"]) == 3

    _, stats = replay_rows(
        report,
        spec=AUTOTAG_SPEC,
        src=source,
        dst=resized,
        path_pattern="sub/*",
        apply=True,
    )

    assert stats.written == 1
    assert stats.skipped["filtered"] == 2
    assert (source / "sub" / "c.txt").exists()
    assert not (source / "a.txt").exists()


def test_replay_report_names_the_written_images(tmp_path: Path):
    """``written`` names the images a UI should reload."""
    resized, source = _dataset(tmp_path, {"a": None, "b": "safe, 1girl"})
    report = _autotag_dry_run(resized, source)
    report_dir = tmp_path / "reports"
    report_path = report_dir / "report.json"
    report_dir.mkdir()
    report_path.write_text(json.dumps(report), encoding="utf-8")

    _, stats, out_path = run_replay(
        spec=AUTOTAG_SPEC,
        report_path=report_path,
        src=source,
        dst=resized,
        report_dir=report_dir,
        apply=True,
    )

    assert out_path.name == REPLAY_REPORT_NAME
    # The dry run's own report survives, so a re-run stays possible.
    assert load_report(report_path)["apply"] is False

    emitted = json.loads(out_path.read_text(encoding="utf-8"))
    assert emitted["written"] == ["a.png"]
    assert emitted["stats"]["written"] == stats.written == 1
    assert emitted["stats"]["from_report"] == str(report_path)
    assert emitted["rows"][0]["image"] == "a.png"
    assert emitted["rows"][0]["caption_path"] == "a.txt"


# ---------------------------------------------------------------------------
# position clauses: derived tree + variants sidecar
# ---------------------------------------------------------------------------


def _clause_proposal(stem: str, original: str, proposed: str) -> ImageProposal:
    return ImageProposal(
        image=f"{stem}.png",
        caption_path=f"{stem}.txt",
        status="proposed",
        original=original,
        proposed=proposed,
    )


def test_position_replay_writes_the_derived_caption_not_the_master(tmp_path: Path):
    original = "safe, 2girls, blue hair"
    proposed = "safe, 2girls. On the left, blue hair."
    resized, source = _dataset(tmp_path, {"a": None})
    (resized / "a.txt").write_text(original, encoding="utf-8")
    (source / "a.txt").write_text(original, encoding="utf-8")
    report = _position_report(
        resized, source, [_clause_proposal("a", original, proposed)]
    )

    _, stats = replay_rows(
        report, spec=POSITION_SPEC, src=source, dst=resized, apply=True
    )

    assert stats.written == 1
    assert (resized / "a.txt").read_text(encoding="utf-8") == proposed
    # GOTCHA the live pass shares: the master is never written by this stage.
    assert (source / "a.txt").read_text(encoding="utf-8") == original


def test_position_replay_drops_the_stale_variants_sidecar(tmp_path: Path):
    """The sidecar outranks ``{stem}.txt`` at encode time, so a stale one is dropped."""
    original = "safe, 2girls, blue hair"
    proposed = "safe, 2girls. On the left, blue hair."
    resized, source = _dataset(tmp_path, {"a": None})
    (resized / "a.txt").write_text(original, encoding="utf-8")
    sidecar = resized / "a.variants.txt"
    sidecar.write_text(f"v0\t{original}\n", encoding="utf-8")
    report = _position_report(
        resized, source, [_clause_proposal("a", original, proposed)]
    )

    replay_rows(report, spec=POSITION_SPEC, src=source, dst=resized, apply=True)

    assert not sidecar.exists()


def test_position_replay_keeps_the_sidecar_on_a_dry_replay(tmp_path: Path):
    original = "safe, 2girls, blue hair"
    resized, source = _dataset(tmp_path, {"a": None})
    (resized / "a.txt").write_text(original, encoding="utf-8")
    sidecar = resized / "a.variants.txt"
    sidecar.write_text(f"v0\t{original}\n", encoding="utf-8")
    report = _position_report(
        resized,
        source,
        [_clause_proposal("a", original, "safe, 2girls. On the left, blue hair.")],
    )

    replay_rows(report, spec=POSITION_SPEC, src=source, dst=resized, apply=False)

    assert sidecar.exists()


def test_position_replay_skips_a_non_proposed_row(tmp_path: Path):
    resized, source = _dataset(tmp_path, {"a": None})
    (resized / "a.txt").write_text("safe, 1girl", encoding="utf-8")
    skipped = _clause_proposal("a", "safe, 1girl", "")
    skipped.status = "skip:single-subject"
    report = _position_report(resized, source, [skipped])

    rows, stats = replay_rows(
        report, spec=POSITION_SPEC, src=source, dst=resized, apply=True
    )

    assert rows == [] and stats.written == 0
    assert stats.skipped["not-writable"] == 1
    assert (resized / "a.txt").read_text(encoding="utf-8") == "safe, 1girl"


# ---------------------------------------------------------------------------
# the point of the whole thing: no model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        "anime_tools.stages.cli.autotag_captions",
        "anime_tools.stages.cli.position_captions",
        "anime_tools.stages.cli.audit_multiview",
    ],
)
def test_replay_cli_does_not_import_torch(tmp_path: Path, module: str, repo_root: Path):
    """End-to-end through ``main()``: ``--from_report --apply`` writes the caption
    and never imports torch."""
    import subprocess

    position = module.endswith("position_captions")
    resized = tmp_path / "resized"
    source = tmp_path / "master"
    resized.mkdir()
    source.mkdir()
    Image.new("RGB", (8, 8)).save(resized / "a.png")
    target = (resized if position else source) / "a.txt"
    target.write_text("safe, 2girls, blue hair", encoding="utf-8")
    proposed = "safe, 2girls. On the left, blue hair."

    if position:
        report = {
            "summary": {
                "applied": False,
                "src": str(source),
                "dst": str(resized),
            },
            "images": [
                asdict(_clause_proposal("a", "safe, 2girls, blue hair", proposed))
            ],
        }
    else:
        report = {
            "apply": False,
            "src": str(source),
            "dst": str(resized),
            "images" if module.endswith("audit_multiview") else "rows": [
                {
                    "image": "a.png",
                    "caption_path": "a.txt",
                    "status": "ok",
                    "verdict": "multiple views",
                    "confidence": "strong",
                    "existing": "safe, 2girls, blue hair",
                    "caption": "safe, 2girls, blue hair",
                    "proposed": proposed,
                }
            ],
        }
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    report_path = report_dir / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    code = (
        "import sys\n"
        f"import {module} as m\n"
        "sys.argv = ['x', '--apply', '--from_report', %r,\n"
        "            '--src', %r, '--dst', %r, '--report_dir', %r]\n"
        "m.main()\n"
        "assert 'torch' not in sys.modules, 'torch imported by the replay path'\n"
    ) % (str(report_path), str(source), str(resized), str(report_dir))
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        check=False,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert target.read_text(encoding="utf-8").strip() == proposed
    emitted = json.loads((report_dir / REPLAY_REPORT_NAME).read_text(encoding="utf-8"))
    assert emitted["written"] == ["a.png"]
