"""The multiview audit's write path shares :func:`replay.apply_one`'s drift
ladder: the statuses it reports, and that it writes the REVISED caption — never
the hand-written master — byte-identically to a ``--from_report`` replay.
"""

from __future__ import annotations

from pathlib import Path

from anime_tools.stages.multiview_audit import (
    MULTIPLE_VIEWS,
    MultiviewFinding,
    apply_findings,
)

CAPTION = "1girl, solo, blue eyes"
PROPOSED = f"1girl, solo, blue eyes, {MULTIPLE_VIEWS}"


def _finding(rel: str, *, caption: str = CAPTION, proposed: str = PROPOSED):
    return MultiviewFinding(
        image=rel.replace(".txt", ".png"),
        caption_path=rel,
        instances=2,
        girls=1,
        boys=None,
        verdict=MULTIPLE_VIEWS,
        confidence="strong",
        caption=caption,
        proposed=proposed,
    )


def test_apply_findings_writes_and_reports_each_skip(tmp_path: Path) -> None:
    src = tmp_path / "resized"
    (src / "sub").mkdir(parents=True)
    (src / "ok.txt").write_text(CAPTION, encoding="utf-8")
    (src / "sub" / "done.txt").write_text(PROPOSED, encoding="utf-8")
    (src / "sub" / "moved.txt").write_text("1girl, solo, red eyes", encoding="utf-8")

    written, skipped = apply_findings(
        [
            _finding("ok.txt"),
            _finding("sub/done.txt"),
            _finding("sub/moved.txt"),
            _finding("gone.txt"),
            _finding("nothing.txt", proposed=CAPTION),
            # Gated out before the ladder runs, so it is not counted at all.
            MultiviewFinding(
                image="weak.png",
                caption_path="weak.txt",
                instances=2,
                girls=1,
                boys=None,
                verdict=MULTIPLE_VIEWS,
                confidence="weak",
                caption=CAPTION,
                proposed=PROPOSED,
            ),
        ],
        resized_dir=src,
    )

    assert written == [("ok.txt", CAPTION, PROPOSED)]
    assert dict(skipped) == {
        "already-applied": 1,
        "drifted": 1,
        "missing-caption": 1,
        "no-proposal": 1,
    }
    # Byte-exact with the replay path, and with every other revised writer:
    # no trailing newline, nothing else touched.
    assert (src / "ok.txt").read_text(encoding="utf-8") == PROPOSED
    assert (src / "sub" / "moved.txt").read_text(encoding="utf-8") == (
        "1girl, solo, red eyes"
    )
    assert not (src / "gone.txt").exists()


def test_apply_findings_is_idempotent(tmp_path: Path) -> None:
    src = tmp_path / "resized"
    src.mkdir()
    (src / "a.txt").write_text(CAPTION, encoding="utf-8")

    first, _ = apply_findings([_finding("a.txt")], resized_dir=src)
    second, skipped = apply_findings([_finding("a.txt")], resized_dir=src)

    assert len(first) == 1
    assert second == []
    assert dict(skipped) == {"already-applied": 1}
    assert (src / "a.txt").read_text(encoding="utf-8") == PROPOSED


def test_apply_findings_never_touches_the_master(tmp_path: Path) -> None:
    """The master is hand-written; a machine verdict off a few crops does not
    get to edit it. It is also read past — resolve_caption is revised-first."""
    master, revised = tmp_path / "image_dataset", tmp_path / "resized"
    master.mkdir()
    revised.mkdir()
    (master / "a.txt").write_text(CAPTION, encoding="utf-8")
    (revised / "a.txt").write_text(CAPTION, encoding="utf-8")

    apply_findings([_finding("a.txt")], resized_dir=revised)

    assert (master / "a.txt").read_text(encoding="utf-8") == CAPTION
    assert (revised / "a.txt").read_text(encoding="utf-8") == PROPOSED


def test_a_write_keeps_the_replaced_text_and_drops_the_stale_sidecar(
    tmp_path: Path,
) -> None:
    """Both are consequences of writing the revised tree: `.variants.txt` wins
    over `.txt` at encode time, and `.history.txt` is the undo the master write
    never had."""
    revised = tmp_path / "resized"
    revised.mkdir()
    (revised / "a.txt").write_text(CAPTION, encoding="utf-8")
    (revised / "a.variants.txt").write_text(CAPTION, encoding="utf-8")

    apply_findings([_finding("a.txt")], resized_dir=revised)

    assert not (revised / "a.variants.txt").exists()
    assert CAPTION in (revised / "a.history.txt").read_text(encoding="utf-8")


# ----- the gate, shared by the write path and the position phase ------------


def test_promotions_and_apply_findings_admit_the_same_findings() -> None:
    """One predicate for both: a threshold that drifted between them would make
    the same corpus give two answers depending on which one ran."""
    from anime_tools.stages.multiview_audit import EXTRA_CHARACTER, promotions

    weak = MultiviewFinding(
        image="weak.png",
        caption_path="weak.txt",
        instances=2,
        girls=1,
        boys=None,
        verdict=MULTIPLE_VIEWS,
        confidence="weak",
        caption=CAPTION,
        proposed=PROPOSED,
    )
    extra = MultiviewFinding(
        image="extra.png",
        caption_path="extra.txt",
        instances=2,
        girls=1,
        boys=None,
        verdict=EXTRA_CHARACTER,
        confidence="strong",
        caption=CAPTION,
        proposed=PROPOSED,
    )
    rows = [_finding("ok.txt"), weak, extra]

    assert promotions(rows) == {"ok.txt": PROPOSED}
    assert promotions(rows, confidences=("strong", "weak")) == {
        "ok.txt": PROPOSED,
        "weak.txt": PROPOSED,
    }
    assert promotions(rows, verdicts=(MULTIPLE_VIEWS, EXTRA_CHARACTER)) == {
        "ok.txt": PROPOSED,
        "extra.txt": PROPOSED,
    }


def test_promotions_touches_no_file_and_drops_an_empty_proposal(tmp_path: Path) -> None:
    """A dry run builds the same map an apply would, so the report is the plan —
    and an empty proposal would promote the image to an empty caption."""
    from anime_tools.stages.multiview_audit import promotions

    src = tmp_path / "image_dataset"
    src.mkdir()
    (src / "a.txt").write_text(CAPTION, encoding="utf-8")

    assert promotions([_finding("a.txt"), _finding("b.txt", proposed="")]) == {
        "a.txt": PROPOSED
    }
    assert (src / "a.txt").read_text(encoding="utf-8") == CAPTION


# ----- the phase the position stage runs first ------------------------------


def _phase(tmp_path, monkeypatch, mode, rows):
    """``_run_audit_phase`` with the sweep stubbed — the orchestration only."""
    import json

    from anime_tools.stages import multiview_audit as audit_mod
    from anime_tools.stages.requests import PositionRequest
    from anime_tools.stages.run import _run_audit_phase

    seen = {}

    def fake_sweep(**kw):
        seen.update(kw)
        return list(rows), audit_mod.MultiviewAuditStats(
            seen=9, audited=4, findings=len(rows)
        )

    monkeypatch.setattr(audit_mod, "run_multiview_audit", fake_sweep)

    class _Tagger:
        def predict(self, crop):  # pragma: no cover - never called
            return {}

    report_dir = tmp_path / "reports"
    out = _run_audit_phase(
        PositionRequest(multiview_audit=mode),
        src=tmp_path / "image_dataset",
        dst=tmp_path / "resized",
        report_dir=report_dir,
        detect_fn=None,
        part_detect_fn=None,
        tagger=_Tagger(),
        vocabulary=None,
    )
    written = report_dir / "audit" / "audit_report.json"
    payload = (
        json.loads(written.read_text(encoding="utf-8")) if written.exists() else None
    )
    return (*out, seen, payload)


def test_the_phase_is_off_by_default_and_runs_nothing(tmp_path, monkeypatch) -> None:
    rows, stats, promoted, seen, payload = _phase(
        tmp_path, monkeypatch, "off", [_finding("a.txt")]
    )
    assert (rows, stats, promoted) == ([], None, {})
    assert seen == {} and payload is None


def test_report_mode_sweeps_and_reports_but_promotes_nothing(
    tmp_path, monkeypatch
) -> None:
    rows, stats, promoted, _seen, payload = _phase(
        tmp_path, monkeypatch, "report", [_finding("a.txt")]
    )
    assert len(rows) == 1 and stats.findings == 1
    assert promoted == {}
    assert payload["summary"]["mode"] == "report"
    assert payload["summary"]["promoted"] == 0
    assert payload["images"][0]["caption_path"] == "a.txt"


def test_apply_mode_promotes_through_the_gate(tmp_path, monkeypatch) -> None:
    weak = MultiviewFinding(
        image="weak.png",
        caption_path="weak.txt",
        instances=2,
        girls=1,
        boys=None,
        verdict=MULTIPLE_VIEWS,
        confidence="weak",
        caption=CAPTION,
        proposed=PROPOSED,
    )
    _rows, _stats, promoted, seen, payload = _phase(
        tmp_path, monkeypatch, "apply", [_finding("a.txt"), weak]
    )
    # The default gate is strong-only, so the weak finding is reported but not fed
    # to the sweep.
    assert promoted == {"a.txt": PROPOSED}
    assert payload["summary"]["promoted"] == 1
    # min_instances pinned to 2, and the sheets/report land under audit/.
    assert seen["options"].min_instances == 2
    assert seen["sheets_dir"] == tmp_path / "reports" / "audit" / "sheets"
