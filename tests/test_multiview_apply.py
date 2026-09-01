"""The multiview audit's write path shares :func:`replay.apply_one`'s drift
ladder: the statuses it reports, and that a write is byte-identical to what a
``--from_report`` replay would write (``proposed + "\\n"``).
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
    src = tmp_path / "image_dataset"
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
        source_dir=src,
    )

    assert written == [("ok.txt", CAPTION, PROPOSED)]
    assert dict(skipped) == {
        "already-applied": 1,
        "drifted": 1,
        "missing-caption": 1,
        "no-proposal": 1,
    }
    # Byte-exact with the replay path: trailing newline, nothing else touched.
    assert (src / "ok.txt").read_text(encoding="utf-8") == PROPOSED + "\n"
    assert (src / "sub" / "moved.txt").read_text(encoding="utf-8") == (
        "1girl, solo, red eyes"
    )
    assert not (src / "gone.txt").exists()


def test_apply_findings_is_idempotent(tmp_path: Path) -> None:
    src = tmp_path / "image_dataset"
    src.mkdir()
    (src / "a.txt").write_text(CAPTION, encoding="utf-8")

    first, _ = apply_findings([_finding("a.txt")], source_dir=src)
    second, skipped = apply_findings([_finding("a.txt")], source_dir=src)

    assert len(first) == 1
    assert second == []
    assert dict(skipped) == {"already-applied": 1}
    assert (src / "a.txt").read_text(encoding="utf-8") == PROPOSED + "\n"
