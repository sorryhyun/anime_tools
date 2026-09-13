"""Caption correction helpers for preprocessing outputs.

:func:`run_correct` is the stage runner over
:class:`~anime_tools.stages.requests.CorrectRequest`; ``cli/correct_captions.py``
is the shell over it.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from anime_tools._env import curation_home, resolve_path
from anime_tools.captions.correction import (
    CaptionCorrectionOptions,
    TagKnowledgeBase,
    correct_caption,
    drop_caption_groups,
    find_tag_csv,
)
from anime_tools.captions.position_clauses import has_clauses
from anime_tools.captions.taxonomy import normalize_tag
from anime_tools.captions.variants import (
    build_erasure_token_pool,
    generate_caption_variants,
    read_variants_sidecar,
    variants_sidecar_path,
    write_variants_sidecar,
)

from ._caption_io import read_caption, write_caption
from ._report import print_dry_run_footer, stage_report_header, write_stage_report
from ._walk_captions import iter_captions
from .resize import resized_tree

if TYPE_CHECKING:
    from anime_tools.stages.requests import CorrectRequest


@dataclass
class PreprocessCaptionStats:
    seen: int = 0
    written: int = 0
    unchanged: int = 0
    no_caption: int = 0
    """Images with neither a revised nor a master caption."""
    from_master: int = 0
    """Captions read from the master because no revised caption existed yet."""
    variants_written: int = 0
    variants_removed: int = 0
    clauses_preserved: int = 0
    """Captions that carried position clauses through the correction."""

    def skip(self, reason: str) -> None:
        """:func:`iter_captions`'s half of the walk. It reports exactly one
        reason, ``no-caption``; the corrector's own skips are statuses on the
        row, not counters here."""
        if reason == "no-caption":
            self.no_caption += 1


@dataclass
class CorrectionProposal:
    """One image's before/after, whether or not it was written.

    Same row shape as :class:`~anime_tools.stages.autotag.AutotagProposal`, and
    for the same reason: ``target_before`` is the *write target*'s own text
    (empty when it does not exist yet), which is the drift baseline a replay
    checks, while ``existing`` is what spoke for the image — the master, for an
    image the corrector is mirroring for the first time.
    """

    image: str = ""
    caption_path: str = ""
    existing: str = ""
    target_before: str = ""
    proposed: str = ""
    status: str = "ok"


@dataclass
class PreprocessCaptionResult:
    """What :func:`write_corrected_preprocess_captions` reports back."""

    stats: PreprocessCaptionStats
    rows: list[CorrectionProposal]


def _resolve_n_rand(num_variants: int, tag_randomize_rate: float) -> int:
    """Size of the identity-randomized r-family that rides alongside v0..v{N-1}.

    The r-family shares v0 as its anchor, so it carries ``N-1`` entries and only
    exists with >=2 variants. Must match the TE writer, or its sidecar lines
    miss the ``prompt_embeds_r*`` keys the loader expects.
    """
    return (num_variants - 1) if (tag_randomize_rate > 0.0 and num_variants >= 2) else 0


def _build_variant_rows(
    corrected: str,
    *,
    num_variants: int,
    tag_dropout_rate: float,
    tag_randomize_rate: float,
    erasure_pool: Collection[str] | None,
    protect_fn: Callable[[str], bool] | None,
) -> list[tuple[str, str]]:
    """``(label, text)`` rows for one image: v0..v{N-1} then r1..r{n_rand}.

    v0 is the *corrected* caption (the anchor that also lives in ``{stem}.txt``);
    the r-family drops its own v0 since it equals that shared anchor.
    """
    rows: list[tuple[str, str]] = []
    v_variants = generate_caption_variants(
        corrected,
        num_variants,
        tag_dropout_rate,
        protect_fn,
    )
    for i, text in enumerate(v_variants):
        rows.append((f"v{i}", text))

    n_rand = _resolve_n_rand(num_variants, tag_randomize_rate)
    if n_rand:
        r_variants = generate_caption_variants(
            corrected,
            num_variants,
            tag_dropout_rate,
            protect_fn,
            tag_randomize_rate=tag_randomize_rate,
            erasure_pool=erasure_pool,
        )
        for j, text in enumerate(r_variants[1:], start=1):  # skip shared v0 anchor
            rows.append((f"r{j}", text))
    return rows


def _sidecar_is_current(
    path: Path, corrected: str, num_variants: int, n_rand: int
) -> bool:
    """True iff an on-disk variant sidecar already matches what we'd generate:
    pristine v0 equals the corrected caption *and* the v/r counts match.

    The draws are stochastic, so rewriting every run would bump the sidecar
    mtime and force a needless TE re-encode. ``v0`` is the corrected caption
    minus the ``@no-artist`` sentinel (the generator strips it from every
    variant), so the comparison is against that, not the raw caption — else a
    caption carrying the sentinel rewrote its sidecar on every run.
    """
    if not path.exists():
        return False
    try:
        rows = read_variants_sidecar(path)
    except OSError:
        return False
    labels = [label for label, _ in rows]
    n_v = sum(1 for label in labels if label.startswith("v"))
    n_r = sum(1 for label in labels if label.startswith("r"))
    if n_v != num_variants or n_r != n_rand:
        return False
    v0 = next((text for label, text in rows if label == "v0"), None)
    return v0 == generate_caption_variants(corrected, 1, 0.0, None)[0]


def write_corrected_preprocess_captions(
    source_dir: Path,
    resized_dir: Path,
    kb: TagKnowledgeBase,
    *,
    options: CaptionCorrectionOptions,
    recursive: bool = True,
    path_pattern: str | None = None,
    correct: bool = True,
    num_variants: int = 0,
    tag_dropout_rate: float = 0.0,
    tag_randomize_rate: float = 0.0,
    qwen3_tokenizer=None,
    t5_tokenizer=None,
    protect_fn: Callable[[str], bool] | None = None,
    apply: bool = True,
) -> PreprocessCaptionResult:
    """Write ``.txt`` captions next to already-resized images.

    The resized tree is the authority over which images are visited, and the
    caption read for each is the **revised** one (``resized_dir / rel``) when it
    exists, the master under ``source_dir`` otherwise — the same rule every
    other caption stage follows (:func:`resolve_caption`). Correcting the
    revised caption in place is what keeps the tags autotag merged and the
    clauses the position rewrite bound: :func:`correct_caption` reorders the
    flat bag around the clauses. The master is never modified, and once an image
    has a revised caption a hand-edit of its master no longer reaches it — the
    revised caption is the one to edit (or delete, to re-mirror).

    ``correct`` (default True) bucket-reorders each caption; ``correct=False``
    mirrors the raw source caption verbatim, less ``options.drop_groups`` —
    a drop is not a reorder, so turning the reorder off must not turn it off
    too. Either way v0 lands in ``{stem}.txt`` and anchors the variant sidecar.

    With ``num_variants > 0`` each image also gets a ``{stem}.variants.txt``
    sidecar the TE step encodes verbatim: v0 the corrected caption, v1..v{N-1}
    shuffled (+ tag-dropped at ``tag_dropout_rate``), and under
    ``tag_randomize_rate > 0`` an r-family with per-tag identity erasure, which
    requires both tokenizers for the dual-single erasure pool.

    ``apply=False`` is the dry run: every caption is corrected and every row is
    reported, but nothing on disk is touched — not the caption, not the variant
    sidecar, not the orphan sidecar of an image that lost its caption. The
    default is True because the in-process callers (the GUI, the trainer's
    wrapper) always apply; the CLI passes ``req.apply``.
    """

    stats = PreprocessCaptionStats()

    # First pass, collected up front (captions are tiny) so the erasure pool can
    # exclude the full real-tag set before any variant is drawn.
    @dataclass
    class _Entry:
        dst: Path
        corrected: str
        row: CorrectionProposal

    entries: list[_Entry] = []
    rows: list[CorrectionProposal] = []

    def _orphaned(image_path: Path, rel_caption: Path) -> None:
        """An image with no caption of either kind: report the row, and drop the
        variant sidecar the caption it was drawn from no longer backs."""
        dst_caption = resized_dir / rel_caption
        rows.append(
            CorrectionProposal(
                image=str(image_path.relative_to(resized_dir)),
                caption_path=str(rel_caption),
                target_before="",
                status="skip:no-caption",
            )
        )
        sidecar = variants_sidecar_path(dst_caption)
        if sidecar.exists():
            if apply:
                sidecar.unlink()
            stats.variants_removed += 1

    for image_path, rel_caption, dst_caption, raw, caption_path in iter_captions(
        resized_dir,
        source_dir,
        path_pattern,
        stats,
        recursive=recursive,
        missing=_orphaned,
    ):
        if caption_path != dst_caption:
            stats.from_master += 1

        if correct:
            corrected = correct_caption(raw, kb, options=options).text
        else:
            corrected = drop_caption_groups(
                raw, kb, options.drop_groups, keep=(options.trigger_word,)
            ).text
        if has_clauses(corrected):
            stats.clauses_preserved += 1
        rows.append(
            row := CorrectionProposal(
                image=str(image_path.relative_to(resized_dir)),
                caption_path=str(rel_caption),
                target_before=read_caption(dst_caption) if dst_caption.exists() else "",
                existing=raw,
                proposed=corrected,
            )
        )
        entries.append(_Entry(dst_caption, corrected, row))

    n_rand = _resolve_n_rand(num_variants, tag_randomize_rate)
    erasure_pool: list[str] | None = None
    if n_rand:
        if qwen3_tokenizer is None or t5_tokenizer is None:
            raise ValueError(
                "tag_randomize_rate > 0 requires qwen3_tokenizer and t5_tokenizer "
                "(load them tokenizer-only) to build the dual-single erasure pool."
            )
        # Split rather than parsed: the pool only cares about the tag *set*,
        # clauses included. Keyed on ``normalize_tag`` like every other tag key.
        real_tags = {
            key
            for e in entries
            for t in e.corrected.split(",")
            if (key := normalize_tag(t))
        }
        erasure_pool = build_erasure_token_pool(
            qwen3_tokenizer, t5_tokenizer, exclude=real_tags
        )
        if not erasure_pool:
            raise ValueError(
                "Identity-randomize requested but the erasure-token pool is empty "
                "(tokenizers lack the expected API or no qualifying tokens)."
            )

    for e in entries:
        if e.dst.exists() and read_caption(e.dst) == e.corrected:
            stats.unchanged += 1
            e.row.status = "skip:unchanged"
        else:
            # No ``drop_variants``: the sidecar is this pass's own output and
            # is rebuilt (or removed) a few lines down.
            if apply:
                write_caption(e.dst, e.corrected, history_by="correct")
            stats.written += 1

        sidecar = variants_sidecar_path(e.dst)
        if num_variants > 0:
            if not _sidecar_is_current(sidecar, e.corrected, num_variants, n_rand):
                variant_rows = _build_variant_rows(
                    e.corrected,
                    num_variants=num_variants,
                    tag_dropout_rate=tag_dropout_rate,
                    tag_randomize_rate=tag_randomize_rate,
                    erasure_pool=erasure_pool,
                    protect_fn=protect_fn,
                )
                if apply:
                    write_variants_sidecar(sidecar, variant_rows)
                stats.variants_written += 1
        elif sidecar.exists():
            # Variants turned off → drop the now-stale sidecar.
            if apply:
                sidecar.unlink()
            stats.variants_removed += 1

    return PreprocessCaptionResult(stats=stats, rows=rows)


TE_NOTE = (
    "\nWritten to the resized captions (the master is untouched). Run "
    "`make preprocess-te` now to re-encode."
)


def resolve_tag_csv(tag_csv: str | None) -> Path:
    """The Danbooru tag KB a caption stage types tags against: ``tag_csv`` when
    given, the curation home's lookup otherwise. A missing one is a
    ``FileNotFoundError`` naming the download, which each shell exits with."""
    csv_path = resolve_path(tag_csv) if tag_csv else find_tag_csv(curation_home())
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(
            "danbooru_tags_classified.csv not found. Run "
            "`python -m anime_tools.downloads danbooru_tags` first "
            "(or the GUI's Settings > Models > Danbooru tag KB)."
        )
    return csv_path


def run_correct(req: CorrectRequest):
    """Correct the revised captions in place — mirroring the master for an image
    that has none yet — plus variant sidecars. Returns ``(rows, stats)``.

    No ``--from_report``: correction is pure text, so re-running it is cheaper
    than the machinery to skip it. The report it leaves is still what the GUI's
    Undo reads (``contract.REPLAY_SHAPES["correct"]``).
    """
    from anime_tools.captions.correction import load_tag_knowledge_base
    from anime_tools.captions.tag_drop_groups import parse_drop_groups

    # Home-anchored like every other runner: a bare ``workspace/resized`` must
    # name the curation home's tree, not the shell's cwd.
    src = resolve_path(req.src)
    dst = resized_tree(req.dst)
    csv_path = resolve_tag_csv(req.tag_csv)

    # The erasure pool (identity-randomize only) needs both tokenizers, loaded
    # tokenizer-only — no encoder weights.
    qwen3_tokenizer = t5_tokenizer = None
    if req.randomizes:
        from anime_tools.captions.tokenizers import (
            load_qwen3_tokenizer_from_dir,
            load_t5_tokenizer_from_dir,
        )

        qwen3_tokenizer = load_qwen3_tokenizer_from_dir(req.qwen3)
        t5_tokenizer = load_t5_tokenizer_from_dir(req.t5_tokenizer_path)

    result = write_corrected_preprocess_captions(
        src,
        dst,
        load_tag_knowledge_base(csv_path),
        options=CaptionCorrectionOptions(
            insert_no_artist=req.caption_insert_no_artist,
            trigger_word=req.caption_trigger_word,
            trigger_at_front=req.caption_trigger_at_front,
            drop_groups=parse_drop_groups(req.caption_drop_groups),
        ),
        recursive=req.recursive,
        path_pattern=req.path_pattern or "*",
        correct=not req.no_correct,
        num_variants=req.caption_shuffle_variants,
        tag_dropout_rate=req.caption_tag_dropout_rate,
        tag_randomize_rate=req.caption_tag_randomize_rate,
        qwen3_tokenizer=qwen3_tokenizer,
        t5_tokenizer=t5_tokenizer,
        apply=req.apply,
    )
    stats, rows = result.stats, result.rows

    report_path = write_stage_report(
        resolve_path(req.report_dir),
        {
            **stage_report_header(
                src=src, dst=dst, path_pattern=req.path_pattern, apply=req.apply
            ),
            "correct": not req.no_correct,
            "stats": {
                "seen": stats.seen,
                "written": stats.written,
                "unchanged": stats.unchanged,
                "from_master": stats.from_master,
                "no_caption": stats.no_caption,
                "variants_written": stats.variants_written,
                "variants_removed": stats.variants_removed,
                "clauses_preserved": stats.clauses_preserved,
            },
            "rows": [asdict(r) for r in rows],
        },
    )

    verb = "written" if req.apply else "would write"
    print(
        "Corrected preprocess captions: "
        f"{stats.written} {verb}, {stats.unchanged} unchanged, "
        f"{stats.from_master} mirrored from the master, "
        f"{stats.no_caption} without a caption, "
        f"{stats.variants_written} variant sidecars, "
        f"{stats.clauses_preserved} position clauses kept "
        f"({stats.seen} resized images)"
    )
    print(f"report: {report_path}")
    print_dry_run_footer(req.apply, TE_NOTE if stats.written else None)
    return rows, stats
