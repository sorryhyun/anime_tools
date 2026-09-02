"""Caption correction helpers for preprocessing outputs."""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path

from anime_tools._walk import walk_images
from anime_tools.captions.correction import (
    CaptionCorrectionOptions,
    TagKnowledgeBase,
    correct_caption,
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
from ._walk_captions import resolve_caption


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
) -> PreprocessCaptionStats:
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
    mirrors the raw source caption verbatim. Either way v0 lands in
    ``{stem}.txt`` and anchors the variant sidecar.

    With ``num_variants > 0`` each image also gets a ``{stem}.variants.txt``
    sidecar the TE step encodes verbatim: v0 the corrected caption, v1..v{N-1}
    shuffled (+ tag-dropped at ``tag_dropout_rate``), and under
    ``tag_randomize_rate > 0`` an r-family with per-tag identity erasure, which
    requires both tokenizers for the dual-single erasure pool.
    """

    stats = PreprocessCaptionStats()
    images = walk_images(resized_dir, recursive=recursive, pattern=path_pattern)
    stats.seen = len(images)

    # First pass, collected up front (captions are tiny) so the erasure pool can
    # exclude the full real-tag set before any variant is drawn.
    @dataclass
    class _Entry:
        src: Path
        dst: Path
        corrected: str

    entries: list[_Entry] = []
    for image_path in images:
        rel_caption = image_path.relative_to(resized_dir).with_suffix(".txt")
        dst_caption = resized_dir / rel_caption
        caption_path = resolve_caption(resized_dir, source_dir, rel_caption)

        if caption_path is None:
            stats.no_caption += 1
            # No caption at all, so a variant sidecar left behind is an orphan.
            sidecar = variants_sidecar_path(dst_caption)
            if sidecar.exists():
                sidecar.unlink()
                stats.variants_removed += 1
            continue
        if caption_path != dst_caption:
            stats.from_master += 1

        raw = read_caption(caption_path)
        corrected = correct_caption(raw, kb, options=options).text if correct else raw
        if has_clauses(corrected):
            stats.clauses_preserved += 1
        entries.append(_Entry(caption_path, dst_caption, corrected))

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
        if e.dst.exists() and e.dst.read_text(encoding="utf-8") == e.corrected:
            stats.unchanged += 1
        else:
            # No ``drop_variants``: the sidecar is this pass's own output and
            # is rebuilt (or removed) a few lines down.
            write_caption(e.dst, e.corrected, history_by="correct")
            stats.written += 1

        sidecar = variants_sidecar_path(e.dst)
        if num_variants > 0:
            if not _sidecar_is_current(sidecar, e.corrected, num_variants, n_rand):
                rows = _build_variant_rows(
                    e.corrected,
                    num_variants=num_variants,
                    tag_dropout_rate=tag_dropout_rate,
                    tag_randomize_rate=tag_randomize_rate,
                    erasure_pool=erasure_pool,
                    protect_fn=protect_fn,
                )
                write_variants_sidecar(sidecar, rows)
                stats.variants_written += 1
        elif sidecar.exists():
            # Variants turned off → drop the now-stale sidecar.
            sidecar.unlink()
            stats.variants_removed += 1

    return stats
