"""Correction, drop groups, variants, history, and the caption index.

Torch-free. Needs the Danbooru tag KB (``python -m anime_tools.downloads
danbooru_tags``); the index part also needs a tagger ``vocab.json``.

    python examples/caption_correction.py

Batch counterpart: the **correct** stage (``examples/stages.py``,
``python -m anime_tools.stages.cli.correct_captions``), which runs exactly this
over the resized tree and writes the revised caption plus sidecars.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from anime_tools._env import curation_home
from anime_tools.captions.correction import (
    CaptionCorrectionOptions,
    correct_caption,
    find_tag_csv,
    load_tag_knowledge_base,
)
from anime_tools.captions.history import push_history, read_history
from anime_tools.captions.index import DEFAULT_VOCAB, build_index
from anime_tools.captions.tag_drop_groups import drop_group_names, parse_drop_groups
from anime_tools.captions.variants import (
    generate_caption_variants,
    read_variants_sidecar,
    variants_sidecar_path,
    write_variants_sidecar,
)
from anime_tools.stages._caption_io import write_caption

MESSY = (
    "long hair, questionable, 1girl, Akita_Neru, smile, vocaloid, "
    "backlighting, long hair, looking at viewer"
)


def main() -> None:
    csv_path = find_tag_csv()
    if csv_path is None:
        raise SystemExit(
            "no tag KB: run `python -m anime_tools.downloads danbooru_tags`"
        )
    kb = load_tag_knowledge_base(csv_path)
    print(f"KB: {len(kb.tags)} tags from {csv_path}")

    # --- correct: bucket order, dedupe, ratings, @no-artist ----------------
    # Buckets: quality → meta → year → rating → count → character → copyright
    # → artist → general. Unknown tags stay, in the general bucket.
    result = correct_caption(
        MESSY, kb, options=CaptionCorrectionOptions(insert_no_artist=True)
    )
    print("\nbefore:", MESSY)
    print("after: ", result.text)
    print(
        "unknown to KB:",
        result.unknown_tags,
        "| inserted @no-artist:",
        result.inserted_no_artist,
    )

    # --- drop groups: strip whole kinds of tag ----------------------------
    # Slugs resolve by tag shape (@ → artist), then Danbooru kind, then the
    # KB's category path; anything else is a literal path prefix.
    print("\ndrop-group slugs:", ", ".join(drop_group_names()))
    dropping = CaptionCorrectionOptions(
        drop_groups=parse_drop_groups("artist,lighting")
    )
    result = correct_caption(MESSY + ", @some_artist", kb, options=dropping)
    print("dropped:", result.dropped_tags, "→", result.text)

    # Clauses are corrected around, never dissolved: only the flat bag is
    # reordered, and a clause emptied by a drop is removed whole.
    clause_caption = (
        "safe, 2girls, backlighting. On the left, red hair. On the right, backlighting"
    )
    print("in clauses:", correct_caption(clause_caption, kb, options=dropping).text)

    # --- variants: the .variants.txt sidecar -------------------------------
    # v0 is the pristine caption (identical to {stem}.txt); v1.. are shuffled
    # with per-tag dropout. A clause is dropped whole or kept whole.
    variants = generate_caption_variants(
        result.text, num_variants=3, tag_dropout_rate=0.2
    )
    for i, v in enumerate(variants):
        print(f"v{i}: {v}")

    with tempfile.TemporaryDirectory() as tmp:
        caption = Path(tmp) / "001.txt"
        write_caption(caption, variants[0], newline=True)
        sidecar = variants_sidecar_path(caption)  # 001.variants.txt
        write_variants_sidecar(sidecar, [(f"v{i}", v) for i, v in enumerate(variants)])
        print("\nsidecar rows:", [label for label, _ in read_variants_sidecar(sidecar)])

        # --- history: what a write replaced -------------------------------
        # Every caption write in the package pushes the old text first, so a
        # run needs no Apply gate: the previous version is a badge, and Undo
        # replays the report backwards. ``history_by`` names the writer.
        write_caption(
            caption, "safe, 1girl, edited", newline=True, history_by="example"
        )
        # Same thing by hand: record the text you are *about to* replace.
        push_history(caption, "safe, 1girl, edited", by="example-by-hand")
        caption.write_text("safe, 1girl, edited twice\n", encoding="utf-8")
        for entry in read_history(caption.with_name("001.history.txt")):
            print(f"  {entry.label('revised')}  {entry.note()}  {entry.text[:40]}…")

    # --- index: typed tags per image → caption_index.json ------------------
    vocab = Path(DEFAULT_VOCAB)
    if vocab.exists():
        src = curation_home() / "image_dataset"
        if src.is_dir():
            index = build_index(src, vocab)
            print(
                f"\nindex: {index['meta']['n_images']} images; characters:",
                sorted(index["groups"]["character"])[:5],
            )
    else:
        print(
            f"\n(skipping the index: no {vocab}; `python -m anime_tools.downloads tagger`)"
        )
    print(
        "CLI: python -m anime_tools.captions.index --src image_dataset --out workspace/captions/caption_index.json"
    )


if __name__ == "__main__":
    main()
