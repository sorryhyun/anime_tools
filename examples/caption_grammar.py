"""The caption grammar: parse, compose, locate and normalize tags.

Torch-free, no models, no dataset — ``python examples/caption_grammar.py``.

A caption is a flat tag bag followed by trailing position clauses::

    safe, 2girls, white socks. On the left, akita neru, yellow eyes. On the right, kasane teto

The **period** delimits clauses and commas delimit tags inside one, so a
``caption.split(",")`` glues a clause header onto the previous tag. Everything
that reads or writes a caption goes through ``anime_tools.captions`` instead.
"""

from __future__ import annotations

from anime_tools.captions import compose_caption, parse_caption
from anime_tools.captions.position_clauses import (
    PositionClause,
    flat_tag_set,
    flatten_caption,
    has_clauses,
    horizontal_names,
    tag_spans,
)
from anime_tools.captions.shuffle import (
    NO_ARTIST_SENTINEL,
    anima_smart_shuffle_caption,
    strip_no_artist_sentinel,
)
from anime_tools.captions.taxonomy import (
    canonical_rating,
    count_of,
    is_artist_tag,
    is_count_tag,
    normalize_tag,
)

CAPTION = (
    "safe, 2girls, akita neru, kasane teto, white socks, @artist_a. "
    "On the left, akita neru, yellow eyes. On the right, kasane teto, drill hair."
)


def main() -> None:
    # --- parse -----------------------------------------------------------
    parsed = parse_caption(CAPTION)
    print("flat bag:", parsed.flat_tags)
    for clause in parsed.clauses:
        print(f"clause {clause.position!r}:", clause.tags)

    # The naive split is the bug the grammar exists to prevent.
    print("\nsplit(',') would read:", CAPTION.split(",")[4:6])

    # --- compose ---------------------------------------------------------
    # Round trip: parse → compose is byte-stable for a canonical caption
    # (clauses end in a period; whitespace around commas is normalized).
    assert compose_caption(parsed.flat_tags, parsed.clauses) == CAPTION

    # Build one from parts. ``horizontal_names(n)`` gives the position words
    # the position stage uses for n subjects in one row.
    names = horizontal_names(3)
    clauses = [
        PositionClause(position=names[0], tags=("red hair",)),
        PositionClause(position=names[2], tags=("blue hair", "glasses")),
    ]
    print("\ncomposed:", compose_caption(["safe", "3girls", "outdoors"], clauses))

    # --- cheap questions about a caption ---------------------------------
    print("\nhas clauses:", has_clauses(CAPTION))
    print("flat set (normalized):", sorted(flat_tag_set(CAPTION)))
    print("flattened:", flatten_caption(CAPTION))

    # --- spans: where each tag sits in the raw string --------------------
    # This is what the GUI editor paints boxes from; it never splits text
    # itself. ``kind`` is tag / header / artist; ``clause`` is -1 in the bag.
    print("\nspans:")
    for span in tag_spans(CAPTION):
        print(
            f"  [{span.start:3d},{span.end:3d}) {span.kind:6s} clause={span.clause:2d} {span.text}"
        )

    # --- taxonomy: the one normal form ------------------------------------
    # Every "does the caption already say this?" check keys off normalize_tag,
    # so the tagger's ``speech bubble`` and a hand-written ``speech_bubble``
    # can never read as two tags.
    print(
        "\nnormalize:",
        normalize_tag("Speech_Bubble"),
        "==",
        normalize_tag("speech bubble"),
    )
    print(
        "count tag:",
        is_count_tag("2girls"),
        "girls =",
        count_of(parsed.flat_tags, "girl"),
    )
    print(
        "artist:",
        is_artist_tag("@artist_a"),
        "rating:",
        canonical_rating("questionable"),
    )

    # --- shuffle: the variant grammar -------------------------------------
    # The @artist prefix stays put, tags shuffle within their clause, and the
    # ``@no-artist`` sentinel marks the artist slot on a caption without one.
    tags = ["safe", "1girl", NO_ARTIST_SENTINEL, "long hair", "smile", "outdoors"]
    shuffled = anima_smart_shuffle_caption(tags)
    print("\nshuffled:", shuffled)
    print("for the tokenizer:", strip_no_artist_sentinel(shuffled))


if __name__ == "__main__":
    main()
