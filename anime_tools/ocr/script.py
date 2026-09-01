"""Which language a recognized line is in, and what tag the caption may say so with.

PP-OCRv6 is **one** model over 50 languages: it returns a string and a
confidence, never a language. So the language is read back off the characters,
which for the four scripts this pipeline cares about is unambiguous enough to be
a lookup rather than a model:

* kana present         → ``ja`` (kanji-with-kana is Japanese, whatever the kanji)
* han, no kana         → ``zh`` (a pure-kanji Japanese line reads as ``zh``; the
  scripts are genuinely identical there, so no classifier can do better)
* latin letters/digits → ``en``
* none of those        → ``other``

The order matters and is the whole subtlety: kana is checked first because
Japanese *contains* han, so a "does it have han?" test run first would call every
Japanese sentence Chinese.

Three languages and no more, because three is what the recognizer can read: the
shipped PP-OCRv6 checkpoint carries 15,565 han characters and both kana, and
**zero hangul**, so a Korean line comes back as an empty string and is dropped
before it ever reaches here. A fourth case would be a branch no input can take.

**The tag map is not the language map**, and that is a fact about Danbooru rather
than about OCR. The site tags text that is *not* Japanese — ``english_text``
(351k posts), ``chinese_text`` (44k), ``korean_text`` (37k) — and has no
``japanese_text`` at all, because Japanese is the unmarked default. Emitting one
would put a tag in every caption that :func:`anime_tools.captions.correction`
types as unknown on every later pass, so ``ja`` maps to nothing: Japanese lands
in the sidecar in full and adds no tag to the bag. That is the same asymmetry
``@no-artist`` encodes on the artist axis — the absence is the statement.

Pure stdlib and no weights, so the classifier is unit-testable without a
100 MB download and the caption side never imports onnxruntime.
"""

from __future__ import annotations

from collections.abc import Iterable

LANGS: tuple[str, ...] = ("en", "ja", "zh", "other")
"""Every value :func:`script_of` can return — the ``--lang`` vocabulary."""

# Han is shared by ``ja`` and ``zh``, so it is deliberately *not* a language on
# its own: :func:`script_of` reaches it only after kana has ruled Japanese out.
_KANA = ((0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9D))
_HAN = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2A6DF))

TAG_FOR_LANG: dict[str, str] = {
    "en": "english text",
    "zh": "chinese text",
    # "ja" is absent on purpose — see the module docstring.
}
"""Language → the Danbooru tag that says the image carries it. Spelled with a
space because :func:`anime_tools.captions.taxonomy.normalize_tag` is what every
"does the caption already say this?" test keys on, and it folds the underscore
the KB spells them with."""

BILINGUAL_TAG = "bilingual text"
"""What two or more *distinct* languages in one image are tagged as (2.9k posts).
Japanese counts toward it even though it has no tag of its own: a Japanese page
with an English sign is bilingual regardless of which halves are nameable."""


def _in(code: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(lo <= code <= hi for lo, hi in ranges)


def script_of(text: str) -> str:
    """The language of one recognized line — one of :data:`LANGS`.

    Whitespace and punctuation vote for nothing, so ``"「こんにちは」"`` is ``ja``
    rather than a tie with the brackets. A line with no letters at all (``"!?"``,
    a row of dots the detector found in a screentone) is ``other``, which is what
    lets ``--lang`` drop that noise without a separate filter.
    """
    han = latin = False
    for ch in text:
        code = ord(ch)
        if _in(code, _KANA):
            return "ja"
        if _in(code, _HAN):
            han = True
        elif ch.isalnum() and code < 0x0250:
            latin = True
    if han:
        return "zh"
    return "en" if latin else "other"


def parse_langs(value: str) -> tuple[str, ...]:
    """``"en,ja,zh"`` → the allowlist, validated against :data:`LANGS`.

    Raises :class:`ValueError` for an unknown code rather than silently keeping
    nothing: a typo in ``--lang`` would otherwise read as an image with no text
    in it, which is indistinguishable from a working run over a clean dataset.
    """
    langs = tuple(
        dict.fromkeys(p.strip().lower() for p in value.split(",") if p.strip())
    )
    unknown = [lang for lang in langs if lang not in LANGS]
    if unknown:
        raise ValueError(
            f"unknown --lang {', '.join(unknown)} (pick from {', '.join(LANGS)})"
        )
    if not langs:
        raise ValueError("--lang needs at least one language")
    return langs


def tags_for(langs: Iterable[str]) -> tuple[str, ...]:
    """The caption tags a set of detected languages earns, in a stable order.

    ``ja`` alone earns none — the image gets a sidecar and the caption is left
    exactly as it was, which is why the stage can run over a Japanese-only corpus
    and honestly report zero proposals.
    """
    found = {lang for lang in langs if lang in LANGS and lang != "other"}
    tags = [TAG_FOR_LANG[lang] for lang in ("en", "zh") if lang in found]
    if len(found) > 1:
        tags.append(BILINGUAL_TAG)
    return tuple(tags)
