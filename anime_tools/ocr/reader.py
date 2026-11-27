"""The manga OCR reader: PaddleOCR-VL-1.6 fine-tuned on manga lettering, SFX and speech.

A line recognizer trained on print reads balloon speech and misses the sound
effects drawn onto the artwork — ``ぱんぱん``, ``びくっ``, ``ばるん`` came back
as ``はんぱん`` / ``でくv`` / nothing from the CTC stack this package retired.
:class:`OcrReader` is the reader for those lines and the speech alike: a crop
in, a string out, no detection. It is the
PaddleOCR-VL-1.6 base with a LoRA on the vision tower + projector **and fully
fine-tuned ERNIE decoder layers** (v7), trained on AnimeText detector boxes
over the Manga109-s train pages with the box's main text as target (official
COO book split), KO / ZH SFX crops pseudo-labelled by the stock model and
proofread KO speech lines (on 4 px-margin crops: COO test SFX exact 81.9 %,
speech 89.1 %, doujin SFX 350 / 593, KO 32 / 48 and ZH 36 / 49 —
``sorryhyun/paddleocr-vl-1.6-manga-lora`` ``v7/``, Manga109-s attribution on the
card). Its crops are detector boxes grown by min(0.12 · long, 0.3 · short, 4 px)
(:func:`tight_pad_box`); the model's second prompt, ``OCR2:`` for a box's smaller
text, is not read here. It reads ``♡`` / ``ー`` / small kana natively, vertical or horizontal, so
it needs no symbol patching, and returns a block's line breaks as ``\\n``
(v4 and earlier joined the lines with a space).

What it is not: a page reader. Feed it the crops a detector has already
boxed — the AnimeText text-block detector (:mod:`anime_tools.ocr.animetext`,
the OCR stage's detector: balloon lines and the SFX on the artwork alike);
:meth:`OcrReader.read_boxes_scored` cuts the padded crops for you. Batch by area: a
1.9 B-parameter model at ~0.25 s per crop is the run's whole wall.

**The decode guard is part of the reader**, not an option. An autoregressive
decoder on a two-glyph crop runs away on ~4 % of inputs (``びく♡`` →
``ぐくーーー…``), so every read passes :func:`guard` — a repetition test and a
length cap tied to the crop's area — and a read that fails comes back as
``None`` rather than as junk. :func:`is_runaway` / :func:`length_cap` are
torch-free so the rule is testable on strings alone.

The guard sees the read in :func:`normalize_read`'s form, and the repetition
test is **ellipsis-blind**: the reader spells an ellipsis as ``......`` or
``・・・・・・`` (never ``…``), and six dots hold the trigram ``...`` four times
over, so the test as first shipped took every dialogue line of nine characters
or more that paused twice — 94 lines on 64 of the 859 sincos pages (2026-09-07),
the longest speech on each page. A dot run is now one ``…`` before anything
counts it.

Weights are two catalog rows (:mod:`anime_tools.downloads`): ``vl16_base``
(the Apache-2.0 base, 1.9 GB) and ``ocr_reader`` (tower LoRA 143 MB + LM
layers 510 MB). :meth:`OcrReader.load` fetches what is missing.

**v6 (opt-in, ``version="v6"``)**: the reader before 0.14.0, trained on the COO
onomatopoeia polygons and the ``<text>`` speech boxes and read on :data:`CROP_PAD`
crops. On the same 4 px crops it reads COO SFX at 85.2 %, speech 88.4 %, doujin
SFX 294 / 593, KO 26 and ZH 32 (catalog row ``ocr_reader_v6``).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import Any

from anime_tools.downloads import OCR_READER_ADAPTER_FILES, OCR_READER_LM_FILE

PROMPT = "OCR:"
"""The crop task of the base model's chat template; the fine-tune kept it."""

DEFAULT_VERSION = "v7"
V6 = "v6"
READER_VERSIONS = (DEFAULT_VERSION, V6)
"""The readers :meth:`OcrReader.load` knows: v7 (:func:`tight_pad_box` crops) and
the opt-in v6 (:data:`CROP_PAD` crops)."""

_CATALOG_ROW = {
    DEFAULT_VERSION: "ocr_reader",
    V6: "ocr_reader_v6",
}

CROP_PAD = 0.12
"""v6's padding around a box, as a fraction of its longer side. The polygons hug
the glyphs and v6 was trained on crops with this margin; a zero-pad crop
clips the strokes of the outer characters."""

TIGHT_PAD_FRAC = 0.12
TIGHT_PAD_SHORT_FRAC = 0.3
TIGHT_PAD_PX = 4
"""The margin of v7: min(0.12 · long side, 0.3 · short side, 4 px). Its training crops were detector boxes cut this way; the 12 % margin pulls a
neighbouring line into a quarter of the COO SFX crops."""

MAX_NEW_TOKENS = 200
"""The tokenizer spends about one token per CJK character; COO's longest
onomatopoeia is 28 characters and a doujin speech balloon runs to ~60, a
page's timetable or profile block to ~160. Past this a decode is a runaway,
and the guard drops it anyway."""

MAX_PIXELS = 1280 * 28 * 28
"""The vision tower's pixel budget per crop (the training and eval setting)."""

MIN_LENGTH_CAP = 12
MIN_GLYPH_PX = 14
"""The smallest glyph a crop is assumed to hold: the length cap is how many of
them fit in the crop's area. At 16 px the cap dropped correct reads of dense
lettering (21 KO balloons at 12–14 px glyphs, a 29-character Manga109 box);
over 483k labelled reads 14 px let through nothing else (2026-11-20)."""
LATIN_GLYPH_AREA = MIN_GLYPH_PX * MIN_GLYPH_PX // 2
"""A Latin letter's share of a line: half a CJK square. Scanlation lettering
at the CJK area overran the cap on plain balloons (``Yesterday's video was
amazing.``, 27 letters on a 15-glyph crop — 37 of the 84 English reads the
guard dropped on 150 hitocrawl ``en`` pages, 2026-10-29)."""

RUNAWAY_RUN = 16
RUNAWAY_UNIT_REPEATS = 8
LOOP_UNIT = 40
"""The runaway test: a glyph run of :data:`RUNAWAY_RUN`, a unit of two to eight
glyphs repeated :data:`RUNAWAY_UNIT_REPEATS` times back to back, or a unit up to
:data:`LOOP_UNIT` long three times back to back. A held vowel, a chant
(``NO, NO, NO, NO!``), a repeated sound effect (``ぐゅぐゅぐゅぐゅ``) and a list's
recurring labels (``:30``, ``「안녕하세요week」``) are lettering, not a decode
loop."""

_LATEX_RE = re.compile(r"\\[(),]|\\\s")
_ELLIPSIS_RE = re.compile(r"[.．・…‥]{2,}")
"""A run of two or more dots of any kind — ASCII, fullwidth, the katakana
middle dot the reader uses for ``…``, or ``…`` / ``‥`` themselves. One
ellipsis, however it was drawn or decoded."""
ELLIPSIS = "…"
HEART_FOLD = str.maketrans({"♥": "♡", "❤": "♡"})
"""The reader emits ``♥`` and ``♡`` for the same drawn heart (sincos records:
332 vs 889); a caption says one glyph, and it is the one the artists draw."""
SHAPE_FOLD = str.maketrans({"◯": "○", "〇": "○", **{c: "×" for c in "☓✕✖╳✗✘"}})
"""A drawn shape is one glyph. The circle: the censor mark (``○○바이러스``, ``オ〇ニー``),
the bullet ``◯`` and the kanji zero ``〇`` (``二〇三六年``) are the same shape in
lettering, and the labels spell them all three ways. The cross likewise: ``☓`` /
``✕`` / ``✖`` / ``╳`` / ``✗`` / ``✘`` → ``×`` (one label wrote ``☓☓☓と☓×☓``).
Latin ``O`` / ``x`` / ``X`` stay."""
_HANGUL_RE = re.compile(r"[\uac00-\ud7a3\u3131-\u318e]")
_HYPHENS = frozenset("-–‒‐－")
_LINE_RUN_RE = re.compile(r"[ー―—─\-–‒‐－]+")
_KO_DASH_FOLD = str.maketrans({c: "-" for c in "―—─–‒‐－"})
_KO_LINE_RUN_RE = re.compile(r"[ー-]{2,}")
_WAVE_FOLD = str.maketrans({c: "~" for c in "〜〰～"})
_WAVE_RUN_RE = re.compile(r"~{2,}")


def is_runaway(text: str) -> bool:
    """The autoregressive failure class: ``ぉぉぉ…`` × 100, ``ふくっ`` × 100.

    The tests of :data:`RUNAWAY_RUN`. Every dot run is one ``…`` first, so
    ``え...な...ない......`` is a pause and not a repeat; a hundred dots alone
    collapse to one and are then no letter at all
    (:func:`~anime_tools.ocr.reread.has_script`).
    """
    t = _ELLIPSIS_RE.sub(ELLIPSIS, "".join(text.split()))
    return bool(
        re.search(rf"(.)\1{{{RUNAWAY_RUN - 1},}}", t)
        or re.search(rf"(.{{2,8}}?)\1{{{RUNAWAY_UNIT_REPEATS - 1},}}", t)
        or re.search(rf"(.{{9,{LOOP_UNIT}}}?)\1\1", t)
    )


def is_latin(text: str) -> bool:
    """Whether a read is Latin lettering: an ASCII letter, and no letter
    outside ASCII. Hearts, digits and punctuation do not decide it."""
    letters = [ch for ch in text if ch.isalpha()]
    return bool(letters) and all(ch.isascii() for ch in letters)


def length_cap(width: int, height: int, *, latin: bool = False) -> int:
    """How many characters a crop of this size can plausibly hold.

    Area over the smallest glyph (:data:`MIN_GLYPH_PX` squared, or
    :data:`LATIN_GLYPH_AREA` for ``latin``), never under
    :data:`MIN_LENGTH_CAP`. A two-glyph crop caps near the floor, so a
    48-token runaway on it is rejected; a multi-column balloon block caps in
    the hundreds and is left to the repetition test — a block's character
    count is not readable off its shape (an aspect rule measured 2026-09-06
    threw away 60 % of the speech reads).
    """
    area = max(width, 0) * max(height, 0)
    glyph = LATIN_GLYPH_AREA if latin else MIN_GLYPH_PX * MIN_GLYPH_PX
    return max(MIN_LENGTH_CAP, round(area / glyph))


def normalize_read(text: str) -> str:
    """The raw decode as a record: NFC-stable glyphs, every heart (``♥``, the
    emoji ``❤`` + variation selector) folded to ``♡``, every circle (``◯``,
    ``〇``) to ``○`` and every cross (``☓``, ``✕`` …) to ``×``
    (:data:`SHAPE_FOLD`), every dot run
    (``......``, ``・・・・・・``, ``‥``) to one ``…``, VL's LaTeX wrapping of a
    measurement (``\\( 156 \\, cm \\)``) stripped, whitespace runs inside a line
    collapsed to one space, and the line breaks the reader wrote kept as one
    ``\\n`` each (blank lines dropped). A caption clause spells a break as
    :data:`~anime_tools.captions.position_clauses.LINE_BREAK`.

    The folds are what make a line comparable: the speech clause dedupes on
    exact text (:func:`~anime_tools.captions.ocr_sfx.dedupe_speech`), and
    ``あ・・・っ♥`` beside ``あ・・・っ♡`` was two lines of the same word."""
    text = text.replace("\ufe0f", "").translate(HEART_FOLD).translate(SHAPE_FOLD)
    text = _LATEX_RE.sub("", text)
    text = unicodedata.normalize("NFC", text)
    text = _ELLIPSIS_RE.sub(ELLIPSIS, text)
    lines = (" ".join(line.split()) for line in text.replace("\r", "\n").split("\n"))
    return "\n".join(line for line in lines if line)


def _ascii_alnum(ch: str) -> bool:
    ch = unicodedata.normalize("NFKC", ch)
    return bool(ch) and ch.isascii() and ch.isalnum()


def _line_run(m: re.Match) -> str:
    if _HYPHENS.issuperset(m[0]):
        s = m.string
        before = s[m.start() - 1] if m.start() else ""
        after = s[m.end()] if m.end() < len(s) else ""
        if _ascii_alnum(before) or _ascii_alnum(after):
            return "-"
    return "ー"


def fold_drawn_lines(text: str) -> str:
    """A drawn-out line as one glyph: a run of ``ー`` / ``―`` / hyphens (any mix, any
    count, one included) → one ``ー`` (``ええーーーっ`` → ``ええーっ``, ``でも――`` →
    ``でもー``, ``―――?`` → ``ー?``, ``二年前──…`` → ``二年前ー…``); every wave (``〜``,
    ``〰``, ``～``) → ``~`` and a ``~`` run → one ``~``. In vertical lettering ``ー`` and
    ``―`` are one stroke, and how many the reader writes for it is not a reading.

    The hyphen stays, as one ``-``, in two places: a run of hyphens alone beside an ASCII
    letter or digit (``747-400D``), and a read holding Hangul, for its dashes and its
    ``ー`` / ``-`` runs (``만세-!``; a lone ``ー`` there stays). A Chinese read folds as
    Japanese does (``——全部剪掉吧`` → ``ー全部剪掉吧``): ideographs alone do not say
    which language a line is in.

    The reader's training target folds the same way from ``TARGET_NORM = 13`` on (the
    trainer's ``textnorm`` fold 7), so a reader trained on it writes this form itself."""
    if _HANGUL_RE.search(text):
        text = _KO_LINE_RUN_RE.sub("-", text.translate(_KO_DASH_FOLD))
    else:
        text = _LINE_RUN_RE.sub(_line_run, text)
    return _WAVE_RUN_RE.sub("~", text.translate(_WAVE_FOLD))


def guard(text: str, width: int, height: int) -> str | None:
    """The read, or ``None`` when it is not one.

    Rejects an empty decode, a runaway (:func:`is_runaway`) and a read longer
    than :func:`length_cap` for the crop's shape. Everything that passes is
    :func:`normalize_read`'s form with its drawn-out lines folded
    (:func:`fold_drawn_lines`): both tests see the read before that fold, so a
    ``ー`` runaway is still one.
    """
    read = normalize_read(text)
    if not read or is_runaway(read):
        return None
    n = sum(1 for ch in read if not ch.isspace())
    if n > length_cap(width, height, latin=is_latin(read)):
        return None
    return fold_drawn_lines(read)


def token_confidence(
    tail: Sequence[int], step_probs: Sequence[float], skip: Iterable[int]
) -> tuple[list[int], float]:
    """The generated ids up to the first end-of-text / pad and the mean of
    their greedy probabilities.

    ``tail`` is one row's generated ids in step order; ``step_probs[i]`` is the
    probability the decoder put on the token it chose at step ``i`` (the
    softmax maximum under greedy decoding). The mean runs over the kept tokens
    only — the stop token is not part of the read — and an empty decode is
    ``0.0``, since nothing was read to be sure of.
    """
    stop = set(skip)
    ids: list[int] = []
    for t in tail:
        if t in stop:
            break
        ids.append(int(t))
    if not ids:
        return ids, 0.0
    probs = [float(p) for p in step_probs[: len(ids)]]
    if not probs:
        return ids, 0.0
    return ids, sum(probs) / len(probs)


def pad_box(
    box: Sequence[int], width: int, height: int, pad: float = CROP_PAD
) -> tuple[int, int, int, int]:
    """``(x0, y0, x1, y1)`` grown by ``pad`` × its longer side, clipped to the
    image."""
    x0, y0, x1, y1 = (int(v) for v in box)
    grow = round(pad * max(x1 - x0, y1 - y0))
    return (
        max(0, x0 - grow),
        max(0, y0 - grow),
        min(width, x1 + grow),
        min(height, y1 + grow),
    )


def tight_pad_box(
    box: Sequence[int], width: int, height: int
) -> tuple[int, int, int, int]:
    """``(x0, y0, x1, y1)`` grown by v7's margin
    (:data:`TIGHT_PAD_PX` capped by the box's sides), clipped to the image —
    the trainer's ``build_sfx_prefix.margin_box``."""
    x0, y0, x1, y1 = (int(v) for v in box)
    w, h = x1 - x0, y1 - y0
    p = min(TIGHT_PAD_FRAC * max(w, h), TIGHT_PAD_SHORT_FRAC * min(w, h), TIGHT_PAD_PX)
    return (
        round(max(0, x0 - p)),
        round(max(0, y0 - p)),
        round(min(width, x1 + p)),
        round(min(height, y1 + p)),
    )


def crop_box(bgr, box: Sequence[int], pad: float = CROP_PAD, *, tight: bool = False):
    """The padded crop of an axis-aligned box, or ``None`` if it is empty.
    ``tight`` cuts v7's margin (:func:`tight_pad_box`) instead of
    ``pad``."""
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = tight_pad_box(box, w, h) if tight else pad_box(box, w, h, pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return bgr[y0:y1, x0:x1]


@cache
def _greedy_recorder() -> type:
    """The logits processor that records the greedy maximum per step.

    Built once and cached, not redefined per read: it subclasses a transformers
    class, so it cannot live at module scope (this module imports without
    torch), and a fresh class object per page is a fresh type per page.
    """
    from transformers import LogitsProcessor

    class _Greedy(LogitsProcessor):
        """Records ``max softmax`` per step; changes nothing."""

        def __init__(self) -> None:
            self.probs: list = []

        def __call__(self, input_ids, scores):
            self.probs.append(scores.float().log_softmax(-1).max(-1).values.exp())
            return scores

    return _Greedy


VL_NOTICES = (
    "Unrecognized keys in `rope_parameters`",
    "`PaddleOCRVLProcessor` defines",
)


class OcrReaderWeightsMissing(RuntimeError):
    """A reader directory the catalog has not filled yet."""

    def __init__(self, model_dir: Path, row: str) -> None:
        super().__init__(
            f"OCR reader weights not found in {model_dir} — run "
            f"`python -m anime_tools.downloads {row}` "
            "(or ⚙ Settings › Models in the GUI)"
        )


@contextmanager
def _without_vl_notices() -> Iterator[None]:
    """Drops what transformers says about the PaddleOCR-VL base itself: a key its
    config carries and the processor class it declares. Other warnings print."""
    import transformers  # noqa: F401  (configures the handlers filtered here)

    def keep(record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(VL_NOTICES)

    handlers = logging.getLogger("transformers").handlers
    for handler in handlers:
        handler.addFilter(keep)
    try:
        yield
    finally:
        for handler in handlers:
            handler.removeFilter(keep)


def _ensure(row_id: str, fetch: bool) -> Path:
    from anime_tools.downloads import ensure

    return ensure(
        row_id, fetch=fetch, missing=lambda d: OcrReaderWeightsMissing(d, row_id)
    )


@dataclass
class OcrReader:
    """The fine-tuned VL crop reader, loaded once, read many."""

    model: Any
    processor: Any
    device: str
    batch_size: int = 16
    min_edge: int = 28
    version: str = DEFAULT_VERSION

    @classmethod
    def load(
        cls,
        *,
        device: str | None = None,
        base_dir: Path | None = None,
        adapter_dir: Path | None = None,
        batch_size: int = 16,
        fetch: bool = True,
        version: str = DEFAULT_VERSION,
    ) -> OcrReader:
        """Base + tower adapter merged + fine-tuned LM layers, in bf16 on ``device``
        (auto when ``None``).

        ``base_dir`` / ``adapter_dir`` default to the catalog rows and are
        fetched when missing (``fetch=False`` raises :class:`OcrReaderWeightsMissing`
        instead). A dir passed explicitly is used as is. ``version`` picks the
        reader (:data:`READER_VERSIONS`): its catalog row and its crop margin.

        bf16 holds on every device, MPS included — measured 2026-09-09 at 860 ms
        a crop there against a CPU's 4960, same decodes.
        """
        from anime_tools._device import resolve_device

        if version not in READER_VERSIONS:
            raise ValueError(
                f"unknown reader version {version!r}; known: {READER_VERSIONS}"
            )
        row = _CATALOG_ROW[version]
        device = resolve_device(device)
        base = Path(base_dir) if base_dir else _ensure("vl16_base", fetch)
        adapter = Path(adapter_dir) if adapter_dir else _ensure(row, fetch)
        for name in (*OCR_READER_ADAPTER_FILES, OCR_READER_LM_FILE):
            if not (adapter / name).is_file():
                raise OcrReaderWeightsMissing(adapter, row)

        import torch
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForImageTextToText, AutoProcessor

        with _without_vl_notices():
            model = AutoModelForImageTextToText.from_pretrained(
                str(base),
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                # the base checkpoint stores a separate lm_head
                tie_word_embeddings=False,
            )
            processor = AutoProcessor.from_pretrained(str(base))
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
        # The decoder layers are a plain state dict under the base model's key
        # names; the adapter only ever touched the tower and projector.
        lm = load_file(str(adapter / OCR_READER_LM_FILE))
        unexpected = model.load_state_dict(lm, strict=False).unexpected_keys
        if unexpected:
            raise RuntimeError(
                f"{OCR_READER_LM_FILE} carries keys the base has no home for: {unexpected[:5]}"
            )
        model = model.to(device).eval()
        return cls(
            model=model,
            processor=processor,
            device=device,
            batch_size=batch_size,
            min_edge=int(processor.image_processor.size["shortest_edge"]),
            version=version,
        )

    @property
    def tight(self) -> bool:
        """Whether boxes are cut with :func:`tight_pad_box` (v7) rather than
        :data:`CROP_PAD` (v6)."""
        return self.version != V6

    @cached_property
    def prompt(self) -> str:
        """The chat-templated ``OCR:`` prompt.

        Templated once for the life of the reader: it depends on nothing but the
        processor, and templating it per batch was a tokenizer round trip inside
        the read loop.
        """
        return self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
                }
            ],
            add_generation_prompt=True,
            tokenize=False,
        )

    def read_raw_scored(self, crops: Sequence) -> list[tuple[str, float]]:
        """Every crop's raw decode with its confidence (BGR ``uint8`` arrays
        in, ``(text, confidence)`` out), in input order. Unguarded — :meth:`read_scored` is the one to call.

        The confidence is the mean, over the tokens the reader emitted, of the
        probability it put on each (:func:`token_confidence`): a logits
        processor records the greedy maximum at every step, so nothing but a
        ``(batch,)`` vector per step is held — never the vocabulary-wide scores
        ``output_scores`` would keep.
        """
        import torch
        from PIL import Image
        from transformers import LogitsProcessorList

        if not crops:
            return []
        greedy = _greedy_recorder()
        order = sorted(
            range(len(crops)), key=lambda i: crops[i].shape[0] * crops[i].shape[1]
        )
        out: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        tok = self.processor.tokenizer
        skip = {tok.eos_token_id, tok.pad_token_id}
        prompt = self.prompt
        for s in range(0, len(order), self.batch_size):
            idx = order[s : s + self.batch_size]
            images = [Image.fromarray(crops[i][:, :, ::-1]) for i in idx]
            inputs = self.processor(
                text=[prompt] * len(images),
                images=images,
                padding=True,
                padding_side="left",
                return_tensors="pt",
                images_kwargs={
                    "size": {"shortest_edge": self.min_edge, "longest_edge": MAX_PIXELS}
                },
            ).to(self.device)
            n = inputs["input_ids"].shape[-1]
            recorder = greedy()
            with torch.inference_mode():
                gen = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                    logits_processor=LogitsProcessorList([recorder]),
                )
            steps = (
                torch.stack(recorder.probs, 1).cpu().tolist()
                if recorder.probs
                else [[] for _ in idx]
            )
            for i, row, probs in zip(idx, gen, steps, strict=True):
                ids, conf = token_confidence(row[n:].tolist(), probs, skip)
                out[i] = (tok.decode(ids).strip(), conf)
        return out

    def read_scored(self, crops: Sequence) -> list[tuple[str, float] | None]:
        """:meth:`read_raw_scored` through :func:`guard`: ``(text, confidence)``
        per crop, or ``None`` for a crop the decoder ran away on."""
        raws = self.read_raw_scored(crops)
        out: list[tuple[str, float] | None] = []
        for (raw, conf), c in zip(raws, crops, strict=True):
            text = guard(raw, int(c.shape[1]), int(c.shape[0]))
            out.append(None if text is None else (text, conf))
        return out

    def read_boxes_scored(
        self, bgr, boxes: Sequence[Sequence[int]], pad: float = CROP_PAD
    ) -> list:
        """One page, many boxes: :func:`crop_box` each, :meth:`read_scored`
        them all.

        One ``(text, confidence)`` or ``None`` per box (an empty box reads as
        ``None``). v7 cuts its own tight margin (``pad`` is unused)."""
        return self.read_pages_scored([(bgr, boxes)], pad)[0]

    def read_pages_scored(
        self,
        pages: Sequence[tuple[Any, Sequence[Sequence[int]]]],
        pad: float = CROP_PAD,
    ) -> list[list]:
        """Many pages in one read: ``[(bgr, boxes), …]`` in, one
        :meth:`read_boxes_scored` answer per page out.

        Every crop of every page goes into one :meth:`read_scored`, so batches
        fill across pages and the pooled crops sort by area together. Read page
        by page, a page's handful of boxes left most ``generate`` calls far
        under ``batch_size`` (240 dataset pages, 2026-11-24: median 2 crops a
        call, 19.4 crops/s; pooled 32.8 at bs 16, 39.3 at bs 32). What
        :class:`~anime_tools.ocr.reread.RereadEngine` calls as ``read_pages``.
        """
        crops, owners = [], []
        out: list[list] = []
        for p, (bgr, boxes) in enumerate(pages):
            out.append([None] * len(boxes))
            for i, box in enumerate(boxes):
                crop = crop_box(bgr, box, pad, tight=self.tight)
                if crop is not None and crop.size:
                    crops.append(crop)
                    owners.append((p, i))
        for (p, i), r in zip(owners, self.read_scored(crops), strict=True):
            out[p][i] = r
        return out


__all__ = [
    "CROP_PAD",
    "DEFAULT_VERSION",
    "ELLIPSIS",
    "HEART_FOLD",
    "LATIN_GLYPH_AREA",
    "MAX_NEW_TOKENS",
    "PROMPT",
    "READER_VERSIONS",
    "RUNAWAY_RUN",
    "RUNAWAY_UNIT_REPEATS",
    "SHAPE_FOLD",
    "V6",
    "OcrReader",
    "OcrReaderWeightsMissing",
    "crop_box",
    "fold_drawn_lines",
    "guard",
    "is_latin",
    "is_runaway",
    "length_cap",
    "normalize_read",
    "pad_box",
    "tight_pad_box",
    "token_confidence",
]
