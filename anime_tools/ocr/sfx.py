"""A manga SFX reader: PaddleOCR-VL-1.6 fine-tuned on hand-lettered onomatopoeia.

A line recognizer trained on print reads balloon speech and misses the sound
effects drawn onto the artwork — ``ぱんぱん``, ``びくっ``, ``ばるん`` came back
as ``はんぱん`` / ``でくv`` / nothing from the CTC stack this package retired.
:class:`SfxReader` is the reader for those lines and the speech alike: a crop
in, a string out, no detection. It is the
PaddleOCR-VL-1.6 base with a LoRA on the language model **and a fully
fine-tuned vision tower**, trained on the Manga109-s COO onomatopoeia
polygons plus a 1 : 1 replay of the ``<text>`` speech boxes (official COO
book split; COO test exact 81.7 %, doujin SFX gate 38 / 71 vs 2 stock —
``sorryhyun/paddleocr-vl-1.6-manga-lora``, Manga109-s attribution on the
card). It reads ``♡`` / ``ー`` / small kana natively, vertical or horizontal,
so it needs no symbol patching.

What it is not: a page reader. Feed it the crops a detector has already
boxed — the AnimeText text-block detector (:mod:`anime_tools.ocr.animetext`,
the OCR stage's detector: balloon lines and the SFX on the artwork alike);
:meth:`SfxReader.read_boxes_scored` cuts the padded crops for you. Batch by area: a
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
(the Apache-2.0 base, 1.9 GB) and ``sfx_reader`` (adapter 24 MB + tower
878 MB). :meth:`SfxReader.load` fetches what is missing.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import Any

from anime_tools.downloads import SFX_READER_ADAPTER_FILES, SFX_READER_TOWER_FILE

PROMPT = "OCR:"
"""The crop task of the base model's chat template; the fine-tune kept it."""

CROP_PAD = 0.12
"""Padding around a box, as a fraction of its longer side. The polygons hug the
glyphs and the model was trained on crops with this margin; a zero-pad crop
clips the strokes of the outer characters."""

MAX_NEW_TOKENS = 80
"""The tokenizer spends about one token per CJK character; COO's longest
onomatopoeia is 28 characters and a doujin speech balloon runs to ~60. Past
this a decode is a runaway, and the guard drops it anyway."""

MAX_PIXELS = 1280 * 28 * 28
"""The vision tower's pixel budget per crop (the training and eval setting)."""

MIN_LENGTH_CAP = 12
MIN_GLYPH_PX = 16
"""The smallest glyph a crop is assumed to hold (the crop builders' 16 px
floor): the length cap is how many of them fit in the crop's area."""

_LATEX_RE = re.compile(r"\\[(),]|\\\s")
_ELLIPSIS_RE = re.compile(r"[.．・…‥]{2,}")
"""A run of two or more dots of any kind — ASCII, fullwidth, the katakana
middle dot the reader uses for ``…``, or ``…`` / ``‥`` themselves. One
ellipsis, however it was drawn or decoded."""
ELLIPSIS = "…"
HEART_FOLD = str.maketrans({"♥": "♡", "❤": "♡"})
"""The reader emits ``♥`` and ``♡`` for the same drawn heart (sincos records:
332 vs 889); a caption says one glyph, and it is the one the artists draw."""


def is_runaway(text: str, *, ngram_repeats: int = 3, run: int = 8) -> bool:
    """The autoregressive failure class: ``ぉぉぉ…`` × 100, ``ふくっ`` × 100.

    A glyph run of ``run`` or more, or (from nine characters on) any 3-gram
    occurring ``ngram_repeats`` or more times. ``ぱんぱん`` (one repeat) and
    ``おおおん`` pass — a doubled unit *is* an onomatopoeia. Every dot run is
    one ``…`` first, so ``え...な...ない......`` is a pause and not a repeat; a
    hundred dots alone collapse to one and are then no letter at all
    (:func:`~anime_tools.ocr.reread.has_script`).
    """
    t = _ELLIPSIS_RE.sub(ELLIPSIS, "".join(text.split()))
    if re.search(rf"(.)\1{{{run - 1},}}", t):
        return True
    if len(t) >= 9:
        grams: dict[str, int] = defaultdict(int)
        for i in range(len(t) - 2):
            grams[t[i : i + 3]] += 1
        if max(grams.values()) >= ngram_repeats:
            return True
    return False


def length_cap(width: int, height: int) -> int:
    """How many characters a crop of this size can plausibly hold.

    Area over the smallest glyph (:data:`MIN_GLYPH_PX` squared), never under
    :data:`MIN_LENGTH_CAP`. A two-glyph crop caps near the floor, so a
    48-token runaway on it is rejected; a multi-column balloon block caps in
    the hundreds and is left to the repetition test — a block's character
    count is not readable off its shape (an aspect rule measured 2026-09-06
    threw away 60 % of the speech reads).
    """
    area = max(width, 0) * max(height, 0)
    return max(MIN_LENGTH_CAP, round(area / (MIN_GLYPH_PX * MIN_GLYPH_PX)))


def normalize_read(text: str) -> str:
    """The raw decode as a record: NFC-stable glyphs, every heart (``♥``, the
    emoji ``❤`` + variation selector) folded to ``♡``, every dot run
    (``......``, ``・・・・・・``, ``‥``) to one ``…``, VL's LaTeX wrapping of a
    measurement (``\\( 156 \\, cm \\)``) stripped, whitespace runs collapsed to
    one space (a column boundary stays a space, so a list stays a list).

    The folds are what make a line comparable: the speech clause dedupes on
    exact text (:func:`~anime_tools.captions.ocr_sfx.dedupe_speech`), and
    ``あ・・・っ♥`` beside ``あ・・・っ♡`` was two lines of the same word."""
    text = text.replace("\ufe0f", "").translate(HEART_FOLD)
    text = _LATEX_RE.sub("", text)
    text = unicodedata.normalize("NFC", text)
    text = _ELLIPSIS_RE.sub(ELLIPSIS, text)
    return " ".join(text.split())


def guard(text: str, width: int, height: int) -> str | None:
    """The read, or ``None`` when it is not one.

    Rejects an empty decode, a runaway (:func:`is_runaway`) and a read longer
    than :func:`length_cap` for the crop's shape. Everything that passes is
    :func:`normalize_read`'s form.
    """
    read = normalize_read(text)
    if not read or is_runaway(read):
        return None
    if sum(1 for ch in read if not ch.isspace()) > length_cap(width, height):
        return None
    return read


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


def crop_box(bgr, box: Sequence[int], pad: float = CROP_PAD):
    """The padded crop of an axis-aligned box, or ``None`` if it is empty."""
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = pad_box(box, w, h, pad)
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


class SfxWeightsMissing(RuntimeError):
    """A reader directory the catalog has not filled yet."""

    def __init__(self, model_dir: Path, row: str) -> None:
        super().__init__(
            f"SFX reader weights not found in {model_dir} — run "
            f"`python -m anime_tools.downloads {row}` "
            "(or ⚙ Settings › Models in the GUI)"
        )


def _ensure(row_id: str, fetch: bool) -> Path:
    from anime_tools.downloads import by_id

    row = by_id()[row_id]
    if row.missing():
        if not fetch:
            raise SfxWeightsMissing(row.dest, row_id)  # type: ignore[arg-type]
        row.fetch()
    assert row.dest is not None
    return row.dest


@dataclass
class SfxReader:
    """The fine-tuned VL crop reader, loaded once, read many."""

    model: Any
    processor: Any
    device: str
    batch_size: int = 16
    min_edge: int = 28

    @classmethod
    def load(
        cls,
        *,
        device: str | None = None,
        base_dir: Path | None = None,
        adapter_dir: Path | None = None,
        batch_size: int = 16,
        fetch: bool = True,
    ) -> SfxReader:
        """Base + adapter merged + fine-tuned tower, in bf16 on ``device``
        (auto when ``None``).

        ``base_dir`` / ``adapter_dir`` default to the catalog rows and are
        fetched when missing (``fetch=False`` raises :class:`SfxWeightsMissing`
        instead). A dir passed explicitly is used as is.

        bf16 holds on every device, MPS included — measured 2026-09-09 at 860 ms
        a crop there against a CPU's 4960, same decodes.
        """
        from anime_tools._device import resolve_device

        device = resolve_device(device)
        base = Path(base_dir) if base_dir else _ensure("vl16_base", fetch)
        adapter = Path(adapter_dir) if adapter_dir else _ensure("sfx_reader", fetch)
        for name in (*SFX_READER_ADAPTER_FILES, SFX_READER_TOWER_FILE):
            if not (adapter / name).is_file():
                raise SfxWeightsMissing(adapter, "sfx_reader")

        import torch
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model = AutoModelForImageTextToText.from_pretrained(
            str(base), dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
        # The tower is a plain state dict under the base model's key names; the
        # adapter only ever touched the language model.
        tower = load_file(str(adapter / SFX_READER_TOWER_FILE))
        unexpected = model.load_state_dict(tower, strict=False).unexpected_keys
        if unexpected:
            raise RuntimeError(
                f"tower.safetensors carries keys the base has no home for: {unexpected[:5]}"
            )
        model = model.to(device).eval()
        processor = AutoProcessor.from_pretrained(str(base))
        return cls(
            model=model,
            processor=processor,
            device=device,
            batch_size=batch_size,
            min_edge=int(processor.image_processor.size["shortest_edge"]),
        )

    @cached_property
    def prompt(self) -> str:
        """The chat-templated prompt every crop is read under.

        One string for the life of the reader: it depends on nothing but the
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
        in, ``(text, confidence)`` out), in input order. Unguarded —
        :meth:`read_scored` is the one to call.

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
    ) -> list[tuple[str, float] | None]:
        """One page, many boxes: :func:`crop_box` each, :meth:`read_scored`
        them all. An empty box reads as ``None``. What
        :class:`~anime_tools.ocr.reread.RereadEngine` calls."""
        crops, owners = [], []
        for i, box in enumerate(boxes):
            crop = crop_box(bgr, box, pad)
            if crop is not None and crop.size:
                crops.append(crop)
                owners.append(i)
        reads = self.read_scored(crops)
        out: list[tuple[str, float] | None] = [None] * len(boxes)
        for i, r in zip(owners, reads, strict=True):
            out[i] = r
        return out


__all__ = [
    "CROP_PAD",
    "ELLIPSIS",
    "HEART_FOLD",
    "MAX_NEW_TOKENS",
    "PROMPT",
    "SfxReader",
    "SfxWeightsMissing",
    "crop_box",
    "guard",
    "is_runaway",
    "length_cap",
    "normalize_read",
    "pad_box",
    "token_confidence",
]
