"""Where every model weight lives — the one spelling of each repo, filename and
destination path. Torch-free, and the leaf of this package: the loaders import
their repo / filename / default path from here (through
:mod:`anime_tools.downloads`, which re-exports it) so a Download button cannot
write where a loader will not look.

Destinations are of two kinds: a path under the curation home (``models/…``),
where a ``--checkpoint`` / ``--tagger_dir`` default points, so the file has to
land exactly there; or the HuggingFace hub cache, for assets whose loader
fetches by repo id — probed with :func:`anime_tools._hf.hf_file_cached`, never
by guessing a cache path.
"""

from __future__ import annotations

import os
from pathlib import Path

from anime_tools._env import models_dir, resolve_path

# The Anima Tagger — our half (vocab / rules / thresholds / groups / sidecar)
# from ``sorryhyun/anima-tagger``. The pixai checkpoint sits under ``pixai/``;
# the repo root still holds the dbv4 one, which every anime_tools <= 0.8
# fetches. The backbone weights are not here: they ship from their upstream
# repo into the hub cache (:mod:`anime_tools.tagger.dbv4_meta`).
TAGGER_HF_REPO = "sorryhyun/anima-tagger"
TAGGER_HF_SUBFOLDER = "pixai"
DEFAULT_TAGGER_DIR = "models/captioners/anima-tagger-pixai"

# SAM 3 — gated on the Hub. The CLIs pass ``--checkpoint`` with
# ``load_from_HF=False``, so the file must land on this path, not merely in the
# hub cache sam3's own downloader would use.
SAM3_REPO = "facebook/sam3"
SAM3_FILENAME = "sam3.pt"
SAM3_DIR = "models/sam3"
DEFAULT_SAM3_CHECKPOINT = f"{SAM3_DIR}/{SAM3_FILENAME}"

# PE-Spatial-B16-512 — weights for the grouping embedder's frozen tower
# (:mod:`anime_tools.vision.pe` builds the architecture).
PE_SPATIAL_REPO = "facebook/PE-Spatial-B16-512"
PE_SPATIAL_FILENAME = "PE-Spatial-B16-512.pt"

# SAM3 subject soft prompt — textual inversion of ``anime girl``, the default
# ``--prompt_embed`` of the position stage and the multiview audit. Not on the
# Hub, so it comes straight from GitHub.
SOFT_PROMPT_GH_REPO = "sorryhyun/anima_lora"
SOFT_PROMPT_DIR = "networks/calibration"
SOFT_PROMPT_FILENAME = "sam3_girl_prompt.safetensors"
DEFAULT_SUBJECT_PROMPT_EMBED = f"{SOFT_PROMPT_DIR}/{SOFT_PROMPT_FILENAME}"
SOFT_PROMPT_URL = (
    f"https://raw.githubusercontent.com/{SOFT_PROMPT_GH_REPO}/main/{SOFT_PROMPT_DIR}"
)

# PaddleOCR-VL-1.6 — the 0.9 B VLM base the OCR reader is fine-tuned from
# (Apache-2.0; ERNIE LM + NaViT tower). A flat checkpoint dir, so the remote
# modeling files ride along with the weights. `anime_tools.ocr.reader` reads it.
VL16_BASE_REPO = "PaddlePaddle/PaddleOCR-VL-1.6"
VL16_BASE_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "configuration_paddleocr_vl.py",
    "image_processing_paddleocr_vl.py",
    "modeling_paddleocr_vl.py",
    "processing_paddleocr_vl.py",
)
VL16_BASE_DIR = "models/paddleocr_vl_1.6"

# The manga OCR reader, **v8_preview** under the repo's ``v8_preview/`` subfolder: a peft
# LoRA on the base's vision tower + projector, its fully fine-tuned ERNIE decoder layers,
# and the embedding / ``lm_head`` rows of two tokens it adds (``♡``, ``💢``), trained on
# AnimeText detector boxes over Manga109-s (the main text of each box), KO / ZH
# pseudo-labelled SFX and KO speech (attribution and citations on the model card). Weights
# only — no Manga109-s image ships. ``v7/`` and ``v6/`` hold the previous versions; v1–v5
# and the two-prompt v7 preview were removed from the head (2026-11-18, 2026-11-24) and stay
# at their commits below.
OCR_READER_REPO = "sorryhyun/paddleocr-vl-1.6-manga-lora"
OCR_READER_SUBFOLDER = "v8_preview"
OCR_READER_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
OCR_READER_LM_FILE = "lm.safetensors"
OCR_READER_VOCAB_FILE = "vocab.safetensors"
"""The rows of the tokens a reader adds to the base tokenizer (``embed`` / ``lm_head``,
the tokens in the file's ``tokens`` metadata); v8_preview has one, v7 and v6 do not."""
OCR_READER_LEGACY_FILES = (*OCR_READER_ADAPTER_FILES, OCR_READER_LM_FILE)
"""What the v7 and v6 rows fetch: no added tokens."""
OCR_READER_FILES = (*OCR_READER_LEGACY_FILES, OCR_READER_VOCAB_FILE)
OCR_READER_DIR = "models/paddleocr_vl_1.6_manga_lora"
OCR_READER_REVISION = "e018dac5690e76d2fd4f2563ab42ba5a62d2f5ca"
"""Hub commit of the reader the package ships — **v8_preview** (``mainsub_1127_2337``:
v7's recipe on the labels reviewed since, 3 epochs on the full data, ``♥`` and ``♡`` read
apart, a drawn-out run read as one ``ー``; weights at ``3ac66a6b``, cards since). ``OCR:``
reads the main text of a detector box cut with a 4 px margin; its ``OCR2:`` prompt for the
rest of the box is not used here. v7 ``169eacbc`` in ``v7/`` (``mainsub_1124_1923``; the
first v7 weights, 0.14.0, at ``59768fc6``; the two-prompt v7 preview at ``1987992e`` in
``v7_preview/``). v6 ``57e7c650`` in ``v6/``: ``vl16_p2b_kozh_nl_swaprelax``, v5 plus
20,745 proofread Korean speech lines; v5 ``fb08a10b`` in ``v5/`` — v4's recipe with the
labels' line breaks kept as ``\\n`` and the crops cut at the page edge; v4 ``926eafb1`` in
``v4/``, its 1-epoch run ``9fe3d41e``; v3 ``26292839``, v2 ``4caffe65``, v1 ``3b5fe022``, at
the repo root of those commits). The adapter file names are shared across versions, so
the catalog row pins this and stamps it under the dest — an install that predates the
pin re-fetches instead of reading an older adapter under the new name."""

# The opt-in v7 (the default from 0.14.0 to 0.16.x) in ``v7/``: the same task and crops,
# no added tokens (hearts read as ``♥``). Its own dest, so it installs beside the default.
OCR_READER_V7_SUBFOLDER = "v7"
OCR_READER_V7_DIR = "models/paddleocr_vl_1.6_manga_lora_v7"
OCR_READER_V7_REVISION = "169eacbcd325e2eb93d9ca7c8606ab4248745df8"

# The opt-in v6 (the default until 0.14.0) in ``v6/``: one prompt on 12 %-margin crops,
# the better reader for polygon crops of sound effects (COO test SFX 85.3 % against
# v8_preview's 80.9 %, both scored with the current folds). Its own dest.
OCR_READER_V6_SUBFOLDER = "v6"
OCR_READER_V6_DIR = "models/paddleocr_vl_1.6_manga_lora_v6"
OCR_READER_V6_REVISION = "57e7c65024d753022299340b3b9b6aafc158a79b"

# The AnimeText text-block detector: a YOLO12-l trained on deepghs/AnimeText
# (735k anime / manga pages, one class), the OCR stage's one detector.
# **Runtime download only, never bundled**: the model card is
# GPL-3.0 and the dataset CC-BY-NC-SA-4.0 — neither may ship inside this MIT
# package, the trainer, or a node. ``threshold.json`` carries the card's F1
# threshold (0.426) for reference; the detector's own default is lower.
ANIMETEXT_REPO = "deepghs/AnimeText_yolo"
ANIMETEXT_SUBFOLDER = "yolo12l_animetext"
ANIMETEXT_WEIGHTS = "model.pt"
ANIMETEXT_FILES = (ANIMETEXT_WEIGHTS, "threshold.json")
ANIMETEXT_DIR = "models/animetext"

# Danbooru tag KB — the ~114k-row classified tag table the correction pass types
# tags against and the GUI's tag panel reads. A CSV in a GitHub repo, so it
# rides ``_fetch_http``. Its descriptions are Korean; the English sibling is
# built, not hosted.
DANBOORU_TAGS_GH_REPO = "Localsmile/danbooru_KR_wiki_tag_search"
DANBOORU_TAGS_URL = f"https://raw.githubusercontent.com/{DANBOORU_TAGS_GH_REPO}/main"

# The Danbooru wiki mirrored as one parquet on the Hub — an input to the
# ``danbooru_tags_en`` row's join, so the 45 MB file stays in the hub cache.
DANBOORU_WIKI_REPO = "isek-ai/danbooru-wiki-2024"
DANBOORU_WIKI_FILE = "data/train-00000-of-00001.parquet"


def default_pe_spatial_path() -> Path:
    """``<models_dir>/pe/PE-Spatial-B16-512.pt``."""
    return models_dir() / "pe" / PE_SPATIAL_FILENAME


def default_vl16_base_dir() -> Path:
    """``<home>/models/paddleocr_vl_1.6`` — the OCR reader's base checkpoint."""
    return resolve_path(VL16_BASE_DIR)


def default_ocr_reader_dir() -> Path:
    """``<home>/models/paddleocr_vl_1.6_manga_lora`` — adapter + fine-tuned
    tower, what :class:`anime_tools.ocr.reader.OcrReader` loads."""
    return resolve_path(OCR_READER_DIR)


def default_ocr_reader_v7_dir() -> Path:
    """``<home>/models/paddleocr_vl_1.6_manga_lora_v7`` — the opt-in v7 reader
    (``OcrReader.load(version="v7")``)."""
    return resolve_path(OCR_READER_V7_DIR)


def default_ocr_reader_v6_dir() -> Path:
    """``<home>/models/paddleocr_vl_1.6_manga_lora_v6`` — the opt-in v6 reader
    (``OcrReader.load(version="v6")``)."""
    return resolve_path(OCR_READER_V6_DIR)


def default_animetext_dir() -> Path:
    """``<home>/models/animetext`` — the AnimeText detector's weights, what
    :class:`anime_tools.ocr.animetext.AnimeTextDetector` loads."""
    return resolve_path(ANIMETEXT_DIR)


def http_timeout() -> float:
    """Socket timeout for the plain-HTTPS rows; shares ``ANIMA_HF_TIMEOUT``."""
    return float(os.environ.get("ANIMA_HF_TIMEOUT", "30"))
