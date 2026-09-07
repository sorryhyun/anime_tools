"""Catalog of every model weight the stages need: repo, destination and an
offline installed-probe. Torch-free; the loaders import their repo / filename /
default path from here so a Download button can't write where a loader won't
look.

Destinations are of two kinds: a path under the curation home (``models/…``),
where a ``--checkpoint`` / ``--tagger_dir`` default points, so the file has to
land exactly there; or the HuggingFace hub cache, for assets whose loader
fetches by repo id — probed with :func:`anime_tools._hf.hf_file_cached`, never
by guessing a cache path. Nothing here is auto-run; loaders still fetch on
first use.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anime_tools._env import models_dir, resolve_path
from anime_tools.captions.correction import TAG_CSV_EN_NAME, TAG_CSV_NAME
from anime_tools.tagger.dbv4_meta import (
    DBV4_BACKBONE_FILES,
    DBV4_ONNX_NAME,
    DBV4_OPTIONAL_FILES,
    DBV4_REQUIRED_FILES,
    DEFAULT_TAGGER_DIR,
    TAGGER_HF_REPO,
    TAGGER_HF_SUBFOLDER,
    backbone_repo_for,
)

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

# MIT text-mask net; its loader reads it straight out of the hub cache, so
# there is no path under models/ to keep in sync.
MIT_TEXT_REPO = "a-b-c-x-y-z/Manga-Text-Segmentation-2025"
MIT_TEXT_FILENAME = "model.pth"

# ComicTextDetector — the text-BLOCK head the MIT stage gates its UNet++ mask
# on (``--ctd-gate``). A manga-image-translator release asset, so it rides
# ``_fetch_http``. The stage has no flag for it: this is the one path it reads.
CTD_GH_REPO = "zyddnys/manga-image-translator"
CTD_ONNX_RELEASE = "beta-0.3"
CTD_ONNX_DIR = "models/mit"
CTD_ONNX_FILENAME = "comictextdetector.pt.onnx"
CTD_ONNX_URL = f"https://github.com/{CTD_GH_REPO}/releases/download/{CTD_ONNX_RELEASE}"

# PaddleOCR-VL-1.6 — the 0.9 B VLM base the SFX reader is fine-tuned from
# (Apache-2.0; ERNIE LM + NaViT tower). A flat checkpoint dir, so the remote
# modeling files ride along with the weights. `anime_tools.ocr.sfx` reads it.
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

# The manga SFX reader: a peft LoRA on the base's language model plus its
# fully fine-tuned vision tower, trained on Manga109-s / COO (attribution and
# citations on the model card). Weights only — no Manga109-s image ships.
SFX_READER_REPO = "sorryhyun/paddleocr-vl-1.6-manga-lora"
SFX_READER_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
SFX_READER_TOWER_FILE = "tower.safetensors"
SFX_READER_FILES = (*SFX_READER_ADAPTER_FILES, SFX_READER_TOWER_FILE)
SFX_READER_DIR = "models/paddleocr_vl_1.6_manga_lora"

# The AnimeText text-block detector: a YOLO12-l trained on deepghs/AnimeText
# (735k anime / manga pages, one class), the OCR stage's one detector.
# **Runtime download only, never bundled**: the model card is
# GPL-3.0 and the dataset CC-BY-NC-SA-4.0 — neither may ship inside this MIT
# package, the trainer, or a node. ``threshold.json`` carries the card's F1
# threshold (0.426) for reference; the detector's own default is lower.
ANIMETEXT_REPO = "deepghs/AnimeText_yolo"
ANIMETEXT_SUBFOLDER = "yolo12l_animetext"
ANIMETEXT_ONNX = "model.onnx"
ANIMETEXT_FILES = (ANIMETEXT_ONNX, "threshold.json")
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


def default_ctd_onnx_path() -> Path:
    """``<home>/models/mit/comictextdetector.pt.onnx`` — where ``--ctd-gate``
    looks and what the catalog row writes; there is no flag for it."""
    return resolve_path(CTD_ONNX_DIR) / CTD_ONNX_FILENAME


def default_vl16_base_dir() -> Path:
    """``<home>/models/paddleocr_vl_1.6`` — the SFX reader's base checkpoint."""
    return resolve_path(VL16_BASE_DIR)


def default_sfx_reader_dir() -> Path:
    """``<home>/models/paddleocr_vl_1.6_manga_lora`` — adapter + fine-tuned
    tower, what :class:`anime_tools.ocr.sfx.SfxReader` loads."""
    return resolve_path(SFX_READER_DIR)


def default_animetext_dir() -> Path:
    """``<home>/models/animetext`` — the AnimeText detector's ONNX, what
    :class:`anime_tools.ocr.animetext.AnimeTextDetector` loads."""
    return resolve_path(ANIMETEXT_DIR)


def http_timeout() -> float:
    """Socket timeout for the plain-HTTPS rows; shares ``ANIMA_HF_TIMEOUT``."""
    return float(os.environ.get("ANIMA_HF_TIMEOUT", "30"))


def _say(msg: str) -> None:
    print(msg, flush=True)


def _size(n: int) -> str:
    return f"{n / 1e6:,.0f} MB" if n >= 1e6 else f"{n / 1e3:,.0f} KB"


@dataclass(frozen=True)
class Pack:
    """A group of rows that install together — what a Download button on a
    Models pane is a button *for*. Every :class:`Asset` names one."""

    id: str
    title: str
    description: str = ""


PACKS: tuple[Pack, ...] = (
    Pack(
        "tagger",
        "Tagger",
        "The Anima Tagger: its checkpoint, the gated dbv4 backbone, and the "
        "ONNX graph traced from it.",
    ),
    Pack(
        "tags",
        "Danbooru tag DB",
        "The ~114k-row tag table caption correction types against, and its "
        "English descriptions.",
    ),
    Pack(
        "masking",
        "Masking",
        "SAM3 subject masks, and the subject soft prompt the position stages "
        "detect with.",
    ),
    Pack(
        "text_mask",
        "Text masking (MIT)",
        "The UNet++ manga text segmenter and its ComicTextDetector gate.",
    ),
    Pack(
        "ocr",
        "OCR",
        "The AnimeText text-block detector and the manga VL reader "
        "(PaddleOCR-VL-1.6 base + the SFX fine-tune).",
    ),
    Pack("grouping", "Grouping", "PE-Spatial-B16-512, the near-twin grouping tower."),
)
"""Display order. A row's :attr:`Asset.pack` is one of these ids; the CLI and
both GUIs accept a pack id wherever they accept a row id."""

PACK_BY_ID: dict[str, Pack] = {p.id: p for p in PACKS}


@dataclass(frozen=True)
class Asset:
    """One downloadable model, resolved against the current curation home."""

    id: str
    title: str
    repo: str
    files: tuple[str, ...]
    """Required files; all of them present means installed."""
    used_by: str
    """Which stages stop working without it."""
    stages: tuple[str, ...] = ()
    """The same, as GUI stage ids, so the stage bar can warn before a run."""
    pack: str = ""
    """Which :data:`PACKS` entry this row installs under. Every catalog row
    names one (pinned by test); the default only spares a caller building a
    row of its own."""
    dest: Path | None = None
    """Directory the files are flattened into; ``None`` = the HF hub cache."""
    url: str = ""
    """Base URL each file hangs off for rows that are not on the Hub. Set it and
    ``repo`` is only a label; requires ``dest``."""
    subfolder: str = ""
    """Path prefix inside the repo (the tagger ships under ``dbv4/``)."""
    repo_type: str = "model"
    """Hub repo kind — ``dataset`` for the Danbooru wiki mirror."""
    derived: tuple[str, ...] = ()
    """Files this row *makes* under ``dest`` via :attr:`build`. The probe asks
    for these, not the downloads, so a hub-cache sweep can't turn a built row
    back to "missing"."""
    build: Callable[[Path, Callable[[str], None]], None] | None = None
    """Post-fetch step that writes :attr:`derived` into ``dest``."""
    optional: tuple[str, ...] = field(default_factory=tuple)
    """Best-effort files: a 404 means this checkpoint doesn't ship one."""
    gated: str = ""
    """Accept-the-terms URL when the repo is gated; empty when it is public."""
    notes: str = ""

    @property
    def location(self) -> str:
        return str(self.dest) if self.dest is not None else "Hugging Face cache"

    def missing(self) -> list[str]:
        """Required files that are not here yet. Never touches the network."""
        if self.derived:
            assert self.dest is not None, "a built row needs a dest to write into"
            return [f for f in self.derived if not (self.dest / f).exists()]
        if self.dest is None:
            from anime_tools._hf import hf_file_cached

            return [f for f in self.files if not hf_file_cached(self.repo, f)]
        return [f for f in self.files if not (self.dest / Path(f).name).exists()]

    @property
    def installed(self) -> bool:
        return not self.missing()

    def to_dict(self) -> dict[str, Any]:
        missing = self.missing()
        return {
            "id": self.id,
            "title": self.title,
            "repo": self.repo,
            "files": list(self.files),
            "used_by": self.used_by,
            "stages": list(self.stages),
            "pack": self.pack,
            "location": self.location,
            "installed": not missing,
            "missing": missing,
            "gated": self.gated,
            "notes": self.notes,
        }

    @property
    def _hint(self) -> str:
        return (
            f"hf auth login, then accept the terms at {self.gated}"
            if self.gated
            else f"python -m anime_tools.downloads {self.id}"
        )

    def _fetch_http(self, log: Callable[[str], None]) -> None:
        """Plain-HTTPS download for a ``url`` row. Bounded: a stalled socket
        must raise, not hang the job slot the GUI runs this in."""
        import urllib.error
        import urllib.request

        assert self.dest is not None, "a url row needs a dest to download into"
        for name in (*self.files, *self.optional):
            src = f"{self.url}/{name}"
            final = self.dest / Path(name).name
            log(f"  {src}")
            part = final.with_name(final.name + ".part")
            try:
                with urllib.request.urlopen(src, timeout=http_timeout()) as r:
                    part.write_bytes(r.read())
            except (OSError, urllib.error.URLError) as exc:
                part.unlink(missing_ok=True)
                if name in self.optional:
                    log(f"    optional — {type(exc).__name__}, skipped")
                    continue
                raise FileNotFoundError(
                    f"{self.title} ({name}): download from {src} failed "
                    f"({type(exc).__name__}: {exc}). Check connectivity, then "
                    f"re-run `{self._hint}`."
                ) from exc
            part.replace(final)
            log(f"    ok  {final}  ({_size(final.stat().st_size)})")

    def fetch(self, log: Callable[[str], None] = _say) -> None:
        """Download every required file; optional ones are best-effort.

        Network and gated-repo failures raise ``FileNotFoundError`` naming the
        asset and the recovery, not a hub traceback.
        """
        from huggingface_hub.utils import EntryNotFoundError

        from anime_tools._hf import hf_download

        if self.dest is not None:
            self.dest.mkdir(parents=True, exist_ok=True)
        if self.url:
            self._fetch_http(log)
            self._build(log)
            return

        hint = self._hint
        # A built row's downloads are inputs and stay in the hub cache; only
        # what ``build`` writes belongs in ``dest``.
        into = None if self.build else self.dest
        for name in (*self.files, *self.optional):
            remote = f"{self.subfolder}/{name}" if self.subfolder else name
            log(f"  {self.repo}/{remote}")
            try:
                got = Path(
                    hf_download(
                        what=f"{self.title} ({name})",
                        hint=hint,
                        repo_id=self.repo,
                        repo_type=self.repo_type,
                        filename=remote,
                        **({"local_dir": str(into)} if into else {}),
                    )
                )
            except EntryNotFoundError:
                if name in self.optional:
                    log(f"    optional — not published by {self.repo}, skipped")
                    continue
                raise
            if into is not None:
                # local_dir keeps the repo's subfolder layout; the loaders want
                # a flat checkpoint dir.
                final = into / Path(name).name
                if got.resolve() != final.resolve():
                    shutil.move(str(got), str(final))
                got = final
            log(f"    ok  {got}  ({_size(got.stat().st_size)})")
        self._build(log)
        if into is not None and self.subfolder:
            # Drop the now-empty `dbv4/` local_dir the files were moved out of.
            leftover = into / self.subfolder
            if leftover.is_dir() and not any(leftover.iterdir()):
                leftover.rmdir()

    def _build(self, log: Callable[[str], None]) -> None:
        """Run the post-fetch step, if this row has one."""
        if self.build is None:
            return
        assert self.dest is not None, "a built row needs a dest to write into"
        self.build(self.dest, log)


def _build_english_tag_csv(dest: Path, log: Callable[[str], None]) -> None:
    """``danbooru_tags_en``'s post-fetch step: join the cached wiki parquet
    against the base CSV and write the English CSV."""
    from anime_tools.tagger.cli.build_english_tag_csv import build

    build(dest / TAG_CSV_NAME, dest / TAG_CSV_EN_NAME, revision=None, log=log)


def _export_dbv4_onnx(dest: Path, log: Callable[[str], None]) -> None:
    """``tagger_onnx``'s post-fetch step: build ``<ckpt_dir>/dbv4.onnx``.

    The backbone files the row just fetched are the export's *input*, read out of
    the hub cache by ``config.json['dbv4']['repo']`` — so this has to run after the
    ``tagger`` row, which is what puts that config there.
    """
    # Checked before the import, which is what pulls torch in: a missing
    # checkpoint should answer in milliseconds.
    if not (dest / "config.json").is_file():
        raise FileNotFoundError(
            f"no tagger checkpoint at {dest} to export against — "
            "`python -m anime_tools.downloads tagger` first"
        )
    from anime_tools.tagger.onnx_export import (
        export_for_checkpoint,
        quiet_exporter_logs,
    )

    quiet_exporter_logs()
    log(f"  tracing the backbone into {dest / DBV4_ONNX_NAME} (a few minutes)")
    out = export_for_checkpoint(dest, overwrite=True)
    log(f"    ok  {out}  ({_size(out.stat().st_size)})")


def catalog() -> tuple[Asset, ...]:
    """The full catalog, rebuilt per call: the home moves with
    ``ANIME_TOOLS_HOME`` and the backbone repo follows the installed
    checkpoint."""
    tagger_dir = resolve_path(DEFAULT_TAGGER_DIR)
    backbone = backbone_repo_for(tagger_dir)
    return (
        Asset(
            id="tagger",
            pack="tagger",
            title="Anima Tagger checkpoint",
            repo=TAGGER_HF_REPO,
            subfolder=TAGGER_HF_SUBFOLDER,
            files=DBV4_REQUIRED_FILES,
            optional=DBV4_OPTIONAL_FILES,
            dest=tagger_dir,
            used_by="Autotag captions · Position captions · Multiview audit",
            stages=("autotag", "position", "audit"),
            notes="Vocab, rules, groups, thresholds, sidecar. Small — the "
            "weights are the backbone below.",
        ),
        Asset(
            id="tagger_backbone",
            pack="tagger",
            title="dbv4 tagger backbone",
            repo=backbone,
            files=DBV4_BACKBONE_FILES,
            used_by="the Anima Tagger checkpoint above",
            stages=("autotag", "position", "audit"),
            gated=f"https://huggingface.co/{backbone}",
            notes="GPL-3.0 and never vendored; the terms auto-approve.",
        ),
        Asset(
            id="tagger_onnx",
            pack="tagger",
            title="Anima Tagger ONNX graph",
            repo=backbone,
            files=DBV4_BACKBONE_FILES,
            derived=(DBV4_ONNX_NAME,),
            build=_export_dbv4_onnx,
            dest=tagger_dir,
            used_by="speed only: Autotag captions · Position captions · Multiview audit",
            # No stage ids on purpose. The stage bar's warning promises "the
            # first Run fetches them itself", and nothing auto-builds this one —
            # without it the three stages run on timm, slower and identical.
            stages=(),
            gated=f"https://huggingface.co/{backbone}",
            notes="Built here, never downloaded: the backbone is GPL-3.0 and "
            "gated, so every user traces their own 539 MB graph (a few minutes, "
            "once). Its presence makes the tagger run on onnxruntime, ~3.7x "
            "faster than timm; ANIMA_TAGGER_BACKEND=torch opts back out without "
            "deleting it.",
        ),
        Asset(
            id="sam3",
            pack="masking",
            title="SAM 3",
            repo=SAM3_REPO,
            files=(SAM3_FILENAME,),
            dest=resolve_path(SAM3_DIR),
            used_by="Position captions · Multiview audit · SAM3 subject masks",
            stages=("position", "audit", "masks_sam"),
            gated=f"https://huggingface.co/{SAM3_REPO}",
            notes="Lands on the --checkpoint default of every SAM3 stage.",
        ),
        Asset(
            id="pe_spatial",
            pack="grouping",
            title="PE-Spatial-B16-512",
            repo=PE_SPATIAL_REPO,
            files=(PE_SPATIAL_FILENAME,),
            dest=default_pe_spatial_path().parent,
            used_by="Build groups (near-twin / same-concept grouping)",
            stages=("groups",),
        ),
        Asset(
            id="soft_prompt",
            pack="masking",
            title="SAM3 subject soft prompt",
            repo=SOFT_PROMPT_GH_REPO,
            url=SOFT_PROMPT_URL,
            files=(SOFT_PROMPT_FILENAME,),
            dest=resolve_path(SOFT_PROMPT_DIR),
            used_by="Position captions · Multiview audit (subject detection)",
            stages=("position", "audit"),
            notes="The default --prompt_embed (161 KB); without it both "
            "stages fall back to the prompt `girl` and find fewer subjects.",
        ),
        Asset(
            id="danbooru_tags",
            pack="tags",
            title="Danbooru tag KB",
            repo=DANBOORU_TAGS_GH_REPO,
            url=DANBOORU_TAGS_URL,
            files=(TAG_CSV_NAME,),
            dest=models_dir(),
            used_by="Correct + mirror captions · the caption panel's tag descriptions",
            stages=("correct",),
            notes="~114k tags with category, post count and a wiki blurb. "
            "Blurbs are Korean; the row below rewrites them in English.",
        ),
        Asset(
            id="danbooru_tags_en",
            pack="tags",
            title="Danbooru tag descriptions (English)",
            repo=DANBOORU_WIKI_REPO,
            repo_type="dataset",
            files=(DANBOORU_WIKI_FILE,),
            derived=(TAG_CSV_EN_NAME,),
            build=_build_english_tag_csv,
            dest=models_dir(),
            used_by="the caption panel's tag descriptions",
            stages=(),
            notes="Optional; needs the row above. Rewrites its blurbs in "
            "English, which the caption panel prefers. The 45 MB mirror it "
            "joins stays in the hub cache.",
        ),
        Asset(
            id="vl16_base",
            pack="ocr",
            title="PaddleOCR-VL-1.6 base",
            repo=VL16_BASE_REPO,
            files=VL16_BASE_FILES,
            dest=default_vl16_base_dir(),
            used_by="OCR text (the manga VL reader's base; anime_tools.ocr.sfx)",
            stages=("ocr",),
            notes="1.9 GB, Apache-2.0. The VLM the SFX reader below is a "
            "fine-tune of; the reader loads this and merges the row below "
            "onto it.",
        ),
        Asset(
            id="sfx_reader",
            pack="ocr",
            title="Manga SFX reader (VL-1.6 LoRA + tower)",
            repo=SFX_READER_REPO,
            files=SFX_READER_FILES,
            dest=default_sfx_reader_dir(),
            used_by="OCR text (the manga VL reader; anime_tools.ocr.sfx)",
            stages=("ocr",),
            notes="0.9 GB (adapter 24 MB + fine-tuned vision tower). Reads every "
            "box the detector finds — balloon speech, hearts, small kana and the "
            "hand-lettered onomatopoeia a print recognizer garbles; a crop reader, "
            "not a detector. Needs the base above; both fetched on first use. "
            "Trained on Manga109-s (COO).",
        ),
        Asset(
            id="animetext_det",
            pack="ocr",
            title="AnimeText text-block detector (YOLO12-l)",
            repo=ANIMETEXT_REPO,
            subfolder=ANIMETEXT_SUBFOLDER,
            files=ANIMETEXT_FILES,
            dest=default_animetext_dir(),
            used_by="OCR text (the text-block detector; anime_tools.ocr.animetext)",
            stages=("ocr",),
            notes="106 MB ONNX. One detector for balloon lines and the SFX drawn "
            "onto the artwork; every box it finds is read by the manga VL reader. "
            "Fetched on first use, never bundled: weights GPL-3.0, training data "
            "CC-BY-NC-SA-4.0 — a shipped build defaulting to it is a licence call.",
        ),
        Asset(
            id="mit_text",
            pack="text_mask",
            title="Manga text segmentation",
            repo=MIT_TEXT_REPO,
            files=(MIT_TEXT_FILENAME,),
            used_by="MIT text masks",
            stages=("masks_mit",),
        ),
        Asset(
            id="ctd_onnx",
            pack="text_mask",
            title="ComicTextDetector text-block head",
            repo=CTD_GH_REPO,
            url=CTD_ONNX_URL,
            files=(CTD_ONNX_FILENAME,),
            dest=default_ctd_onnx_path().parent,
            used_by="MIT text masks (the --ctd-gate precision pass)",
            stages=("masks_mit",),
            notes="95 MB. Without it --ctd-gate degrades to raw UNet++ masks, "
            "which false-positive on decorative line art.",
        ),
    )


def by_id() -> dict[str, Asset]:
    return {a.id: a for a in catalog()}


def by_pack(rows: tuple[Asset, ...] | None = None) -> dict[str, tuple[Asset, ...]]:
    """Rows bucketed by pack, in :data:`PACKS` order, catalog order inside each;
    a pack with no rows is left out. ``rows`` defaults to the catalog."""
    rows = catalog() if rows is None else rows
    out: dict[str, tuple[Asset, ...]] = {}
    for pack in PACKS:
        picked = tuple(a for a in rows if a.pack == pack.id)
        if picked:
            out[pack.id] = picked
    return out


def expand(names: list[str] | tuple[str, ...]) -> list[str]:
    """Row ids and/or pack ids → row ids, catalog order, deduped.

    Raises ``KeyError`` naming the first token that is neither, so a typo fails
    loudly rather than downloading nothing.
    """
    assets = by_id()
    packed = by_pack()
    picked: list[str] = []
    for name in names:
        if name in packed:
            ids = [a.id for a in packed[name]]
        elif name in assets:
            ids = [name]
        else:
            raise KeyError(
                f"unknown model or pack id {name!r} — rows: {', '.join(assets)}; "
                f"packs: {', '.join(packed)}"
            )
        picked.extend(i for i in ids if i not in picked)
    order = list(assets)
    return sorted(picked, key=order.index)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m anime_tools.downloads",
        description="Pre-fetch the model weights the stages need. With no ID, "
        "downloads everything that is missing; with IDs, re-fetches exactly "
        "those (a repair). An ID is a row (`sam3`) or a pack (`ocr` — every row "
        "that installs under it). Every loader still auto-fetches on first use — "
        "this only moves the wait somewhere you chose.",
    )
    p.add_argument(
        "ids",
        nargs="*",
        metavar="ID",
        help="Row or pack ids to fetch (default: every missing row); "
        f"packs: {', '.join(p.id for p in PACKS)}",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Show the catalog, grouped by pack, and exit",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    assets = by_id()

    if args.list:
        for pack_id, rows in by_pack().items():
            pack = PACK_BY_ID[pack_id]
            print(f"[{pack.id}] {pack.title} — {pack.description}")
            for a in rows:
                mark = "installed" if a.installed else "MISSING  "
                print(f"  {mark}  {a.id:<16} {a.repo:<48} → {a.location}")
        return 0

    try:
        ids = expand(args.ids)
    except KeyError as e:
        print(e.args[0], file=sys.stderr)
        return 2

    picked = [assets[i] for i in ids] or [a for a in catalog() if not a.installed]
    if not picked:
        print("every model is already installed.")
        return 0

    failed: list[tuple[str, Exception]] = []
    for a in picked:
        print(f"\n{a.title}  [{a.repo}] → {a.location}", flush=True)
        try:
            a.fetch()
        except Exception as e:  # noqa: BLE001 — one gated repo must not
            # abort the rest; every failure is reported at the end.
            failed.append((a.title, e))
            print(f"  FAILED: {e}", flush=True)

    print(flush=True)
    for title, e in failed:
        print(f"{title}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    print(f"{len(picked) - len(failed)}/{len(picked)} model(s) ready.", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
