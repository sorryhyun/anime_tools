"""Every model weight the stages need: where it comes from, where it lands, and
whether it is already here.

Torch-free, so one answer serves the GUI's Settings dialog, this module's own
CLI (``python -m anime_tools.downloads``, what the Download button runs) and the
loaders themselves. :mod:`anime_tools.vision.pe`,
:mod:`anime_tools.masking.cli.generate_masks_mit` and the SAM3 CLIs take their
repo / filename / default path from here, so a button can never put a checkpoint
somewhere the loader won't look for it.

Two kinds of destination exist and they are **not** interchangeable:

* a path under the curation home (``models/…``) — what a ``--checkpoint`` /
  ``--tagger_dir`` flag defaults to, so the file has to be exactly *there*;
* the HuggingFace hub cache — for assets whose loader fetches them by repo id
  (the gated dbv4 backbone, the MIT text-mask net). Presence is probed with
  :func:`anime_tools._hf.hf_file_cached`, never by guessing a cache path.

Nothing here is auto-run: every loader still fetches what it needs on first
use. Pre-fetching just moves the wait to a moment the user chose.
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
    DBV4_OPTIONAL_FILES,
    DBV4_REQUIRED_FILES,
    DEFAULT_TAGGER_DIR,
    TAGGER_HF_REPO,
    TAGGER_HF_SUBFOLDER,
    backbone_repo_for,
)

# SAM 3 — gated on the Hub. Our CLIs pass ``--checkpoint`` explicitly with
# ``load_from_HF=False``, so the file has to land on this path, not merely in
# the hub cache sam3's own downloader would use.
SAM3_REPO = "facebook/sam3"
SAM3_FILENAME = "sam3.pt"
SAM3_DIR = "models/sam3"
DEFAULT_SAM3_CHECKPOINT = f"{SAM3_DIR}/{SAM3_FILENAME}"

# PE-Spatial-B16-512 — the grouping embedder's frozen tower
# (:mod:`anime_tools.vision.pe` builds the architecture, this is its weights).
PE_SPATIAL_REPO = "facebook/PE-Spatial-B16-512"
PE_SPATIAL_FILENAME = "PE-Spatial-B16-512.pt"

# SAM3 subject soft prompt — the textual inversion of ``anime girl`` that the
# position stage and the multiview audit pass as ``--prompt_embed`` by default.
# A trained artifact of the trainer repo, not a checkpoint anyone re-hosts on
# the Hub, so it is fetched straight from GitHub.
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
# on (``--ctd-gate``). Published as a release asset of manga-image-translator
# rather than on the Hub, so it rides ``_fetch_http`` like the soft prompt does.
# The stage has no flag for it: this is the one path it looks at.
CTD_GH_REPO = "zyddnys/manga-image-translator"
CTD_ONNX_RELEASE = "beta-0.3"
CTD_ONNX_DIR = "models/mit"
CTD_ONNX_FILENAME = "comictextdetector.pt.onnx"
CTD_ONNX_URL = f"https://github.com/{CTD_GH_REPO}/releases/download/{CTD_ONNX_RELEASE}"

# PP-OCRv6 — text detection + recognition, as the official ONNX mirrors of the
# Paddle inference models. ONNX and not Paddle because `inference.yml` beside
# each graph carries everything the wrapping code needs (the recognizer's
# 18,708-character dictionary, the detector's DB thresholds), so onnxruntime and
# opencv — both already here — are the whole runtime, and a second deep-learning
# framework does not enter a py3.13 / numpy>=2 / torch stack for one 19M model.
# Like the CTD net below, neither has a flag: `anime_tools.ocr._onnx` reads these
# paths, so a Download button cannot write where the loader does not look.
PPOCR_DET_REPO = "PaddlePaddle/PP-OCRv6_medium_det_onnx"
PPOCR_REC_REPO = "PaddlePaddle/PP-OCRv6_medium_rec_onnx"
PPOCR_FILES = ("inference.onnx", "inference.yml")
PPOCR_DIR = "models/ppocr"

# Danbooru tag KB — the ~114k-row classified tag table every correction pass
# types its tags against, and what the GUI's click-a-tag panel reads. A CSV in a
# GitHub repo, so it rides ``_fetch_http`` like the soft prompt does. Its
# descriptions are Korean; the English sibling is *built* from the Danbooru
# wiki, not hosted, which is why only the base file is a row here.
DANBOORU_TAGS_GH_REPO = "Localsmile/danbooru_KR_wiki_tag_search"
DANBOORU_TAGS_URL = f"https://raw.githubusercontent.com/{DANBOORU_TAGS_GH_REPO}/main"

# The Danbooru wiki mirrored as one parquet on the Hub — an *input*, not a
# product: the row below joins it against the base CSV and writes the English
# sibling, so the 45 MB parquet stays in the hub cache.
DANBOORU_WIKI_REPO = "isek-ai/danbooru-wiki-2024"
DANBOORU_WIKI_FILE = "data/train-00000-of-00001.parquet"


def default_pe_spatial_path() -> Path:
    """``<models_dir>/pe/PE-Spatial-B16-512.pt`` — the trainer's ``models/pe/``
    when run in-tree, ``ANIME_TOOLS_MODELS`` standalone."""
    return models_dir() / "pe" / PE_SPATIAL_FILENAME


def default_ctd_onnx_path() -> Path:
    """``<home>/models/mit/comictextdetector.pt.onnx`` — the only place the MIT
    stage's ``--ctd-gate`` looks, and what the catalog row writes."""
    return resolve_path(CTD_ONNX_DIR) / CTD_ONNX_FILENAME


def default_ppocr_det_dir() -> Path:
    """``<home>/models/ppocr/det`` — the only place the OCR stage's detector is
    read from, and what the ``ppocr_det`` row writes."""
    return resolve_path(PPOCR_DIR) / "det"


def default_ppocr_rec_dir() -> Path:
    """``<home>/models/ppocr/rec`` — the recognizer's half of the same."""
    return resolve_path(PPOCR_DIR) / "rec"


def http_timeout() -> float:
    """Socket timeout for the plain-HTTPS rows, sharing ``ANIMA_HF_TIMEOUT``
    with :mod:`anime_tools._hf`."""
    return float(os.environ.get("ANIMA_HF_TIMEOUT", "30"))


def _say(msg: str) -> None:
    print(msg, flush=True)


def _size(n: int) -> str:
    return f"{n / 1e6:,.0f} MB" if n >= 1e6 else f"{n / 1e3:,.0f} KB"


@dataclass(frozen=True)
class Asset:
    """One downloadable model, resolved against the current curation home."""

    id: str
    title: str
    repo: str
    files: tuple[str, ...]
    """Required files; all of them present means installed."""
    used_by: str
    """Which stages stop working without it — the reason a row exists."""
    stages: tuple[str, ...] = ()
    """The same, as GUI stage ids, so the stage bar can warn before a run stalls
    on a multi-GB first-use fetch."""
    dest: Path | None = None
    """Directory the files are flattened into; ``None`` = the HF hub cache."""
    url: str = ""
    """Base URL each file hangs off, for the rows that are not on the Hub (a
    GitHub raw prefix). Set it and ``repo`` is only a label; requires ``dest``."""
    subfolder: str = ""
    """Path prefix inside the repo (the tagger ships under ``dbv4/``)."""
    repo_type: str = "model"
    """Hub repo kind — ``dataset`` for the Danbooru wiki mirror."""
    derived: tuple[str, ...] = ()
    """Files this row *makes* under ``dest`` via :attr:`build`. They, not the
    downloads, are what the row is: the probe asks for them, and the downloaded
    inputs stay cache detail so a hub-cache sweep can't turn a built row back
    to "missing"."""
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
        """Plain-HTTPS download for a ``url`` row. Bounded and fail-fast: a
        stalled socket must raise, not hang the job slot the GUI runs this in."""
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

        Network and gated-repo failures come back as a ``FileNotFoundError``
        that names the asset and the recovery instead of a hub traceback.
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
        # A built row's downloads are inputs, so they stay in the hub cache;
        # only what ``build`` writes belongs in ``dest``.
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
            # Don't leave the empty `dbv4/` local_dir mirrored in the
            # checkpoint dir after the files were moved out of it.
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
    """``danbooru_tags_en``'s post-fetch step: the wiki parquet is in the hub
    cache by now, so this is the join that writes the English CSV."""
    from anime_tools.tagger.cli.build_english_tag_csv import build

    build(dest / TAG_CSV_NAME, dest / TAG_CSV_EN_NAME, revision=None, log=log)


def catalog() -> tuple[Asset, ...]:
    """The full catalog, resolved against the *current* curation home.

    Rebuilt per call, not cached at import: the home moves with
    ``ANIME_TOOLS_HOME`` and the backbone repo follows the installed checkpoint.
    """
    tagger_dir = resolve_path(DEFAULT_TAGGER_DIR)
    backbone = backbone_repo_for(tagger_dir)
    return (
        Asset(
            id="tagger",
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
            title="dbv4 tagger backbone",
            repo=backbone,
            files=DBV4_BACKBONE_FILES,
            used_by="the Anima Tagger checkpoint above",
            stages=("autotag", "position", "audit"),
            gated=f"https://huggingface.co/{backbone}",
            notes="GPL-3.0 and never vendored; the terms auto-approve.",
        ),
        Asset(
            id="sam3",
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
            title="PE-Spatial-B16-512",
            repo=PE_SPATIAL_REPO,
            files=(PE_SPATIAL_FILENAME,),
            dest=default_pe_spatial_path().parent,
            used_by="Build groups (near-twin / same-concept grouping)",
            stages=("groups",),
        ),
        Asset(
            id="soft_prompt",
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
            id="ppocr_det",
            title="PP-OCRv6 text detection",
            repo=PPOCR_DET_REPO,
            files=PPOCR_FILES,
            dest=default_ppocr_det_dir(),
            used_by="OCR text",
            stages=("ocr",),
            notes="62 MB. Finds the text lines the recognizer below reads.",
        ),
        Asset(
            id="ppocr_rec",
            title="PP-OCRv6 text recognition",
            repo=PPOCR_REC_REPO,
            files=PPOCR_FILES,
            dest=default_ppocr_rec_dir(),
            used_by="OCR text",
            stages=("ocr",),
            notes="77 MB. English, Chinese and Japanese: its dictionary is "
            "both kana plus 15,565 han characters, and no hangul.",
        ),
        Asset(
            id="mit_text",
            title="Manga text segmentation",
            repo=MIT_TEXT_REPO,
            files=(MIT_TEXT_FILENAME,),
            used_by="MIT text masks",
            stages=("masks_mit",),
        ),
        Asset(
            id="ctd_onnx",
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m anime_tools.downloads",
        description="Pre-fetch the model weights the stages need. With no ID, "
        "downloads everything that is missing; with IDs, re-fetches exactly "
        "those (a repair). Every loader still auto-fetches on first use — this "
        "only moves the wait somewhere you chose.",
    )
    p.add_argument(
        "ids",
        nargs="*",
        metavar="ID",
        help="Model ids to fetch (default: every missing one)",
    )
    p.add_argument("--list", action="store_true", help="Show the catalog and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    assets = by_id()

    if args.list:
        for a in catalog():
            mark = "installed" if a.installed else "MISSING  "
            print(f"{mark}  {a.id:<16} {a.repo:<48} → {a.location}")
        return 0

    unknown = [i for i in args.ids if i not in assets]
    if unknown:
        print(
            f"unknown model id: {', '.join(unknown)}  (known: {', '.join(assets)})",
            file=sys.stderr,
        )
        return 2

    picked = [assets[i] for i in args.ids] or [a for a in catalog() if not a.installed]
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
