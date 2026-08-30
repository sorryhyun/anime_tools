"""Every model weight the stages need: where it comes from, where it lands, and
whether it is already here.

Torch-free on purpose, so one answer serves three callers: the GUI's Settings
dialog (which lists the catalog and puts a Download button on each row), this
module's own CLI — ``python -m anime_tools.downloads`` — which is what that
button runs in a subprocess, and the loaders themselves.
:mod:`anime_tools.vision.pe`, :mod:`anime_tools.masking.cli.generate_masks_mit`
and the SAM3 CLIs take their repo / filename / default path from here, so a
button can never put a checkpoint somewhere the loader won't look for it. The
tagger's own facts live in the other torch-free metadata module,
:mod:`anime_tools.tagger.dbv4_meta`, and are re-used below.

Two kinds of destination exist and they are **not** interchangeable:

* a path under the curation home (``models/…``) — what a ``--checkpoint`` /
  ``--tagger_dir`` flag defaults to, so the file has to be exactly *there*;
* the HuggingFace hub cache — for the assets whose loader fetches them by repo
  id (the gated dbv4 backbone, the MIT text-mask net). Presence is probed with
  :func:`anime_tools._hf.hf_file_cached`, never by guessing a cache path.

Nothing here is auto-run: every loader still fetches what it needs on first
use. Pre-fetching just moves the wait (and any gated-repo refusal) to a moment
the user chose.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anime_tools._env import models_dir, resolve_path
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

# MIT / ComicTextDetector text-mask net; its loader reads it straight out of
# the hub cache, so there is no path under models/ to keep in sync.
MIT_TEXT_REPO = "a-b-c-x-y-z/Manga-Text-Segmentation-2025"
MIT_TEXT_FILENAME = "model.pth"


def default_pe_spatial_path() -> Path:
    """``<models_dir>/pe/PE-Spatial-B16-512.pt`` — the trainer's ``models/pe/``
    when run in-tree, ``ANIME_TOOLS_MODELS`` standalone."""
    return models_dir() / "pe" / PE_SPATIAL_FILENAME


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
    dest: Path | None = None
    """Directory the files are flattened into; ``None`` = the HF hub cache."""
    subfolder: str = ""
    """Path prefix inside the repo (the tagger ships under ``dbv4/``)."""
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
            "location": self.location,
            "installed": not missing,
            "missing": missing,
            "gated": self.gated,
            "notes": self.notes,
        }

    def fetch(self, log: Callable[[str], None] = _say) -> None:
        """Download every required file; optional ones are best-effort.

        Network and gated-repo failures come back as the ``FileNotFoundError``
        :func:`anime_tools._hf.hf_download` raises, which names the asset and
        the recovery instead of dumping a hub traceback.
        """
        from huggingface_hub.utils import EntryNotFoundError

        from anime_tools._hf import hf_download

        hint = (
            f"hf auth login, then accept the terms at {self.gated}"
            if self.gated
            else f"python -m anime_tools.downloads {self.id}"
        )
        if self.dest is not None:
            self.dest.mkdir(parents=True, exist_ok=True)
        for name in (*self.files, *self.optional):
            remote = f"{self.subfolder}/{name}" if self.subfolder else name
            log(f"  {self.repo}/{remote}")
            try:
                got = Path(
                    hf_download(
                        what=f"{self.title} ({name})",
                        hint=hint,
                        repo_id=self.repo,
                        filename=remote,
                        **({"local_dir": str(self.dest)} if self.dest else {}),
                    )
                )
            except EntryNotFoundError:
                if name in self.optional:
                    log(f"    optional — not published by {self.repo}, skipped")
                    continue
                raise
            if self.dest is not None:
                # local_dir keeps the repo's own subfolder layout; the loaders
                # want a flat checkpoint dir.
                final = self.dest / Path(name).name
                if got.resolve() != final.resolve():
                    shutil.move(str(got), str(final))
                got = final
            log(f"    ok  {got}  ({_size(got.stat().st_size)})")
        if self.dest is not None and self.subfolder:
            # local_dir mirrored the repo's layout and we moved the files out of
            # it; don't leave an empty `dbv4/` sitting in the checkpoint dir.
            leftover = self.dest / self.subfolder
            if leftover.is_dir() and not any(leftover.iterdir()):
                leftover.rmdir()


def catalog() -> tuple[Asset, ...]:
    """The full catalog, resolved against the *current* curation home.

    Rebuilt per call rather than cached at import: the home moves with
    ``ANIME_TOOLS_HOME``, and the backbone repo follows whichever tagger
    checkpoint is actually installed.
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
            notes="Our half of the tagger — vocab, rules, groups, thresholds, "
            "sidecar. Small; the weights are the backbone below.",
        ),
        Asset(
            id="tagger_backbone",
            title="dbv4 tagger backbone",
            repo=backbone,
            files=DBV4_BACKBONE_FILES,
            used_by="the Anima Tagger checkpoint above",
            gated=f"https://huggingface.co/{backbone}",
            notes="GPL-3.0 and gated, never vendored: sign in with a Hugging "
            "Face token, then accept the terms on the repo page (auto-approve).",
        ),
        Asset(
            id="sam3",
            title="SAM 3",
            repo=SAM3_REPO,
            files=(SAM3_FILENAME,),
            dest=resolve_path(SAM3_DIR),
            used_by="Position captions · Multiview audit · SAM3 subject masks",
            gated=f"https://huggingface.co/{SAM3_REPO}",
            notes=f"Gated. Lands on the --checkpoint default, {DEFAULT_SAM3_CHECKPOINT}.",
        ),
        Asset(
            id="pe_spatial",
            title="PE-Spatial-B16-512",
            repo=PE_SPATIAL_REPO,
            files=(PE_SPATIAL_FILENAME,),
            dest=default_pe_spatial_path().parent,
            used_by="Build groups (near-twin / same-concept grouping)",
        ),
        Asset(
            id="mit_text",
            title="Manga text segmentation",
            repo=MIT_TEXT_REPO,
            files=(MIT_TEXT_FILENAME,),
            used_by="MIT text masks",
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
