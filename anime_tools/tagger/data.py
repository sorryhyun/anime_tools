"""Anima Tagger dataset manifest (``dataset.json`` emitted by ``--mode build_vocab``).

Only :class:`TaggerManifest` survives here. The PE dual-encoder feature/token
cache builders and the bucketed dual dataset that used to share this module
went to ``_archive/anima_tagger_training/pe_backend_removed_2026_08_30/`` with
the in-house PE tagger head (curation split Phase 0); the live dbv4 sidecar
trainer (``anime_tools/tagger/cli/train_sidecar.py``) keeps its own cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class TaggerManifest:
    """Trainable-sample manifest emitted by ``--mode build_vocab``."""

    stems: List[str]
    image_paths: List[Path]
    tag_indices: List[List[int]]
    rating_indices: List[int]
    people_count_indices: List[int]
    train_stems: List[str]
    val_stems: List[str]
    n_tags: int
    n_ratings: int
    n_people_counts: int

    @classmethod
    def from_path(cls, path: Path) -> "TaggerManifest":
        with open(path) as f:
            d = json.load(f)
        # ``people_count_indices`` / ``n_people_counts`` were added late; default
        # to a zero-length head so old manifests signal "no people supervision"
        # instead of KeyError-ing (they rebuild on next ``build_vocab``).
        people_idx = list(d.get("people_count_indices") or [])
        n_people = int(d.get("n_people_counts", 0))
        if people_idx and not n_people:
            n_people = max(people_idx) + 1
        return cls(
            stems=list(d["stems"]),
            image_paths=[Path(p) for p in d["image_paths"]],
            tag_indices=[list(idxs) for idxs in d["tag_indices"]],
            rating_indices=list(d["rating_indices"]),
            people_count_indices=people_idx,
            train_stems=list(d["split"]["train"]),
            val_stems=list(d["split"]["val"]),
            n_tags=int(d["n_tags"]),
            n_ratings=int(d["n_ratings"]),
            n_people_counts=n_people,
        )

    def stem_index(self) -> Dict[str, int]:
        return {s: i for i, s in enumerate(self.stems)}
