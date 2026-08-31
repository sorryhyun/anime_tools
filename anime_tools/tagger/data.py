"""Anima Tagger checkpoint dir + dataset manifest, as their readers see them.

:class:`TaggerCheckpoint` is the one read side of a checkpoint directory —
``config.json`` / ``vocab.json`` / ``dataset.json``, the shared "run ``--mode
build_vocab`` first" exit when a required one is absent, and the
``index -> name`` map every consumer needs. :class:`TaggerManifest` is the
trainable-sample list inside ``dataset.json``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from anime_tools._json import read_json

# Which files a checkpoint dir holds, by the short name ``require`` uses.
CHECKPOINT_FILES: dict[str, str] = {
    "config": "config.json",
    "vocab": "vocab.json",
    "dataset": "dataset.json",
}


@dataclass
class TaggerManifest:
    """Trainable-sample manifest emitted by ``--mode build_vocab``."""

    stems: list[str]
    image_paths: list[Path]
    tag_indices: list[list[int]]
    rating_indices: list[int]
    people_count_indices: list[int]
    train_stems: list[str]
    val_stems: list[str]
    n_tags: int
    n_ratings: int
    n_people_counts: int

    @classmethod
    def from_path(cls, path: Path) -> TaggerManifest:
        return cls.from_dict(read_json(path))

    @classmethod
    def from_dict(cls, d: dict) -> TaggerManifest:
        # Default to a zero-length head so a pre-people-count manifest signals
        # "no people supervision" instead of KeyError-ing.
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

    def stem_index(self) -> dict[str, int]:
        return {s: i for i, s in enumerate(self.stems)}


@dataclass(frozen=True)
class TaggerCheckpoint:
    """The three JSON files of a tagger checkpoint dir, read once.

    ``config`` and ``dataset`` are ``None`` when the file is absent and the
    caller did not ``require`` it — a checkpoint published for inference ships
    no ``dataset.json``, and the vocab-build modes run before ``config.json``
    exists.
    """

    path: Path
    vocab: dict
    config: dict | None = None
    dataset: dict | None = None

    @classmethod
    def from_dir(
        cls,
        path: str | Path,
        *,
        require: Iterable[str] = ("vocab",),
        backend: str | None = None,
    ) -> TaggerCheckpoint:
        """Read ``path``'s checkpoint files; exit if a ``require``d one is missing.

        ``require`` names files from :data:`CHECKPOINT_FILES`; anything present
        but unrequired is read too, so a caller that merely *prefers* the
        manifest (``derive_groups``' co-occurrence source) just tests for
        ``None``. ``backend`` asserts ``config.json[backend]`` — and implies
        ``require=("config", …)``, since there is nothing to assert otherwise.
        """
        d = Path(path)
        wanted = set(require) | ({"config"} if backend else set())
        unknown = wanted - CHECKPOINT_FILES.keys()
        if unknown:
            raise ValueError(f"unknown checkpoint file(s): {sorted(unknown)}")
        missing = [
            str(d / CHECKPOINT_FILES[k])
            for k in CHECKPOINT_FILES
            if k in wanted and not (d / CHECKPOINT_FILES[k]).exists()
        ]
        if missing:
            raise SystemExit(
                f"need {' and '.join(missing)} — run --mode build_vocab first"
            )
        loaded = {
            k: read_json(d / name)
            for k, name in CHECKPOINT_FILES.items()
            if (d / name).exists()
        }
        ckpt = cls(
            path=d,
            vocab=loaded.get("vocab") or {},
            config=loaded.get("config"),
            dataset=loaded.get("dataset"),
        )
        if backend and (ckpt.config or {}).get("backend") != backend:
            raise SystemExit(f"{d} is not a {backend}-backed checkpoint")
        return ckpt

    @property
    def tags(self) -> list[dict]:
        """``vocab["tags"]`` — one row per kept tag, ``index`` = output slot."""
        return list(self.vocab.get("tags") or [])

    @property
    def n_tags(self) -> int:
        return len(self.vocab.get("tags") or [])

    def idx_to_name(self) -> dict[int, str]:
        """Model output slot → tag name."""
        return {int(t["index"]): t["name"] for t in self.tags}

    def manifest(self) -> TaggerManifest:
        """``dataset.json`` as a :class:`TaggerManifest` (requires it present)."""
        if self.dataset is None:
            raise SystemExit(
                f"need {self.path / CHECKPOINT_FILES['dataset']} — "
                "run --mode build_vocab first"
            )
        return TaggerManifest.from_dict(self.dataset)
