"""Loading the Anima Tagger the way every caption stage loads it — once per
process.

``load_anima_tagger`` caches on ``(checkpoint dir, device)``, so autotag followed
by position in one interpreter reads the weights once. The clause vocabulary is
read from the checkpoint the tagger actually loaded, so a caller cannot pair one
model's predictions with another's clause gates.

Torch is imported inside the functions: a request's ``--from_report`` replay
path must stay torch-free, so call these at the model-load site only.
"""

from __future__ import annotations

from pathlib import Path

from anime_tools._device import resolve_device
from anime_tools._env import resolve_path
from anime_tools._progress import phase
from anime_tools.tagger.dbv4_meta import DEFAULT_TAGGER_DIR

_LOADED: dict[tuple[str, str], tuple] = {}


def load_anima_tagger(
    tagger_dir: str | Path | None, device: str | None, *, quiet: bool = False
):
    """``(tagger, ckpt_dir)`` for a checkpoint dir (``None`` = the shipped
    default, fetched when absent) on ``device`` (``None`` = auto)."""
    from anime_tools.tagger.tagger import AnimaTagger, ensure_tagger_checkpoint

    # One phase over the fetch and the build: the first run's backbone download
    # is the quiet stretch a daemon stall watchdog would otherwise read as a wedge.
    with phase("load tagger"):
        ckpt_dir: Path = ensure_tagger_checkpoint(
            resolve_path(str(tagger_dir or DEFAULT_TAGGER_DIR))
        )
        dev = resolve_device(device)
        key = (str(ckpt_dir), dev)
        loaded = _LOADED.get(key)
        if loaded is None:
            if not quiet:
                print(f"Loading Anima Tagger from {ckpt_dir} ({dev})...", flush=True)
            loaded = _LOADED[key] = (AnimaTagger(ckpt_dir, device=dev), ckpt_dir)
    return loaded


def load_tagger(req, *, quiet: bool = False):
    """``(tagger, vocabulary, ckpt_dir)`` for a request carrying ``tagger_dir``
    and ``device`` (:class:`~anime_tools.stages.requests.TaggerRequest`)."""
    from anime_tools.stages.position_captions import load_clause_vocabulary

    tagger, ckpt_dir = load_anima_tagger(req.tagger_dir, req.device, quiet=quiet)
    return tagger, load_clause_vocabulary(ckpt_dir), ckpt_dir
