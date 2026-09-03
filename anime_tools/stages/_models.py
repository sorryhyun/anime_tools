"""Loading the Anima Tagger the way every caption stage loads it — once per
process.

``load_anima_tagger`` caches on ``(checkpoint dir, device)``, so autotag followed
by position in one interpreter reads the weights once. The clause vocabulary is
read from the checkpoint the tagger actually loaded, so a caller cannot pair one
model's predictions with another's clause gates. ``release_models`` is the
inverse: a driver that runs several stages in one process (the trainer's
daemon job) calls it before handing the GPU to something else.

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


def release_models() -> None:
    """Drop every per-process model cache this package keeps — the tagger
    (:data:`_LOADED`) and SAM3 (``masking._sam3._LOADED``) — and return the
    freed VRAM to the driver.

    For a chain that runs stages in-process and then starts a *different* GPU
    process (the trainer's daemon job runs autotag in-process and the VAE
    encode as a child): a resident tagger + SAM3 would otherwise sit in VRAM
    for the whole child. A no-op when nothing was loaded; importing torch only
    when it already is.
    """
    import sys

    from anime_tools.masking import _sam3

    _LOADED.clear()
    _sam3._LOADED.clear()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
