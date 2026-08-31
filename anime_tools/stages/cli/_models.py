"""Loading the Anima Tagger the way every stage CLI loads it.

Four call sites (the two caption stages and the two review sheets) resolved the
checkpoint dir, downloaded it if missing, built the tagger on the resolved
device and read the clause vocabulary out of the *same* dir — in four copies of
the same five lines. The last part is the one worth pinning: the vocabulary is
read from the checkpoint the tagger actually loaded, never from a second path,
so a caller cannot pair one model's predictions with another's clause gates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anime_tools._device import resolve_device
from anime_tools._env import resolve_path
from anime_tools.tagger.dbv4_meta import DEFAULT_TAGGER_DIR


def load_tagger(args: argparse.Namespace, *, quiet: bool = False):
    """``(tagger, vocabulary, ckpt_dir)`` for a namespace with the model flags.

    Imports torch transitively, so this is called at the model-load site and
    never from ``parse_args`` — the ``--from_report`` replay paths return
    before reaching it and must stay torch-free.
    """
    from anime_tools.stages.position_captions import load_clause_vocabulary
    from anime_tools.tagger.tagger import AnimaTagger, ensure_tagger_checkpoint

    ckpt_dir: Path = ensure_tagger_checkpoint(
        resolve_path(args.tagger_dir or DEFAULT_TAGGER_DIR)
    )
    if not quiet:
        print(f"Loading Anima Tagger from {ckpt_dir}...", flush=True)
    tagger = AnimaTagger(ckpt_dir, device=resolve_device(args.device))
    return tagger, load_clause_vocabulary(ckpt_dir), ckpt_dir
