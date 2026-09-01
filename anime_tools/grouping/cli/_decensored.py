"""Paths shared by the sincos decensor pass.

``match_decensored`` writes the manifest and ``apply_decensored`` consumes it, so
both read these six paths, resolved off
:func:`~anime_tools._env.curation_home` at import. Not flag-configurable.
"""

from __future__ import annotations

from anime_tools import workspace as WS
from anime_tools._env import curation_home

__all__ = [
    "BACKUP_DIR",
    "DECEN_DIR",
    "LORA_CACHE",
    "OUT_DIR",
    "RESIZED",
    "ROOT",
    "SINCOS_DIR",
]

ROOT = curation_home()
# The censored originals, in the caption master.
SINCOS_DIR = ROOT / "image_dataset" / "sincos"
# The uncensored drop the matcher pairs against (pixiv IDs, unrelated names).
DECEN_DIR = ROOT / "sincos_decensored"
# matches.csv + review.html + the descriptor caches.
OUT_DIR = ROOT / "output" / "curate" / "sincos_decensored"
# Apply-side only: where a replaced original is kept, and the two image-derived
# caches that must be invalidated for a stem whose pixels changed.
BACKUP_DIR = OUT_DIR / "backup_censored"
LORA_CACHE = ROOT / WS.WORKSPACE / "lora" / "sincos"
RESIZED = ROOT / WS.RESIZED / "sincos"
