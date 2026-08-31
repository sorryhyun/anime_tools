"""Where the sincos decensor pass reads and writes.

``match_decensored`` produces the manifest and ``apply_decensored`` consumes it,
so the two agree on six paths or neither works. They were declared twice — the
matcher writing ``OUT_DIR / "matches.csv"`` and the applier reading it back from
its own copy of the same expression. One home instead, resolved off
:func:`~anime_tools._env.curation_home` at import like the rest of the curation
CLIs.

The tree is fixed rather than flagged on purpose: this is a one-off cleanup pass
over one artist directory, not a stage.
"""

from __future__ import annotations

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
LORA_CACHE = ROOT / "post_image_dataset" / "lora" / "sincos"
RESIZED = ROOT / "post_image_dataset" / "resized" / "sincos"
