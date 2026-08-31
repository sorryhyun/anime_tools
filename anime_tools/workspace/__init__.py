"""The workspace layout: where the tools write, and where Export publishes to.

Curation used to write straight at the trainer's paths — the clause rewrite into
``post_image_dataset/resized/``, the mask generators into
``post_image_dataset/masks/``, autotag and the multiview audit into the caption
master under ``image_dataset/``. There was no moment at which curation was
*done*: an ``--apply`` was a publish, and the only thing between a half-finished
run and the trainer was that you had not started the trainer yet.

The workspace makes that moment explicit::

    <home>/
      image_dataset/                  INPUT -- read-only for the tools
      workspace/                      everything the tools produce
        master/<rel>.txt                revised master
        resized/<rel>.{png,txt,variants.txt}
        masks/<rel>/{stem}_mask.png
        captions/<stage>/report.json    the diffs
        groups/groups.json
        export/report.json              the export ledger
      post_image_dataset/             OUTPUT -- written only by Export
        resized/  masks/

The invariant, which ``tests/test_workspace.py`` pins on the roots and a later
phase pins on the stages themselves:

    **No stage writes outside** ``workspace/``. Export is the only thing that
    touches ``image_dataset/`` or ``post_image_dataset/``.

Nothing here reads the filesystem or the settings file: these are the *default*
paths, which ⚙ Settings may override per root. Stdlib only, torch-free, and
imported by :mod:`anime_tools.gui.dataset` so the GUI and the migrate CLI cannot
disagree about the layout.
"""

from __future__ import annotations

WORKSPACE = "workspace"
"""The workspace directory, home-relative. ``_env.workspace_dir()`` resolves it
(and honours ``ANIME_TOOLS_WORKSPACE``); this is the spelling that goes into the
default roots below, which are home-relative by contract."""

SOURCE_ROOT = "image_dataset"
"""The input tree. Read-only for the tools from the workspace phase onward."""

EXPORT_ROOT = "post_image_dataset"
"""The output tree, and the one the trainer reads (``docs/contract.md`` §2).
Written only by Export."""

DEFAULT_ROOTS: dict[str, str] = {
    # Listed input -> workspace -> output, which is the order the ⚙ Settings
    # dialog shows them in.
    "src": SOURCE_ROOT,
    "master": f"{WORKSPACE}/master",
    "dst": f"{WORKSPACE}/resized",
    "masks": f"{WORKSPACE}/masks",
    "out": EXPORT_ROOT,
}
"""The five dataset roots, home-relative.

``src`` / ``dst`` / ``masks`` keep the names they have always had — the three
trees the sidebar joins by the same relative path, and the names
``gui.stages.ROOT_FIELDS`` binds stage flags to — so only their *defaults* moved
under the workspace. ``master`` (the revised-master overlay) and ``out`` (the
export destination) are additive.
"""

OUTPUT_ROOTS = frozenset({"master", "dst", "masks"})
"""The roots a stage run may create on the way past.

Everything a stage writes, and nothing else: not ``src``, which is input, and
not ``out``, which only Export writes and which mkdirs its own destination.
``gui.server.make_output_dirs`` intersects a stage's bound roots with this.
"""

EXPORT_ROOTS = frozenset({"out"})
"""Roots only Export writes, and which nothing else may create on its behalf.

The mirror of :data:`OUTPUT_ROOTS`, and the reason ⚙ Settings' "make my roots"
gesture stops short of one root: an export tree that exists should mean an
export happened. Export makes its own destination.
"""

REPORTS_SUBDIR = "captions"
"""The workspace subdirectory the stage reports land in, as the tail of every
CLI's ``--report_dir`` default (``captions/autotag``, ``captions/position``, …).

Not spelled out anywhere else: ``gui.server.report_root`` derives the root from
the parent of ``dst``, which *is* the workspace, and each stage keeps its own
tail. This constant exists so the migrate CLI knows what to move.
"""

RESIZED = DEFAULT_ROOTS["dst"]
MASKS = DEFAULT_ROOTS["masks"]
REPORTS = f"{WORKSPACE}/{REPORTS_SUBDIR}"
GROUPS = f"{WORKSPACE}/groups"
"""The three workspace paths the CLI defaults are written in terms of.

Every ``--dst`` / ``--report_dir`` / ``--out`` default in the package is one of
these plus the stage's own tail, so the CLI half of the workspace and the GUI
half cannot disagree — and so ``gui.stages.report_subpath``, which drops the
first component of a report default to get that tail, keeps seeing exactly one
component in front of it.
"""

LEGACY_ROOTS: dict[str, str] = {
    # root name -> where its default used to point, before the workspace.
    "dst": f"{EXPORT_ROOT}/resized",
    "masks": f"{EXPORT_ROOT}/masks",
}
"""Pre-workspace defaults for the two roots that moved.

An install with explicit roots saved in ⚙ Settings is unaffected by the change
of defaults — which is exactly why the migrate CLI has to *warn* about one
rather than silently move the tree out from under it.
"""

LEGACY_DIRS: tuple[str, ...] = (REPORTS_SUBDIR, "groups")
"""Workspace subdirectories that used to sit under :data:`EXPORT_ROOT` and are
not roots, so nothing in Settings names them: the stage reports and the grouping
manifest. Both follow ``dst``'s parent today, so moving the tree is all it takes.
"""
