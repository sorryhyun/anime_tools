"""The workspace layout: where the tools write, and where Export publishes to.

The workspace is what makes "curation is done" an explicit moment rather than a
side effect of the last ``--apply``::

    <home>/
      image_dataset/                  INPUT -- read-only for the tools
      workspace/                      everything the tools produce
        master/<rel>.txt                revised master
        resized/<rel>.{png,txt,variants.txt}
        masks_sam/<rel>/{stem}_mask.png  each generator's own tree
        masks_mit/<rel>/{stem}_mask.png
        masks/<rel>/{stem}_mask.png      the merge of them
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
and honours ``ANIME_TOOLS_WORKSPACE``."""

SOURCE_ROOT = "image_dataset"
"""The input tree. Read-only for the tools from the workspace phase onward."""

EXPORT_ROOT = "post_image_dataset"
"""The output tree, and the one the trainer reads (``docs/contract.md`` §2).
Written only by Export."""

DEFAULT_ROOTS: dict[str, str] = {
    # input -> workspace -> output, the order ⚙ Settings shows them in.
    "src": SOURCE_ROOT,
    "master": f"{WORKSPACE}/master",
    "dst": f"{WORKSPACE}/resized",
    "masks": f"{WORKSPACE}/masks",
    "out": EXPORT_ROOT,
}
"""The five dataset roots, home-relative.

``src`` / ``dst`` / ``masks`` are the three trees the sidebar joins by the same
relative path, and the names ``gui.stages.ROOT_FIELDS`` binds stage flags to.
``master`` (the revised-master overlay) and ``out`` (the export destination) are
additive.
"""

OUTPUT_ROOTS = frozenset({"master", "dst", "masks"})
"""The roots a stage run may create on the way past — not ``src`` (input) and
not ``out`` (Export-only, and it mkdirs its own destination).
``gui.server.make_output_dirs`` intersects a stage's bound roots with this.
"""

EXPORT_ROOTS = frozenset({"out"})
"""Roots only Export writes, and which nothing else may create on its behalf.

The mirror of :data:`OUTPUT_ROOTS`, and why ⚙ Settings' "make my roots" gesture
stops short of one root: an export tree that exists should mean an export
happened.
"""

REPORTS_SUBDIR = "captions"
"""The workspace subdirectory the stage reports land in, as the tail of every
CLI's ``--report_dir`` default (``captions/autotag``, ``captions/position``, …).

``gui.server.report_root`` derives the root from the parent of ``dst``; this
constant exists so the migrate CLI knows what to move.
"""

RESIZED = DEFAULT_ROOTS["dst"]
MASKS = DEFAULT_ROOTS["masks"]
REPORTS = f"{WORKSPACE}/{REPORTS_SUBDIR}"
GROUPS = f"{WORKSPACE}/groups"
OCR_SUBDIR = "ocr"
OCR = f"{WORKSPACE}/{OCR_SUBDIR}"
"""The five workspace paths the CLI defaults are written in terms of.

Every ``--dst`` / ``--report_dir`` / ``--out`` default is one of these plus the
stage's own tail, so the CLI and GUI halves cannot disagree — and
``gui.stages.report_subpath``, which drops the first component of a report
default to get that tail, keeps seeing exactly one component in front of it.

:data:`OCR` is a tree of its own rather than a sidecar beside each caption
because what it holds is not a caption: it is the text that is *in the picture*,
and keeping it out of the resized tree is what lets it be published, deleted or
regenerated without touching a single caption.
"""

MASKS_SAM = f"{WORKSPACE}/masks_sam"
MASKS_MIT = f"{WORKSPACE}/masks_mit"
"""Each mask generator's own output tree, and ``merge_masks``' two inputs.

Not roots, and deliberately *not* :data:`MASKS`: both generators name a mask
``{stem}_mask.png`` under the same relative path, so one shared directory means
the second run overwrites the first, and the merge — a pixel-wise minimum over
the two trees, i.e. the union of what they mask — has nothing left to combine.
:data:`MASKS` is the merged answer, the root the sidebar joins and Export
publishes. Written here rather than left to the operator because these three
paths are one fact in three CLIs: ``merge_masks`` reads exactly what the two
generators write, and ``tests/test_masking_plan.py`` pins that they agree.
"""

LEGACY_ROOTS: dict[str, str] = {
    # root name -> where its default pointed before the workspace.
    "dst": f"{EXPORT_ROOT}/resized",
    "masks": f"{EXPORT_ROOT}/masks",
}
"""Pre-workspace defaults for the two roots that moved.

An install with explicit roots saved in ⚙ Settings is unaffected by the change
of defaults — which is why the migrate CLI *warns* about one rather than
silently moving the tree out from under it.
"""

LEGACY_DIRS: tuple[str, ...] = (REPORTS_SUBDIR, "groups")
"""Workspace subdirectories that are not roots, so nothing in Settings names
them. Both follow ``dst``'s parent, so moving the tree is all it takes."""
