"""The workspace layout: where the tools write, and where Export publishes to::

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

The invariant: **no stage writes outside** ``workspace/``; Export is the only
thing that touches ``image_dataset/`` or ``post_image_dataset/``.

Nothing here reads the filesystem or the settings file: these are the *default*
paths, which ⚙ Settings may override per root. Stdlib only, torch-free.
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
relative path, and the names ``gui.stages.ROOT_FIELDS`` binds stage flags to;
``master`` is the revised-master overlay and ``out`` the export destination.
"""

OUTPUT_ROOTS = frozenset({"master", "dst", "masks"})
"""The roots a stage run may create on the way past — not ``src`` (input) and
not ``out`` (Export-only, and it mkdirs its own destination).
``gui.server.make_output_dirs`` intersects a stage's bound roots with this.
"""

EXPORT_ROOTS = frozenset({"out"})
"""Roots only Export writes, and which nothing else may create on its behalf —
⚙ Settings' "make my roots" gesture stops short of these."""

REPORTS_SUBDIR = "captions"
"""The workspace subdirectory the stage reports land in, as the tail of every
CLI's ``--report_dir`` default (``captions/autotag``, ``captions/position``, …).
``gui.server.report_root`` derives the root from the parent of ``dst``."""

RESIZED = DEFAULT_ROOTS["dst"]
MASKS = DEFAULT_ROOTS["masks"]
REPORTS = f"{WORKSPACE}/{REPORTS_SUBDIR}"
GROUPS = f"{WORKSPACE}/groups"
OCR_SUBDIR = "ocr"
OCR = f"{WORKSPACE}/{OCR_SUBDIR}"
"""The workspace paths the CLI defaults are written in terms of.

Every ``--dst`` / ``--report_dir`` / ``--out`` default is one of these plus the
stage's own tail, which keeps ``gui.stages.report_subpath`` — it drops the first
component of a report default to get that tail — seeing exactly one component in
front of it.

:data:`OCR` is its own tree, not a sidecar beside each caption: what it holds is
text *in the picture*, so it can be published, deleted or regenerated without
touching a caption.
"""

MASKS_SAM = f"{WORKSPACE}/masks_sam"
MASKS_MIT = f"{WORKSPACE}/masks_mit"
"""Each mask generator's own output tree, and ``merge_masks``' two inputs.

Not roots, and separate from :data:`MASKS`: both generators name a mask
``{stem}_mask.png`` under the same relative path, so a shared directory would
have the second run overwrite the first. :data:`MASKS` holds the merge (a
pixel-wise minimum over the two trees, i.e. the union of what they mask) and is
the root the sidebar joins and Export publishes.
"""

LEGACY_ROOTS: dict[str, str] = {
    # root name -> where its default pointed before the workspace.
    "dst": f"{EXPORT_ROOT}/resized",
    "masks": f"{EXPORT_ROOT}/masks",
}
"""Pre-workspace defaults for the two roots that moved. A root explicitly saved
in ⚙ Settings is unaffected, so the migrate CLI warns about one instead of
moving it."""

LEGACY_DIRS: tuple[str, ...] = (REPORTS_SUBDIR, "groups")
"""Workspace subdirectories that are not roots, so nothing in Settings names
them. Both follow ``dst``'s parent, so moving the tree is all it takes."""
