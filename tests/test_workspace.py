"""The workspace layout, and the invariant it exists to make true.

    No stage writes outside ``workspace/``. Export is the only thing that
    touches ``image_dataset/`` or ``post_image_dataset/``.

Phase 1 can pin the half of that which is visible in the *defaults*: every path
a stage CLI would write to if you ran it bare lands in the workspace, and the
GUI's roots agree with the CLIs about where the workspace is. The other half —
that a stage's ``--apply`` really touches nothing outside it — needs the master
overlay, and is pinned by ``test_workspace_boundary.py`` when that lands.
"""

from __future__ import annotations

import pytest

from anime_tools import workspace as WS
from anime_tools.gui import dataset as D
from anime_tools.gui import stages as S
from anime_tools.workspace import migrate as M

# The stage CLIs whose defaults are output paths. ``grouping`` and the two probe
# CLIs are in here too: a default that writes is a default that must be in the
# workspace, whichever package it lives in.
WRITERS = (
    "anime_tools.stages.cli.autotag_captions",
    "anime_tools.stages.cli.position_captions",
    "anime_tools.stages.cli.audit_multiview",
    "anime_tools.stages.cli.audit_apply_curated",
    "anime_tools.stages.cli.resize_images",
    "anime_tools.stages.cli.correct_captions",
    "anime_tools.grouping.cli.build_groups",
)


def _defaults(module_path: str) -> dict[str, str]:
    """``{dest: default}`` for every string default of one stage parser."""
    import importlib

    parser = importlib.import_module(module_path).build_parser()
    return {
        a.dest: a.default
        for a in parser._actions
        if isinstance(a.default, str) and a.dest != "help"
    }


# ---- the layout ---------------------------------------------------------


def test_the_workspace_roots_are_under_the_workspace():
    """`master` / `dst` / `masks` are the workspace; `src` and `out` are not.

    The whole framing rests on this split, so it is worth saying out loud
    rather than leaving it implied by three string literals.
    """
    for name in WS.OUTPUT_ROOTS:
        assert WS.DEFAULT_ROOTS[name].startswith(f"{WS.WORKSPACE}/"), name
    assert WS.DEFAULT_ROOTS["src"] == WS.SOURCE_ROOT
    assert WS.DEFAULT_ROOTS["out"] == WS.EXPORT_ROOT
    for name in (WS.SOURCE_ROOT, WS.EXPORT_ROOT):
        assert not name.startswith(WS.WORKSPACE)


def test_output_and_export_roots_do_not_overlap():
    """A root is either something a stage may create or Export's alone."""
    assert WS.OUTPUT_ROOTS.isdisjoint(WS.EXPORT_ROOTS)
    assert WS.OUTPUT_ROOTS | WS.EXPORT_ROOTS | {"src"} == set(WS.DEFAULT_ROOTS)


def test_the_gui_imports_the_layout_rather_than_restating_it():
    assert D.DEFAULT_ROOTS is WS.DEFAULT_ROOTS
    assert D.OUTPUT_ROOTS is WS.OUTPUT_ROOTS
    assert D.EXPORT_ROOTS is WS.EXPORT_ROOTS


# ---- the CLI defaults ---------------------------------------------------


@pytest.mark.parametrize("module_path", WRITERS)
def test_no_cli_default_writes_to_the_export_tree(module_path):
    """Running a stage bare must not publish.

    This is the CLI half of the invariant: before the workspace, a hand-run
    ``python -m anime_tools.stages.cli.position_captions --apply`` wrote
    straight into the tree the trainer reads. Now every default that names an
    output names the workspace.
    """
    for dest, default in _defaults(module_path).items():
        assert not default.startswith(WS.EXPORT_ROOT), f"{module_path} --{dest}"


@pytest.mark.parametrize("module_path", WRITERS)
def test_report_defaults_keep_one_component_in_front_of_their_tail(module_path):
    """``report_subpath`` drops the first component to get a stage's own tail.

    That is what lets one ``report_root`` setting move every stage's report
    while each keeps a directory of its own — and it only works while the root
    is exactly one component. Moving the defaults under ``workspace/`` kept
    that true; spelling one ``workspace/captions/x`` as ``a/b/captions/x``
    would silently hand the GUI the wrong tail.
    """
    for dest, default in _defaults(module_path).items():
        if dest not in ("report_dir", "out"):
            continue
        head, _, tail = default.partition("/")
        assert head == WS.WORKSPACE, f"{module_path} --{dest} = {default}"
        assert tail and S.report_subpath(default) == tail


def test_the_resized_default_is_the_dst_root():
    """One tree, named twice — the CLI's ``--dst`` and the GUI's ``dst`` root."""
    dst = _defaults("anime_tools.stages.cli.autotag_captions")["dst"]
    assert dst == WS.DEFAULT_ROOTS["dst"] == WS.RESIZED


def test_grouping_reads_the_resized_tree_by_default():
    """The resized tree is the one decode substrate, not just the caption
    stages' input: ``build_groups`` walks it too, so grouping sees the pixels
    training sees and shares the geometry every other stage embeds."""
    assert (
        _defaults("anime_tools.grouping.cli.build_groups")["source_dir"] == WS.RESIZED
    )


def test_every_pixel_reading_stage_is_bound_to_the_resized_tree():
    """The GUI half of the same rule.

    A stage that opens an image reads ``dst``; ``src`` is left to ``autotag``,
    which falls back to the master for a caption, and to Export, which
    publishes back over it. Pinned because the binding is
    also what earns each stage its resize preflight — drop one back to ``src``
    and it silently walks a tree nothing else agrees with.
    """
    pixel_stages = {
        "autotag",
        "position",
        "correct",
        "audit",
        "masks_sam",
        "masks_mit",
        "groups",
    }
    for stage_id in pixel_stages:
        bound = S.ROOT_FIELDS[stage_id]
        assert "dst" in bound.values(), f"{stage_id} does not read the resized tree"


# ---- migrate ------------------------------------------------------------


def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIME_TOOLS_HOME", str(tmp_path))
    monkeypatch.delenv("ANIME_TOOLS_WORKSPACE", raising=False)
    return tmp_path


def test_migrate_plans_only_what_is_there(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    (home / WS.EXPORT_ROOT / "resized").mkdir(parents=True)
    (home / WS.WORKSPACE / "masks").mkdir(parents=True)
    (home / WS.EXPORT_ROOT / "masks").mkdir(parents=True)

    got = {
        name: status for name, _, _, status in M.plan_moves(home, home / WS.WORKSPACE)
    }
    assert got == {
        "resized": "would-move",
        "masks": "both-exist",  # never merged: picking a winner is not its call
        "captions": "absent",
        "groups": "absent",
    }


def test_migrate_moves_the_tree_and_is_idempotent(tmp_path, monkeypatch, capsys):
    home = _home(tmp_path, monkeypatch)
    old = home / WS.EXPORT_ROOT / "resized" / "sub"
    old.mkdir(parents=True)
    (old / "a.txt").write_text("1girl", encoding="utf-8")

    assert M.main([]) == 0  # dry run writes nothing
    assert (old / "a.txt").exists()
    assert "DRY RUN" in capsys.readouterr().out

    assert M.main(["--apply"]) == 0
    moved = home / WS.WORKSPACE / "resized" / "sub" / "a.txt"
    assert moved.read_text(encoding="utf-8") == "1girl"
    assert not (home / WS.EXPORT_ROOT / "resized").exists()

    assert M.main(["--apply"]) == 0  # nothing left to move
    assert moved.exists()


def test_migrate_names_the_saved_roots_that_still_pin_the_old_paths():
    """Only the *defaults* moved, so an explicitly-saved legacy root is stranded.

    Warned about rather than rewritten: the user typed those paths into the
    settings file, and a migration is not the place to edit them.
    """
    assert M.pinned_roots({"dst": WS.LEGACY_ROOTS["dst"]}) == [
        ("dst", WS.LEGACY_ROOTS["dst"])
    ]
    # Blank or absent means "follow the defaults", which is not pinned; and a
    # root the user moved somewhere of their own is not this script's business.
    assert M.pinned_roots({}) == []
    assert M.pinned_roots({"dst": "", "masks": None}) == []
    assert M.pinned_roots({"dst": "elsewhere/resized"}) == []
