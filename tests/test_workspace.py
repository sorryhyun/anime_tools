"""The workspace layout: no stage writes outside ``workspace/``, and Export is
the only thing that touches ``image_dataset/`` or ``post_image_dataset/``.

Pins the half visible in the *defaults* — every path a stage CLI would write to
if run bare lands in the workspace, and the GUI's roots agree with the CLIs.
"""

from __future__ import annotations

import pytest

from anime_tools import workspace as WS
from anime_tools.gui import dataset as D
from anime_tools.gui import stages as S
from anime_tools.workspace import migrate as M

# The stage CLIs whose defaults are output paths, whichever package they live in.
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
    """`master` / `dst` / `masks` are the workspace; `src` and `out` are not."""
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
    """Running a stage bare must not publish: no default names the export tree."""
    for dest, default in _defaults(module_path).items():
        assert not default.startswith(WS.EXPORT_ROOT), f"{module_path} --{dest}"


@pytest.mark.parametrize("module_path", WRITERS)
def test_report_defaults_keep_one_component_in_front_of_their_tail(module_path):
    """``report_subpath`` drops the first component to get a stage's own tail, so
    one ``report_root`` setting moves every report — which only works while the
    root is exactly one component.
    """
    for dest, default in _defaults(module_path).items():
        if dest not in ("report_dir", "out"):
            continue
        head, _, tail = default.partition("/")
        assert head == WS.WORKSPACE, f"{module_path} --{dest} = {default}"
        assert tail and S.report_subpath(default) == tail


def test_the_resized_default_is_the_dst_root():
    """The CLI's ``--dst`` and the GUI's ``dst`` root are one tree."""
    dst = _defaults("anime_tools.stages.cli.autotag_captions")["dst"]
    assert dst == WS.DEFAULT_ROOTS["dst"] == WS.RESIZED


def test_grouping_reads_the_resized_tree_by_default():
    """``build_groups`` walks the resized tree, like every other pixel stage."""
    assert (
        _defaults("anime_tools.grouping.cli.build_groups")["source_dir"] == WS.RESIZED
    )


def test_every_pixel_reading_stage_is_bound_to_the_resized_tree():
    """A stage that opens an image reads ``dst``; ``src`` is left to ``autotag``
    (master caption fallback) and Export. The binding is also what earns each
    stage its resize preflight.
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
        "masks": "both-exist",  # never merged
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
    """An explicitly-saved legacy root is warned about, not rewritten."""
    assert M.pinned_roots({"dst": WS.LEGACY_ROOTS["dst"]}) == [
        ("dst", WS.LEGACY_ROOTS["dst"])
    ]
    # Blank or absent means "follow the defaults", which is not pinned.
    assert M.pinned_roots({}) == []
    assert M.pinned_roots({"dst": "", "masks": None}) == []
    assert M.pinned_roots({"dst": "elsewhere/resized"}) == []
