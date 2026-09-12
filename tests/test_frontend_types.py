"""``frontend/src/types.ts`` is hand-written; the constants in it are not.

The file mirrors what ``gui/stages.py::schema()``, ``gui/jobs.py`` and
``update.py`` put on the wire, and that prose is worth keeping by hand — a
generated file would lose the reason each field exists. What is *not* worth
keeping by hand is the handful of literal copies inside it: a root list, two
mask axes, three string unions and four constants, each of which is a Python
value spelled again in TypeScript with nothing checking the two agree. This is
the most-churned file in the repo, so "nothing checks" is not a small gap.

So the copies are checked and the prose is left alone. Each assertion reads the
literal out of the ``.ts`` source rather than running the bundler: a value that
the browser would only discover as a blank dropdown fails here instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "frontend" / "src"
TYPES = SRC / "types.ts"
EN = SRC / "i18n" / "en.ts"


def _string_array(source: str, name: str) -> tuple[str, ...]:
    """``export const <name> = ["a", "b"] as const;`` → ``("a", "b")``."""
    m = re.search(rf"export const {name}\s*=\s*\[(.*?)\]\s*as const", source, re.DOTALL)
    assert m, f"{name} is not spelled as an `as const` array any more"
    return tuple(re.findall(r'"([^"]*)"', m.group(1)))


def _string_union(source: str, name: str) -> tuple[str, ...]:
    """``export type <name> = "a" | "b";`` → ``("a", "b")``, one line or many."""
    m = re.search(rf"export type {name}\s*=\s*([^;]+);", source, re.DOTALL)
    assert m, f"{name} is not spelled as a string union any more"
    return tuple(re.findall(r'"([^"]*)"', m.group(1)))


def _const(source: str, name: str) -> str:
    m = re.search(rf'export const {name}\s*=\s*"([^"]*)"', source)
    assert m, f"{name} is not spelled as a string constant any more"
    return m.group(1)


def test_root_names_are_the_workspace_roots():
    from anime_tools import workspace as WS

    assert _string_array(TYPES.read_text(encoding="utf-8"), "ROOT_NAMES") == tuple(
        WS.DEFAULT_ROOTS
    )


def test_the_mask_axes_are_the_masking_requests():
    from anime_tools.masking.requests import MASK_KINDS, MASK_ROLES

    src = TYPES.read_text(encoding="utf-8")
    assert _string_array(src, "MASK_ROLES") == tuple(MASK_ROLES)
    assert _string_array(src, "MASK_KINDS") == tuple(MASK_KINDS)


def test_stage_ids_are_the_registry_in_its_order():
    """``i18n/en.ts``'s ``StageId``, which every overlay's keys are typed
    against — so a stage the registry grows is a type error in three
    translations before it is a missing label."""
    from anime_tools.stages.registry import STAGES

    assert _string_union(EN.read_text(encoding="utf-8"), "StageId") == tuple(
        s.id for s in STAGES
    )


def test_every_field_kind_the_server_emits_is_declared():
    """The schema the browser actually receives, not the set of kinds
    ``_request._kind`` could in principle return."""
    from anime_tools.gui import stages as S

    declared = set(_string_union(TYPES.read_text(encoding="utf-8"), "FieldKind"))
    emitted = {f["kind"] for s in S.STAGES for f in S.schema(s)["fields"]}
    assert emitted <= declared, f"undeclared: {sorted(emitted - declared)}"
    assert "masks" in emitted, "the masks kind is the one no other stage emits"


def test_job_states_are_the_states_a_job_reports():
    """Every branch of ``Job.state``, reached by building the job rather than by
    reading the property's source."""
    from anime_tools.gui.jobs import Job

    def job(**kw) -> str:
        return Job(id="j", stage="s", steps=[], home=ROOT, **kw).state

    reachable = {
        job(),
        job(exit_code=0),
        job(exit_code=1),
        job(exit_code=0, cancelled=True),
    }
    declared = set(_string_union(TYPES.read_text(encoding="utf-8"), "JobState"))
    assert reachable == declared


def test_the_update_unions_are_the_update_modules():
    from anime_tools import update as U

    src = TYPES.read_text(encoding="utf-8")
    assert set(_string_union(src, "UpdateState")) == {
        U.CURRENT,
        U.AVAILABLE,
        U.AHEAD,
        U.UNKNOWN,
    }
    # `install_kind` answers one of these three; only the first may be rewritten.
    assert set(_string_union(src, "InstallKind")) == {"uv-tool", "checkout", "other"}
    assert U.install_kind() in _string_union(src, "InstallKind")


@pytest.mark.parametrize(
    "ts_name, py_module, py_name",
    [
        ("REPLAY_FIELD", "anime_tools.gui.stages", "REPLAY_FIELD"),
        ("REPORT_SETTING", "anime_tools.gui.stages", "REPORT_SETTING"),
        ("MASK_SETTING", "anime_tools.gui.stages", "MASK_SETTING"),
        ("ANALYSIS_KIND", "anime_tools.stages._analysis", "ANALYSIS_SUBDIR"),
    ],
)
def test_the_wire_constants_match_their_python_owner(ts_name, py_module, py_name):
    import importlib

    expected = getattr(importlib.import_module(py_module), py_name)
    assert _const(TYPES.read_text(encoding="utf-8"), ts_name) == expected
