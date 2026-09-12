"""What one request is answered against: the settings-derived values, who is
asking, and the caching rule every file this server hands out goes out under.

:class:`RunContext` is the settings file read **once** per request. Every route
used to recompute ``roots_for`` / ``stage_defaults`` / ``report_root`` /
``mask_root`` in its own combination and then pass all four into
``S.build_argv``; here they are one object, and :meth:`RunContext.bindings` is
the four keyword arguments a form is resolved and turned into argv with. The
free functions below are still the implementations, because two routes need a
value computed against *empty* settings — the Settings dialog's placeholders.

``D.DatasetError`` propagates to the app-wide 400 handler, so nothing here
catches it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import Request

from anime_tools.gui import dataset as D
from anime_tools.gui import stages as S
from anime_tools.gui.jobs import Step
from anime_tools.gui.settings import load_settings
from anime_tools.stages.resize import DEFAULT_MIN_PIXELS

__all__ = [
    "LOOPBACK_HOSTS",
    "NO_CACHE",
    "RunContext",
    "is_loopback",
    "make_output_dirs",
    "mask_root",
    "preprocess_min_pixels",
    "preprocess_steps",
    "report_root",
    "root_paths",
    "roots_for",
    "stage_defaults",
]

NO_CACHE = {"cache-control": "no-cache"}
"""Every file this server hands out is one something else rewrites.

Starlette sends a ``last-modified`` and no ``cache-control``, which is the case
a browser answers by *inventing* a freshness window — a tenth of the file's age
— and serving its cached copy without asking. So a page rebuilt by ``make
frontend`` goes on running yesterday's bundle, and an image a stage rewrote in
place goes on being drawn as it was. ``no-cache`` keeps the copy and makes the
load revalidate: the file's own ETag turns an unchanged one into a 304, so the
1.7 MB font is not re-sent for the sake of a header. Not ``no-store``, which
would throw the copy away and re-send it every time."""

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"})
"""Who may open a window on this desktop: only the machine it is drawn on."""


def is_loopback(request: Request) -> bool:
    client = request.client
    return client is not None and client.host in LOOPBACK_HOSTS


# ---- settings-derived values, as pure functions ---------------------------- #
# Each takes the settings mapping rather than reading it, so one request reads the
# file once — which is what :class:`RunContext` is.


def roots_for(settings: Mapping[str, Any], **overrides: str) -> D.Roots:
    """The dataset roots for this request: overrides win, blanks fall back to
    the saved roots, then to :data:`D.DEFAULT_ROOTS`.

    The containment bases are computed here, from the settings already in hand,
    and handed down: otherwise every root recomputes them and each of those is
    another read of the settings file this object exists to read once.
    """
    saved = settings.get(D.SETTINGS_KEY) or {}
    merged = {**saved, **{k: v for k, v in overrides.items() if v}}
    return D.resolve_roots(merged, bases=D.dataset_bases(settings))


def stage_defaults(settings: Mapping[str, Any]) -> dict[str, str]:
    """The Settings dialog's stage defaults (``S.SETTING_FIELDS``). Blanks are
    dropped, so an emptied field means "the CLI's own default"."""
    got = settings.get(S.SETTINGS_KEY) or {}
    return {
        k: str(got[k]).strip()
        for k in S.SETTING_FIELDS.values()
        if str(got.get(k) or "").strip()
    }


def _root_beside(settings: Mapping[str, Any], key: str, sibling: Path) -> str:
    """A Settings root that stages hang their own tails off, home-relative:
    whatever ``key`` holds, and *beside* ``sibling`` when it holds nothing.

    *Beside* stops at the curation home: a ``sibling`` of one component (a bare
    ``resized``, or a pre-workspace export root) has the home itself
    for a parent, which would strew the tails across the project root. The
    workspace root is used instead.
    """
    got = str((settings.get(S.SETTINGS_KEY) or {}).get(key) or "").strip()
    if got:
        return got
    beside = PurePosixPath(D.rel_to_home(sibling)).parent.as_posix()
    return D.WS.WORKSPACE if beside in (".", "", "/") else beside


def report_root(settings: Mapping[str, Any], roots: D.Roots) -> str:
    """Where every stage's report lands, home-relative — the root only; each stage
    appends its own tail (``S.Field.report``), so no two share a ``--report_dir``
    and one stage's ``--from_report`` cannot read another's report.

    Blank means *beside the* ``dst`` *root* (:func:`_root_beside`).
    """
    return _root_beside(settings, S.REPORT_SETTING, roots.dst)


def mask_root(settings: Mapping[str, Any], roots: D.Roots) -> str:
    """Where the generator's *own* mask tree lands, home-relative — the root only;
    the generator appends its own tail (``S.Field.mask``), so a second tree merged
    in beside it cannot overwrite its ``{stem}_mask.png`` at the same relative
    path.

    Blank means *beside the* ``masks`` *root* (:func:`_root_beside`).
    """
    return _root_beside(settings, S.MASK_SETTING, roots.masks)


def preprocess_min_pixels(settings: Mapping[str, Any]) -> int:
    """The resize floor the preflight runs at, from the Settings *Preprocess*
    block — the stage's own default when blank, 0 when turned off.

    Read here rather than off the schema dump, because the item route must answer
    while the schemas are still loading.
    """
    got = (settings.get(S.PREPROCESS_SETTINGS_KEY) or {}).get("min_pixels")
    if got is None or str(got).strip() == "":
        return DEFAULT_MIN_PIXELS
    try:
        return max(0, int(got))
    except (TypeError, ValueError):
        return DEFAULT_MIN_PIXELS


def root_paths(roots: D.Roots) -> dict[str, str]:
    """The dataset roots, home-relative, for the fields bound to them
    (``S.ROOT_FIELDS``)."""
    return {k: v["path"] for k, v in roots.as_dict().items()}


# ---- the request's own snapshot ------------------------------------------- #


@dataclass(frozen=True)
class RunContext:
    """The settings file as one request sees it, read once.

    Built by :meth:`load`; :attr:`defaults` is a field rather than a property so
    :meth:`scoped_to` can narrow it without a second read of anything.
    """

    settings: Mapping[str, Any]
    roots: D.Roots
    defaults: dict[str, str]
    """The ⚙ Settings stage values a form's fields fall back to."""
    report_root: str
    mask_root: str

    @classmethod
    def load(cls, **overrides: str) -> RunContext:
        """One settings read, then everything derived from it. ``overrides`` are
        per-request root spellings (``src`` / ``dst`` / ``masks``)."""
        settings = load_settings()
        roots = roots_for(settings, **overrides)
        return cls(
            settings=settings,
            roots=roots,
            defaults=stage_defaults(settings),
            report_root=report_root(settings, roots),
            mask_root=mask_root(settings, roots),
        )

    @property
    def root_paths(self) -> dict[str, str]:
        return root_paths(self.roots)

    @property
    def min_pixels(self) -> int:
        return preprocess_min_pixels(self.settings)

    def bindings(self) -> dict[str, Any]:
        """The four keyword arguments ``S.resolved_schema`` and ``S.build_argv``
        bind a stage form with — what used to be four recomputed values."""
        return {
            "roots": self.root_paths,
            "settings": self.defaults,
            "report_root": self.report_root,
            "mask_root": self.mask_root,
        }

    def scoped_to(self, rel: str) -> RunContext:
        """The same context with the run narrowed to one dataset image."""
        return replace(
            self, defaults={**self.defaults, S.SCOPE_FIELD: D.item_pattern(rel)}
        )


# ---- what a run writes ---------------------------------------------------- #


def make_output_dirs(stage: S.Stage, report: str | None, roots: D.Roots) -> None:
    """Create the directories this run *writes* to, so a fresh home does not need
    a mkdir tour before the first job.

    Only the outputs the GUI itself chose: the workspace roots this stage binds
    (``D.OUTPUT_ROOTS`` ∩ ``S.ROOT_FIELDS``) and the report directory. Never
    ``src``, never a free-text path off the stage form (that would hide a typo),
    and never ``out``, so an export tree that exists means an export happened.
    """
    for name in set(S.ROOT_FIELDS.get(stage.id, {}).values()) & D.OUTPUT_ROOTS:
        D.ensure_output_dir(getattr(roots, name))
    if report:
        D.ensure_output_dir(Path(report).parent)


def preprocess_steps(
    stage: S.Stage, ctx: RunContext, *, schemas: Mapping[str, Any]
) -> list[Step]:
    """The resize preflight for ``stage``, or nothing.

    It runs with the *same* ``ctx.defaults`` the stage got, so a per-image Apply
    resizes exactly that image; its own knobs come from the Settings
    ``preprocess`` block. A stage whose preflight is unavailable runs alone.
    """
    pre_id = S.preprocess_for(stage.id)
    pre = S.BY_ID.get(pre_id or "")
    sc = schemas.get(pre_id or "")
    if pre is None or sc is None or not sc["available"]:
        return []
    saved = ctx.settings.get(S.PREPROCESS_SETTINGS_KEY) or {}
    values = S.form_values(sc["fields"], saved)
    argv = S.build_argv(sc, values, **ctx.bindings())
    make_output_dirs(
        pre, S.report_path(pre, sc["fields"], values, ctx.report_root), ctx.roots
    )
    return [Step(pre.module, argv, pre.id)]
