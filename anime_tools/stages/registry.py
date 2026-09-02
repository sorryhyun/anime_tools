"""The stage registry: every stage the package exposes, by id, with the request
class that is its surface and the ``python -m`` module that runs it.

Torch-free and import-light on purpose — it names request classes and runners
as ``module:name`` strings and resolves one only when asked
(:meth:`Stage.request_class`, :meth:`Stage.runner`), so the GUI server, the
trainer and the tests can list the stages without importing any of them. The GUI's own bindings (which
flags are dataset roots, which are ⚙ Settings values, which fold under
Advanced) stay in :mod:`anime_tools.gui.stages`; this is the part that is true
of a stage whoever is driving it.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["BY_ID", "PANELS", "STAGES", "Stage", "request_class", "resolve"]


def resolve(spec: str):
    """The object a ``module:name`` string names, imported now."""
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


def request_class(spec: str) -> type:
    """The request class ``module:Class`` names, imported now."""
    return resolve(spec)


@dataclass(frozen=True, kw_only=True)
class Stage:
    id: str
    title: str
    request: str
    """``module:Class`` — the stage's request dataclass, resolved lazily."""
    run: str
    """``module:function`` — the in-process runner, ``run_<stage>(request)``;
    the CLI shell in :attr:`module` is ``run(Request.from_argv())``. Resolved
    lazily, since the runners' modules are the ones that import the models."""
    module: str
    """The ``python -m`` module: the CLI shell over :attr:`request`."""
    panel: str
    """Which dock button this stage lives under; several stages share one, and
    the panel picks between them."""
    extra: str = ""
    """Feature area (``tagger`` / ``stages`` / ...); informational only."""
    report: tuple[str, str | None] | None = None
    """``(dest, filename)``: the form field naming the report dir (or file when
    ``filename`` is None), so the GUI can fetch the result after a run."""
    notes: str = ""
    short: str = ""
    """Label for the in-panel picker; defaults to :attr:`title`."""
    hidden: bool = False
    """Keep this stage out of the dock. It still has a schema and an argv, so
    it can run as a preflight and be configured from Settings."""

    def request_class(self) -> type:
        """The request class, imported now. Raises ``ImportError`` when the
        stage's dependencies are missing, which the GUI reports as *unavailable*."""
        return request_class(self.request)

    def runner(self) -> Callable[[Any], Any]:
        """The runner, imported now. Same ``ImportError`` contract as
        :meth:`request_class`; the runner's own model imports stay inside its
        body, so this import alone loads no weights."""
        return resolve(self.run)


_STAGES = "anime_tools.stages.requests"
_RUN = "anime_tools.stages.run"
_MASKING = "anime_tools.masking.requests"
_GROUPING = "anime_tools.grouping.requests"

STAGES: tuple[Stage, ...] = (
    Stage(
        id="resize",
        title="Resize to buckets",
        request=f"{_STAGES}:ResizeRequest",
        run=f"{_RUN}:run_resize",
        module="anime_tools.stages.cli.resize_images",
        panel="Resize",
        extra="stages",
        report=("report_dir", "report.json"),
        hidden=True,
        notes=(
            "Runs automatically before every stage that reads the resized "
            "tree. These defaults apply to all of them."
        ),
    ),
    Stage(
        id="autotag",
        title="Autotag captions",
        request=f"{_STAGES}:AutotagRequest",
        run=f"{_RUN}:run_autotag",
        module="anime_tools.stages.cli.autotag_captions",
        panel="Autotag",
        extra="tagger",
        report=("report_dir", "report.json"),
        notes=(
            "Writes the revised caption under the resized tree; the master is "
            "read as a fallback and never edited. `missing` skips any image a "
            "caption already speaks for."
        ),
    ),
    Stage(
        id="position",
        title="Position captions",
        request=f"{_STAGES}:PositionRequest",
        run=f"{_RUN}:run_position",
        module="anime_tools.stages.cli.position_captions",
        panel="Curate",
        extra="stages",
        report=("report_dir", "report.json"),
        short="Position",
    ),
    Stage(
        id="correct",
        title="Correct + mirror captions",
        request=f"{_STAGES}:CorrectRequest",
        run=f"{_RUN}:run_correct",
        module="anime_tools.stages.cli.correct_captions",
        panel="Curate",
        extra="tokenizers",
        notes=(
            "Corrects the revised caption in place; the master is read only for "
            "an image that has none yet, and is never edited."
        ),
        short="Correct",
    ),
    Stage(
        id="audit",
        title="Multiview audit",
        request=f"{_STAGES}:AuditRequest",
        run=f"{_RUN}:run_audit",
        module="anime_tools.stages.cli.audit_multiview",
        panel="Curate",
        extra="stages",
        report=("report_dir", "report.json"),
        short="Audit",
    ),
    Stage(
        id="ocr",
        title="OCR text",
        request=f"{_STAGES}:OcrRequest",
        run=f"{_RUN}:run_ocr",
        module="anime_tools.stages.cli.ocr_captions",
        panel="OCR",
        extra="stages",
        report=("report_dir", "report.json"),
        notes=(
            "Writes {stem}.ocr.txt into the OCR tree, mirroring the resized "
            "tree. Reads and writes no caption."
        ),
    ),
    Stage(
        id="groups",
        title="Build groups",
        request=f"{_GROUPING}:GroupRequest",
        run="anime_tools.grouping.groups:run_groups",
        module="anime_tools.grouping.cli.build_groups",
        panel="Groups",
        extra="grouping",
        report=("out", None),
    ),
    Stage(
        id="masks_sam",
        title="SAM3 subject masks",
        request=f"{_MASKING}:SamMaskRequest",
        run="anime_tools.masking.sam:run_sam_masks",
        module="anime_tools.masking.cli.generate_masks",
        panel="Masks",
        extra="masking",
        short="Subject",
    ),
    Stage(
        id="masks_mit",
        title="Text masks",
        request=f"{_MASKING}:MitMaskRequest",
        run="anime_tools.masking.mit:run_mit_masks",
        module="anime_tools.masking.cli.generate_masks_mit",
        panel="Masks",
        extra="masking",
        short="Text",
        notes=(
            "Two detectors, each behind its own switch: SAM3 on a prompt "
            "(balloons) and the UNet++ segmenter (lettering). Their masks are "
            "unioned."
        ),
    ),
    Stage(
        id="masks_merge",
        title="Merge masks",
        request=f"{_MASKING}:MergeMasksRequest",
        run="anime_tools.masking.merge:run_merge_masks",
        module="anime_tools.masking.cli.merge_masks",
        panel="Masks",
        extra="masking",
        short="Merge",
    ),
    Stage(
        id="export",
        title="Export workspace",
        request=f"{_STAGES}:ExportRequest",
        run=f"{_RUN}:run_export",
        module="anime_tools.stages.cli.export_workspace",
        panel="Export",
        extra="stages",
        report=("report_dir", "report.json"),
        notes=(
            "The only stage that writes outside the workspace: Run copies to the "
            "tree the trainer reads, skipping what is already identical. Undo "
            "restores the text it overwrote; a replaced pixel is not undoable."
        ),
    ),
)

BY_ID: dict[str, Stage] = {s.id: s for s in STAGES}

PANELS: tuple[str, ...] = tuple(dict.fromkeys(s.panel for s in STAGES if not s.hidden))
"""The dock's buttons, in registry order. Hidden stages contribute none."""
