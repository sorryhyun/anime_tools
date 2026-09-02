"""The constants both halves of the seam spell: this package's CLIs and GUI, and
the trainer (``anima_lora``) that shells out to them.

**Stdlib only.** Nothing here imports another ``anime_tools`` module, so the GUI
server, the resident autotag worker's driver and the trainer can all read it
without pulling a stage (and its torch) into their process. Pinned by
``tests/test_boundary.py::test_contract_is_torch_free``.

Everything below is part of the CLI/stdio surface a consumer was written
against. That surface is append-only within one :data:`CONTRACT_VERSION`; a
removed or renamed name bumps it, and the trainer asserts the version it
expects. The file formats themselves are in ``docs/contract.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

CONTRACT_VERSION = 1
"""Bumped on any incompatible change to a name in this module or a stage's flags."""

# --- resident autotag worker (``tagger/cli/autotag_server.py``) ---------------
# Only these sentinel lines reach **stdout**; logging goes to stderr. The driver
# reads live, so each is one line, flushed.
AUTOTAG_READY = "ANIMA_AUTOTAG_READY"
"""The worker has loaded its model and is serving requests."""
AUTOTAG_RESULT_PREFIX = "ANIMA_AUTOTAG_RESULT\t"
"""``prefix + caption`` — one per request, also the single-image CLI's output."""
AUTOTAG_ERROR_PREFIX = "ANIMA_AUTOTAG_ERROR\t"
"""``prefix + message`` — the request failed; the worker keeps serving."""

# --- batch autotag (``stages/autotag.py``) ------------------------------------
AUTOTAG_MODES = ("missing", "merge", "overwrite")
"""``--mode``: only ``missing`` is non-destructive (see ``docs/anima_tagger.md``)."""

# --- tagger checkpoint layout (``tagger/dbv4_meta.py``) -----------------------
# Our half of the tagger — vocab / rules / groups / thresholds / sidecar — is
# fetched into ``models/captioners/`` when a required file is missing.
TAGGER_REQUIRED_FILES = ("config.json", "model.safetensors", "vocab.json", "rules.yaml")
TAGGER_OPTIONAL_FILES = ("thresholds.safetensors", "groups.yaml")
# A dbv4-backed checkpoint carries no ``model.safetensors`` (the weights come
# from the gated upstream repo); the sidecar pair is optional.
DBV4_REQUIRED_FILES = ("config.json", "vocab.json", "rules.yaml")
DBV4_OPTIONAL_FILES = TAGGER_OPTIONAL_FILES + ("sidecar.safetensors", "sidecar.json")
DBV4_BACKBONE_FILES = ("model.safetensors", "selected_tags.csv", "meta.json")
"""What the backbone loader pulls from the upstream repo."""

# --- stage reports and their replay (``stages/replay.py``) --------------------
REPLAY_REPORT_NAME = "apply_report.json"
"""What ``--from_report --apply`` writes, beside the dry run's ``report.json``
and never over it."""

GATE_ATTR = "gui_gate"
"""The attribute an argparse group carries when it is a *drawer*: the dest of the
boolean that switches the whole group on. Stamped by
``masking._masks.gated_group``, read by ``gui.stages.fields_of``."""


@dataclass(frozen=True)
class ReplaySpec:
    """How to read one stage's report and where its proposals get written.

    ``target_root`` is the tree ``caption_path`` is relative to: ``"src"`` for
    the stages that write the caption master, ``"dst"`` for the revised caption.
    """

    stage: str
    # Report container keys — ``rows``/``stats`` (autotag) or ``images``/``summary``
    # (position clauses, multiview audit).
    rows_key: str = "rows"
    stats_key: str = "stats"
    # A row is writable when its status matches, and/or ``row_filter`` says so.
    # The multiview audit gates on verdict/confidence instead of a status field,
    # so it supplies a closure over the CLI's own gate.
    ok_status: str | None = None
    row_filter: Callable[[Mapping[str, object]], bool] | None = None
    before_field: str = "existing"
    after_field: str = "proposed"
    target_root: str = "src"
    # Passed straight to ``_caption_io.write_caption``; ``history_by`` names who
    # the superseded version is filed under, so a replay pushes the same history
    # entry the stage's own apply would have.
    newline: bool = False
    drop_variants: bool = False
    history_by: str | None = None


REPLAY_SHAPES: dict[str, ReplaySpec] = {
    # The proposal lands on the **revised** caption (``--dst``); the master is the
    # read-only fallback the tagger merged into, so the drift baseline is the
    # target's own text (``target_before``), not what spoke for the image.
    "autotag": ReplaySpec(
        stage="autotag_captions",
        rows_key="rows",
        stats_key="stats",
        ok_status="ok",
        before_field="target_before",
        after_field="proposed",
        target_root="dst",
        drop_variants=True,
        history_by="autotag",
    ),
    # ``drop_variants`` mirrors the stage's own write: a stale
    # ``{stem}.variants.txt`` outranks ``{stem}.txt`` at encode time.
    "position": ReplaySpec(
        stage="position_captions",
        rows_key="images",
        stats_key="summary",
        ok_status="proposed",
        before_field="original",
        after_field="proposed",
        target_root="dst",
        drop_variants=True,
        history_by="position",
    ),
    # The writable set is the verdict/confidence gate, not a row ``status``, so
    # ``row_filter`` is left open here and closed over the gate at replay time.
    # ``apply_findings`` writes ``proposed + "\n"``; a replay must be
    # byte-identical to it.
    "audit": ReplaySpec(
        stage="audit_multiview",
        rows_key="images",
        stats_key="summary",
        before_field="caption",
        after_field="proposed",
        target_root="src",
        newline=True,
    ),
}
"""GUI stage id → the shape of the report its CLI writes. OCR is absent: it writes
only its own sidecar tree and touches no caption."""
