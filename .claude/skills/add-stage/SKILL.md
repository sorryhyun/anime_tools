---
name: add-stage
description: Checklist for adding a pipeline stage (or a flag to one) — request dataclass, runner,
CLI shell, registry row, GUI binding maps, frontend labels, the tests that pin the surface.
Load before creating a stage, renaming a flag, or wiring a new knob into the GUI.
---

# Adding a stage

A stage is one request dataclass plus one runner, and everything else is generated from or
bound to them. The order below is the dependency order; nothing is hand-typed twice.

## 1. The request

Declare the request in the package's `requests.py` (`stages/`, `masking/` or `grouping/`), a
dataclass over `anime_tools/_request.py::Request`. Every field is `arg(default, help=…,
group=…, gate=…, choices=…)`; the class docstring is the `--help` description. Rules:

- Torch-free. The module is imported by the GUI server and by the registry test with torch
  poisoned. Model imports go inside the runner's body.
- Validation in `__post_init__`, raising `ValueError`; the shell turns it into
  `parser.error`, the GUI into a 400. A missing input tree is a `FileNotFoundError`.
- Flag spelling follows the package: `stages/` underscores (`FLAG_SEP = "_"`), `masking/`
  and `grouping/` hyphens. The other spelling is generated as an alias.
- A `store_false` switch names its flag in `off=` metadata (`skip_en` is `--keep_en`).
- A drawer is a switch field with `gate=<its own name>` and a `group` title, and every knob
  inside it carries `gate=<the switch>`. The GUI folds a shut drawer and drops its knobs from the
  argv; `tests/test_masking_plan.py` shows the shape.
- Shared flags keep one spelling across stages — `path_pattern`, `tagger_dir`, `checkpoint`,
  `prompt_embed`, `device` — because the GUI fills each from one Settings value.
  `tests/test_registry_requests.py::test_shared_flags_keep_one_spelling` pins the list; the
  device flag literal exists once, in `_device.py::DEVICE_HELP`.
- Roots and report paths default in terms of `anime_tools/workspace/__init__.py` (`RESIZED`,
  `REPORTS`, …) so the CLI and GUI defaults can't drift.
- SAM3 detection knobs are the nested `DetectionRequest` (`GROUP = "detection"`); reuse it, and
  build the options object field by field through `.options()` so a knob with no request field
  is an error.

## 2. The runner and the shell

`run_<stage>(req)` in `stages/run.py` (or the package's own module for masking/grouping):
preflight, model load through `stages/_models.py` (cached per process; `release_models()`
empties it), the library call, `report.json` via `cli/_report.py`, progress through
`cli/_args.py::make_progress` (the `  [done/total] detail` line is the GUI's bar and the daemon's
`progress.jsonl`), and the printed epilogue. Dry-run by default; `--apply` writes. A
`phase("…")` bracket from `_progress.py` around a model load keeps the daemon's stall watchdog
quiet.

Caption reads go through `_walk_captions.resolve_caption` (revised first, master fallback);
caption writes through `_caption_io.write_caption`, which pushes the replaced text onto
`{stem}.history.txt`. Anything that proposes a text change and applies it later goes through
`replay.apply_one`, and its report shape is one of `contract.REPLAY_SHAPES` (add one there if
none fits; the GUI's `proposals.py` reads the same objects).

The CLI in `cli/` is a shell and nothing more: `build_parser()` returns `Request.parser()`,
`main()` calls `run_<stage>(Request.from_argv(parser, argv))`. Copy `stages/cli/ocr_captions.py`.

## 3. Registration

- `stages/registry.py`: a `Stage(id, title, request="module:Class", run="module:function",
  module="python -m path", panel=…, report=("report_dir", "report.json"), short=…, notes=…)`.
  `panel` is the dock button; `hidden=True` keeps a preflight-only stage out of the dock.
- `stages/__init__.py` `__all__` (lazy export) if it lives in `stages/`.
- `gui/stages.py`: a `ROOT_FIELDS` row (which dests are dataset roots), `BASIC_FIELDS` if some
  knobs should fold under Advanced, `MASK_FIELDS` / `REPORT_INPUTS` / `PANEL_FIELDS` when they
  apply, `NO_PREFLIGHT` if the stage must not have `resize` run in front of it.
- `frontend/src/i18n/{en,ko,ja,zh}.ts`: `stage.panels` / `stage.titles` / `stage.shorts` entries
  keyed by the id. A missing id falls back to the English the server sent, but `en.ts` is the
  schema, so add it there first and `tsc` names the other three. Then `make frontend`.
- `docs/<stage>.md` + a row in `docs/README.md`; `examples/` gets a script if the API is new.
- If the trainer will call it, the trainer repo — the dependency is one-way, so the wrapper lives
  there and imports this package, never the reverse.

## 4. Tests to update

- `tests/test_registry_requests.py::CASES` needs a non-default request per stage id; the
  parametrized tests then cover round-trip, defaults, the torch-poisoned import, the runner
  resolution and the dual flag spelling for free.
- `tests/test_stage_requests.py` (or `test_masking_requests.py`) for the stage's own validation.
- `tests/test_gui.py` sees the new schema through the registry; a bound field needs a case there
  if the binding is new.
- `tests/test_boundary.py` runs unchanged but will fail if the request module imports torch.

Run `uv run pytest -q tests/test_registry_requests.py tests/test_boundary.py tests/test_gui.py`
before the stage's own tests.
