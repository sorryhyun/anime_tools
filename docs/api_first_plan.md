# API-first boundary plan (2026-09-02)

The trainer (`anima_lora`) drives this package by hand-assembling argv strings
and shelling out to `python -m anime_tools.<pkg>.cli.<name>`. Nothing on either
side checks that the flags it emits exist in the parser they hit. On
2026-09-02 the trainer's pin was 88 commits behind and its `make mask` already
failed against HEAD (`generate_masks --config` was replaced by `--prompts` /
`--focus-prompts` / `--threshold` / `--dilate`).

This plan makes a **typed Python API the primary surface** and demotes every
CLI to a thin shell over it. It is the same rule the trainer already follows
(`scripts/preprocess/*.py` are shells over `library/preprocess/`, inference is
driven by a `GenerationRequest` whose `.to_args()` yields the argv).

The trainer-side half of this plan is
`anima_lora/docs/proposal/anime_tools_api_first.md`; the release that ships it
is tracked in `anima_lora/docs/v2_release_plan.md` (Track D).

## Design

One frozen dataclass per stage, in a **torch-free module**, with the heavy
imports deferred into the call:

```python
from anime_tools.masking import SamMaskRequest, generate_sam_masks

req = SamMaskRequest(
    image_dir="post_image_dataset/resized",
    mask_dir="workspace/masks_sam",
    prompts=("speech bubble", "text bubble"),
    threshold=0.7, dilate=3, path_pattern="artist_a/*",
)
report = generate_sam_masks(req)   # in-process
argv = req.to_argv()               # for a subprocess / the trainer's daemon
```

Every request class carries the same four things:

| Member | Contract |
|---|---|
| fields | one per knob, typed, defaulted; field name == argparse `dest` |
| `to_argv()` | canonical spelling (`--path_pattern`), omits defaults |
| `from_namespace(ns)` | the CLI's only job after `parse_args()` |
| `run()` / module-level `run_<stage>(req)` | imports torch, cv2, sam3 **inside** |

Three consumers derive from it, so the request is the single contract:

- **CLI**: `build_parser()` is generated from the dataclass fields (help from
  field metadata), `main()` is `run(Request.from_namespace(parse_args()))`.
  Every flag gets both spellings (`--foo_bar` canonical, `--foo-bar` alias)
  from one helper, ending the per-package hyphen/underscore split.
- **Web GUI**: `gui/stages.py` builds its form schema from the dataclass
  instead of reaching into `parser._actions` / `_action_groups`. The
  "declaration order is part of the contract" fragility in
  `stages/cli/_args.py` goes away.
- **Trainer**: builds the request and either calls it in-process or submits
  `req.to_argv()` to its daemon. No hand-written flag strings.

A torch-free `anime_tools.contract` module holds the constants both GUIs and
the trainer currently copy by string: the autotag stdio sentinels, autotag
`MODES`, the dbv4 checkpoint file set, `REPLAY_REPORT_NAME`, `GATE_ATTR`, the
replay `SHAPES`, and a `CONTRACT_VERSION` integer bumped on any incompatible
change to a request class.

## Stage inventory

| Stage | Library function today | CLI file | Work |
|---|---|---|---|
| SAM masks | none (logic in `masking/cli/generate_masks.py::main`) | 213 lines | extract `run_sam_masks`, `SamMaskRequest` |
| MIT text masks | none (~350 lines in the CLI) | 453 lines | extract `run_mit_masks`, `MitMaskRequest`; share the SAM3 load with the stage above |
| merge masks | `masking._masks.mask_run` (partial) | 85 lines | `MergeMasksRequest` |
| autotag | `stages.autotag.run_autotag_captions` | wrap | `AutotagRequest` |
| position clauses | `stages.position_captions.run_position_captions` | wrap | `PositionRequest` (+ the `_detection` block as a nested dataclass) |
| correct | `stages.captions.write_corrected_preprocess_captions` | wrap | `CorrectRequest` |
| OCR | `stages.ocr` | wrap | `OcrRequest` |
| resize | `stages.resize.run_resize_images` | wrap | `ResizeRequest` |
| export | `stages.export_workspace` | wrap | `ExportRequest` |
| grouping | `grouping.groups.build_groups` | wrap | `GroupRequest` |
| audit | `stages.multiview_audit` | wrap | `AuditRequest` |

`masking/__init__.py`, `grouping/__init__.py` and `stages/__init__.py` export
nothing today; each gains a lazy PEP 562 `__getattr__` like `tagger/__init__.py`.

## Phases

- [x] **P0. Hygiene.** `__version__` reads `importlib.metadata` (it says
  `0.1.0`; `pyproject.toml` says `0.3.1`). Land `anime_tools/contract.py` with
  the constants above and `CONTRACT_VERSION = 1`; switch `gui/proposals.py`,
  `gui/stages.py` and `tagger/cli/autotag_server.py` to import from it.
  `tests/test_boundary.py` gains `test_contract_is_torch_free`.
- [ ] **P1. Masking.** Move the two mask mains into `masking/sam.py` /
  `masking/mit.py` as `run_*` functions over request dataclasses in
  `masking/requests.py` (torch-free). Add `masking/_sam3.py::load_sam3()` with
  a process-level cache so MIT's SAM gate reuses the subject-mask model when
  both run in one process. CLIs become shells; flags unchanged (aliases).
- [ ] **P2. Caption stages.** Request classes over the existing `run_*`
  functions for autotag, position, correct, OCR, resize, export, audit. The
  detection block (`stages/cli/_detection.py`) becomes a nested dataclass
  shared by position and audit, which is what
  `test_the_two_sam3_stages_declare_identical_detection_flags` pins today.
  `stages/cli/_models.py::load_tagger` takes the request instead of a
  namespace and caches by checkpoint dir, so autotag followed by position in
  one process loads the tagger once.
- [ ] **P3. Grouping.** `GroupRequest` over `build_groups`.
- [ ] **P4. Registry and GUI.** Move `STAGES` out of `gui/stages.py` into
  `stages/registry.py` (torch-free: it only names request classes and
  modules). `gui/stages.py::schema()` is derived from the dataclass; drop the
  argparse-private introspection and the on-disk schema cache, which existed
  only because importing the CLIs was expensive. Keep `build_argv()` as a
  thin call to `Request.to_argv()`.
- [ ] **P5. Daemon-friendly progress.** `anime_tools/_progress.py`
  (stdlib): when `ANIMA_DAEMON_JOB_DIR` is set, the stage's `progress()`
  callback also appends `{"event": "step", "step": i, "total": n, "detail":
  …}` lines to `<job_dir>/progress.jsonl`, the file the trainer's daemon
  stall watchdog and `get_progress` already read. Today a quiet SAM3 or
  tagger load can be killed after 120 s under the daemon.
- [ ] **P6. Round-trip tests.** For every registered stage:
  `Request.from_namespace(build_parser().parse_args(req.to_argv())) == req`
  on a non-default instance, and `import anime_tools.<pkg>.requests` under
  `sys.modules` torch-poisoning (the pattern `test_gui_server_is_torch_free`
  uses). These two tests replace most of `tests/test_stage_cli_args.py`.

Phases P1 to P3 keep every existing flag working, so the trainer can bump its
pin at any point after P0 without waiting for the rest.

## Invariants

- A request module imports without torch, cv2, sam3, onnxruntime or timm.
  Pinned by test.
- `to_argv()` and `build_parser()` are generated from the same field list;
  neither is hand-written per stage.
- The CLI flag surface is append-only within a `CONTRACT_VERSION`; a removed
  or renamed field bumps it. The trainer asserts the version it was written
  against.
- Dependency direction is unchanged: the trainer imports this package, never
  the reverse (`tests/test_boundary.py`).
- The resident autotag worker's stdio protocol
  (`tagger/cli/autotag_server.py`) is **not** wrapped in a request; it is a
  streaming worker, not a one-shot stage. Its sentinels move to `contract.py`
  and nothing else changes.

## Decisions left open

- **Web GUI job runner and the trainer's daemon.** `gui/jobs.py` is a
  single-slot `Popen` runner; on a box that also runs `anima_lora`, the two
  contend for the GPU with no queue between them. The runner could submit to
  the daemon over HTTP when its pidfile is present (`~/.anima/daemon.json`,
  stdlib `urllib`, no import of trainer code). This is a runtime dependency,
  not an import, so it does not violate the boundary test, but it does
  couple this package's GUI to the trainer's process model. Default: leave
  it, revisit if contention is reported.
- **Write target of batch autotag.** Since `7395a58` autotag writes the
  revised caption, not the master. The trainer's correction pass still
  rewrites the revised caption from the master and only preserves position
  clauses, so a trainer-side `caption-autotag --apply` followed by
  `preprocess` loses the tags. Resolution is on the trainer side (its
  proposal, T0), but the contract table in `docs/contract.md` must state
  which side owns the revised caption before the pin bumps.
