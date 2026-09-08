# anime_tools/stages/

Caption-master stages and their thin CLIs: `resize.py`, `autotag.py` (modes
`missing`/`merge`/`overwrite`; only `missing` is non-destructive), `position_captions.py` (SAM3
instances → reading order → mask-blanked crops → tagger → clause rewrite; see
`docs/position_captions.md`), `captions.py` (correction + mirror), `multiview_audit.py`
(`docs/multiview_audit.md`), `ocr.py` (one path: the AnimeText
text-block detector, detect-only, every box read by the manga VL reader through
`anime_tools.ocr.reread.RereadEngine`, plus the optional `--mask_dir` components as `0.000`-score
lines; torch lives in `run.py::_vl_engine`, never in the ONNX detector load),
`export_workspace.py`. Adding one is the `add-stage`
skill; the caption grammar these stages write is the `captions` skill.

## Requests and runners

**The surface is a request object per stage** (`requests.py`, torch-free): `ResizeRequest`,
`AutotagRequest`, `PositionRequest`, `CorrectRequest`, `OcrRequest`, `AuditRequest`,
`ExportRequest`, run by `run.py::run_<stage>(req)`, which is the old CLI main minus the parsing
(preflight, model load, the library call, `report.json`, the printed epilogue). Same base as
masking's (`anime_tools/_request.py`), with two differences: flags are spelled with underscores
(`FLAG_SEP = "_"`), and a `store_false` switch names its one flag in `off` metadata (`skip_en` is
`--keep_en`). **The parser is generated from the class** (`Request.parser()` →
`_request.build_parser`): every field is declared through `arg(default, help=…, group=…, gate=…,
choices=…)`, the class docstring is the `--help` description, and every flag with a separator
takes the other spelling as an alias (`--path_pattern` / `--path-pattern`). The CLIs in `cli/` are
one-line shells (`build_parser()` = `Request.parser()`, `main()` = `run_<stage>(from_argv())`).

The SAM3 detection flags are one nested `DetectionRequest` (`GROUP = "detection"`) and the audit's
verdict gate one nested `MultiviewRequest` (`GROUP = "multiview audit"`), both shared by
`PositionRequest` and `AuditRequest`; `.options()` on either builds the `PositionCaptionOptions`
field by field, so an option field with no request field is an error, not a silent default. The
audit pins `min_instances=2` (`MultiviewRequest.MIN_INSTANCES`, applied by the shared
`requests.audit_options`) rather than exposing it. Validation lives in `__post_init__`
(autotag mode, `--flatten` vs `--from_report`, the randomize tokenizers, resize tiers); a missing
input tree is a `FileNotFoundError` the shell turns into `SystemExit`. `__init__.py` exposes all
fifteen names lazily; `tests/test_registry_requests.py` round-trips every registered stage's
request through its parser and imports the request half torch-poisoned;
`tests/test_stage_requests.py` keeps the stage-specific pins.

**`registry.py`** is the stage list — `Stage(id, title, request="module:Class",
run="module:function", module, panel, …)` for all eleven stages, masking and grouping included —
resolved lazily (`Stage.request_class()`, `Stage.runner()`), so the GUI server and the trainer can
enumerate stages without importing one, and a driver can go from a stage id to the in-process
`run_<stage>(req)` call without naming a runner.

`_models.py::load_anima_tagger` caches the tagger per `(checkpoint dir, device)`, so autotag
followed by position in one process loads it once, and `release_models()` (exported from
`stages`) empties that cache and SAM3's for a driver that runs stages in-process and then hands
the GPU to another process (the trainer's daemon job does, before its VAE/TE children);
`detector.py::build_detect_fn` builds the SAM3 detector from a `DetectionRequest` (the A/B,
review and probe CLIs share it).

## The audit phase

`PositionRequest.multiview_audit` (`off` / `report` / `apply`) runs the multiview audit as the
position stage's **first** phase, over the complement population — `is_audit_target` is defined as
exactly what `is_candidate` rejects as `single-subject`. `run.py::_run_audit_phase` reuses the
already-resident SAM3 + tagger, detects under `req.audit_options()`, and in `apply` mode hands
`multiview_audit.promotions()` to `run_position_captions(promoted=…)`, which substitutes the
promoted caption **before** `is_candidate` sees it.

**Order is load-bearing, not a preference.** `multiple views` is what moves an image out of the
`single-subject` rejection AND what `is_repeated_subject_layout` reads to arm the `view_invariant`
gate; audit after the sweep and both arrive too late. `admitted()` is the one verdict/confidence
gate the write path and the promotion path share.

**The two write different trees on purpose.** `apply_findings` (the standalone stage) writes the
caption master; the phase writes the revised tree through `run_position_captions`. Revised-first
(`_walk_captions.resolve_caption`) means a master write reaches nothing downstream once a revised
caption exists — see the gotcha in `docs/multiview_audit.md`.

## Export

**Export is the only thing that writes outside the workspace.** Six artifact kinds
(`image`/`caption`/`variants`/`mask`/`master`/`index`), each decided against the destination
(`identical` by byte compare for text, `(size, mtime_ns)` for pixels). It always copies, takes no
`--from_report`, and `revert_export` restores text it overwrote — an overwritten pixel reports
`not-undoable`. `--combine_ocr` (GUI: the "Combine OCR" drawer, `--ocr_dir` inside it) is the one
knob that makes a row a *render* rather than a copy: a `caption` / `variants` row whose image has a
`{stem}.ocr.txt` publishes the text with the OCR clauses attached (the lines held to `--ocr_min_det`
and `--ocr_min_glyph`, both recorded on the row), carries `ocr` + `text` in the
report (`text` re-derived at decide time — a sidecar deleted since the plan publishes the caption
bare; revert compares against the recorded text), and counts under `stats.combined`. Exporting
again without the knob takes the clause back. The trainer must `make preprocess-te` after either.

## Reports, replay, shared scaffolding

Stages are dry-run by default **from the CLI** and write `report.json`; `--apply` writes for real.
The GUI always passes it. `tests/test_registry_requests.py` pins one spelling per shared flag,
since the GUI fills `--path_pattern` / `--tagger_dir` / `--checkpoint` / `--prompt_embed` from one
Settings value each.

- `cli/_args.py::make_progress` (the `  [done/total] detail` line the GUI's progress bar parses;
  under the trainer's daemon the same callback also streams every call to the job's
  `progress.jsonl` through `_progress.py`), `cli/_report.py`, `replay.run_replay_cli` (reads
  `from_report` / `path_pattern` / `apply` off the request).
- `_caption_io.py` — `read_caption`/`write_caption`, the trailing-newline invariant, the
  `.variants.txt` drop, and `history_by`.
- `_walk_captions.py` — `resolve_caption`/`iter_captions`: revised caption first, master as
  read-only fallback. `autotag` walks images rather than captions, so it calls `resolve_caption`
  itself; `cli/ab_position_captions` deliberately reads the master only.
- `replay.apply_one` — the one drift-guarded write: `no-proposal` → `missing-caption` →
  `already-applied` → `drifted` → `would-write`/`written`. The GUI's Undo is this call with the
  two texts swapped (`gui/proposals.py`); the replay shapes are `contract.REPLAY_SHAPES`, bound by
  the three stage CLIs as `REPLAY_SPEC`.

Every `--apply` that touches captions must be followed by the trainer's TE re-encode.

## Tests

`test_registry_requests` and `test_stage_requests` (the surface), `test_stage_replay`,
`test_resize_images` (bucket numbers and the `anima_resize_*` PNG keys), `test_autotag_captions`,
`test_position_captions` (the promotion path included), `test_correct_captions`,
`test_multiview_apply` (the gate and `_run_audit_phase`), `test_ocr`,
`test_export`.
