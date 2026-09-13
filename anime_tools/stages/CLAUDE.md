# anime_tools/stages/

Caption-master stages and their thin CLIs: `resize.py`, `autotag.py` (modes
`missing`/`merge`/`overwrite`; only `missing` is non-destructive), `position_captions.py` (SAM3
instances → reading order → mask-blanked crops → tagger → clause rewrite; see
`docs/position_captions.md`), `captions.py` (correction + mirror), `drop_groups.py` (whole tag
groups cut from the revised caption, nothing else moved), `multiview_audit.py`
(`docs/multiview_audit.md`), `ocr.py` (one path: the AnimeText
text-block detector, detect-only, every box read by the manga VL reader through
`anime_tools.ocr.reread.RereadEngine`, plus the optional `--mask_dir` components as `0.000`-score
lines; the detector and the VL reader share the one device `run_ocr` resolves),
`export_workspace.py`. Adding one is the `add-stage`
skill; the caption grammar these stages write is the `captions` skill.

## Requests and runners

The surface is a request object per stage (`requests.py`, torch-free): `ResizeRequest`,
`AutotagRequest`, `PositionRequest`, `CorrectRequest`, `DropGroupRequest`, `OcrRequest`,
`AuditRequest`, `ExportRequest`, run by `run_<stage>(req)` **in the stage's own module** —
the old CLI main minus
the parsing (preflight, model load, the library call, `report.json`, the printed epilogue).
Same base as
masking's (`anime_tools/_request.py`), with two differences: flags are spelled with underscores
(`FLAG_SEP = "_"`), and a `store_false` switch names its one flag in `off` metadata (`skip_en` is
`--keep_en`). The parser is generated from the class (`Request.parser()` →
`_request.build_parser`): every field is declared through `arg(default, help=…, group=…, gate=…,
choices=…)`, the class docstring is the `--help` description, and every flag with a separator
takes the other spelling as an alias (`--path_pattern` / `--path-pattern`). The CLIs in `cli/` are
one-line shells (`build_parser()` = `Request.parser()`, `main()` = `run_<stage>(from_argv())`).

A runner lives beside the library function it drives, because `registry.py` addresses it as
`module:function` and resolves it lazily — nothing has to import eight stages to name one:

| Runner | Module | Also there |
|---|---|---|
| `run_resize` | `resize.py` | `resized_tree()` + `RESIZE_FIRST`, the "run Resize first" preflight every other stage borrows |
| `run_autotag` | `autotag.py` | `run_autotag_captions`, whose `tag_batch` is one forward per `--batch_size` images |
| `run_position` | `position_captions.py` | `summarize()` (the report's own counts), `_run_flatten` |
| `run_correct` | `captions.py` | `resolve_tag_csv`, the KB lookup both text stages share |
| `run_drop_groups` | `drop_groups.py` | `drop_tag_groups`; the cut is `correction.drop_caption_groups`, Correct's drop minus its reorder |
| `run_audit` | `multiview_audit.py` | `run_audit_phase` (the same sweep as the position stage's phase 1) |
| `run_ocr` | `ocr.py` | `_vl_engine`; the library walk is `read_tree` |
| `run_export` | `export_workspace.py` | the library call is `publish` |

`anime_tools/stages/__init__.py` re-exports all eight lazily (`_RUNNERS` is the same mapping), so
`from anime_tools.stages import run_autotag` still costs one module. `_report.py` (the `report.json`
header, the dry-run footer) and `_progress.py` (`make_progress`) are the two things every runner
shares; they were under `cli/` until the library depended on its own CLI package.

Every default and help string a request field names comes from a leaf — `_options.py` here
(`PositionCaptionOptions`, the resize geometry, the autotag batch size, the audit's verdicts and
witness floors) and
`masking/_prompts.py` for the SAM3 prompt vocabulary — never from the stage module that uses it,
which re-exports it instead. `registry.py` is import-light on purpose and the GUI resolves every
request class to build its form, so reaching into a stage for one number used to cost that build
numpy, PIL and yaml (360 modules); `tests/test_registry_requests.py` measures it and pins the
re-exports as the same objects.

The SAM3 detection flags are one nested `DetectionRequest` (`GROUP = "detection"`) and the audit's
verdict gate one nested `MultiviewRequest` (`GROUP = "multiview audit"`), both shared by
`PositionRequest` and `AuditRequest`; `.options()` on either builds the `PositionCaptionOptions`
field by field, so an option field with no request field is an error, not a silent default. The
audit pins `min_instances=2` (`MultiviewRequest.MIN_INSTANCES`, applied by the shared
`requests.audit_options`) rather than exposing it. Validation lives in `__post_init__`
(autotag mode, `--flatten` vs `--from_report`, the randomize tokenizers, resize tiers); a missing
input tree is a `FileNotFoundError` the shell turns into `SystemExit`. `__init__.py` exposes all
nineteen names lazily; `tests/test_registry_requests.py` round-trips every registered stage's
request through its parser and imports the request half torch-poisoned;
`tests/test_stage_requests.py` keeps the stage-specific pins.

`registry.py` is the stage list — `Stage(id, title, request="module:Class",
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
position stage's first phase, over the complement population — `is_audit_target` is defined as
exactly what `is_candidate` rejects as `single-subject`. `multiview_audit.run_audit_phase` reuses
the
already-resident SAM3 + tagger, detects under `req.audit_options()`, and in `apply` mode hands
`multiview_audit.promotions()` to `run_position_captions(promoted=…)`, which substitutes the
promoted caption before `is_candidate` sees it.

The audit runs before the sweep: `multiple views` is what moves an image out of the
`single-subject` rejection and what `is_repeated_subject_layout` reads to arm the `view_invariant`
gate. `admitted()` is the one verdict/confidence gate the write path and the promotion path share.

The dock has no audit button: `registry.py` marks the stage `hidden`, and the GUI reaches it as
this phase through the position form's `multiview_audit`. The standalone CLI, its request and
its replay stay; only the button went.

Both phases also leave a per-image record under the report directory, for the GUI's analysis
badge: `run_position_captions(analysis_dir=<report_dir>/analysis)` and
`run_multiview_audit(analysis_dir=<report_dir>/audit/analysis)` write `<stem>.json` (the row, plus
`labels` — which of its lists the mask indexes) and `<stem>.png` (8-bit instance labels, `i + 1`
where instance `i` is) through `_analysis.py`, from the detections `propose_for_image` /
`audit_image` hand their `mask_sink`. An image the next run walks without a row loses its pair, so
the files are always the latest word on the image.

Both write the revised tree, like every other caption stage — the phase through
`run_position_captions`, the standalone stage through `apply_findings`. No stage writes the
hand-written master (Export and the GUI caption editor are its only writers), and revised-first
(`_walk_captions.resolve_caption`) means a master write would be read past anyway once a revised
caption exists.

## Export

Export is the only thing that writes outside the workspace. Six artifact kinds
(`image`/`caption`/`variants`/`mask`/`master`/`index`), each decided against the destination
(`identical` by byte compare for text, `(size, mtime_ns)` for pixels). It always copies, takes no
`--from_report`, and `revert_export` restores text it overwrote — an overwritten pixel reports
`not-undoable`.

The `caption` row reads the ladder, not one file (`_caption_source`): the revised caption, else the
master — overlay (`workspace/master/`) first, hand-written (`--src`) behind it, the same
overlay-first rule `gui.dataset.caption_paths` and `resolve_caption` read. Nothing copies a master
into the resized tree, so an image no caption stage has touched would otherwise publish
captionless, which the trainer reads as unlabelled. The `variants` sidecar stays a revised-tree
artifact and is looked up there whichever rung the caption came from.

It also takes no `--path_pattern` — `ExportRequest` is the one stage request that is not a
`DatasetRequest`, and `plan_export` walks the whole resized tree. A publish is the workspace or
it is a partial dataset landing beside a stale tree the trainer reads as all of it. That is also
what makes it the one stage the GUI cannot narrow to the open image: `scoped` is
"has `path_pattern`", so Export shows a single Run and `POST /api/jobs` refuses a `rel` for it.

`--combine_ocr` (GUI: the "Combine OCR" drawer, `--ocr_dir` inside it) is the one
knob that makes a row a render rather than a copy: a `caption` / `variants` row whose image has a
`{stem}.ocr.txt` publishes the text with the OCR clauses attached (the lines held to `--ocr_min_det`
and `--ocr_min_glyph`, both recorded on the row), carries `ocr` + `text` in the
report (`text` re-derived at decide time — a sidecar deleted since the plan publishes the caption
bare; revert compares against the recorded text), and counts under `stats.combined`. Exporting
again without the knob takes the clause back. The trainer must `make preprocess-te` after either.

`--excluded_dir` adds a second, smaller plan over `workspace/_excluded/`
(`anime_tools/exclude/`): `_excluded/resized` walked as the live resized tree is,
`_excluded/masks` looked up against it by the same rule, published under `<out>/_excluded/`.
Same kinds, so the compare and the revert are unchanged; no `master` row and no OCR combine.
The rows carry `excluded`, which is all that tells them apart in the report, and a workspace
where nothing is excluded publishes nothing extra. `resize` reads the same tree's ledger into
its `--skip` — that is where an exclusion is enforced, since every other stage walks
`workspace/resized/` and an excluded image has left it.

## Reports, replay, shared scaffolding

Stages are dry-run by default from the CLI and write `report.json`; `--apply` writes for real.
The GUI always passes it. `tests/test_registry_requests.py` pins one spelling per shared flag,
since the GUI fills `--path_pattern` / `--tagger_dir` / `--checkpoint` / `--prompt_embed` from one
Settings value each.

- `_progress.py::make_progress` (the `  [done/total] detail` line the GUI's progress bar parses;
  under the trainer's daemon the same callback also streams every call to the job's
  `progress.jsonl` through `anime_tools/_progress.py`), `_report.py`, `replay.run_replay_cli` (reads
  `from_report` / `path_pattern` / `apply` off the request).
- `_caption_io.py` — `read_caption`/`write_caption`, the trailing-newline invariant, the
  `.variants.txt` drop, and `history_by`.
- `_walk_captions.py` — `resolve_caption`/`iter_captions`: revised caption first, master as
  read-only fallback. `autotag` walks images rather than captions, so it calls `resolve_caption`
  itself; `cli/ab_position_captions` deliberately reads the master only.
- `replay.apply_one` — the one drift-guarded write: `no-proposal` → `missing-caption` →
  `already-applied` → `drifted` → `would-write`/`written`. Its baseline is what the *write
  target* held, which is why every row records `target_before` beside the caption that spoke for
  the image (`existing` / `original` / `caption`): those differ exactly when the stage read the
  master, and an empty `target_before` is the row whose write CREATES the revised caption.
  `replay.undo_one` is the inverse — `apply_one` with the two texts swapped, except for that row,
  where the inverse of a create is a delete (the caption and its sidecars go, leaving the master
  as the ladder had it). The GUI's Undo (`gui/proposals.py`) and the audit's `revert_curated` are
  both that call; the replay shapes are `contract.REPLAY_SHAPES`, bound by the five stage CLIs as
  `REPLAY_SPEC`.
  `correct` carries a shape and no `--from_report` either: the pass is pure text, so re-running
  it is cheaper than the machinery to skip it, but its report is still what the GUI's Undo
  replays backwards.

Every `--apply` that touches captions must be followed by the trainer's TE re-encode.

## Tests

`test_registry_requests` and `test_stage_requests` (the surface), `test_stage_replay`,
`test_resize_images` (bucket numbers and the `anima_resize_*` PNG keys), `test_autotag_captions`,
`test_position_captions` (the promotion path included), `test_correct_captions`,
`test_multiview_apply` (the gate and `_run_audit_phase`), `test_ocr`,
`test_export`.
