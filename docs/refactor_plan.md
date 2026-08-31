# Dedup / refactor plan

A full-codebase duplication review (2026-08-31) found that the shared cores this
package already has — `stages/replay.py`, `captions/taxonomy.py`, `_walk.py`,
`_env.py`, `tagger/cli/eval_metrics.py` — are declared but not consistently
used: sibling files re-implement them by hand, and in several places that
duplication had already drifted into live bugs. The bugs were fixed first
(path traversal in `/api/files`/`/api/ls`, hand `split(",")` in the caption
index and grouping tag reader, the people-count regex drift, the audit stage's
missing `mask_containment_threshold`, grouping's stem-collision keying, the
GUI cold-start dead-end, underscore-blind clause matching,
`tag_groups.from_dict` validation drift). This plan is the structural follow-up
that makes those classes of drift impossible rather than merely fixed.

Phases are ordered so each one is independently shippable, guarded by existing
tests, and earlier phases create the homes later phases move into. Within a
phase the work is one PR-sized change.

**Out of scope, deliberately:**

- `gui/proposals.SHAPES` stays a hand-copy of the CLI `ReplaySpec`s (pinned by
  `tests/test_gui_proposals.py`) — importing the CLI modules would pull torch
  into the GUI. Phase 6 removes only the *unpinned* copies that rode along.
- `buckets.py` stays a copy of the trainer's free-fit half (pinned by
  `tests/test_resize_images.py`); the dependency direction forbids the import.
- Anything in `docs/contract.md`: no file-format, report-shape, or caption
  grammar changes anywhere in this plan. Refactors move code, not bytes.
- `bench/` cleanups ride along only where a phase touches the same seam
  (noted inline); nothing here blocks on bench.

## Phase 1 — stage CLI scaffolding (`stages/cli/`) — **done**

The largest pool: ~300+ duplicated lines across four parsers, with real teeth
because `gui/stages.py` introspects `build_parser()` — a dropped `dest=` or a
drifted default silently changes the GUI form.

Landed as `_args.py` / `_detection.py` / `_models.py` / `_report.py` plus
`replay.run_replay_cli` and `position_captions.options_from_flag_string`; net
−330 lines across the seven CLIs. The one-off before/after schema snapshot is
now a standing test, `tests/test_stage_cli_args.py` — including a field-by-field
check that every `PositionCaptionOptions` field still moves when its flag does,
which is the failure the audit's drifted copy actually shipped.

1. **`stages/cli/_args.py`** — `add_dataset_args(p)` (`--src`, `--dst`,
   `--path_pattern`), `add_apply_args(p)` (`--apply`, `--from_report`,
   `--report_dir`), `add_model_args(p)` (`--tagger_dir`, `--device`). The
   four parsers (`autotag_captions`, `position_captions`, `audit_multiview`,
   `resize_images`) currently re-type these with byte-identical help strings
   and the dual `--foo/--foo-bar` + `dest=` spelling. Actions, dests, defaults
   and help must come out identical, so the GUI schema dump is unchanged —
   assert that by snapshotting the schema JSON for each stage before/after.
2. **`stages/cli/_detection.py`** — `add_detection_args(p)`: the 15-flag
   detection group declared in both `position_captions.py` and
   `audit_multiview.py` (already drifted: the audit copy lost the long-form
   aliases and most help text). Lives beside `build_detect_fn`, which the
   audit already imports — the two CLIs already share the namespace contract,
   they should share its declaration. Restoring the aliases to the audit is a
   deliberate (tiny, additive) CLI change.
3. **Split `build_options_from_args`** (`position_captions.py:433`) into
   `detection_options(args, **overrides)` + the clause half; the audit calls
   `detection_options(args, min_instances=2)` instead of its inline rebuild.
   (The missing-field bug is already fixed; this removes the copy itself.)
4. **`options_from_flag_string(flags)`** in `position_captions.py`: the
   verbatim `sys.argv`-swap helper currently duplicated in
   `ab_position_captions.py:106` and `review_position_captions.py:124`.
5. **`replay.run_replay_cli(args, spec, src, dst, report_dir, *,
   after_write_note)`** — the `--from_report` wrapper (`try run_replay /
   except StaleReportError → SystemExit`, three prints, stage-specific
   epilogue) copied in three CLIs. The audit also moves its inline
   `ReplaySpec` to a module-level `REPLAY_SPEC` like the other two.
6. **`stages/cli/_report.py`** — `write_stage_report(report_dir, payload)`
   (mkdir + `json.dumps(indent=2, ensure_ascii=False)` + write) and
   `print_dry_run_footer(apply, note)` (the literally-repeated "Dry run — no
   captions written…" string), used by all five report writers. Share the
   `{"src","dst","path_pattern","applied"}` header construction with
   `replay.build_replay_report` — they are the keys `validate_report` checks.
7. **Small shared bits**: `_models.py::load_tagger(args)` → `(tagger,
   vocabulary, ckpt_dir)` (four identical copies); export `BOX_COLORS` from
   `stages/multiview_sheet.py` (three copies of the same 8-tuple; one already
   claims to be "shared"); one `progress` closure; the `np.bool` shim moves to
   the masking SAM3 module in Phase 3, stage CLIs import it from there.

Guards: full pytest; the before/after GUI schema snapshot; `--help` diff for
each CLI should show only the audit's restored aliases.

Outcome: the GUI schema is byte-identical for every stage but `audit`, whose
only changes are `help` text and five restored `--foo-bar` aliases — every
`dest`, default, kind and the field order are unchanged, so no saved form value
and no stored command line moved. Two epilogue strings were unified onto the
canonical wording (autotag printed `report → …` and a lowercase dry-run line);
`audit_multiview`'s `ReplaySpec` is now module-level `REPLAY_SPEC` with only its
verdict/confidence gate closed over at replay time, so `test_gui_proposals`
pins it by object comparison like the other two instead of scraping the source.

## Phase 2 — caption write + apply semantics (`stages/`) — **done**

The byte-exactness contract ("`audit_multiview` writes a trailing newline; the
other two do not") was enforced by comments across four files, and the
drift-guarded apply loop (read target → compare recorded before →
`already-applied` / `drifted` / write) was written four times.

Landed as `stages/_caption_io.py` (`read_caption` / `write_caption`),
`stages/_walk_captions.py` (`resolve_caption` / `iter_captions`) and
`replay.apply_one`. `replay_rows`, `apply_findings`, `apply_curated` and
`revert_curated` are now loops over `apply_one`; the audit's walk is the shared
`iter_captions` instead of a hand-rolled copy under a comment claiming it
matched. New standing test: `tests/test_multiview_apply.py`.

Deviation from the item-3 sketch below: **`autotag` and the two review CLIs did
not adopt `iter_captions`**, and deliberately so — the derived-first rule is not
their rule. `autotag` reads (and writes) the master; `ab_position_captions`
reads the master on purpose, since after one `--apply` the derived side already
carries clauses and every rule in it would skip; `review_position_captions`
reads master and derived as the two *sides* of its diff. They adopted
`read_caption`/`write_caption` only, and `_walk_captions`'s docstring records
why the walk stops there.

1. **`stages/_caption_io.py::write_caption(path, text, *, newline=False,
   drop_variants=False)`** — the mkdir/write/unlink-variants triplet from
   `replay._write_caption`, `position_captions._write_derived_caption`,
   `multiview_audit.apply_findings`, `autotag.run_autotag_captions`. The
   newline invariant becomes an argument in one function instead of a comment
   in four files.
2. **`replay.apply_one(target, before, after, *, apply, newline,
   drop_variants) -> status`** — the single drift-guarded write primitive.
   `replay.replay_rows` becomes a loop over it; `multiview_audit`'s
   `apply_findings` / `apply_curated` / `revert_curated` become loops over it
   (and gain the full status vocabulary — today `apply_findings` silently
   `continue`s, so callers can't say why a row was skipped; surfacing those
   statuses in the report is a strict information gain, keys unchanged).
3. **Promote `position_captions._iter_captions`** to
   `stages/_walk_captions.py::iter_captions(resized_dir, source_dir, pattern,
   stats, progress)` — its docstring already says "shared so they can't
   disagree", but `multiview_audit` hand-rolls the same resolution while
   claiming "same caption resolution as run_position_captions". `autotag` and
   the two review CLIs adopt the same helper (a bare `resolve_caption(src,
   dst, rel)` covers the CLI read-only cases).

Guards: the replay tests already pin byte-exactness; add one test asserting
`apply_findings` reports the drifted/already-applied statuses.

Outcome: `apply_findings` returns `(written, skipped)` where `skipped` is a
`Counter` over `apply_one`'s statuses, surfaced as the audit report's new
`summary.apply_skipped` — every existing key is unchanged, and a gated row that
was refused because the master drifted is no longer indistinguishable from one
the verdict/confidence gate rejected. One behaviour change rode along, in
`revert_curated`: reverting *to* an empty before-text now reports `no-proposal`
instead of blanking the caption file.

## Phase 3 — give `masking/` a package core — **done**

`masking/` had zero non-CLI modules, which was the root cause of ~85 lines of
copied skeleton between `generate_masks.py` and `generate_masks_mit.py`, four
copies of SAM3 construction, and nine copies of the `np.bool` shim.

Landed as `masking/_sam3.py` (`load_sam3` / `make_processor`, and the shim as
that module's import side effect) and `masking/_masks.py` (the flag blocks, the
plan, the write tail, the read side); net −157 lines across the five masking
CLIs, the four stage CLIs and the seven bench scripts. New standing test:
`tests/test_masking_plan.py`.

1. **`masking/_sam3.py::load_sam3(checkpoint, device, *,
   confidence_threshold=None, disable_act_ckpt=False)`** — one constructor
   (`build_kwargs` → `build_sam3_image_model` → `Sam3Processor`), with the
   `np.bool` compat shim as an import side effect at this single SAM3 entry
   point. Adopted by `generate_masks`, `probe_sam_masks`,
   `stages/cli/position_captions.build_detect_fn`, and
   `bench/sam3_soft_prompt/common.py`.
2. **`masking/_masks.py`** — `add_mask_io_args(parser)` (the word-identical
   `--image-dir/--mask-dir/--force/--device/--workers/--recursive/
   --path-pattern` block), `plan_mask_jobs(image_dir, mask_dir, *, recursive,
   pattern, force)` (walk → `{stem}_mask.png` mirror → force/skip → mkdir),
   `save_mask` / `write_ignore_mask` (the invert-and-save tail, one home for
   the `detected=1 → alpha=0` comment), `mask_path_for(image_path, image_dir,
   mask_dir)` + `iter_masks(mask_dir)` for the read side (`merge_masks.py`,
   `gui/dataset.py:207` — the GUI's flat-layout fallback stays, documented).
3. **Delete the `mask_fill` copies** in `probe_nms_pairs.py:84` and
   `probe_sam_masks.py:186`; import `mask_box_fill` from
   `stages/instance_detection` (the copies clamp to a different bound —
   the canonical one wins).

Guards: masking has thinner test coverage; add a `plan_mask_jobs` unit test
(mirroring + force semantics) as part of the move.

Outcome: the two generators' argparse schemas are byte-identical (checked
against `HEAD` through `gui.stages.dump_schemas`, for every stage), which is why
`_masks.py` exposes five small contiguous flag blocks rather than one
`add_mask_io_args` — the two parsers interleave their own flags between them and
declaration order is what the GUI form follows. Deviations from the sketch:
`write_ignore_mask` is a thin wrapper over a `write_mask(path, keep)` primitive,
because the SAM3 generator's `focus_prompts` path writes the *keep* mask
directly and only the inversion is worth naming; and `gui/dataset.mask_path`
adopted `mask_name` alone rather than `mask_path_for`, since it starts from a
dataset-relative path rather than an image under an `--image-dir`. The
`np.bool` shim went from nine copies to one: only `_sam3.py` still spells it,
and the seven bench scripts get it by importing `bench/sam3_soft_prompt/
common.py`, whose own `load_sam3` is now the shared loader plus the two things
only a trainer wants (frozen parameters, activation checkpointing off).
`probe_sam_masks`'s inline box-fill became a `Detection` handed to
`mask_box_fill`, so the number it prints is the one `dedupe_detections` ranks on
— it clamps to the mask's shape rather than the image's, which is the point when
the probe is reporting a misaligned mask.

## Phase 4 — one tag-shape vocabulary (`captions/taxonomy.py`) — **done**

`taxonomy.py` was already the declared torch-free home of tag-shape primitives.
The strays moved in: `normalize_tag`, `count_of` (+ its `exact_count` half),
`SINGLE_COUNT_NAMES` / `is_solo_names` / `solo_multi_indices`. Every count
regex outside `taxonomy.py` is gone — `tests/test_tag_taxonomy.py` now asserts
that by scanning the package for the pattern, in place of the test that used to
pin the four copies byte-for-byte.

1. **`normalize_tag`** — today three behaviours: `correction.py:226`
   (canonical: strip/underscore-fold/lower/collapse), `grouping/features.py:45`
   (equivalent), and the weak `strip().lower()` family in
   `position_clauses.tag_keys` / `flatten_caption` / `clause_rewrite`.
   Canonical version lands in `taxonomy`; `correction` and `features` import
   it. The `position_clauses` comparison keys were fixed for the
   underscore-blindness bug; finishing the consolidation means their key
   functions call the same normalizer (comparison keys only — written output
   is untouched).
2. **Solo / people-count predicate** — `SINGLE_COUNT_NAMES`,
   `is_solo_names(names)`, `solo_multi_indices(vocab_tags)` in `taxonomy`,
   replacing the four copies in `tagger/cli/derive_groups.py`,
   `tagger/cli/role_markers.py`, `tagger/tagger.py`, and wiring
   `captions/group_router.py` to the same constants. (The regex drift is
   already fixed; this removes the copies so it can't recur. tagger → captions
   is the allowed import direction.)
3. **`count_of(tags, gender) -> int | None`** — one count parser replacing
   `caption_layout.py`'s four regexes and its three near-identical count
   bodies (`caption_subject_count`, `caption_boy_count`, the inline girls copy
   in `caption_panel_ceiling`). Also reconciles the `multiple_girls`
   space/underscore disagreement with `_COUNT_RE` — via `normalize_tag`, both
   forms parse the same.
4. **Delete `shuffle._is_artist_tag`** (verbatim leftover of
   `taxonomy.is_artist_tag` from the trainer split); replace the three
   hand-typed `("On the ", "In the ")` header literals in `shuffle.py` /
   `variants.py` with `position_clauses.CLAUSE_PREFIXES` / `is_clause_header`.

Guards: caption tests are the densest in the suite; run with the captions
skill's grammar rules in mind — no written-caption bytes may change.

Outcome: no rendering changed, and two comparisons got stricter in the way the
consolidation was for. `position_clauses.tag_keys` and `flatten_caption` now key
on `normalize_tag` like `clause_rewrite._cmp_key` always did, so a tag the
rewrite moved out in space form no longer comes back into the bag beside its own
underscore spelling (pinned by
`test_flatten_folds_the_underscore_spelling_of_a_tag_it_returns`); and
`caption_boy_count` finally sees `multiple_boys`, which the hand-typed
`"multiple boys" in tags` had been reading as "no boys" — the upper bound of the
count gate now drops for that caption, as it always did for the space form.
`multiview_audit`'s two `GIRLS_COUNT_RE` uses came along: the count reads through
`count_of` and the splice through `exact_count`. Left alone deliberately:
`tagger.py`'s `_GIRLS_COUNT_RE`, which is a different rule (it takes 6 out of
`6+girls` as a character cap, where `count_of` reads that as "unknown"), and the
underscore members of `caption_layout`'s `LAYOUT_TAGS`, now unreachable through
`flat_tag_set` but still correct for a direct caller holding raw tags.

## Phase 5 — captions/grouping/tagger structural strays — **done**

Grab-bag of medium items, each independent. Landed as `captions/vocab_io.py`,
`tag_groups.resolved_from_dict`, `tagger/data.TaggerCheckpoint`, a rewritten
`tagger/feature_cache.py`, `grouping/cli/_decensored.py` and `_json.py`; new
standing tests `tests/test_vocab_io.py`, `tests/test_tagger_checkpoint.py`,
`tests/test_json_io.py` plus four walk-parity cases in
`tests/test_grouping_features.py`.

1. **`captions/vocab_io.py`** (torch-free): `load_vocab(path)`,
   `names_by_category(vocab)`, and `resolved_from_dict(...)` — the missing
   inverse of `tag_groups.resolved_to_dict` — replacing the four hand parsers
   of `vocab.json` (`index.py:121`, `clause_vocabulary.py:371`,
   `tagger/tagger.py:379`, `group_router.py:114`). `group_router.from_vocab`
   then only builds tensors. Fold the `load_rules`/`from_dict` twin in
   `tag_rules.py` the same way `tag_groups` was folded in the bug fix.
2. **Grouping walks through `_walk.py`** — `features.iter_images` delegates to
   the shared glob (one `IMAGE_EXTS`, reconciling the `.bmp`/`.jxl`/`.avif`
   disagreement across `_walk.py` / `features.py` / `tagger/cli/constants.py`);
   `gather_members`' second walk collapses onto it. (Rel-path keying already
   landed with the stem-collision fix.)
3. **`resolve_path` at the grouping/index CLI boundary** —
   `grouping/cli/build_groups.py` and `captions/index.py` anchor their
   relative defaults like every stage CLI does, instead of silently writing to
   `$CWD`. The `match_decensored`/`apply_decensored` pair's shared
   `SINCOS_DIR`/`DECEN_DIR`/`OUT_DIR` triple moves to one module.
4. **`tagger/data.py::TaggerCheckpoint.from_dir(path, *, require=…)`** — the
   config/vocab/dataset read + existence check + "run build_vocab first" exit
   + `idx_to_name` rebuild repeated across five tagger CLIs (and
   `bench/tagger_external/calibration_check.py`).
5. **`tagger/feature_cache.py`** grows `dbv4_cache_path(arch)`,
   `load_dbv4_cache(path, stems)`, `multi_hot_from_manifest(...)`,
   `DEFAULT_SWEEP` — the module already says "change the layout here,
   propagate everywhere", yet the path template is hardcoded in
   `train_sidecar.py` and `calibration_check.py`. Bench ride-along: delete
   `probe_position_rescore.load_external`/preprocess copies in favour of
   `Dbv4Backend` + `preprocess_dbv4`.
6. **`_device.resolve_device` adoption** — replace the ~9 ad-hoc
   `cuda if torch.cuda.is_available()` sites (all of `tagger/`,
   `grouping/embedder.py`, `vision/pe.py`); none of them get the module's
   broken-driver fallback today. Mechanical.
7. **`_json.py::write_json_report / read_json_report`** — one JSON I/O pair
   for the ~8 write and ~6 read sites that currently disagree on `indent`,
   `ensure_ascii` (one site mojibakes Korean paths), and `mkdir`. Keep
   `indent=2, ensure_ascii=False` as the canonical shape; `index.py`'s
   `indent=1` changes byte shape of `caption_index.json` — acceptable (not a
   contract file), but called out.
8. **`grouping/features.cached_descriptors(paths, compute, cache_path,
   version)`** — unify `match_decensored.build`'s `.npz` cache lifecycle with
   `embed_members`'s; `match_decensored` also stops missing non-`.webp`
   sources by using the shared walk. Lowest urgency in this phase.


Outcome: no caption byte, report key or contract format changed. Four things
moved that are worth knowing about.

- **Two comparisons got wider, on purpose.** `grouping/features.IMAGE_EXTS` is
  now derived from `_walk.IMAGE_EXTENSIONS`, so grouping sees `.bmp` (and stops
  claiming `.jxl`/`.avif` when no Pillow plugin can decode them) — the same
  pool the stages and the GUI see. `tagger/cli/constants.IMAGE_EXTS` keeps its
  preference order for `find_image_for_caption` but appends the walker's
  remainder, so a caption whose only sibling is a `.bmp` is no longer dropped
  from the vocab build. `match_decensored` walks through the same glob instead
  of filtering to `.webp`.
- **Four JSON files change byte shape** (not content): `caption_index.json`,
  the A/B and review report artifacts and the mask probe's `probe.json` were
  written at `indent=1` and are now at the canonical `indent=2`. Eight
  checkpoint reads that used a bare `open(path)` — the platform locale
  codepage, which is not UTF-8 on Windows — now read UTF-8 explicitly, and
  `groups.json` + the sidecar meta stop escaping non-ASCII paths to `\uXXXX`.
- **`feature_cache.py` lost its dead half.** The `feature_cache_root` /
  `cache_dir_for` helpers addressed the PE token caches, which were reclaimed
  with that trainer in 2026-08; nothing imported them. The module is now the
  dbv4 hidden-state cache it claimed to own.
- **Item 8 was not done as written, deliberately.** A shared
  `cached_descriptors(paths, compute, cache_path, version)` would have had one
  caller: `embed_members`' cache is per-image, content-addressed, unsigned and
  written from a thread pool mid-stream, while `match_decensored`'s is one
  whole-directory blob stamped `(newest mtime, count, version)`. They share the
  words "npz cache" and nothing else, so unifying them would have invented an
  abstraction rather than removed a copy. What was real in that item — the
  `.webp`-only walk — is fixed, and `build`'s cache read now leaves its handle
  closed and treats a corrupt cache as "recompute" instead of raising, which is
  the stance the rest of the package's caches take.

## Phase 6 — GUI backend + frontend — **done**

Landed as `gui/proposals.py` on the replay primitives, five pure helpers hoisted
out of `create_app` plus two app-wide exception handlers, and a frontend split
into `components/SettingsDialog.tsx`, `components/StatusLine.tsx`,
`components/ClauseRow.tsx` and `state.ts` (`createJobFollower` / `persisted`).
`App.tsx` 1236 → 855 lines; three new GUI tests in `tests/test_gui.py` and the
`SHAPES` pin collapsed to one whole-object comparison.

1. **`gui/proposals.py` imports the replay primitives it copied** —
   `load_report`, `report_rows` (wrapping `StaleReportError` in
   `ProposalError`), and the undo drift-ladder aligned with
   `replay.apply_one` from Phase 2. Verified: `stages/replay.py` imports
   torch-free, so `test_boundary`'s no-torch assertion holds. Only `SHAPES`
   remains a (pinned) hand-copy.
2. **`gui/server.py`** — register `@app.exception_handler` for
   `DatasetError` / `ProposalError` (the two 404 sites raise explicitly),
   replacing nine copies of `except → HTTPException`; a `_proposals(job_id)`
   helper for the duplicated report-read; load settings **once per request**
   and pass down (today one `POST /api/jobs` re-reads the settings file ~7×
   with no consistency guarantee). Structural: hoist the pure helpers
   (`_roots`/`_stage_defaults`/`_root_paths`/`_make_output_dirs`/
   `_preprocess_steps`) out of the 463-line `create_app` closure; routers are
   optional polish.
3. **`frontend/src/App.tsx` diet** (1138 → ~700 lines): move
   `SettingsDialog` + `ModelRow` to component files (with one `<SettingRow>`
   for the repeated label/input/hint row and one collect-diff-send helper for
   the OK handler); one `createJobFollower({onLine, onDone})` replacing the
   twin `attach`/`attachDownload` SSE followers and their mirrored signal
   trios; one `lastRun()` freshness memo replacing the three flipped-polarity
   derivations that gate Apply; one `startJob()` + `toStatus(e)` in `api.ts`
   replacing the four copies of catch→status; a `persisted(key, init)` helper
   for the five localStorage signal pairs.
4. **Shared fragments + types** — `<StatusLine>`, `<ClauseRow>` (CaptionCard/
   CaptionDiff markup), a `JobStatus` type in `types.ts` replacing four inline
   `{ text; state? }` declarations; pass the resolved `Stage` object to
   `StagePanel` instead of `stages`+`curId` (kills five `find(s => s.id …)`
   lookups and halves StagePanel's prop surface).

Guards: `make frontend` must be rerun and the committed bundle diffed in the
same change (CI fails on drift); GUI pytest suite; manual smoke via `make gui`
for the run→diff→apply loop.


Outcome: no status string, response key or HTTP status changed. Three things
are worth knowing.

- **`SHAPES` is now `dict[str, ReplaySpec]`.** The out-of-scope note above was
  half right: the three *instances* must stay hand-copied, because importing a
  stage CLI pulls torch into the server, but the `ReplaySpec` **class** —
  along with `load_report`, `report_rows` and `apply_one` — comes from
  `stages/replay.py`, which is torch-free by construction and always was
  importable here. `test_gui_proposals` went from seven field comparisons to
  one equality, so a field added to `ReplaySpec` later cannot slip past the
  mirror.
- **Two adoption behaviours got slightly better, deliberately.** The server
  adopts a `DatasetError` raised anywhere in a route as a 400 rather than a
  500; the frontend's "is something already running?" guard tests `running`
  instead of a job id, and an id outlives its job, so a second
  externally-started job can now be adopted where before only the first could.
- **The dock height stays hand-rolled.** `persisted` writes on every change,
  and the height moves on every pointermove of a drag; it saves once on
  pointerup instead. Item 6.3's "five localStorage signal pairs" is therefore
  four.

## Sizing and order rationale

Phases 1–2 first: they carry the highest drift risk (GUI schema coupling, the
byte-exactness contract) and the densest existing test cover. Phase 3 unblocks
nothing but is self-contained and deletes the most raw lines per hour. Phase 4
touches grammar-adjacent code, so it goes after the caption bug fixes have
settled and rides on the caption test suite. Phases 5–6 are independent
grab-bags; item 6.1 depends on Phase 2's `apply_one`.

Rough net effect: ~900–1100 duplicated lines deleted, four "mirrors of a
mirror" (count regex, detection options, caption resolution, drift ladder)
reduced to one definition each, and every invariant currently enforced by a
comment ("kept byte-exact", "shared so they can't disagree", "mirrors the
trainer") enforced by an import instead.

## Where it landed — **all six phases done, 2026-08-31**

The four "mirrors of a mirror" are one definition each, and so are nine more
the review found on the way: the report shape, the drift ladder, the count
regex, the tag normalizer, the vocab reader, the checkpoint reader, the dbv4
cache, the image walk, the device probe, the JSON file, the threshold sweep,
the SSE follower and the persisted signal. What is left duplicated is
duplicated on purpose, and each copy now says so where it lives:

| Copy | Why it stays | Pinned by |
|---|---|---|
| `gui/proposals.SHAPES` (three `ReplaySpec` instances) | importing a stage CLI pulls torch into the server | `tests/test_gui_proposals.py` |
| `buckets.py` free-fit geometry | the dependency direction forbids importing the trainer | `tests/test_resize_images.py` |
| `_walk.py` vs the trainer's `image_utils` | same | the trainer's own parity test |
| `vision/pe.py` tower | kept diffable against upstream | — |
| `grouping`'s two `.npz` caches | different shapes, one caller each (Phase 5, item 8) | — |

Two later-phase items were not done as written, both because doing them would
have invented an abstraction rather than removed a copy: Phase 5's
`cached_descriptors` (one caller) and Phase 6's fifth persisted signal (the
dock height, which must not write at frame rate). Both are recorded in their
phase's outcome.
