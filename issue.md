# Codebase review — findings and cleanup backlog

Written 2026-09-12 against `71bdec4`. A whole-repo review of `anime_tools/` (30.5k lines of
Python), `frontend/src/` (8.7k lines of TS/TSX) and `tests/` (15.6k lines), looking for structural
problems, duplication, inefficiency and drift rather than for features.

Items under "Reported, not individually reproduced" come from the same review pass but were not
each checked by hand — treat them as leads, not as verdicts.

**Only open items are listed.** Everything fixed on 2026-09-12 — the twelve confirmed defects
(three P0, nine P1), the four "close the pairs" items, the five big-file splits and the eleven
efficiency items — has been removed from the sections below; what each one was, and what closing it
still needs from the trainer repo, is at the bottom under "Landed, and what the trainer owes".

## The one theme

The core abstractions here are good and, in several cases, better than what most projects of this
size manage. `_request.py` turns one field declaration into a parser, an argv and a GUI form.
`registry.py` names stages lazily so nothing heavy is imported to list them. `replay.apply_one` is
one drift-guarded write shared by four callers. `_walk`, `_json`, `_device`, `buckets` each claim to
be the single answer to a question.

The recurring problem was that **most of these abstractions had a second, partial implementation
sitting next to them**, and the invariants in `CLAUDE.md` described the abstraction rather than the
pair. That produced most of what this file originally held. One pair is left:

| The one answer | The second one beside it |
|---|---|
| `parse_caption` / `compose_caption` | `split(",")` on caption text: `variants.py:186`, `autotag.py:130`/`:211`, `captions.py:254`, `tag_rules.py:108` |

It is also the weakest of the six that table held on 2026-09-12: all five of its remaining sites
split a *tagger's* comma-joined output rather than a caption. The other five closed that day — the
progress format (`_progress.progress_line` / `ProgressBar`, no `tqdm` anywhere in the package),
`correct`'s dry-run default, `gui.Field` carrying its `Arg`, grouping's walk, and the registry's
laziness surviving resolution — and are recorded under "Landed" below. Closing a pair is still
worth more than any individual bug in this file, because a pair is a place where a future change
can be made correctly in one spot and still be wrong.

## Structural

- [ ] **`frontend/src/types.ts` is a 650-line hand-maintained mirror of the Python dataclasses**,
      with no drift check, and it is the single most-churned file in the repo (44 touches in 60
      days, ahead of `gui/server.py` at 42). `ROOT_NAMES`, `MASK_ROLES`, `MASK_KINDS` and `StageId`
      each copy a Python constant.
      Fix: emit it from `schema()` during `scripts/build_frontend.sh`, or add a test asserting the
      key sets match. CI already diffs the bundle, so generation would be enforced for free.

- [ ] **Flag separator: unify on `_`.** The hyphen/underscore split costs about 15 lines
      (`FLAG_SEP`, `flag_of(sep)`, `spellings()`, and a `replace` in `gui/stages.py`) and two
      docstring paragraphs. It is not even honoured inside `masking/`, where two probe CLIs and
      `exclude/_cli.py` declare underscore flags. Since every flag already accepts both spellings,
      setting `FLAG_SEP = "_"` package-wide breaks no saved command line.

## Cleanup

- [ ] **Dead code, confirmed by grep:** `captions/correction.py:434 correct_many` and
      `grouping/features.py:57 caption_text` have zero references; `exclude/_ledger.py:181 is_excluded`,
      `masking/_sam3.py:183 add_prompt_embed_arg` and `masking/cli/merge_masks.py:11 DEFAULT_INPUTS`
      are referenced only by their own export line. `frontend/src/components/Report.tsx` (97 lines)
      is imported nowhere. `ocr/engine.py crop_quad`, `ocr/sfx.py read_raw`/`read`/`read_boxes`, and
      `dbv4_backend.py:266 normalization` have no callers.
- [ ] **Stale counts, confirmed.** Three files — root `CLAUDE.md:110`, `stages/CLAUDE.md:68`,
      `gui/CLAUDE.md:15` — say the registry lists "eleven" stages. It has listed ten since
      `9413178` removed the ComicTextDetector text-mask stage, a commit that *did* edit
      `CLAUDE.md` and still missed all three copies of the number. Likewise `stages/CLAUDE.md:63`
      says `__init__` exposes "fifteen names" where `__all__` has 17, and `README.md:106` plus
      `frontend/CLAUDE.md:47` say "three Settings dialogs" where `frontend/src/config.ts:12` has
      four — `frontend/CLAUDE.md` then contradicts itself 120 lines later.
      Every one of these is a number buried mid-sentence in prose. The fix is a test asserting the
      documented counts against `len(STAGES)` and `len(__all__)`, or writing no counts at all.
- [ ] **Other stale prose:** `tagger/CLAUDE.md:10` says `groups.json`, the code says `groups.yaml`;
      `masking/cli/merge_masks.py:12` and `masking/CLAUDE.md` still say "two generators" (one since
      0.5); `_masks.py:141` names the removed ComicTextDetector stage; `_device.py:8` spends five of
      56 lines on the deleted `anime_tools._onnx`; `ocr/engine.py:139` still says "overlaps a
      `session.run`" from the retired ONNX backend; `position_captions.py:10` says
      `cli/position_captions.py` owns model loading, which `run.py` does.
- [ ] **`requests.py` help strings double as a changelog** — `bag_relax` runs 11 lines, others cite
      measurements and "the pre-2026-08-19 behaviour". `docs/position_captions.md` is the declared
      home for the numbers; help text should be one sentence.
- [ ] **`path_filter.py` is a 25-line internal leaf with one caller** and no CLI or re-export.
      Rename `_path_filter.py` or fold it into `_walk.py`, so the `_`-prefix convention stays
      meaningful.
- [ ] **`stages/captions.py` re-implements the caption walk** (`:172`) instead of using
      `iter_captions`, and compares raw `read_text()` to the corrected text at `:220` where every
      other stage compares a stripped `read_caption`. A legacy file with a trailing newline is
      therefore judged "changed" and rewritten, with a history push, on every run.
- [ ] **Test fixtures are duplicated rather than shared.** `conftest.py` holds only `repo_root` and
      `chdir`, while `home(tmp_path, monkeypatch)` is redefined in six files, a `TestClient` fixture
      in five, and a `_png` writer in five more under four different signatures. Promoting `home`,
      `client` and a `png()` factory would cut roughly 150 lines.
- [ ] **One test asserts on source text rather than behaviour** — `test_registry_requests.py:527`
      rglobs the package for the literal `"--device"`, so it breaks on any refactor that preserves
      behaviour. The `inspect.getsource` grep in `test_ocr.py` was the other one; it went with the
      split (it was reading `stages/run.py`) and now stubs both model loads instead.
- [ ] **Two tests write outside `tmp_path`** — `tests/test_gui.py:556` (cleaned up in a `finally`)
      and `tests/test_gui_dataset.py:461` (not cleaned, leaking into the pytest basetemp parent).

## Docs, packaging and CI

The health check is good: **1096 tests pass in 15.8 s**, CPU-only; `ruff check` and `ruff format
--check` are both clean across 239 files; `scripts/wrap_md.py --check` is clean over every tracked
markdown file; and there are **zero** `TODO`/`FIXME`/`XXX`/`HACK` markers in `anime_tools/`,
`frontend/src/` and `tests/`. The problems are elsewhere.

- [ ] **`ruff check` does not gate CI.** `.github/workflows/ci.yml:21` sets
      `continue-on-error: true`, because the lint rules live in the maintainer's user-level config
      and the comment calls the default set "advisory only". Lint is therefore unreproducible for
      any contributor and enforced for nobody.
      Fix: move the rules into `[tool.ruff]` in `pyproject.toml` and drop the flag.

- [ ] **No Windows CI job**, despite `install.ps1`, a PowerShell branch in the Makefile,
      `triton-windows`, the cu132 index, `.lnk` shortcut writing and five win32-branching modules.
      No macOS job either, despite the MPS and no-triton paths. Nothing in the matrix tests the two
      platforms with the most bespoke code.

- [ ] **A GUI-only install still costs about 4.5 GB of CUDA.** `sam3 @ git+…` is a hard dependency,
      so it drags torch into every install including the documented torch-free server path. A fresh
      `uv sync` is roughly 5.3 GB across 96 packages, dominated by nvidia at 2.7 GB, torch at 1.1 GB
      and triton at 691 MB. `matplotlib` has one importer, `opencv-python` (~190 MB) has four, and
      `pycocotools` plus `psutil` exist only to patch sam3's under-declared imports.
      Fix: move the model stack behind an extra or a dependency group, so the invariant the test
      suite already enforces is also true of the install.

- [ ] **About 12 GB of irreplaceable data sits inside the git checkout.** `models/` (6.9 GB),
      `workspace/` (3.3 GB) and `_archive/` (2.1 GB) are all gitignored but live in the worktree,
      because the curation home is the checkout during development. One `git clean -xfd` destroys
      the weights, the dataset and the archive together.
      Fix: point `ANIME_TOOLS_HOME` at a sibling directory for dev, and say so in the README.

- [ ] **The guidebook filenames are non-ASCII.** `가이드북.md`, `ガイドブック.md` and `指南书.md`
      are looked up correctly and shipped as package data, but git's default `core.quotePath`
      octal-escapes them, so `git ls-files '*.md' | xargs …` crashes without `-z` (reproduced).
      Windows non-UTF-8 consoles and macOS NFD normalization are the other two hazards, the latter
      silently breaking the dict lookup.
      Fix: ASCII filenames (`guidebook.ko.md`) behind the `lang → filename` map that already exists.

- [ ] **Stale README claims.** `README.md:20` says "with every extra" where `pyproject.toml` has no
      `[project.optional-dependencies]` at all and root `CLAUDE.md:28` says "no extras"; the
      `ANIME_TOOLS_VERSION` example still reads `v0.3.1` against a current `0.6.5`; and the layout
      tree matches neither the guidebook's nor `workspace/__init__.py`'s, though both cite the
      latter as owner.

- [ ] **The same fact is spelled in up to fifteen files.** Measured: the `workspace/resized/` tree
      appears in 15, the torch-free-server rule in 9, the dry-run rule in 7, `history.txt`/Undo in
      7, the numpy override in 7, and the `split(",")` rule in 6. Root `CLAUDE.md`'s "Shared infra"
      section is a third copy of each module's own docstring and has the highest drift rate in the
      repo.
      Fix: reduce the root file's "Shared infra" section and the README's Development and layout
      blocks to pointers. The module docstrings and `examples/` are the more current owners.

- [ ] **The prose style helps maintainers and costs newcomers.** Docstrings are 16% of the Python
      lines, tracked markdown is 371 KB, and 57 KB of `CLAUDE.md` loads before any file is read. The
      *why* is genuinely exceptional and should stay — `_device.py` on why MPS is second, the
      `pyproject.toml` note naming the real ROCm lock collision, the `(size, mtime_ns)` stamp
      rationale. The *what* as narrative is what rots: all six stale claims above are a number or
      name buried mid-sentence.
      Fix: tabulate the *what*, keep the *why* in prose. `gui/CLAUDE.md`'s 14-bullet "Other server
      pieces" and `stages/CLAUDE.md`'s 20 unbroken lines on Export are the two worst offenders.

- [ ] **Housekeeping.** The live `.venv` still holds onnxruntime, onnx and opencv-python-headless,
      about 440 MB that is not in `uv.lock`, left over from the 2026-09-09 removal; `uv sync
      --reinstall` reclaims it. `bench/tagger_external/README.md` points at a script archived into
      the gitignored `_archive/`, which nobody else has. `design/` is 136 KB with no runtime role
      and carries a second copy of the Pretendard font.

The committed frontend bundle is worth keeping as is. It is touched by 39% of commits and the
history holds several 2.4 MB blobs from the era when the font was inlined, but `.git` is still only
14 MB, git installs need no bun, and CI's drift check makes it safe. Splitting the font out was the
right fix.

## Reported, not individually reproduced

These came out of the same review but were not each checked by hand.

- `_request.py:362` emits the positive spelling for a `store_false` field whose value is `True`,
  while `build_parser:277` declared only the `off` spelling. Unreachable today because every
  `_off()` call defaults to `True`, but it is a latent argv/parser mismatch.
- `_request.py:209` requires a gate field to be declared before the fields it gates, or the drawer
  silently loses its group. Raising on an unknown gate dest would make it loud.
- `grouping/cli/match_decensored.py:130` stamps a whole directory with `(max mtime, count)`, so an
  in-place edit that moves neither is a silent stale cache hit. The per-image stamp in
  `features.py:137` is the correct one.
- `gui/routes/settings.py::put_settings` lets `PUT /api/settings` write any key, including the
  `update_check` cache that only the server is supposed to own.
- `JobManager.jobs` grows without bound, holding up to 20,000 lines per job even though the log is
  already on disk.
- `runner.ts:131` fires `finished()` and `reloadTouched()` concurrently and both append a
  "changed N" suffix, so a clean replay reads "exit 0 — 12 changed — 12 changed".
- `gui/stages.py:613` drops `root`/`setting`/`report`/`auto` fields from `form_values` but not
  `mask`, so `mask_dir` lingers in the persisted form.
- The browser re-implements the server's field-hiding rule in `FieldRow.tsx:16`, though
  `gui/CLAUDE.md:30` says the server settles it. Ship `hidden` from `schema()` instead.
- Several dataset routes accept `src`/`dst`/`masks`/`pattern`/`limit` overrides the browser never
  sends.
- `masking`'s drawer mechanism (`gate` + `GATE_ATTR`) is documented in `masking/CLAUDE.md` but no
  masking request uses one; the only carrier is Export's Combine OCR.
- `grouping/matching.py:32` has no production caller and exists as an oracle for one test; worth
  saying so in the docstring.
- `tagger/cli/*` is largely checkpoint-build tooling that needs a private corpus and hardcodes
  `v5_reference` numbers, yet ships in the wheel. `readback.py`'s only consumer is a bench script in
  the trainer repo. Candidates for `bench/tagger_build/`.
- `vision/pe.py` carries two model configs with no caller, an attention-pooling path nothing
  selects, and training-only `DropPath`/checkpointing that costs a `timm.layers` import.

## What is genuinely well done

Worth keeping in mind before any refactor moves these.

- **`_request.py`.** One small module turns a field declaration into the flag spelling, default,
  help, grouping, the GUI schema and the argv, with `to_argv`/`from_namespace` pinned as inverses by
  a test. This is rare, and it is the reason the GUI and CLI cannot drift on flags.
- **`replay.apply_one` / `undo_one`.** One drift ladder, with the `target_before` versus `original`
  distinction written down, reused by four writers. It is what makes running without an Apply gate
  defensible.
- **`grouping/features.py:137`.** The `(size, mtime_ns) + version` stamp, the "anything wrong means
  recompute, never an error" rule, and the note at `:213` about capturing the stamp *before* the
  decode — the subtle part most caches get wrong.
- **`exclude.py:233 rel_key`** applies Windows path rules on every host because they are the strict
  ones, with the reason recorded; `artifacts:290` dedupes by `(st_dev, st_ino)` rather than by
  spelling.
- **`masking/_sam3.py`.** Three unavoidable monkey-patches, each documenting why the obvious
  alternative is wrong, pinned by a test.
- **`vision/yolo12.py:435`.** The unpickler stub is a clean licence firewall, and `load_yolo12`
  refuses checkpoint mismatches with explicit allowed exceptions.
- **i18n parity is actually enforced** — `ko`/`ja`/`zh` are typed against the `en` dictionary and
  `bun run check` runs in CI, so a missing key fails the build.
- **`ClientWatch` + `/api/alive`** is a correct and minimal exit-with-the-window, including the
  reasoning about throttled timers in hidden tabs.
- **The hygiene baseline is genuinely clean.** 1096 tests in 15.8 seconds on CPU, lint and format
  clean, markdown at its wrap fixpoint, and not one `TODO` or `FIXME` in the whole tree. Every
  `--flag` named in `docs/*.md` and in the guidebook resolves to a real source symbol, and every
  import in `examples/` resolves. `examples/` is the best-maintained documentation layer here.

## Suggested order

Grouped so that each block is one coherent sitting. The five blocks that are done — correctness,
durability, closing the pairs, splitting the big files, and the efficiency sweep — are under
"Landed" below, so this is only what is left.

1. **Make the guards real.** Move the ruff rules into `pyproject.toml` and let `ruff check` gate
   CI; add a Windows job; generate or test `types.ts` against `schema()`; test the documented
   counts against `len(STAGES)` and `len(__all__)`.
2. **Cleanup.** Share the test fixtures, delete the dead symbols, fix the stale prose and the
   README's four wrong claims in one sweep.

Two items sit outside that order because they are one-line changes with outsized downside: move
`ANIME_TOOLS_HOME` out of the checkout before the next `git clean`, and rename the guidebooks to
ASCII.

The remaining blocks are safe to do incrementally, and the cleanup is much less likely to regress
now that the second implementations are gone.

## Landed, and what the trainer owes

Each of these was a block of items in the sections above and has been removed from them. What they
were, so a reader of the git history can find them:

### The five big files, 2026-09-12 (`0.7.0`)

No behaviour changed — the CLIs, every flag, the wire schema and the reports are what they were —
so this was entirely about where a future change lands.

- **`stages/run.py` (919 lines) is gone.** Each `run_<stage>(req)` lives in its stage's own module,
  which `registry.py` already addressed as `module:function` and resolves lazily. The two pieces
  that were orchestration rather than runner moved with it: `position_captions.summarize()` holds
  the report's counts (the runner keeps the header and the knobs, which are the request's to
  report), and `multiview_audit.run_audit_phase` sits beside `promotions`, its only caller. Two
  library functions took the aliases `run.py` already gave them so a runner could keep its name:
  `stages/ocr.py::run_ocr` → `read_tree`, `export_workspace.py::run_export` → `publish`.
  `cli/_args.py` and `cli/_report.py` became `stages/_progress.py` and `stages/_report.py`, so the
  library half no longer depends on its own CLI package.

- **`gui/server.py` 1129 → 207.** The API is `gui/routes/` (`settings`, `jobs`, `dataset`,
  `desktop`), each an `APIRouter` reading its state off `request.app.state` — `watch` included,
  which `/api/alive` used to close over as a `create_app` argument. `launch.py` took `main`,
  `pick_port` and the Chromium app window. The settings-derived values are `gui/_context.py`'s
  `RunContext`, one settings read per request, where `bindings()` is the four keyword arguments
  `resolved_schema` / `build_argv` were handed separately and `scoped_to(rel)` is the per-image
  narrowing. The free `roots_for` / `report_root` / `mask_root` stay, because Settings'
  placeholders are those same values computed against *empty* settings.

- **`tagger/tagger.py` 684 → 496**, leaving the model, with two torch-free leaves beside it:
  `tagger/schema.py` (what a tag means to a checkpoint — the four tuples, `TagEntry`,
  `dedupe_count_tags`; every value of it baked into `vocab.json` at build time) and
  `tagger/fetch.py` (getting one onto disk, with `is_dbv4_dir` public, which is what the ComfyUI
  node was apologising for in a comment). `tagger.py` re-exports both, but the vocab build, both
  `cli/autotag*` entry points, the node and `stages/_models.py` were repointed at the leaf, since
  not importing torch to read a tuple is the whole point; `test_boundary` pins all three.

- **`downloads.py` (624) and `exclude.py` (540) are packages** whose `__init__.py` re-exports the
  surface they had, so every `from anime_tools.downloads import …` is unchanged and `python -m`
  reaches both (`_cli.py` behind a two-line `__main__.py`, so importing the package runs nothing).
  `downloads/`: `_locations.py` (where each weight lives), `_assets.py` (`Asset` / `Pack` + the
  fetch engine), `_catalog.py` (the rows and the lookups). `exclude/`: `_ledger.py` (the state and
  `rel_key`), `_artifacts.py` (what an image is made of), `_move.py` (the engine).

One real bug fell out: `test_the_tagger_loads_once_per_process` patched
`tagger.tagger.ensure_tagger_checkpoint` while `_models.py` imports it from `fetch`, so the test
was doing a live hub fetch — 10.7 s to 1.0 s once it patched the owning module. Two tests were
rewritten rather than repointed, both named under *Cleanup* as source-text assertions: the OCR
device test (an `inspect.getsource` grep for `_vl_engine(req, engine, resized_dir, device)`, now
stubbing both loads to assert one probe and one device reaching each) and the listing-cap test,
which read `app.routes` — this FastAPI does not flatten an included router.

`CONTRACT_VERSION` stays at 2: no name in `contract.py` moved and no stage flag changed.

### Closing the pairs, 2026-09-12

Two new leaves came out of it — `anime_tools/stages/_options.py` (every default a stage request
field
names) and `anime_tools/masking/_prompts.py` (what a SAM3 prompt is, for both packages) — plus
five regression tests: the import-light measurement and the re-export identity
(`test_registry_requests`), the one-way masking seam (`test_boundary`), the stem collision and the
shared `path_pattern` (`test_grouping_features`), and the `Arg`-driven coercion plus the
`Field`-is-its-`Arg` shape (`test_gui`). Nothing in `frontend/` moved: the schema the browser
receives is byte-identical.

### The efficiency sweep, 2026-09-12

Eleven items, no behaviour change: every one is pinned by a test asserting the *same* answer off
fewer passes. The recurring shape was a loop asking a cheap question the expensive way — per image
what the batch already had, per tag what the kept handful could answer, per root what one settings
read already said.

- **The tagger batches.** `AnimaTagger.predict_batch` / `predict_caption_batch` are one
  `Dbv4Backend.forward` over a list (`_heads_forward_batch`, which `_heads_forward` is now the
  one-image case of), with the post-processing split out as `_predict_row` / `_caption_of` so a
  batch of one answers exactly what `predict` does. `stages/autotag.py` drives it: the pass is two
  halves now, a caption-resolution walk that says which images the tagger has to see and a decision
  walk that pulls each candidate's tags out of a buffer filled `--batch_size` (default
  `_options.DEFAULT_BATCH_SIZE`, 8) at a time. The report, the skip reasons and the `[done/total]`
  line are what the one-at-a-time pass produced. `build_tag_fn` is `build_tag_batch`.
- **`dbv4_backend.forward_tensor` normalises on the device**, after the transfer that had to
  happen anyway, with `_mean` / `_std` living there.
- **`predict` reads its tensors once, as lists**, instead of a `float(t[i])` per vocabulary row;
  the five category filters behind it key off the kept handful (`_cat_of`, `_scored`) rather than
  walking all ~13k `tag_entries` apiece, and `predict_caption`'s slotting does the same
  (`_slot_of` / `_emit_key`).
- **`downloads.by_id` / `expand` take the rows**, so `_cli.main` builds the catalog once instead of
  four times, each of which resolved the home and read the installed checkpoint's config.
- **`exclude_many` / `restore_many`** read and write the one ledger document once however many
  images move; `exclude_one` / `restore_one` are the single-image case, and the CLI checks every
  rel first and then makes one pass.
- **`grouping._grid_match_edges` chunks the Stage-A prefilter** (`cls_chunk`) like the pooling and
  the pair match above it, row-major so the edge list is the one the n x n pass produced.
- **An in-process Export decides each row once** (`export_one(..., decided=True)`); a replay of a
  saved report still re-decides, because that disk has moved on.
- **`caption_layout` takes a `ParsedCaption`** (`position_clauses.as_parsed`), so `is_candidate`
  parses once instead of three times and `propose_for_image` / `audit_image` hand their own parse
  to every layout question.
- **`OcrEngine.read_iter(..., with_pixels=True)`** yields the page it already decoded, so
  `RereadEngine` no longer decodes every page a second time. The chunk's pixels were alive anyway.
- **The SFX reader templates its prompt once** (`SfxReader.prompt`, a `cached_property`) and the
  greedy logits recorder is one cached class (`_greedy_recorder`) rather than a fresh type per page.
- **One settings read per request.** `dataset_bases` / `reachable` / `resolve_roots` take what the
  caller already has, so `RunContext.load()` reads `.anime_tools_gui.json` once rather than five
  times.

`AutotagRequest` gained `--batch_size`, which is the only new flag; `CONTRACT_VERSION` is
unchanged and nothing in `contract.py` moved.

### The confirmed defects, 2026-09-12

**P0.** The caption corrector round-tripping its tag bag through a comma split; `correct` writing
with no dry run and no report (so no Undo); a custom clause vocabulary never reaching the
multiview verdict.

**P1.** The slug regex that was a character class of backslash, `s` and slash rather than the
whitespace class; `probe_nms_pairs` reading an undeclared `args.score_threshold`;
`--mode predict` gated on a file a dbv4 checkpoint never has; the job log stream indexing past a
trimmed buffer; unbounded decoded-image retention in the SAM3 run; non-atomic `write_json` and
three unserialised settings writers; three stages with no GUI progress bar; `--batch-size` on the
mask stage batching nothing; `--workers 0` crashing it.

Ten regression tests came with them. `correct` is now an `ApplyRequest` with
`REPLAY_SHAPES["correct"]`, `write_json` is temp-file-plus-`os.replace`, settings go through
`gui.settings.edit_settings()`, and `_progress.py` owns both the printed `  [done/total] detail`
line and the daemon stream.

### The trainer side (`anima_lora`)

Three of those fixes cross the seam, so they are not done until the trainer moves too. The first
is blocking and the trainer's tree is already red against its own pinned dependency.

- [ ] **`from anime_tools.stages import run` no longer resolves.** `stages/run.py` is gone
      (the split, 2026-09-12); each runner lives in its stage's module and `stages/__init__.py`
      re-exports it, so the trainer's *production* path is untouched —
      `preprocess.py` does `from anime_tools.stages import release_models` and `_common.py` goes
      through `Stage.runner()`, both still correct. What breaks is one test:
      `tests/test_anime_tools_cli_contract.py:416` does `from anime_tools.stages import run as
      pkg_run` and patches `pkg_run.release_models`. The runner cache lives in
      `anime_tools.stages._models`, so that line becomes
      `from anime_tools.stages import _models as pkg_run`.

- [ ] **Release `anime_tools` and bump the pin — blocking.** `correct` is dry-run-by-default now,
      so `scripts/tasks/preprocess.py` has to pass `apply=True` (done, at both construction sites:
      the `request_from_form` path and the `CorrectRequest(**fields)` path, with one test
      assertion updated). But `pyproject.toml:141` pins `tag = "v0.6.4"` and `anime-tools-git`
      is default-on (`default-groups`, `:115`), so a plain
      `uv sync` installs a package whose `CorrectRequest` has no `apply` field:
      `TypeError: CorrectRequest.__init__() got an unexpected keyword argument 'apply'` on the
      first `make preprocess-captions` or `preprocess-te`. Seven of
      `tests/test_preprocess_tasks.py` fail there today; all 33 pass with
      `PYTHONPATH=../anime_tools`. Tag a release here, then bump the tag and `uv.lock` in the same
      commit as the `apply=True` change — they cannot land separately.
      Without the flag the correction silently stops writing and TE caches the un-corrected
      caption, which is the same class of bug as the P0 it came from.

- [ ] **Teach the Qt progress bar the package's line format.** No stage in `anime_tools` emits
      `tqdm` any more, and `gui/progress.py:29`'s `TQDM_RE` only matches tqdm's
      `NN%|bar| cur/tot`. The tabs that drive a bar off a daemon job's *stdout* — `_job_mixin.py:90`
      feeding `TqdmProgressTracker.feed`, used by `tabs/image_tab.py:245` for `curate-group` and
      `tabs/preprocess/tab.py:1097` for the mask stages — therefore sit in indeterminate "busy"
      mode for the whole run. One extra alternative in the regex fixes it:
      `^\s*\[(?P<cur>\d+)/(?P<tot>\d+)\]\s*(?P<label>.*)$` (label *after* the counts, unlike
      tqdm's). The alternative fix is to move those tabs onto `JsonlProgressReader`: the mask
      merge and the grouping pass now write `step` lines to `progress.jsonl` where before only
      `masks_sam` did, so the structured stream covers all of them. `_job_mixin.py:90`'s docstring
      ("preprocess/mask … emit no progress.jsonl, so tqdm is the only progress signal") is stale
      either way.

- [ ] **One stale sentence in `scripts/tasks/masking.py:191`.** "The SAM3 checkpoint and batch
      size are the request defaults" — `SamMaskRequest` has no `batch_size` any more. Nothing
      passes it (the yaml rule path forwards only `threshold` / `dilate`, and no saved GUI card
      carries it), so this is prose only.

Nothing else crosses: `contract.py` only gained a `REPLAY_SHAPES` key, `buckets.py` is untouched,
and the `progress.jsonl` line shapes the daemon's reader filters on are unchanged.
