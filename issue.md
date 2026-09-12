# Codebase review — findings and cleanup backlog

Written 2026-09-12 against `71bdec4`. A whole-repo review of `anime_tools/` (30.5k lines of
Python), `frontend/src/` (8.7k lines of TS/TSX) and `tests/` (15.6k lines), looking for structural
problems, duplication, inefficiency and drift rather than for features.

Items under "Reported, not individually reproduced" come from the same review pass but were not
each checked by hand — treat them as leads, not as verdicts.

**The "Confirmed defects" section is gone: all twelve of its items (three P0, nine P1) were fixed
on 2026-09-12 and are no longer listed.** What they were and what closing them needs from the
trainer repo is at the bottom, under "Landed, and what the trainer owes".

## The one theme

The core abstractions here are good and, in several cases, better than what most projects of this
size manage. `_request.py` turns one field declaration into a parser, an argv and a GUI form.
`registry.py` names stages lazily so nothing heavy is imported to list them. `replay.apply_one` is
one drift-guarded write shared by four callers. `_walk`, `_json`, `_device`, `buckets` each claim to
be the single answer to a question.

The recurring problem is that **most of these abstractions have a second, partial implementation
sitting next to them**, and the invariants in `CLAUDE.md` describe the abstraction rather than the
pair. That is what produces nearly every item below:

| The one answer | The second one beside it |
|---|---|
| `parse_caption` / `compose_caption` | `split(",")` on caption text: `variants.py:186`, `autotag.py:130`/`:211`, `captions.py:254`, `tag_rules.py:108` |
| `_request.Arg` + generated parser | `gui/stages.Field` + `_coerce`, a parallel type system |
| `_walk.walk_images` | `grouping/features.py:87` calling `glob_images_pathlib` directly |
| `registry.py`, deliberately import-light | `stages/requests.py`, which pulls numpy, PIL and yaml |

Fixing the pairs is worth more than any individual bug below, because each pair is a place where a
future change can be made correctly in one spot and still be wrong.

Two rows are off that table since 2026-09-12: the progress format now has one implementation
(`_progress.progress_line` / `ProgressBar`, no `tqdm` anywhere in the package), and `correct` is
dry-run-by-default with a report like its six siblings. `correction.py`'s own two caption splits
went with the first P0.

## Structural

- [ ] **`stages/requests.py` defeats the registry's laziness.** Measured: `import
      anime_tools.stages.registry` pulls no numpy, PIL, yaml, cv2 or torch; `import
      anime_tools.stages.requests` pulls numpy, PIL and yaml, at 360 modules. The cause is module-
      level imports of `masking._sam3` (`:29`) for four constants, `stages.multiview_audit` (`:35`)
      for four defaults, and `stages.position_captions` for `PositionCaptionOptions`. The GUI
      resolves request classes to build its form, so it pays this on every schema build.
      Fix: move `prompt_list` and the help constants to `contract.py`, and `PositionCaptionOptions`
      plus the `DEFAULT_*` constants to a leaf `stages/_options.py`.

- [ ] **`gui/stages.Field` and `_coerce` are a second type system over the same field list.**
      `Field` restates eight of `Arg`'s attributes, `field_of` copies them across one by one, and
      `_coerce` re-implements as a `kind` switch the coercion `build_parser` already encodes in
      `type=`. Adding an `Arg.kind` means remembering a matching `_coerce` branch.
      Fix: have `Field` carry its `Arg`, and dispatch `_coerce` on `a.type` / `a.nargs`.

- [ ] **`stages/run.py` holds seven runners (882 lines) away from the stage modules**, while
      `registry.py` already addresses runners as `module:function`. `run_position`'s 55-line summary
      dict (`:335`) belongs in `position_captions.summarize()`, and `_run_audit_phase` belongs beside
      `promotions` in `multiview_audit.py`. What is left per runner is roughly 30 lines.

- [ ] **`stages/run.py:25` imports from `stages/cli/_args.py` and `cli/_report.py`** — the library
      depends on its own CLI package. Move both to `stages/_report.py` and `stages/_progress.py`.

- [ ] **The masking ↔ stages seam is two-way.** `masking/sam.py:36` imports from
      `stages/instance_detection.py`, while `stages/requests.py:29` imports from `masking/_sam3.py`.
      The root cause is naming: `instance_detection.py:7` claims "Detector-agnostic — nothing here
      imports SAM3" yet owns `resolve_prompt_embed`, `SOFT_PROMPT_KEYS` and `load_soft_prompt`.
      Fix: move those three into `masking/_sam3.py`. The docstring becomes true, the seam becomes
      one-way, and masking loses its only `stages` import.

- [ ] **Grouping bypasses the one image walk.** `grouping/features.py:87` calls
      `glob_images_pathlib` directly while `masking/_masks.py:66` uses `walk_images`, so grouping
      gets neither the same-stem collision assertion nor `path_pattern`. A test currently asserts
      the two agree, which is exactly the assertion that breaks first.

- [ ] **`frontend/src/types.ts` is a 650-line hand-maintained mirror of the Python dataclasses**,
      with no drift check, and it is the single most-churned file in the repo (44 touches in 60
      days, ahead of `gui/server.py` at 42). `ROOT_NAMES`, `MASK_ROLES`, `MASK_KINDS` and `StageId`
      each copy a Python constant.
      Fix: emit it from `schema()` during `scripts/build_frontend.sh`, or add a test asserting the
      key sets match. CI already diffs the bundle, so generation would be enforced for free.

- [ ] **`gui/server.py` is one 1129-line file with ~620 lines of route closures inside
      `create_app`.** Natural split: routers for jobs, dataset, desktop (`/api/pick`, `/reveal`,
      `/ls`) and settings, reading `request.app.state`, plus a `launch.py` for `main`/`pick_port`.
      Settings-derived values (`roots_for`, `stage_defaults`, `report_root`, `mask_root`) are
      recomputed in varying combinations per handler; a `RunContext` built once per request would
      replace four kwargs with one object.

- [ ] **`tagger/tagger.py` mixes three concerns** — checkpoint fetching, torch-free vocab schema
      constants, and the model with a 150-line `predict`. Consequence: `cli/vocab.py` and
      `comfyui/anima_tagger/nodes.py` import torch to read four tuples, and `cli/constants.py`
      apologises for it in a comment. Split a torch-free `tagger/schema.py` and a `tagger/fetch.py`.

- [ ] **`downloads.py` (624) and `exclude.py` (540) each hold three things.** Downloads is
      constants, a 190-line catalog literal, a fetch engine and a CLI — and `Asset.build` wires
      `tagger.cli.build_english_tag_csv` into a module advertised as a torch-free weight catalog.
      Exclude is the ledger, the artifact map, and a move engine with a 110-line CLI.

- [ ] **Flag separator: unify on `_`.** The hyphen/underscore split costs about 15 lines
      (`FLAG_SEP`, `flag_of(sep)`, `spellings()`, and a `replace` in `gui/stages.py`) and two
      docstring paragraphs. It is not even honoured inside `masking/`, where two probe CLIs and
      `exclude.py` declare underscore flags. Since every flag already accepts both spellings,
      setting `FLAG_SEP = "_"` package-wide breaks no saved command line.

## Efficiency

- [ ] **No batched tagger inference.** `tagger.py:479` wraps a single image and `stages/autotag.py`
      calls `predict_caption` per image with no prefetch, even though `Dbv4Backend.forward` already
      takes a list and `train_sidecar.build_cache` already drives a DataLoader. Add `predict_batch`.
- [ ] **`dbv4_backend.py:335` normalises on CPU in float32** over the whole batch, then moves to the
      device. Move first, normalise on device.
- [ ] **`tagger.py:520`** runs two Python loops of `float(tensor[i])` per image where one
  `.tolist()`
      would do, then walks `tag_entries` five more times for category filters.
- [ ] **`downloads.py` rebuilds the catalog 4+ times per invocation** (`expand` calls `by_id` and
      `by_pack`, each rebuilding; `main` calls all three again), each running a disk probe.
- [ ] **`exclude.py` rewrites the whole ledger per image.** `main:513` loops over `args.rels`
      calling `exclude_one`, so N exclusions cost N full reads and N full JSON rewrites. Add
      `exclude_many`.
- [ ] **`grouping/groups.py:112` builds the full n×n CLS similarity unchunked**, while the pooling
      directly above it is chunked at 256. A 5k-image bucket is 25M floats plus a 2×12.5M index
      tensor.
- [ ] **Export decides every row twice per in-process run**, so each text row's source and
      destination bytes are read twice and each combined row re-renders `with_ocr_clause` twice.
- [ ] **Repeated caption re-parsing** — `caption_layout.is_candidate` parses three times, and
      `propose_for_image` parses the same caption about five more times. Cheap next to SAM3, but a
      `ParsedCaption` entry point removes it.
- [ ] **`ocr/reread.py:334` decodes every page a second time**, having already been decoded in
      `ocr/engine.py:227` a chunk earlier. `read_iter` could yield the pixels it already has.
- [ ] **`ocr/sfx.py:338` re-runs `apply_chat_template` per batch** and redefines `_Greedy` per call.
      Both belong in `load`.
- [ ] **Per-request settings re-reads in the GUI.** `roots_for()` ends up parsing
      `.anime_tools_gui.json` five times per request, because `reachable` → `dataset_bases` →
      `load_settings()` runs once per root, on top of the caller's own read.

## Cleanup

- [ ] **Dead code, confirmed by grep:** `captions/correction.py:434 correct_many` and
      `grouping/features.py:57 caption_text` have zero references; `exclude.py:229 is_excluded`,
      `masking/_sam3.py:183 add_prompt_embed_arg` and `masking/cli/merge_masks.py:11 DEFAULT_INPUTS`
      are referenced only by their own export line. `frontend/src/components/Report.tsx` (97 lines)
      is imported nowhere. `ocr/engine.py crop_quad`, `ocr/sfx.py read_raw`/`read`/`read_boxes`, and
      `dbv4_backend.py:266 normalization` have no callers.
- [ ] **Stale counts, confirmed.** Three files — root `CLAUDE.md:110`, `stages/CLAUDE.md:41`,
      `gui/CLAUDE.md:15` — say the registry lists "eleven" stages. It has listed ten since
      `9413178` removed the ComicTextDetector text-mask stage, a commit that *did* edit
      `CLAUDE.md` and still missed all three copies of the number. Likewise `stages/CLAUDE.md:36`
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
- [ ] **Two tests assert on source text rather than behaviour** — `tests/test_ocr.py:260` greps the
      runner with `inspect.getsource` for the exact call string `_vl_engine(req, engine,
      resized_dir, device)`, and `tests/test_registry_requests.py:459` rglobs the package for
      `"--device"`. Both break on any refactor that preserves behaviour.
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
- `gui/server.py:419` lets `PUT /api/settings` write any key, including the `update_check` cache
  that only the server is supposed to own.
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

Grouped so that each block is one coherent sitting. The original blocks 1 (correctness) and 2
(durability) are done — see "Landed" below — and block 3 has lost its first item, so the
numbering starts where the work does.

1. **Close the pairs.** Grouping onto `walk_images`; `Field` onto `Arg`; move the prompt-embed
   helpers so the masking seam is one-way; lighten `requests.py` so the registry's laziness
   survives resolution. (The progress helper is done: `_progress.progress_line` / `ProgressBar`.)
2. **Split the big files.** `server.py` into routers, `run.py` into per-stage runners,
   `tagger.py` into schema/fetch/model, `downloads.py` and `exclude.py` into catalog/engine/CLI.
3. **Make the guards real.** Move the ruff rules into `pyproject.toml` and let `ruff check` gate
   CI; add a Windows job; generate or test `types.ts` against `schema()`; test the documented
   counts against `len(STAGES)` and `len(__all__)`.
4. **Cleanup.** Share the test fixtures, delete the dead symbols, fix the stale prose and the
   README's four wrong claims in one sweep.

Two items sit outside that order because they are one-line changes with outsized downside: move
`ANIME_TOOLS_HOME` out of the checkout before the next `git clean`, and rename the guidebooks to
ASCII.

Block 1 is where the maintainability return is. Blocks 2 through 4 are safe to do incrementally,
and the cleanup is much less likely to regress once block 1 has removed the second
implementations.

## Landed, and what the trainer owes

The twelve items that were under "Confirmed defects" — three P0, nine P1 — were fixed on
2026-09-12 and removed from this file. What they were, so a reader of the git history can find
them:

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
