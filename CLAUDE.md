# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

`anime_tools` is the dataset-curation half split out of the `anima_lora` trainer: caption grammar +
correction, the Anima Tagger (dbv4), position clauses, multiview audit, PE-Spatial grouping,
SAM3/MIT masking, PP-OCRv6 text recognition, and a web GUI over all of it. It is consumed as a git
dependency (no PyPI).

**The dependency direction is one-way: the trainer imports this package, never the reverse.**
No module here may import `library.*`, `networks`, `train`, `gui`, `scripts`, `bench` —
`tests/test_boundary.py` greps for it. That test also pins that `anime_tools.captions.*`,
`tagger.dbv4_meta` and `gui.create_app()` work without `torch` in `sys.modules`.

`docs/contract.md` describes the `anime_tools` ↔ `anima_lora` seam (file formats, grammar, shared
code). Changing a format there means changing both repos, so check before editing one.

## Commands

```bash
uv sync                       # torch/sam3 are plain dependencies (no extras); on CPU add
                              #   --index https://download.pytorch.org/whl/cpu
make install                  # bun (frontend bundler) + uv sync + git hooks
make gui                      # anime-tools-gui dev server (GUI_HOST / GUI_PORT / GUI_ARGS)
make frontend                 # rebuild the committed anime_tools/gui/static/ bundle; CI fails on drift
uv run pytest -q              # CPU-only unless ANIMA_TEST_GPU=1
uv run ruff check . && uv run ruff format --check .   # keep clean; config is user-level, not in pyproject
python3 scripts/wrap_md.py **/*.md                    # semantic-wrap markdown at 100 cols (--check to report)
```

Python >= 3.13. `[tool.uv] override-dependencies = ["numpy>=2.0"]` overrides sam3's stale `numpy<2`
pin. `onnxruntime` is a plain dependency split by marker — `onnxruntime` on macOS,
`onnxruntime-gpu` everywhere else — because the two are the same import from two conflicting
distributions and OCR (unlike the CTD text-mask gate, which falls back to `cv2.dnn`) has no
fallback.

`make hooks` points `core.hooksPath` at `scripts/hooks`. Pre-commit formats staged files only
(`ruff check --fix --exit-zero` + `ruff format`, prettier on `frontend/`, `scripts/wrap_md.py` on
`.md`) and never blocks the commit. It re-stages in place, so a file with unstaged edits gets those
in the commit too — the hook names them on the way past. Bypass with `--no-verify`.

`scripts/wrap_md.py` only ever *splits* a line over 100 columns, never joins two, and breaks on
sentence/clause boundaries rather than at the column — a greedy fill re-wraps everything below an
edited sentence, which is the churn it exists to stop. Fenced code, tables, headings, quotes, link
definitions and unbreakable tokens (long URLs) are skipped. `tests/test_doc_width.py` asserts the
fixpoint (`wrap_text(t) == t`), not a width.

CLIs are `python -m` modules: `anime_tools.tagger.cli`, `anime_tools.stages.cli.*`,
`anime_tools.grouping.cli.*`, `anime_tools.masking.cli.*`, `anime_tools.downloads`. The
`make caption-*` / `make preprocess-*` targets in the docs live in the **trainer** repo, which wraps
these CLIs; the Makefile here only has dev targets.

## Where things get written

The tools write `workspace/`; **Export** publishes from there to the trainer's paths.
`anime_tools/workspace/__init__.py` is the layout (`DEFAULT_ROOTS`, `RESIZED`, `MASKS`, `REPORTS`,
`GROUPS`, `OUTPUT_ROOTS`, `EXPORT_ROOTS`) and every CLI default is written in terms of it, so the
CLI and GUI halves can't drift.

`resize` populates `workspace/resized/`, and **every stage that opens an image reads that tree** —
masking and grouping included, so there is one geometry in the pipeline. An image only in the master
tree is invisible to the rest, which is why the GUI runs resize as an automatic preflight. Two
consequences are pinned by tests: the near-twin feature cache needs its `(size, mtime_ns)` stamp
because resize rewrites files under a key that doesn't move, and `resize`'s `min_pixels` skip means
"invisible to the pipeline", so it names each dropped file rather than counting it.

Caption stages write the **revised** caption under `workspace/resized/` and read it first — the
correction pass included, which corrects it in place; the hand-written master under
`image_dataset/` is a read-only fallback for an image that has no revised caption yet.
Export, the GUI's caption editor and the multiview
audit's `--apply` (which adds `multiple views` to the master, report holding the before-text) are
the only writers of `image_dataset/`. Each write pushes the replaced text
onto `{stem}.history.txt`, which is what makes a run
safe without an Apply gate: the old version is a badge in the panel and Undo replays the report
backwards.

`python -m anime_tools.workspace.migrate` moves a pre-workspace tree over. Any `--apply` that
touches captions must be followed by the trainer's TE re-encode.

## Architecture

### `captions/` (torch-free)

The single caption grammar: `<flat tag bag>. On the left, …. In the …, ….` — periods delimit
clauses, commas delimit tags inside one. **Never `split(",")` a caption**; go through
`position_clauses.parse_caption` / `compose_caption`.

- `taxonomy.py` is the one tag-*shape* vocabulary (pure stdlib): `normalize_tag` is the key every
  "does the caption already say this?" comparison uses, so two danbooru spellings of a tag can never
  read as two tags. `tests/test_tag_taxonomy.py` greps `stages/` for the bare `.lower()` that should
  be this call. Also `is_count_tag`/`count_of`/`exact_count` (the one count regex) and
  `SINGLE_COUNT_NAMES`/`is_solo_names`/`solo_multi_indices` (the `softmax_when_solo` predicate).
- `vocab_io.py` is the one reader of a checkpoint's `vocab.json`.
- `_sidecar.py` is the one tab-delimited sidecar format (`sidecar_header` / `sidecar_path` /
  `read_rows` / `write_rows`) — the multi-dot-stem rule (`with_name`, not `with_suffix`) and
  hand-edit tolerance (blank, `#`, wrong-arity lines skipped) live there. Three sidecars sit on it:
  `variants.py` (`.variants.txt`, `v0` = pristine), `history.py` (`.history.txt`, capped at
  `HISTORY_LIMIT`, sequences never renumbered), `ocr_sidecar.py` (`.ocr.txt`). OCR is not a caption
  — it names words *in the picture*, so nothing downstream encodes it and it's absent from the
  contract.
- `correction.py` + `taxonomy.py` / `tag_rules.py` / `tag_groups.py` do Danbooru-KB correction and
  bucket ordering; `index.py` builds `caption_index.json`; `shuffle.py` owns the `@no-artist`
  sentinel and Anima-prefix shuffle.
- Gate/group sets for position clauses are **data** in `captions/data/clause_vocabulary.yaml` —
  retune there, not in Python.

### `tagger/`

`AnimaTagger` = vocab/thresholds/optional sidecar head over the external `animetimm/*.dbv4-full`
caformer. Checkpoint dir = `config.json`, `vocab.json`, `rules.yaml`, `groups.json`,
`thresholds.safetensors`, optional `sidecar.safetensors`; GPL backbone weights are fetched at load
via `_hf.py` under the user's HF token, never vendored.

`data.py::TaggerCheckpoint.from_dir` is the one read of a checkpoint dir. `feature_cache.py` owns
the dbv4 hidden-state cache (a cache built for another manifest is misaligned row-for-row, so every
reader checks the stem list in the safetensors metadata). `derive_groups.derive_from_args` +
`write_merged_groups` are the one groups-derivation path, shared by `--mode derive_groups --apply`
and `build_vocab --derive_groups`. `tagger/cli/autotag.py` is single-image/stdout only — batch
tagging is `stages/autotag.py`.

### `stages/`

Caption-master stages and their thin CLIs: `resize.py`, `autotag.py` (modes
`missing`/`merge`/`overwrite`; only `missing` is non-destructive), `position_captions.py` (SAM3
instances → reading order → mask-blanked crops → tagger → clause rewrite; see
`docs/position_captions.md`), `captions.py`, `multiview_audit.py`, `ocr.py`,
`export_workspace.py`.

**The surface is a request object per stage** (`stages/requests.py`, torch-free): `ResizeRequest`,
`AutotagRequest`, `PositionRequest`, `CorrectRequest`, `OcrRequest`, `AuditRequest`,
`ExportRequest`,
run by `stages/run.py::run_<stage>(req)`, which is the old CLI main minus the parsing (preflight,
model load, the library call, `report.json`, the printed epilogue). Same base as masking's
(`anime_tools/_request.py`), with two differences: flags are spelled with underscores
(`FLAG_SEP = "_"`), and a `store_false` switch names its one flag in `off` metadata
(`skip_en` is `--keep_en`). **The parser is generated from the class** (`Request.parser()` →
`_request.build_parser`): every field is declared through `arg(default, help=…, group=…, gate=…,
choices=…)`, the class docstring is the `--help` description, and every flag with a separator
takes the other spelling as an alias (`--path_pattern` / `--path-pattern`). The CLIs in `cli/` are
one-line shells (`build_parser()` = `Request.parser()`, `main()` = `run_<stage>(from_argv())`).
The SAM3 detection flags are one nested `DetectionRequest` (`GROUP = "detection"`) shared by
`PositionRequest` and `AuditRequest`; `.options()` on either builds the `PositionCaptionOptions`
field by field, so an option field with no request field is an error, not a silent default. The
audit pins `min_instances=2` there rather than exposing it. Validation lives in `__post_init__`
(autotag mode, `--flatten` vs `--from_report`, the randomize tokenizers, resize tiers); a missing
input tree is a `FileNotFoundError` the shell turns into `SystemExit`. `stages/__init__.py` exposes
all fifteen names lazily; `tests/test_registry_requests.py` round-trips every registered stage's
request through its
parser and imports the request half torch-poisoned; `tests/test_stage_requests.py` keeps the
stage-specific pins.

**`stages/registry.py`** is the stage list — `Stage(id, title, request="module:Class",
run="module:function", module, panel, …)` for all eleven stages, masking and grouping included —
resolved lazily (`Stage.request_class()`, `Stage.runner()`), so the GUI server and the trainer can
enumerate stages without importing one, and a driver can go from a stage id to the in-process
`run_<stage>(req)` call without naming a runner.

`stages/_models.py::load_anima_tagger` caches the tagger per `(checkpoint dir, device)`, so
autotag followed by position in one process loads it once; `stages/detector.py::build_detect_fn`
builds the SAM3 detector from a `DetectionRequest` (the A/B, review and probe CLIs share it).

**Export is the only thing that writes outside the workspace.** Six artifact kinds
(`image`/`caption`/`variants`/`mask`/`master`/`index`), each decided against the destination
(`identical` by byte compare for text, `(size, mtime_ns)` for pixels). It always copies, takes no
`--from_report`, and `revert_export` restores text it overwrote — an overwritten pixel reports
`not-undoable`.

Stages are dry-run by default **from the CLI** and write `report.json`; `--apply` writes for real.
The GUI always passes it.

Shared scaffolding (`tests/test_registry_requests.py` pins one spelling per shared flag, since the
GUI fills `--path_pattern` / `--tagger_dir` / `--checkpoint` / `--prompt_embed` from one Settings
value each):

- `cli/_args.py::make_progress` (the `  [done/total] detail` line the GUI's progress bar parses;
  under the trainer's daemon the same callback also streams every call to the job's
  `progress.jsonl` through `_progress.py`),
  `cli/_report.py`, `replay.run_replay_cli` (reads `from_report` / `path_pattern` / `apply` off the
  request).
- `_caption_io.py` — `read_caption`/`write_caption`, the trailing-newline invariant, the
  `.variants.txt` drop, and `history_by`.
- `_walk_captions.py` — `resolve_caption`/`iter_captions`: revised caption first, master as
  read-only fallback. `autotag` walks images rather than captions, so it calls `resolve_caption`
  itself; `ab_position_captions` deliberately reads the master only.
- `replay.apply_one` — the one drift-guarded write: `no-proposal` → `missing-caption` →
  `already-applied` → `drifted` → `would-write`/`written`.

### `grouping/`

`features.Embedder` protocol (`cls[B,D]` f32 L2-normed + `grid16[B,16,16,D]` f16); default embedder
is PE-Spatial-B16-512 from the vendored tower in `vision/pe.py`. Feature cache at
`$NEAR_TWIN_CACHE` (default `~/.cache/near_twin/`) is curation-private, keyed by parent-dir hash +
stem and stamped with `(size, mtime_ns)` + `FEATURE_CACHE_VER`; anything wrong with an entry means
recompute, never an error. `cli/match_decensored.py` has a different cache shape and they are
deliberately not unified.

`features.read_tags` is the one caption read on this side (through `parse_caption`, keyed by
`normalize_tag`). The surface is `GroupRequest` (`grouping/requests.py`, torch-free, hyphenated
flags) run by `groups.run_groups`, which resolves `--embedder`'s `module:callable` and calls
`build_groups`; `--source-dir` defaults to `workspace/resized/`. Output `groups.json` is
`MANIFEST_VERSION = 2`.

### `masking/`

SAM3 subject masks, SAM3/UNet++/ComicTextDetector text masks, merged into 8-bit L
`{stem}_mask.png` mirroring the source subdir.

**The surface is a request object per stage** (`requests.py`, torch-free): `SamMaskRequest`,
`MitMaskRequest`, `MergeMasksRequest`, run by `sam.py::run_sam_masks`, `mit.py::run_mit_masks`,
`merge.py::run_merge_masks`. Each field is one flag of the matching CLI, whose parser is
generated from the class (`Request.parser()`, hyphenated: `FLAG_SEP = "-"`, with the underscore
spelling as an alias);
`to_argv()` / `from_namespace()` come from `anime_tools/_request.py` and are inverses over it
(`tests/test_registry_requests.py` round-trips every one, and a default argv must read back as a
default request). Validation lives in `__post_init__`; the CLIs in `cli/` are shells that parse,
build the request, and turn its `ValueError` into `parser.error`. `load_sam3` caches
per process on its arguments, so a text-mask pass after a subject-mask pass reuses the model.
`masking/__init__.py` exposes all six names lazily. Two private cores:

- `_sam3.py` is the **only** place SAM3 is constructed, the one declaration of `--checkpoint` /
  `--prompt_embed`, and the home of `ground_with_soft_prompt` (a soft prompt *is* the text encoder's
  output, so the encode is skipped), `prompt_list` (`none`/`off` = no prompts) and `detect_union`.
  It installs the `np.bool` alias sam3 needs as an import side effect, with sam3 imports deferred
  into functions so importing it stays torch-free. Two more shims run inside those functions:
  `stub_edt_kernel` pre-seeds `sam3.model.edt` (the one module that imports triton, which has no
  macOS build) with a stand-in that refuses to run, and `shim_sam3_for_cpu` redirects the image
  model's two build-time `"cuda"` literals to CPU when torch has none. Neither fakes `triton`
  itself: torch guards its own import of it and would take a fake one for real.
- `_masks.py` owns the mask layout — `plan_mask_jobs`, `write_mask`/`write_ignore_mask`
  (`detected=1 → alpha=0`), the read side (`mask_name`/`mask_path_for`/`iter_masks`), and
  `mask_run`, the scaffolding both generators wrap their inner loop in (it reads the walk fields
  of `MaskWalkRequest` by attribute). It deliberately does **not** import `_sam3`, because
  `gui/dataset.py` imports `mask_name` and would otherwise drag in that side effect.

The subject-mask CLI takes prompts, not a config: `--prompts` (masked out) / `--focus-prompts` (keep
only, default `girl`) / `--prompt_embed`. The text-mask CLI is two detectors over one walk, each
behind its own switch, unioned before the single dilation: `--use-sam` grounds SAM3 on
`--sam-prompts`, `--use-mit` runs the UNet++ segmenter. They answer different questions (a balloon
is a shape, a letter is a stroke), neither subsumes the other, and both being off is the one argv
the stage refuses.

Each switch is a **drawer**: the switch field carries `gate=<its own name>` plus the drawer's
`group` title, and every knob inside it carries `gate=<the switch>` (`MitMaskRequest`). The GUI
folds a shut drawer's knobs away and drops them from the argv; the generated parser also stamps the
argparse group with `contract.GATE_ATTR` for anyone introspecting it directly.
`tests/test_masking_plan.py` pins the shape.

The three mask directories are **one ⚙ Settings value, not three form fields**: both generators name
a mask `{stem}_mask.png` at the same relative path, so a shared directory would have the second run
overwrite the first and leave the merge one tree to union. Each `--mask-dir` is its own tree, all
three hanging off `MASK_SETTING` (`mask_root`).

### `gui/` (torch-free)

The `anime-tools-gui` web panel. `frontend/CLAUDE.md` owns the browser half; this section is the
server side of the same seam.

**`stages.py` turns each stage's request dataclass into a form schema and a form payload back
into the request's argv.** The registry is `stages/registry.py` (re-exported); `schema()` walks
`_request.args_of(Request)` — the same field list the CLI parser is generated from, so a flag's
kind, default, help, group and drawer reach the form without argparse in between — and
`build_argv()` coerces the payload into a namespace, reads it with `Request.from_namespace` (so the
request's own validation runs server-side, as a 400) and spells it with `to_argv()`. Both happen
in-process: the request modules are torch-free by test, and building all eleven schemas takes
~0.1 s, so there is no child interpreter and no cache. Field binding:

| Map | Bound to | Shown? |
|---|---|---|
| `ROOT_FIELDS` (`--src`, `--dst`, `--image-dir`, …) | dataset roots | hidden |
| `SETTING_FIELDS` (`--path_pattern`, `--tagger_dir`, `--checkpoint`, `--prompt_embed`) | Settings stage defaults | hidden |
| `REPORT_SETTING` / `REPORT_INPUTS` (`--report_dir`, groups' `--out`, Export's `--index`) | `report_root` + the stage's own tail | hidden |
| `MASK_FIELDS` | `mask_root` + tail | hidden |
| `PANEL_FIELDS` (Export's `--out`, `--index`) | as above, but a per-run choice | **shown** |
| `AUTO_FIELDS` (`--device`) | resolved in the child by `_device.resolve_device` | neither shown nor sent |
| `BASIC_FIELDS` | — | shown; everything else in that stage folds under `advanced (n)` |

Report and mask roots are split per-stage (`report_subpath` / `mask_subpath`) so one stage's
`--from_report` can't read another's report, and so the two generators don't overwrite each other. A
blank root means *beside* the relevant dataset root (`_root_beside`). A drawer's gate, a required
field and an already-hidden field are never folded — the server settles that, not the browser.

Other server pieces:

- `dataset.py` joins the trees (`src`/`dst`/`masks`/`master`/`out`) into the sidebar's image→caption
  tree by relative path, reads/writes single captions, renders thumbnails. Only `master` and
  `revised` are writable. An image's captions are a **ladder** (`CAPTION_LADDER`, one `Rung` per
  caption kind), shipped to the browser as per-row dot flags and as `caption_versions`' ordered
  list,
  where a sidecar rung expands into one entry per caption it holds (`v0`, `v1`, `revised@2`…). The
  sidebar draws one listing in two orderings, `tree` and `groups`; `load_groups` reads
  `<report_root>/<GROUPS_SUBPATH>` and answers **rels only**, so filters and pending dots mean the
  same thing in both modes. A missing groups manifest is not an error; an unparseable one is a 400.
- **The browser never splits a caption**: clause structure and every tag's `[start, end)` come from
  `/api/dataset/item` and `/api/dataset/parse` via `position_clauses.tag_spans`, which is what lets
  the editor stay a real `<textarea>` with boxes painted behind it.
- `proposals.py` is `stages/replay.py` seen from the server: `load_report` / `report_rows` /
  `apply_one` are imported; an Undo is `apply_one` with the two texts swapped, and Export branches
  to
  `revert_export` at the top. `proposals.SHAPES` is `contract.REPLAY_SHAPES`, the same objects the
  three stage CLIs bind as their `REPLAY_SPEC` (importing a stage CLI would pull torch in);
  `tests/test_gui_proposals.py` pins the identity.
- `jobs.py` runs one `python -m` subprocess at a time over SSE. A job is a *sequence* of `Step`s
  sharing one slot, log and stream, because `preprocess_for()` puts `resize` in front of every stage
  bound to the `dst` root; a failing step stops the chain. `masks_merge` and the `NO_PREFLIGHT`
  names
  sit outside it.
  A running stage tells the browser nothing but its stdout, so the panel's progress bar and log
  window are read straight off it — `stages/cli/_args.py::make_progress`'s `  [done/total] detail`
  and the `── step i/n: label ──` header this module prints in front of each step of a sequence are
  the two formats parsed there, and a stage printing neither simply has no bar.
- `tags.py` merges the two Danbooru KB files (base CSV = taxonomy; optional `.en.csv` replaces only
  the description) for `/api/tags/describe`, cached on both mtimes; a missing KB answers
  `installed: false` rather than erroring.
- `nativepick.py` opens the *host's* file chooser (zenity/kdialog/osascript/PowerShell) as a
  subprocess for `POST /api/pick`. `/api/ls` is the fallback for headless or remote browsers.
- What the panel may **read** is `dataset.dataset_bases()`: the curation home plus any root the
  *saved* settings pin outside it (`reachable()`, lexical). What it may **create** is narrower —
  `owned()`, under the home only — so a typo in an external root is a missing root, not a new empty
  directory.
- ⚙ Settings is **three dialogs, not one tabbed one** (`SETTINGS_PANES`): roots, stage defaults +
  preflight, models. Only the open pane is mounted, so `SettingsOut` carries `null` for the other
  two.
- The panel's own chrome is translated (`frontend/src/i18n/`), and so is the dock's navigation —
  the panel buttons and the stage names on them, keyed by this registry's own ids. Everything else
  the server owns (a stage's doc and notes, argparse labels and help, the model catalog) ships as it
  arrives.

### `downloads.py` (torch-free)

The model catalog — one `Asset` per checkpoint (tagger + gated dbv4 backbone, SAM3, PE-Spatial, MIT
text net, ComicTextDetector, SAM3 subject soft prompt, Danbooru tag KB) with repo, files,
destination and an offline `installed` probe. It is the **single source of truth for weight
locations**: `vision/pe.py`, `masking/mit.py` and `_sam3`'s flag defaults import
theirs from here, and `default_ctd_onnx_path()` has no flag at all — a path you could point
elsewhere is a Download button that writes where the loader doesn't look.

Most rows are HF-hub fetches; the soft prompt, the CTD net and the tag KB go through
`Asset._fetch_http`. `danbooru_tags_en` is the one `derived` row: it *builds* its CSV from the
Danbooru wiki mirror, so its probe asks for the file it writes and the 45 MB parquet stays in the
hub
cache. `python -m anime_tools.downloads [ID…]` fetches; the GUI's Models pane runs exactly that.

### Smaller pieces

- **`buckets.py`** (torch-free, numpy-free): the free-fit token-band geometry, a copy of the
  trainer's. `stages/resize.py` must land an image on the *same* `(W, H)` as the trainer's
  `make preprocess-resize` or each side re-encodes the other's PNGs; `tests/test_resize_images.py`
  pins the numbers and the `anima_resize_*` PNG text keys.
- **`contract.py`** (stdlib-only leaf, pinned by `test_contract_is_torch_free`): the constants both
  sides of the seam spell — autotag stdio sentinels and modes, tagger checkpoint file sets,
  `REPLAY_REPORT_NAME`, `GATE_ATTR` (the stamp the generated parser leaves on a drawer's argparse
  group), `ReplaySpec` + `REPLAY_SHAPES`, `CONTRACT_VERSION`. Anything the GUI server or the
  trainer needs without importing a stage goes here; the stage re-exports it.
- **Shared infra** (tiny copies, not trainer imports): `_env.py` (`curation_home()` =
  `ANIME_TOOLS_HOME` → `ANIMA_HOME` → CWD; `models_dir()`; `workspace_dir()`; `resolve_path`),
  `_walk.py` (the one image walk — `IMAGE_EXTENSIONS` / `glob_images_pathlib` / `walk_images`),
  `_json.py` (UTF-8 both ways, `ensure_ascii=False`, `indent=2` — a bare `open()` reads in the
  platform codepage, which isn't UTF-8 on Windows), `_device.py` (`DEVICE_HELP` for the request
  fields, `add_device_arg` for the hand-written CLIs, and the one `cuda if available` probe; the
  flag literal exists once, pinned by `tests/test_registry_requests.py`),
  `_hf.py` (tests patch this path), `path_filter.py` (the one `path_pattern` implementation),
  `_progress.py` (stdlib: with `ANIMA_DAEMON_JOB_DIR` set, `step()` appends the daemon's own
  `{"ev": "step", "global_step", "total_steps", "detail"}` line to `<job_dir>/progress.jsonl` and
  `phase(name)` brackets a model load with a 30 s heartbeat so the daemon's stall watchdog does
  not kill a quiet SAM3/tagger/OCR/embedder load; every loader and progress callback goes through
  it, and without the variable it is a no-op).
- **`comfyui/anima_tagger/`** (not installed — `packages.find` only includes `anime_tools*`): the
  ComfyUI node. Imports `AnimaTagger` plus `ensure_tagger_checkpoint` and the `dbv4_meta` constants
  from the installed package and vendors nothing, so the gated-backbone fetch happens in the loader
  node rather than mid-predict. What stays local is the ComfyUI shell.
- **`design/`** (not installed, no runtime role): the GUI's design system as a published Claude
  Design canvas. Every value on the boards is lifted from `frontend/src/styles.css` — a number that
  drifts from the source is a bug. `boards/<Name>.html` + `canvas.json`, built by `design/build.py`;
  see `design/README.md`. Edits made in the published editor do **not** flow back.

## Working on captions

Load the `captions` skill (`.claude/skills/captions/`) before parsing/editing captions or touching
`captions/` / `stages/` code — it carries the grammar rules, autotag modes, and the position-clause
move rules/gates in detail. Docs: `docs/anima_tagger.md`, `docs/position_captions.md`,
`docs/multiview_audit.md`.
