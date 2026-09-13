# CLAUDE.md

Guidance for Claude Code when working in this repository. This file is the part that applies
from any directory: what the package is, the commands, the workspace layout, and the invariants
that bite everywhere. Each package under `anime_tools/` carries its own `CLAUDE.md` with that
module's architecture, and procedures live in `.claude/skills/`; the map is at the bottom.

## What this is

`anime_tools` is the dataset-curation half split out of the `anima_lora` trainer: caption grammar +
correction, the Anima Tagger (dbv4), position clauses, multiview audit, PE-Spatial grouping,
SAM3 masking, OCR (AnimeText detector + manga VL reader), and a web GUI over all of it.
It is consumed as a git
dependency (no PyPI).

**The dependency direction is one-way: the trainer imports this package, never the reverse.**
No module here may import `library.*`, `networks`, `train`, `gui`, `scripts`, `bench` —
`tests/test_boundary.py` greps for it. That test also pins that `anime_tools.captions.*`,
`tagger.dbv4_meta` and `gui.create_app()` work without `torch` in `sys.modules`.

The `anime_tools` ↔ `anima_lora` seam is a shared surface — file formats, the caption grammar, and
the code both sides spell (`contract.py`, `buckets.py`) — so changing one of them means changing
both repos in step.

## Commands

```bash
uv sync                       # torch/sam3 are plain dependencies (no extras); on CPU add
                              #   --index https://download.pytorch.org/whl/cpu
                              # Windows: the default `cuda-windows` group binds torch to cu132
make install                  # bun (frontend bundler) + uv sync + git hooks
make gui                      # anime-tools-gui dev server (GUI_HOST / GUI_PORT / GUI_ARGS)
make frontend                 # rebuild the committed anime_tools/gui/static/ bundle; CI fails on drift
uv run pytest -q              # CPU-only unless ANIMA_TEST_GPU=1
uv run ruff check . && uv run ruff format --check .   # keep clean; config is user-level, not in pyproject
python3 scripts/wrap_md.py **/*.md                    # semantic-wrap markdown at 100 cols (--check to report)
```

Python >= 3.13. `[tool.uv] override-dependencies = ["numpy>=2.0"]` overrides sam3's stale `numpy<2`
pin. **Every model in this package runs on torch** — onnxruntime, `onnx` and `onnxscript` were
dropped 2026-09-09 along with the tagger's exported graph, and the AnimeText detector moved onto
the vendored `vision/yolo12.py`; the `model-catalog` skill has the why.

`make hooks` points `core.hooksPath` at `scripts/hooks`. Pre-commit formats staged files only
(`ruff check --fix --exit-zero` + `ruff format`, prettier on `frontend/`, `scripts/wrap_md.py` on
`.md`) and never blocks the commit. It re-stages in place, so a file with unstaged edits gets those
in the commit too — the hook names them on the way past. Bypass with `--no-verify`.

`scripts/wrap_md.py` only ever splits a line over 100 columns, never joins two, and breaks on
sentence/clause boundaries rather than at the column — a greedy fill re-wraps everything below an
edited sentence, which is the churn it exists to stop. Fenced code, tables, headings, quotes, link
definitions and unbreakable tokens (long URLs) are skipped. `tests/test_doc_width.py` asserts the
fixpoint (`wrap_text(t) == t`), not a width, over every tracked `.md` — these files included.

CLIs are `python -m` modules: `anime_tools.tagger.cli`, `anime_tools.stages.cli.*`,
`anime_tools.grouping.cli.*`, `anime_tools.masking.cli.*`, `anime_tools.downloads`. The
`make caption-*` / `make preprocess-*` targets in the docs live in the trainer repo, which wraps
these CLIs; the Makefile here only has dev targets.

## Where things get written

The tools write `workspace/`; Export publishes from there to the trainer's paths.
`anime_tools/workspace/__init__.py` is the layout (`DEFAULT_ROOTS`, `RESIZED`, `MASKS`, `REPORTS`,
`GROUPS`, `OUTPUT_ROOTS`, `EXPORT_ROOTS`) and every CLI default is written in terms of it, so the
CLI and GUI halves can't drift.

`resize` populates `workspace/resized/`, and every stage that opens an image reads that tree —
masking and grouping included, so there is one geometry in the pipeline. An image only in the master
tree is invisible to the rest, which is why the GUI runs resize as an automatic preflight. Two
consequences are pinned by tests: the near-twin feature cache needs its `(size, mtime_ns)` stamp
because resize rewrites files under a key that doesn't move, and `resize`'s `min_pixels` skip means
"invisible to the pipeline", so it names each dropped file rather than counting it.

Caption stages write the revised caption under `workspace/resized/` and read it first —
the correction pass included, which corrects it in place; the hand-written master under the
`src` root is a read-only fallback for an image that has no revised caption yet.
Export, the GUI's caption editor and the multiview audit's `--apply` (which adds `multiple views`
to the master, report holding the before-text) are the only writers of the `src` root. Each
write pushes the replaced text onto `{stem}.history.txt`, which is what makes a run safe without
an Apply gate: the old version is a badge in the panel and Undo replays the report backwards.

Excluding an image is the one gesture that moves files *out* of those trees:
`anime_tools/exclude/` puts every file it has (resized, mask, OCR sidecar) under
`workspace/_excluded/<tree>/<rel>` and writes the rel into `workspace/_excluded/excluded.json`.
The ledger is the state, and `resize` is where it bites — it adds every listed rel to its own
`--skip`, which is the only place an exclusion has to be enforced, since every other stage
walks `workspace/resized/` and the image has left it. Export republishes the whole tree under
`<out>/_excluded/`, beside what the trainer reads rather than inside it. The GUI's ⊘ button and
`python -m anime_tools.exclude` are the two ways in; the source image and its hand-written
master are never touched, so putting one back is the same move reversed.

`python -m anime_tools.workspace.migrate` moves a pre-workspace tree over. Any `--apply` that
touches captions must be followed by the trainer's TE re-encode.

## Invariants

Each of these is implemented in one package but bites from any of them.

- Never `split(",")` a caption. The grammar `<flat tag bag>. On the left, …. In the …, ….`
  has one parser, `captions/position_clauses.py` (`parse_caption` / `compose_caption` /
  `tag_spans`); the browser gets clause structure from the server. `taxonomy.normalize_tag` is
  the key for every "does the caption already say this?" comparison. Details: the `captions`
  skill.
- Export is the only thing that writes outside the workspace, and it always copies; every
  other stage reads `workspace/resized/` and writes under `workspace/`.
- The surface is a request object per stage. Every stage — captions, masking, grouping — is
  a torch-free dataclass over `anime_tools/_request.py` (`arg(default, help=, group=, gate=,
  choices=)`, validation in `__post_init__`, parser generated by `Request.parser()`,
  `to_argv()` / `from_namespace()` inverses), a `run_<stage>(req)` runner, and a one-line CLI
  shell. `stages/registry.py` lists all eleven stages, resolved lazily; the GUI form schema and argv
  come from the same field list. `stages/` spells flags with underscores, `masking/` and
  `grouping/` with hyphens, and either spelling is an alias. A field's default and help come
  from a leaf — `stages/_options.py`, `masking/_prompts.py` — and never from the stage module
  that uses it: resolving a request class is how the GUI builds its form, and reaching into a
  stage for one number costs that build numpy, PIL and yaml. Procedure: the `add-stage` skill.
- Dry-run by default from the CLI; `--apply` writes. Every run leaves `report.json`; the
  GUI always applies and relies on `{stem}.history.txt` plus `replay.apply_one` (the one
  drift-guarded write) for Undo.
- An exclusion is enforced in one place. `exclude/` empties the live trees, but only
  `resize` could refill them, so only `resize` reads the ledger. A stage that learns to
  check it separately is a second answer to the same question — the fix for one that
  processes an excluded image is that it is not walking `workspace/resized/`.
- Torch stays out of the server path. `captions/`, `gui/`, `contract.py`, `downloads/`,
  every `requests.py` and `registry.py` import without torch; model imports live inside runner
  bodies. `tests/test_boundary.py` and `tests/test_registry_requests.py` pin it.
- Weight locations are spelled once, in `downloads/_locations.py`; loaders import their
  defaults from
  it. Procedure: the `model-catalog` skill.
- One device answer, `_device.py::resolve_device`: `cuda`, then `mps`, then `cpu`, and
  whatever `--device` said if it said anything. Every stage's model takes the device its
  runner resolved rather than probing again — the OCR stage's two models are the case
  that made it a rule.
- Progress is stdout, in one format, from one place. `_progress.py::progress_line` prints
  `  [done/total] detail` — what the GUI's bar parses — and forwards the same step to the
  trainer's daemon under `ANIMA_DAEMON_JOB_DIR`. A stage that hands a callback down uses
  `stages/_progress.py::make_progress`; a stage that walks its own loop uses
  `_progress.ProgressBar` (`advance` / `note` / `tick`). No stage uses `tqdm`: its carriage
  returns are noise in the GUI's log and match nothing its bar reads. A stage that prints
  neither has no bar.
- Markdown is wrapped at 100 columns by `scripts/wrap_md.py`; run it on any `.md` you edit.

## Shared infra (stdlib-level leaves, not trainer imports)

- `contract.py` (pinned torch-free): the constants both sides of the seam spell — autotag
  stdio sentinels and modes, tagger checkpoint file sets, `REPLAY_REPORT_NAME`, `GATE_ATTR` (the
  stamp the generated parser leaves on a drawer's argparse group), `ReplaySpec` + `REPLAY_SHAPES`,
  `CONTRACT_VERSION`. Anything the GUI server or the trainer needs without importing a stage goes
  here; the stage re-exports it.
- `buckets.py` (torch-free, numpy-free): the free-fit token-band geometry — the owner
  since 2026-09-03: the trainer's `library/datasets/buckets.py` re-exports these names, so
  `stages/resize.py` and the trainer's `make preprocess-resize` land an image on the same
  `(W, H)` by construction. `tests/test_resize_images.py` pins the numbers and the
  `anima_resize_*` PNG text keys.
- `_env.py` (`curation_home()` = `ANIME_TOOLS_HOME` → `ANIMA_HOME` → CWD; `models_dir()`;
  `workspace_dir()`; `resolve_path`), `_walk.py` (the one image walk — `IMAGE_EXTENSIONS` /
  `glob_images_pathlib` / `walk_images`, and the one `path_pattern` implementation,
  `filter_paths_by_glob`), `_json.py` (UTF-8 both ways, `ensure_ascii=False`,
  `indent=2` — a bare `open()` reads in the platform codepage, which isn't UTF-8 on Windows),
  `_device.py` (`DEVICE_HELP` for the request fields, `add_device_arg` for the hand-written
  CLIs, and the one device probe — `cuda`, then `mps`, then `cpu`; the flag literal exists
  once), `_hf.py` (tests patch this path), `_progress.py` (the printed `  [done/total]` line —
  `progress_line`,
  `ProgressBar` — plus the daemon stream: with `ANIMA_DAEMON_JOB_DIR` set, `step()` appends to
  the daemon's `progress.jsonl` and `phase(name)` brackets a model load with a 30 s heartbeat;
  without the variable that half is a no-op and the line still prints).
- `update.py` (torch-free, stdlib-only): the self-update — what is installed
  (`__version__`), what GitHub's latest release is, which of the three install shapes this
  process runs out of (`uv tool` / checkout / other) and the one `uv tool install --force` that
  moves between them. Only the shape `install.sh` makes is ever rewritten; the GUI runs it as a
  job (`gui/updates.py`, the Update pane), and `make update` is not a target here — the trainer's
  `scripts/update.py` merges a tarball over a working tree, which this package does not have.
- `shortcut.py` (torch-free, stdlib-only): the double-clickable GUI launcher, one file per platform
  — a `.lnk` written through `WScript.Shell`, a `chmod +x` `.command`, or a `Terminal=true`
  `.desktop` — each pinning the folder it lands in as that launcher's curation home, since the GUI
  reads the home from its working directory. `anime-tools-shortcut` is the console script;
  both bootstrap installers run it best-effort in the directory they ran in and parse `launcher:
  <path>` off its stdout for the closing message.
- `comfyui/anima_tagger/` (not installed — `packages.find` only includes `anime_tools*`): the
  ComfyUI node, importing `AnimaTagger` from the installed package and vendoring nothing.
- `design/` (not installed, no runtime role): the GUI's design system as a published Claude
  Design canvas; every value is lifted from `frontend/src/styles.css`, and edits made in the
  published editor do not flow back. See `design/README.md`.

## Map

| Working on | Read first |
|---|---|
| `anime_tools/captions/` — grammar, sidecars, correction, clause rewrite | `anime_tools/captions/CLAUDE.md` + the `captions` skill; `docs/position_captions.md` |
| `anime_tools/stages/` — the eight caption stages, requests, registry, replay, Export | `anime_tools/stages/CLAUDE.md`; `docs/multiview_audit.md` |
| `anime_tools/tagger/` — checkpoint, backends, feature cache, calibration | `anime_tools/tagger/CLAUDE.md`; `docs/anima_tagger.md` |
| `anime_tools/masking/` — SAM3 construction, mask layout, drawers | `anime_tools/masking/CLAUDE.md`; `docs/masking.md` |
| `anime_tools/grouping/` — embedders, feature cache, `groups.json` | `anime_tools/grouping/CLAUDE.md`; `docs/grouping.md` |
| `anime_tools/vision/` — the vendored towers: PE-Spatial (`pe.py`), YOLO12 (`yolo12.py`) | each module's own doc |
| `anime_tools/gui/` — schema/argv binding, dataset ladder, jobs, settings | `anime_tools/gui/CLAUDE.md` |
| `frontend/` — the Solid browser half | `frontend/CLAUDE.md` |
| `anime_tools/ocr/` + `stages/ocr.py` — the AnimeText text-block detector over the resized tree, every box read by `ocr/sfx.py`, the manga VL crop reader (fine-tuned PaddleOCR-VL-1.6, decode guard built in) | `anime_tools/stages/CLAUDE.md`; the sidecar rule in the `captions` skill; `ocr/sfx.py`'s module doc |
| `anime_tools/exclude/` — taking an image out of the pipeline, and its ⊘ button | this file's "Where things get written"; `anime_tools/gui/CLAUDE.md` |
| `anime_tools/downloads/` — adding or moving a weight | the `model-catalog` skill |
| `anime_tools/update.py` — the self-update and its GUI pane | the `release` skill; `anime_tools/gui/CLAUDE.md` |
| A new stage, a renamed flag, a GUI knob | the `add-stage` skill |
| A version bump, the installer, `release.yml` | the `release` skill |
| An edit to `anime_tools/gui/guidebooks/guidebook.md` | the `translator` agent — it re-syncs the ko/ja/zh guidebooks beside it |
| Tests in `tests/` | the nested file of the package under test; `test_registry_requests` and `test_boundary` span all of them |

A nested `CLAUDE.md` loads when you read a file in its directory; a skill loads when you ask for
it. If a task touches a package only through `tests/` or `docs/`, open the nested file yourself.

- Run the test suite at most twice per task (here or in `../anime_tools`): once
  after the change, once after fixing what it caught. Re-running it as a progress
  check is noise — read the failure and fix it. Needing a third run means the change
  wants rethinking, not another loop; if a run is genuinely required beyond that, say
  why. Scope a re-run to the affected file (`pytest tests/test_x.py`) rather than
  sweeping the whole suite again.