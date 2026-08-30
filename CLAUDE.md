# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`anime_tools` is the dataset-curation half split out of the `anima_lora` trainer: caption grammar + correction, the Anima Tagger (dbv4), position clauses, multiview audit, PE-Spatial grouping, SAM3/MIT masking. It is consumed as a git dependency (no PyPI). **Dependency direction is one-way: the trainer imports this package; this package never imports the trainer** (`library.*`, `networks`, `train`, `gui`, `scripts`, `bench`) — `tests/test_boundary.py` greps every module for that, and also asserts `anime_tools.captions.*` + `tagger.dbv4_meta` import without pulling `torch`.

`docs/contract.md` is the frozen `anime_tools` ↔ `anima_lora` contract (file formats, grammar, seams). Changing any row there is a two-repo change by design — don't edit formats casually.

## Commands

```bash
uv sync                       # torch/sam3 are plain dependencies (no extras since 0.3); GPU torch comes from PyPI on Linux, use --index https://download.pytorch.org/whl/cpu for CPU
make install                  # scripts/ensure_bun.sh (bun for the frontend build) + uv sync
make gui                      # anime-tools-gui dev server, opens the browser (GUI_HOST / GUI_PORT / GUI_ARGS)
uv run pytest -q              # tests/ ; CPU-only unless ANIMA_TEST_GPU=1 (conftest blanks CUDA_VISIBLE_DEVICES)
uv run pytest -q tests/test_position_captions.py -k "layout"   # single file / test
uv run pytest -q -n auto      # pytest-xdist is in the dev group
uv run ruff check . && uv run ruff format --check .   # no ruff config in pyproject; rules come from the user-level config. ~37 remaining findings (shebangs, blind-except in best-effort fallbacks, B023 false positives) are deliberate
```

Python >= 3.13. No extras — everything (torch, sam3, timm, fastapi) is a plain dependency. `[tool.uv] override-dependencies = ["numpy>=2.0"]` deliberately overrides sam3's stale `numpy<2` pin. `onnxruntime` is intentionally not pinned (gpu/cpu wheels conflict; the CTD text-mask gate falls back to `cv2.dnn`).

CLIs are `python -m` modules, e.g. `python -m anime_tools.tagger.cli --mode …`, `python -m anime_tools.stages.cli.position_captions`, `python -m anime_tools.grouping.cli.build_groups --source-dir …`, `python -m anime_tools.masking.cli.generate_masks`. The `make caption-autotag` / `make caption-position` / `make preprocess-*` targets mentioned in the docs and the `captions` skill live in the **trainer repo**, which wraps these CLIs; the Makefile here only has `gui`.

## Architecture

- **`captions/`** (torch-free): the single caption grammar. A caption is `<flat tag bag>. On the left, …. In the …, ….` — periods delimit clauses, commas delimit tags inside one. **Never `split(",")` a caption**; go through `position_clauses.parse_caption` / `compose_caption`. `shuffle.py` owns the `@no-artist` sentinel and Anima-prefix shuffle; `correction.py` + `taxonomy.py`/`tag_rules.py`/`tag_groups.py` do Danbooru-KB correction and bucket ordering; `variants.py` writes the tab-delimited `.variants.txt` sidecar (`v0` = pristine); `index.py` builds `caption_index.json`; `tag_drop_groups.py` implements `--caption_drop_groups`. Gate/group sets for position clauses are **data** in `captions/data/clause_vocabulary.yaml` (loaded by `clause_vocabulary.py`) — retune there, not in Python.
- **`tagger/`**: `AnimaTagger` = vocab/thresholds/optional sidecar head over the external `animetimm/*.dbv4-full` caformer (`dbv4_backend.py`). Checkpoint dir = `config.json`, `vocab.json`, `rules.yaml`, `groups.json`, `thresholds.safetensors`, optional `sidecar.safetensors`; GPL backbone weights are fetched at load via `_hf.py` under the user's HF token, never vendored. `tagger/cli/autotag.py` is single-image/stdout only — batch tagging is `stages/autotag.py`.
- **`stages/`**: caption-master stages and their thin CLIs — `autotag.py` (modes `missing`/`merge`/`overwrite`; only `missing` is non-destructive), `position_captions.py` (SAM3 instances → reading order → mask-blanked crops → tagger → v2 rewrite that *moves* bound tags out of the bag; five move rules + four gates, see `docs/position_captions.md`), `captions.py` (mirror master → `post_image_dataset/resized/` with correction + variants, re-attaching clauses), `multiview_audit.py`. Stages are dry-run by default and write `report.json`; `--apply` writes for real.
- **`grouping/`**: `features.Embedder` protocol (`cls[B,D]` f32 L2-normed + `grid16[B,16,16,D]` f16); default embedder is PE-Spatial-B16-512 from the vendored tower in **`vision/pe.py`** (`load_pe_spatial()`, Hub fetch). Feature cache lives at `$NEAR_TWIN_CACHE` (default `~/.cache/near_twin/`) and is curation-private. Output `groups.json` is `MANIFEST_VERSION = 2`.
- **`masking/`**: SAM3 subject masks, MIT/ComicTextDetector text masks, merge → 8-bit L `{stem}_mask.png` mirroring the source subdir.
- **`comfyui/anima_tagger/`** (not part of the installed package — `packages.find` only includes `anime_tools*`): the Anima Tagger ComfyUI node (`AnimaTaggerLoader` + `AnimaTaggerCaption`, `ANIMA_TAGGER` socket). Imports `anime_tools.tagger.AnimaTagger` from the installed package, vendors nothing; users link/copy the directory into `custom_nodes/`. Its `pyproject.toml` is the Comfy-registry manifest (`comfy node publish` runs from that directory). Moved here from the standalone `ComfyUI-Anima-Tagger` repo 2026-08-30.
- **Shared infra** (tiny copies, not trainer imports): `_env.py` (`curation_home()` = `ANIME_TOOLS_HOME` → `ANIMA_HOME` → CWD; `models_dir()` = `ANIME_TOOLS_MODELS` → `<home>/models`; `resolve_path` anchors bare relatives), `_walk.py`, `_hf.py` (tests patch this path), `path_filter.py::filter_paths_by_glob` (the one `path_pattern` implementation).

Where things are written matters: autotag/position/correction stages write the **derived** caption under `post_image_dataset/resized/`; the hand-written master under `image_dataset/` is only read as fallback (autotag `missing` is the exception that creates masters). Any `--apply` that touches captions must be followed by the trainer's TE re-encode.

## Working on captions

Load the `captions` skill (`.claude/skills/captions/`) before parsing/editing captions or touching `captions/` / `stages/` code — it carries the grammar rules, autotag modes, and the position-clause move rules/gates in detail. Docs: `docs/anima_tagger.md`, `docs/position_captions.md`, `docs/multiview_audit.md`, `docs/tagger_caformer_backend.md`.
