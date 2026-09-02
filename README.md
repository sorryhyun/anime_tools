# anime_tools

Dataset curation for anime diffusion training — the half of
[`anima_lora`](https://github.com/sorryhyun/anima_lora) that produces the
caption master and sidecars, split out so it can be used on datasets bound for
*any* trainer and installed without the trainer's DiT/VAE stack.

| Sub-package | What | Install |
|---|---|---|
| `anime_tools.captions` | The caption **grammar** (`parse_caption` / `compose_caption` — never `split(",")` a caption), tag taxonomy + Danbooru-KB correction, `--caption_drop_groups`, shuffle/dropout **variants sidecars**, `caption_index.json` builder | base (torch-free) |
| `anime_tools.tagger` | **Anima Tagger** — a vocab/threshold/sidecar head over the external `animetimm/*.dbv4-full` caformer tagger, emitting Anima-format tags (`rating, count, characters, copyrights, @artists, generals`). CLIs: `python -m anime_tools.tagger.cli --mode …`, `…cli.autotag`, `…cli.autotag_server`, `…cli.train_sidecar` |
| `anime_tools.stages` | Caption stages: batch **autotag**, **position clauses** (SAM3 crops → tagger → v2 rewrite), correction + variants mirror, **multiview audit** |
| `anime_tools.grouping` | Near-twin / same-concept **grouping** on PE-Spatial-B16-512 features (`anime_tools.vision.pe`, weights fetched from the Hub) → `groups.json`; decensor match tools. CLI: `python -m anime_tools.grouping.cli.build_groups --source-dir …` |
| `anime_tools.masking` | Training masks: SAM3 subject masks, MIT / ComicTextDetector text masks, merge. CLIs: `python -m anime_tools.masking.cli.{generate_masks,generate_masks_mit,merge_masks}` |

## Install

One line, no checkout — installs [uv](https://astral.sh/uv) if missing, then
`uv tool install`s the latest release with every extra and puts
`anime-tools-gui` on PATH:

```bash
curl -fsSL https://github.com/sorryhyun/anime_tools/releases/latest/download/install.sh | sh
```

Windows (PowerShell) 

```
irm https://github.com/sorryhyun/anime_tools/releases/latest/download/install.ps1 | iex
```

`ANIME_TOOLS_VERSION=v0.3.1` pins a tag, `TORCH_INDEX=https://download.pytorch.org/whl/cu130` picks
a torch
index (PyPI's Linux wheel is already CUDA; Windows defaults to CPU). Update
with `uv tool upgrade anime-tools`.

As a library dependency (what the trainer does):

```bash
uv add "anime-tools @ git+https://github.com/sorryhyun/anime_tools"   # git dependency; no PyPI
```

The repo is the product (kohya-ss/sd-scripts style): pin a tag, or use a
`[tool.uv.sources]` path override for a live checkout.

sam3 pins `numpy>=1.26,<2` and the pin is stale (see `[tool.uv]` in `pyproject.toml`), but uv
reads `tool.uv` only from the workspace root — a project that *depends* on anime-tools has to
repeat the override in its own `pyproject.toml`, or the resolve fails on numpy:

```toml
[tool.uv]
override-dependencies = ["numpy>=2.0"]
```

The installers above pass the same override on the command line.

## Web GUI

```bash
cd <your dataset folder>   # image_dataset/, workspace/, models/ live here
anime-tools-gui --open     # http://127.0.0.1:8790
```

A small standalone panel on your dataset. The **sidebar is the dataset**: every
source image, and under each one its captions as a ladder — the hand-written
`master`, every version the `revised` caption used to be (`revised@N`), that
caption, then the generated `v0…vN` variants. `master` and `revised` are
editable; every write, by hand or by a stage, keeps what it replaced as a
version badge, and **Undo** puts it back from the run's report. The **stage
runner is the bottom dock**: its buttons are the stage list, the form is
generated from the CLI's own `--help`, and a Run works on the selected image or
the whole batch as one `python -m …` subprocess while the dataset stays on
screen. The ☰ menu holds three Settings dialogs (dataset roots, stage defaults,
models with a Download button and the Hugging Face sign-in) and a language
switch (English, Korean, Japanese, Chinese). `--host 0.0.0.0` exposes it on the
LAN for a headless GPU box (no auth — use your own tunnel), `--home` overrides
the curation home.

The end-to-end walkthrough — install, the curation home, the panel, every stage
in the order you run it, and the hand-off to the trainer — is the
[**guidebook**](docs/guidelines/guidebook.md).

FastAPI + uvicorn are plain dependencies; the trainer's own PySide6 GUI remains
the rich editor.

## Layout of a curated dataset

```
image_dataset/**/{stem}.png + {stem}.txt        caption master   ← hand-written; read-only for the tools
workspace/                                       everything the tools write
  resized/{stem}.{png,txt,variants.txt}            resized image, revised caption, shuffle / dropout variants
  master/{stem}.txt                                revised master (Export publishes it back to image_dataset/)
  masks_sam/ masks_mit/ masks/                     each mask generator's tree, and their merge
  captions/<stage>/report.json  groups/groups.json  ocr/  export/report.json
post_image_dataset/resized/ masks/               what the trainer reads   ← written only by Export
models/captioners/anima-tagger-dbv4/            tagger checkpoint (auto-fetched from sorryhyun/anima-tagger)
models/sam3/  models/pe/  models/mit/  models/ppocr/   SAM3 / PE-Spatial / text-mask / OCR weights
networks/calibration/sam3_girl_prompt.safetensors  SAM3 subject soft prompt (default `--prompt_embed`)
```

`python -m anime_tools.downloads --list` says which weights are present and where each one goes.

Every artifact the trainer reads is a **file**; the formats are frozen in
[`docs/contract.md`](docs/contract.md). Paths resolve against the curation
home: `ANIME_TOOLS_HOME` → `ANIMA_HOME` → current directory
(`ANIME_TOOLS_MODELS` overrides the model dir).

## Docs

[`docs/README.md`](docs/README.md) is the index. Start with the
[guidebook](docs/guidelines/guidebook.md) (end-to-end walkthrough for users); the
rest are per-piece references:

- [`docs/contract.md`](docs/contract.md) — the `anime_tools` ↔ `anima_lora` contract (dependency
  direction, file formats, grammar, seams).
- [`docs/anima_tagger.md`](docs/anima_tagger.md) — the tagger: vocab build, dbv4 backend,
  sidecar head, calibration.
- [`docs/position_captions.md`](docs/position_captions.md) — position-clause grammar, rewrite rules,
  gates and knobs.
- [`docs/multiview_audit.md`](docs/multiview_audit.md) — multi-view / multi-panel caption audit.
- [`docs/grouping.md`](docs/grouping.md) — near-twin grouping, `groups.json`, the feature cache,
  decensor match tools.
- [`docs/masking.md`](docs/masking.md) — SAM3 subject masks, text masks, merge, where a mask lives.
- [`comfyui/anima_tagger/`](comfyui/anima_tagger/) — the Anima Tagger ComfyUI node (loader +
  captioner; link the directory into `custom_nodes/`).
- `.claude/skills/captions/` — the Claude Code skill for the caption pipeline.
- [`examples/`](examples/) — one runnable script per feature, API beside CLI.

## Development

```bash
make install       # install bun (frontend bundler) + uv sync
uv run pytest -q
make gui           # dev server on http://127.0.0.1:8790, opens your browser
make frontend      # rebuild anime_tools/gui/static/ from frontend/ (Solid, needs bun)
make frontend-dev  # bun dev server with hot reload on :5173, proxying /api to `make gui`
```

Every target runs on Windows too, from cmd, PowerShell or Git Bash: the
Makefile pins PowerShell as its shell there (GNU Make otherwise falls back to
`cmd.exe` when Git Bash is absent, where the recipes mean something else) and
calls the `.ps1` twin of each setup script — `scripts/ensure_bun.ps1`,
`scripts/build_frontend.ps1`. You still need GNU Make itself
(`winget install GnuWin32.Make`, or scoop/choco); everything else the targets
touch — uv, bun, python — is cross-platform.

The GUI frontend lives in `frontend/` (TypeScript / Solid) and builds into
`anime_tools/gui/static/` — `index.html` with script and CSS inlined, plus the
bundled Pretendard beside it — which is **committed**: git installs ship it
as-is, and CI fails if it drifts from `frontend/src`. Users never need bun. bun
is the whole toolchain (`frontend/build.ts` drives `Bun.build`); there is no
Vite, webpack or Rollup in the tree.

`tests/test_boundary.py` pins the one rule of the split: this package never
imports the trainer (`library.*`, `networks`, `train`).

## License

MIT (see `LICENSE`). The dbv4 backbone weights are GPL-3.0 and gated upstream —
they are fetched at load under the user's Hugging Face token, never vendored.
