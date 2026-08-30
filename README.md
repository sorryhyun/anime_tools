# anime_tools

Dataset curation for anime diffusion training — the half of
[`anima_lora`](https://github.com/sorryhyun/anima_lora) that produces the
caption master and sidecars, split out so it can be used on datasets bound for
*any* trainer and installed without the trainer's DiT/VAE stack.

| Sub-package | What | Install |
|---|---|---|
| `anime_tools.captions` | The caption **grammar** (`parse_caption` / `compose_caption` — never `split(",")` a caption), tag taxonomy + Danbooru-KB correction, `--caption_drop_groups`, shuffle/dropout **variants sidecars**, `caption_index.json` builder | base (torch-free) |
| `anime_tools.tagger` | **Anima Tagger** — a vocab/threshold/sidecar head over the external `animetimm/*.dbv4-full` caformer tagger, emitting Anima-format tags (`rating, count, characters, copyrights, @artists, generals`). CLIs: `python -m anime_tools.tagger.cli --mode …`, `…cli.autotag`, `…cli.autotag_server`, `…cli.train_sidecar` | `[tagger]` |
| `anime_tools.stages` | Caption-master stages: batch **autotag**, **position clauses** (SAM3 crops → tagger → v2 rewrite), correction + variants mirror, **multiview audit** | `[stages]` |
| `anime_tools.grouping` | Near-twin / same-concept **grouping** on PE-Spatial-B16-512 features (`anime_tools.vision.pe`, weights fetched from the Hub) → `groups.json`; decensor match tools. CLI: `python -m anime_tools.grouping.cli.build_groups --source-dir …` | `[grouping]` |
| `anime_tools.masking` | Training masks: SAM3 subject masks, MIT / ComicTextDetector text masks, merge. CLIs: `python -m anime_tools.masking.cli.{generate_masks,generate_masks_mit,merge_masks}` | `[masking]` |

## Install

One line, no checkout — installs [uv](https://astral.sh/uv) if missing, then
`uv tool install`s the latest release with every extra and puts
`anime-tools-gui` on PATH:

```bash
curl -fsSL https://github.com/sorryhyun/anime_tools/releases/latest/download/install.sh | sh
```

Windows (PowerShell): `irm https://github.com/sorryhyun/anime_tools/releases/latest/download/install.ps1 | iex`

`ANIME_TOOLS_VERSION=v0.2.0` pins a tag, `ANIME_TOOLS_EXTRAS=gui` trims the
extras, `TORCH_INDEX=https://download.pytorch.org/whl/cu130` picks a torch
index (PyPI's Linux wheel is already CUDA; Windows defaults to CPU). Update
with `uv tool upgrade anime-tools`.

As a library dependency (what the trainer does):

```bash
uv add "anime-tools[all] @ git+https://github.com/sorryhyun/anime_tools"   # git dependency; no PyPI
```

The repo is the product (kohya-ss/sd-scripts style): pin a tag, or use a
`[tool.uv.sources]` path override for a live checkout.

## Web GUI

```bash
cd <your dataset folder>   # image_dataset/, post_image_dataset/, models/ live here
anime-tools-gui --open     # http://127.0.0.1:8765
```

A small standalone panel over the stage CLIs: pick a stage, fill the form
(generated from the CLI's own `--help`), **Dry run**, read the log and
`report.json`, then **Apply**. Stages run as `python -m …` subprocesses, one at
a time; the server never loads a model. `--host 0.0.0.0` exposes it on the LAN
for a headless GPU box (no auth — use your own tunnel), `--home` overrides the
curation home. Sign in to Hugging Face under ⚙ Settings once: the tagger
backbone and SAM3 weights are gated. Needs the `gui` extra (FastAPI + uvicorn);
the trainer's own PySide6 GUI remains the rich editor — this one only runs
stages. Design notes: [`docs/gui_plan.md`](docs/gui_plan.md).

## Layout of a curated dataset

```
image_dataset/**/{stem}.png + {stem}.txt        caption master   ← autotag / position / correction write here
post_image_dataset/resized/{stem}.txt           derived caption  ← stages.captions mirrors + corrects
post_image_dataset/resized/{stem}.variants.txt  shuffle / dropout variants (tab-delimited, v0 = pristine)
post_image_dataset/captions/caption_index.json  typed-tag index (character / copyright / artist / count)
models/captioners/anima-tagger-dbv4/            tagger checkpoint (auto-fetched from sorryhyun/anima-tagger)
```

Every artifact the trainer reads is a **file**; the formats are frozen in
[`docs/contract.md`](docs/contract.md). Paths resolve against the curation
home: `ANIME_TOOLS_HOME` → `ANIMA_HOME` → current directory
(`ANIME_TOOLS_MODELS` overrides the model dir).

## Docs

- [`docs/contract.md`](docs/contract.md) — the `anime_tools` ↔ `anima_lora` contract (dependency direction, file formats, grammar, seams).
- [`docs/anima_tagger.md`](docs/anima_tagger.md) — the tagger: vocab build, dbv4 backend, sidecar head, calibration.
- [`comfyui/anima_tagger/`](comfyui/anima_tagger/) — the Anima Tagger ComfyUI node (loader + captioner; link the directory into `custom_nodes/`).
- [`docs/position_captions.md`](docs/position_captions.md) — position-clause grammar, v2 rewrite rules and gates, evidence.
- [`docs/multiview_audit.md`](docs/multiview_audit.md) — multi-view / multi-panel caption audit.
- `.claude/skills/captions/` — the Claude Code skill for the caption pipeline.

## Development

```bash
uv sync --all-extras
uv run pytest -q
```

`tests/test_boundary.py` pins the one rule of the split: this package never
imports the trainer (`library.*`, `networks`, `train`).

## License

MIT (see `LICENSE`). The dbv4 backbone weights are GPL-3.0 and gated upstream —
they are fetched at load under the user's Hugging Face token, never vendored.
