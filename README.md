# anime_tools

Dataset curation for anime diffusion training — the half of
[`anima_lora`](https://github.com/sorryhyun/anima_lora) that produces the
caption master and sidecars, split out so it can be used on datasets bound for
*any* trainer and installed without the trainer's DiT/VAE stack.

| Sub-package | What | Install |
|---|---|---|
| `anime_tools.captions` | The caption **grammar** (`parse_caption` / `compose_caption` — never `split(",")` a caption), tag taxonomy + Danbooru-KB correction, `--caption_drop_groups`, shuffle/dropout **variants sidecars**, `caption_index.json` builder | base (torch-free) |
| `anime_tools.tagger` | **Anima Tagger** — a vocab/threshold/sidecar head over the external `animetimm/*.dbv4-full` caformer tagger, emitting Anima-format tags (`rating, count, characters, copyrights, @artists, generals`). CLIs: `python -m anime_tools.tagger.cli --mode …`, `…cli.autotag`, `…cli.autotag_server`, `…cli.train_sidecar` |
| `anime_tools.stages` | Caption-master stages: batch **autotag**, **position clauses** (SAM3 crops → tagger → v2 rewrite), correction + variants mirror, **multiview audit** |
| `anime_tools.grouping` | Near-twin / same-concept **grouping** on PE-Spatial-B16-512 features (`anime_tools.vision.pe`, weights fetched from the Hub) → `groups.json`; decensor match tools. CLI: `python -m anime_tools.grouping.cli.build_groups --source-dir …` |
| `anime_tools.masking` | Training masks: SAM3 subject masks, MIT / ComicTextDetector text masks, merge. CLIs: `python -m anime_tools.masking.cli.{generate_masks,generate_masks_mit,merge_masks}` |

## Install

One line, no checkout — installs [uv](https://astral.sh/uv) if missing, then
`uv tool install`s the latest release with every extra and puts
`anime-tools-gui` on PATH:

```bash
curl -fsSL https://github.com/sorryhyun/anime_tools/releases/latest/download/install.sh | sh
```

Windows (PowerShell): `irm https://github.com/sorryhyun/anime_tools/releases/latest/download/install.ps1 | iex`

`ANIME_TOOLS_VERSION=v0.2.0` pins a tag, `TORCH_INDEX=https://download.pytorch.org/whl/cu130` picks a torch
index (PyPI's Linux wheel is already CUDA; Windows defaults to CPU). Update
with `uv tool upgrade anime-tools`.

As a library dependency (what the trainer does):

```bash
uv add "anime-tools @ git+https://github.com/sorryhyun/anime_tools"   # git dependency; no PyPI
```

The repo is the product (kohya-ss/sd-scripts style): pin a tag, or use a
`[tool.uv.sources]` path override for a live checkout.

## Web GUI

```bash
cd <your dataset folder>   # image_dataset/, post_image_dataset/, models/ live here
anime-tools-gui --open     # http://127.0.0.1:8790
```

A small standalone panel on your dataset. The **sidebar is the dataset**: every
source image, and under each one its captions as child nodes —

```
▾ chars/alice
  ▾ chars_alice_001.png
      master      image_dataset/…/chars_alice_001.txt      (editable)
      derived     post_image_dataset/resized/…/….txt       (editable)
      variants    …/….variants.txt                         (generated, read-only)
```

Selecting an image shows it (source / resized / mask) beside its captions,
each with its flat tag bag and position clauses broken out — parsed by
`captions.position_clauses` server-side, never by splitting on commas in the
browser. Master and derived are editable: type, watch the clause preview
update, **Save**. Editing is a file write, so it must still be followed by the
trainer's TE re-encode; a derived edit also says so, because it leaves the
`.variants.txt` sidecar stale. `↑`/`↓` (or `j`/`k`) walk the images, the filter
box narrows the tree, and `#<rel>|<kind>` in the URL is a link to one caption.

The **stage runner is the bottom dock**: its button strip *is* the stage list
(grouped: captions / grouping / masking), so one click picks a stage and a
second click on the open one folds the dock away. Fill the form (generated from
the CLI's own `--help`), **Dry run**, then **Apply** — the dataset stays on
screen throughout, and refreshes when the job finishes. The run's newest output
line shows in the stage bar; the log / `report.json` / job-history panels are
being reworked. Stages run as `python -m …` subprocesses, one at a time; the server
never loads a model. `--host 0.0.0.0` exposes it on the LAN for a headless GPU
box (no auth — use your own tunnel), `--home` overrides the curation home. ⚙
Settings holds the three dataset roots, a one-time Hugging Face sign-in (the
tagger backbone and SAM3 weights are gated), and a **Models** list: one row per
checkpoint the stages need, saying whether it is here and where it goes, with a
Download button that runs `python -m anime_tools.downloads` as an ordinary job
(so it shares the one slot with the stages and streams into the same bar).
Nothing has to be pre-fetched — every loader still fetches on first use; the
buttons only move the wait, and any gated-repo refusal, to a moment you picked.
The panel speaks **English, Korean, Japanese and Chinese**: the ☰ menu's
*Language* row switches it in place, and a first visit follows the browser's own
language list. Only the GUI's own chrome is translated — a stage's title, its
form fields and their help text come from that CLI's `--help` and stay in
English, and captions, tags and paths are data, never translated.

FastAPI + uvicorn are plain dependencies; the trainer's own PySide6 GUI remains
the rich editor.

## Layout of a curated dataset

```
image_dataset/**/{stem}.png + {stem}.txt        caption master   ← autotag / position / correction write here
post_image_dataset/resized/{stem}.txt           derived caption  ← stages.captions mirrors + corrects
post_image_dataset/resized/{stem}.variants.txt  shuffle / dropout variants (tab-delimited, v0 = pristine)
post_image_dataset/captions/caption_index.json  typed-tag index (character / copyright / artist / count)
models/captioners/anima-tagger-dbv4/            tagger checkpoint (auto-fetched from sorryhyun/anima-tagger)
models/sam3/sam3.pt, models/pe/…             SAM3 / PE-Spatial weights (`python -m anime_tools.downloads --list`)
networks/calibration/sam3_girl_prompt.safetensors  SAM3 subject soft prompt (default `--prompt_embed`)
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
make install       # install bun (frontend bundler) + uv sync
uv run pytest -q
make gui           # dev server on http://127.0.0.1:8790, opens your browser
make frontend      # rebuild anime_tools/gui/static/index.html from frontend/ (Solid, needs bun)
make frontend-dev  # bun dev server with hot reload on :5173, proxying /api to `make gui`
```

Every target runs on Windows too, from cmd, PowerShell or Git Bash: the
Makefile pins PowerShell as its shell there (GNU Make otherwise falls back to
`cmd.exe` when Git Bash is absent, where the recipes mean something else) and
calls the `.ps1` twin of each setup script — `scripts/ensure_bun.ps1`,
`scripts/build_frontend.ps1`. You still need GNU Make itself
(`winget install GnuWin32.Make`, or scoop/choco); everything else the targets
touch — uv, bun, python — is cross-platform.

The GUI frontend lives in `frontend/` (TypeScript / Solid) and builds into one
self-contained `anime_tools/gui/static/index.html` that is **committed** — git
installs ship it as-is, and CI fails if it drifts from `frontend/src`. Users
never need bun. bun is the whole toolchain (`frontend/build.ts` drives
`Bun.build` and inlines script, CSS and the bundled Pretendard into the one
file); there is no Vite, webpack or Rollup in the tree.

`tests/test_boundary.py` pins the one rule of the split: this package never
imports the trainer (`library.*`, `networks`, `train`).

## License

MIT (see `LICENSE`). The dbv4 backbone weights are GPL-3.0 and gated upstream —
they are fetched at load under the user's Hugging Face token, never vendored.
