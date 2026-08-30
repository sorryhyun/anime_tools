# Plan: a simple web GUI for `anime_tools` + a one-line installer

Status: v1 shipped 2026-08-30 (`anime_tools/gui/`, `install.sh` / `install.ps1`,
`.github/workflows/release.yml`). Deviations from the plan below:

- Schemas are collected in a **child interpreter** (`stages.load_schemas()`):
  `masking.cli.generate_masks_mit` imports torch at module level, so the server
  process imports no stage module at all. `build_argv` works from the cached
  field list.
- The frontend is the **vanilla `index.html`** fallback (committed, no bun
  build); the Solid/bun app is still milestone 3.
- No `sse-starlette`: SSE is a plain `StreamingResponse`.
- Settings live in `<home>/.anime_tools_gui.json`; the HF token goes through
  `huggingface_hub.login` and is never read back.

## Goal

Let someone curate a dataset (autotag → position captions → correct/mirror →
multiview audit → groups → masks) **without installing the trainer**, from a
browser, with an install line as short as `anima_lora`'s:

```sh
curl -fsSL https://github.com/sorryhyun/anime_tools/releases/latest/download/install.sh | sh
anime-tools-gui          # opens http://127.0.0.1:8790
```

## Why web, not Qt

- The trainer already has the *rich* PySide6 GUI whose Preprocess tab wraps
  every stage here. This GUI is the small standalone one; it must not become a
  second copy of that.
- Curation usually runs on a headless GPU box. A browser tab over SSH beats X
  forwarding and PySide6 wheels (~150 MB) — `fastapi + uvicorn` is ~10 MB.
- Image / mask / contact-sheet previews are a static route + `<img>`.
- The trainer's daemon is already aiohttp; a web panel can be mounted into it
  later. Qt widgets could not.

## Why the installer can't be a static binary (yaar-style)

`yaar` ships one `bun build --compile` binary because its runtime deps are
zero. Ours are torch + timm + sam3 + Hub-fetched weights; nothing compiles that
away. So `uv` resolving the wheel *is* the install. Web vs. Qt changes only what
is inside the wheel, not the install line.

## Design

### Packaging

```toml
[project.optional-dependencies]
gui = ["fastapi>=0.115", "uvicorn>=0.30", "sse-starlette>=2"]
all = ["anime-tools[stages,grouping,masking,gui]"]

[project.scripts]
anime-tools-gui = "anime_tools.gui.server:main"

[tool.setuptools.package-data]
"anime_tools.gui" = ["static/**/*"]
```

- The trainer's `anime-tools-git` group keeps `[stages,grouping,masking]` — it
  must **not** pull `gui` (no second web stack in the trainer venv).
- `tests/test_boundary.py` gains: `anime_tools.gui.*` imports neither the
  trainer nor `torch`. The server process never loads a model; stages run as
  subprocesses.

### Backend — `anime_tools/gui/`

| file | role |
|---|---|
| `server.py` | FastAPI app + `main()` (argparse: `--host 127.0.0.1 --port 8790 --home`, `--open`). Localhost only by default; `--host 0.0.0.0` is the opt-in for remote use. |
| `stages.py` | The **stage registry**: `(id, title, module, build_parser)` for each CLI, plus the parser→JSON schema converter (`store_true` → bool, `type=int/float` → number, `choices` → enum, `help` → tooltip, dest names ending in `_dir`/`_path`/`dir` → path picker hint). |
| `jobs.py` | `JobManager`: spawn `sys.executable -m <module> <args>` with `ANIME_TOOLS_HOME` set, tail stdout to a ring buffer + on-disk log, one running job at a time (GPU), cancel = kill process tree (psutil, same as trainer's `process.py`). |
| `static/` | built frontend (see below). |

Endpoints:

```
GET  /api/stages                 → [{id, title, doc, schema}]
POST /api/jobs {stage, args}     → {id}
GET  /api/jobs/{id}              → {state, started, exit_code, argv}
GET  /api/jobs/{id}/log  (SSE)   → stdout lines, then `done`
GET  /api/jobs/{id}/report       → the stage's report.json, if any
GET  /api/files?path=…           → image/mask/contact-sheet bytes (restricted to curation_home() and the job's source dir)
GET  /api/settings, PUT …        → last source dir, HF token presence (token itself stays in the HF cache; never returned)
```

Apply is a separate, explicit control: the form's **Dry run** button posts
without `--apply`; **Apply** posts with it and requires a confirm. Stages are
dry-run by default already, so the UI only re-exposes that.

### CLI refactor (prerequisite, no behaviour change)

Every CLI builds its parser inline in `main()`. Lift each into
`build_parser() -> argparse.ArgumentParser`; `main()` becomes
`build_parser().parse_args()` + existing body. Stages in scope for v1:

1. `stages.cli.autotag_captions`
2. `stages.cli.position_captions` (+ `review_position_captions` as a read-only report view)
3. `stages.cli.correct_captions`
4. `stages.cli.audit_multiview` / `audit_apply_curated`
5. `grouping.cli.build_groups`
6. `masking.cli.generate_masks` / `generate_masks_mit` / `merge_masks`

Probe/bench CLIs (`probe_*`, `ab_*`, `tagger.cli.*`) stay CLI-only.

### Frontend — `frontend/` (TypeScript, Solid, built with bun)

- Layout: stage list on the left; selected stage = generated form + Dry run /
  Apply / Cancel + live log + report tab (renders `report.json` as a table,
  with image thumbnails where the report names files).
- No router, no state library; ~1k lines target.
- `bun` is a **dev/CI dependency only**. `bun run build` emits
  `anime_tools/gui/static/`; the release workflow builds it and the wheel
  ships it. Users never run bun. `static/` is gitignored; a `make`-less
  `scripts/build_frontend.sh` is the one build entry.
- Fallback if we don't want a JS toolchain yet: a single vanilla
  `index.html` + `<script>` using `EventSource`. Same API, swap later.

### Installer — `install.sh` / `install.ps1`

Mirror `anima_lora/install.sh` minus CUDA-toolkit and tarball steps:

1. install `uv` if missing (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
2. `uv tool install "anime-tools[all] @ git+https://github.com/sorryhyun/anime_tools@${ANIME_TOOLS_VERSION:-<latest tag>}"`
   - Windows / CPU-only hosts: pass `--index https://download.pytorch.org/whl/<cu…>` via `TORCH_INDEX` env; default PyPI torch is CUDA on Linux and CPU on Windows.
3. print `anime-tools-gui` + `uv tool upgrade anime-tools` for updates.

Try-before-install: `uvx --from "anime-tools[all] @ git+…" anime-tools-gui`.

### Release workflow — `.github/workflows/release.yml`

On tag: bun build frontend → `uv build` (sdist+wheel with `static/`) → attach
`install.sh`/`install.ps1` + wheel to the GitHub release. `install.sh` may
install from the release wheel URL instead of git to skip the frontend build
on the user's side (git install would ship an empty `static/`).

## Milestones

1. **Parsers**: `build_parser()` in the six stage CLIs; tests that
   `build_parser().parse_args([...])` matches current behaviour. (½ day)
2. **Server**: `gui/` package, registry, JobManager, endpoints; vanilla
   `index.html` so it is usable immediately; boundary test. (1 day)
3. **Frontend**: Solid app, build script, `static/` packaging. (1–2 days)
4. **Installer + release**: scripts, workflow, README section. (½ day)
5. Later: mount into the trainer daemon; caption viewer (torch-free
   `captions/` makes this cheap); groups.json / mask overlay viewers.

## Non-goals

- Editing captions or configs by hand (trainer's Dataset tab).
- Training, TE re-encode (`--apply` runs still need the trainer's re-encode;
  the UI says so in the Apply confirm).
- Multi-user / auth. Localhost by default; remote use is the user's tunnel.
