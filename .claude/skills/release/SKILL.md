---
name: release
description: Cutting an anime_tools release — version bump, annotated tag, what release.yml
builds and attaches, how install.sh resolves a version, what CI checks before it. Load when asked
to release, bump the version, or debug the installer or the release workflow.
---

# Releasing

Releases are tag-driven: `.github/workflows/release.yml` runs on a `v*` tag, rebuilds the
frontend and refuses a stale committed bundle (`git diff --exit-code -- anime_tools/gui/static/`),
runs `uv build`, and publishes a GitHub release with generated notes carrying `install.sh`,
`install.ps1`, the wheel and the sdist. An untagged push never changes what users install.

## Cutting one

1. `main` is green: `uv run pytest -q`, `uv run ruff format --check .`, `make frontend` leaves
   no diff under `anime_tools/gui/static/`, `cd frontend && bun run check && bun run format:check`.
   `ci.yml` runs the same set with CPU torch (`uv sync --index …/whl/cpu`, all dependencies, no
   skipped torch tests; `ruff check` is advisory there because the rule set is user-level).
2. Bump `version` in `pyproject.toml` and commit.
3. `git tag -a vX.Y.Z -m "vX.Y.Z"` and `git push origin main vX.Y.Z`.
4. Watch the `release` workflow; the release page should show the four files.

## The installer

`install.sh` / `install.ps1` install `uv` if missing, then `uv tool install` the package from the
release tag into an isolated venv and put `anime-tools-gui` on PATH — no git checkout, no CUDA
toolkit (the torch wheel bundles its runtime). Version resolution: `ANIME_TOOLS_VERSION=vX.Y.Z`
or the first argument, else `releases/latest` through the GitHub API. `TORCH_INDEX` adds an extra
index for CPU-only or Windows hosts. A static binary is not an option (torch + sam3), which is why
the GUI is a server that spawns stage CLIs rather than a bundle.

## The in-app updater

`anime_tools/update.py` + the GUI's ☰ → Update pane are the installer read backwards: the pane
compares `__version__` with `releases/latest` from the GitHub API and installs a newer tag with the
same `uv tool install --force … @<tag>` (numpy override included) that `install.sh` runs. So a
release is only reachable from inside the app if it is a *GitHub release* on a `v*` tag — an
untagged push, or a tag whose workflow failed before publishing, is invisible to every installed
copy. The check is cached six hours in `.anime_tools_gui.json` and is off entirely when the pane's
checkbox is; unauthenticated GitHub allows 60 requests an hour per IP.

Two things a release must not break: the tag name has to stay orderable
(`update.parse_version` reads `v?N(.N)*` and calls anything else — a `+local`, an `rc1` —
"unknown", which shows the version row without offering a button), and the package must stay
installable straight from the tag, since that command is what the button runs. A checkout is never
updated from the pane; it shows `git pull && uv sync` instead. `tests/test_gui_updates.py` pins the
ordering, the cache and the refusals, and never touches the network.

## What a release must keep true

- The GUI server process stays torch-free (`tests/test_boundary.py::test_gui_server_is_torch_free`).
- The package is consumed as a git dependency by the trainer; a change to anything on the shared
  surface — file formats, the caption grammar, `contract.py`, `buckets.py` — needs the trainer
  bumped in step, and `contract.CONTRACT_VERSION` moved.
- **No bare `[tool.uv.sources]`.** uv honors a git dependency's own sources, so a bare torch
  index here lands in the trainer's lock and collides with its Windows backend groups (0.6.1
  did; 0.6.2 bound it to the `cuda-windows` group). Sources stay group- or extra-conditioned.
- `packages.find` includes only `anime_tools*`: `comfyui/`, `design/`, `examples/` and
  `frontend/` sources are not in the wheel; `anime_tools/gui/static/*` is.
