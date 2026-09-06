---
name: release
description: Cutting an anime_tools release — version bump, annotated tag, what release.yml
builds and attaches, how install.sh resolves a version, what CI checks before it. Load when asked
to release, bump the version, or debug the installer or the release workflow.
---

# Releasing

Releases are **tag-driven**: `.github/workflows/release.yml` runs on a `v*` tag, rebuilds the
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

## What a release must keep true

- The GUI server process stays torch-free (`tests/test_boundary.py::test_gui_server_is_torch_free`).
- The package is consumed as a git dependency by the trainer; a change to anything in
  `docs/contract.md` needs the trainer bumped in step, and `contract.CONTRACT_VERSION` moved.
- `packages.find` includes only `anime_tools*`: `comfyui/`, `design/`, `examples/` and
  `frontend/` sources are not in the wheel; `anime_tools/gui/static/*` is.
