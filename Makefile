# The dataset make targets (caption-autotag, preprocess-*, ...) live in the trainer repo.
# This Makefile only hosts developer conveniences for this checkout.
GUI_HOST ?= 127.0.0.1
GUI_PORT ?= 8790
GUI_ARGS ?=

# Windows has no sh(1) to count on -- GNU Make falls back to cmd.exe when Git Bash
# is absent, and the recipes then mean something else entirely -- so pin PowerShell
# and call the .ps1 twin of each setup script. Every other line (uv / bun / pytest)
# is identical in both shells.
ifeq ($(OS),Windows_NT)
SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command
REQUIRE_UV := if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Write-Host "uv not found; install it: https://docs.astral.sh/uv/getting-started/installation/"; exit 1 }
ENSURE_BUN := & ./scripts/ensure_bun.ps1
BUILD_FRONTEND := & ./scripts/build_frontend.ps1
FRONTEND_DEV := Set-Location frontend; bun install; $$env:ANIME_TOOLS_API = "http://$(GUI_HOST):$(GUI_PORT)"; bun run dev
else
REQUIRE_UV := command -v uv >/dev/null 2>&1 || { echo "uv not found; install it: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
ENSURE_BUN := scripts/ensure_bun.sh
BUILD_FRONTEND := scripts/build_frontend.sh
FRONTEND_DEV := cd frontend && bun install && ANIME_TOOLS_API=http://$(GUI_HOST):$(GUI_PORT) bun run dev
endif

.PHONY: install gui sync test frontend frontend-dev
install:  ## one-shot dev setup: bun (frontend bundler) + uv sync
	@$(REQUIRE_UV)
	$(ENSURE_BUN)
	uv sync
	@echo ""
	@echo "ready -- run 'make gui' to open the web GUI"
sync:
	uv sync
gui:  ## run the web GUI from this checkout and open it in a browser
	uv run anime-tools-gui --host $(GUI_HOST) --port $(GUI_PORT) --open $(GUI_ARGS)
test:
	uv run pytest -q -n auto
frontend:  ## rebuild anime_tools/gui/static/index.html from frontend/ (needs bun)
	$(BUILD_FRONTEND)
frontend-dev:  ## bun dev server with hot reload, proxying /api to a running `make gui`
	$(FRONTEND_DEV)
