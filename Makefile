# The dataset make targets (caption-autotag, preprocess-*, ...) live in the trainer repo.
# This Makefile only hosts developer conveniences for this checkout.
GUI_HOST ?= 127.0.0.1
GUI_PORT ?= 8790
GUI_ARGS ?=

.PHONY: install gui sync test frontend frontend-dev
install:  ## one-shot dev setup: bun (frontend bundler) + uv sync
	@command -v uv >/dev/null 2>&1 || { \
		echo "uv not found; install it: https://docs.astral.sh/uv/getting-started/installation/" >&2; \
		exit 1; }
	scripts/ensure_bun.sh
	uv sync
	@echo
	@echo "ready -- run 'make gui' to open the web GUI"
sync:
	uv sync
gui:  ## run the web GUI from this checkout and open it in a browser
	uv run anime-tools-gui --host $(GUI_HOST) --port $(GUI_PORT) --open $(GUI_ARGS)
test:
	uv run pytest -q -n auto
frontend:  ## rebuild anime_tools/gui/static/index.html from frontend/ (needs bun)
	scripts/build_frontend.sh
frontend-dev:  ## bun dev server with hot reload, proxying /api to a running `make gui`
	cd frontend && bun install && ANIME_TOOLS_API=http://$(GUI_HOST):$(GUI_PORT) bun run dev
