# The dataset make targets (caption-autotag, preprocess-*, ...) live in the trainer repo.
# This Makefile only hosts developer conveniences for this checkout.
GUI_HOST ?= 127.0.0.1
GUI_PORT ?= 8790
GUI_ARGS ?=

.PHONY: gui sync test frontend frontend-dev
sync:
	uv sync
gui:  ## run the web GUI from this checkout (uv sync first)
	uv run anime-tools-gui --host $(GUI_HOST) --port $(GUI_PORT) $(GUI_ARGS)
test:
	uv run pytest -q -n auto
frontend:  ## rebuild anime_tools/gui/static/index.html from frontend/ (needs bun)
	scripts/build_frontend.sh
frontend-dev:  ## Vite dev server with HMR, proxying /api to a running `make gui`
	cd frontend && bun install && bun run dev
