#!/usr/bin/env sh
# Build the Solid frontend into anime_tools/gui/static/index.html (one self-contained file).
# Needs bun (https://bun.sh) and nothing else -- bun is the bundler (frontend/build.ts),
# not just the runner. Users never run this: the built file is committed and shipped in
# the wheel, and CI fails if it drifts from frontend/src.
set -eu
cd "$(dirname "$0")/../frontend"
bun install --frozen-lockfile
bun run check
bun run build
