#!/usr/bin/env sh
# Build the Solid frontend into anime_tools/gui/static/ -- index.html with its script and
# stylesheet inlined, plus the woff2 it points at.
# Needs bun (https://bun.sh) and nothing else -- bun is the bundler (frontend/build.ts),
# not just the runner. Users never run this: the built file is committed and shipped in
# the wheel, and CI fails if it drifts from frontend/src.
set -eu
cd "$(dirname "$0")/../frontend"
# The bundle is committed and CI diffs it, so every builder must run the SAME bun:
# minified output differs between bun releases. frontend/.bun-version is the pin
# (CI reads it too); `bun upgrade --version X` or scripts/ensure_bun.sh installs it.
want=$(tr -d '[:space:]' < .bun-version)
have=$(bun --version)
if [ "$have" != "$want" ]; then
	echo "build_frontend: bun $have found, but the committed bundle is built with bun $want" >&2
	echo "  (frontend/.bun-version) -- run: bun upgrade --version $want" >&2
	exit 1
fi
bun install --frozen-lockfile
bun run check
bun run build
