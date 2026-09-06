#!/usr/bin/env sh
# Install bun (https://bun.sh) unless it is already on PATH.
# bun is the frontend bundler (frontend/build.ts); it is not needed to *run* the GUI,
# only to rebuild anime_tools/gui/static/index.html from frontend/src.
set -eu

if command -v bun >/dev/null 2>&1; then
	echo "bun: $(command -v bun) ($(bun --version))"
	exit 0
fi

if [ -x "$HOME/.bun/bin/bun" ]; then
	echo "bun: $HOME/.bun/bin/bun ($("$HOME/.bun/bin/bun" --version)) -- not on PATH"
	echo "     add it with: export PATH=\"\$HOME/.bun/bin:\$PATH\""
	exit 0
fi

# The pinned bundler version (frontend/.bun-version): the committed bundle is built
# with it and scripts/build_frontend.sh refuses any other.
want=$(tr -d '[:space:]' < "$(dirname "$0")/../frontend/.bun-version")
echo "bun not found; installing bun $want from https://bun.sh/install ..."
if command -v curl >/dev/null 2>&1; then
	curl -fsSL https://bun.sh/install | bash -s "bun-v$want"
elif command -v wget >/dev/null 2>&1; then
	wget -qO- https://bun.sh/install | bash -s "bun-v$want"
else
	echo "need curl or wget to install bun; install it manually: https://bun.sh" >&2
	exit 1
fi

echo
echo "bun installed to \$HOME/.bun/bin -- add it to your shell PATH:"
echo "    export PATH=\"\$HOME/.bun/bin:\$PATH\""
