#!/usr/bin/env sh
# anime_tools bootstrap installer (Linux / macOS).
#
#   curl -fsSL https://github.com/sorryhyun/anime_tools/releases/latest/download/install.sh | sh
#
# Installs uv if missing, then `uv tool install`s anime-tools with every extra
# (tagger, SAM3 stages, grouping, masking, web GUI) into an isolated venv and
# puts `anime-tools-gui` on PATH. No git checkout, no CUDA toolkit: the torch
# wheel bundles its CUDA runtime.
#
# Options (env vars, since args are awkward through a pipe):
#   ANIME_TOOLS_VERSION=v0.2.0   install a specific tag      (default: latest release)
#   ANIME_TOOLS_EXTRAS=gui       extras to install           (default: all)
#   TORCH_INDEX=https://download.pytorch.org/whl/cu130
#                                extra index for torch (Windows/CPU-only hosts;
#                                PyPI's Linux torch is already CUDA)
set -eu

REPO="sorryhyun/anime_tools"
VERSION="${ANIME_TOOLS_VERSION:-${1:-}}"
EXTRAS="${ANIME_TOOLS_EXTRAS:-all}"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || die "curl is required"

# 1. uv ----------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  say "installing uv (https://astral.sh/uv)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1090
  [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv install failed; open a new shell and re-run"

# 2. resolve the release tag -------------------------------------------------
if [ -z "$VERSION" ]; then
  say "resolving latest release of $REPO"
  VERSION=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
    | grep -m1 '"tag_name"' \
    | sed -E 's/.*"tag_name"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')
  [ -n "$VERSION" ] || die "could not resolve latest release tag from GitHub API"
fi

# 3. install -----------------------------------------------------------------
say "installing anime-tools[$EXTRAS] @ $VERSION (resolves torch + sam3; may take a while)"
set -- --python 3.13
[ -n "${TORCH_INDEX:-}" ] && set -- "$@" --index "$TORCH_INDEX"
uv tool install --force "$@" "anime-tools[$EXTRAS] @ git+https://github.com/$REPO@$VERSION"
uv tool update-shell >/dev/null 2>&1 || true

cat <<MSG

$(printf '\033[1;32m✓ anime-tools %s installed\033[0m' "$VERSION")

Next steps:
  cd <your dataset folder>      # image_dataset/, post_image_dataset/, models/ live here
  anime-tools-gui --open        # web GUI on http://127.0.0.1:8765
                                #   (sign in to Hugging Face under ⚙ Settings — the
                                #    tagger backbone and SAM3 weights are gated)

CLI:      python -m anime_tools.tagger.cli --help   (inside: uv tool run --from anime-tools python …)
Update:   uv tool upgrade anime-tools
Remove:   uv tool uninstall anime-tools
MSG
