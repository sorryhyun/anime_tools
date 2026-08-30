# anime_tools bootstrap installer (Windows PowerShell).
#
#   irm https://github.com/sorryhyun/anime_tools/releases/latest/download/install.ps1 | iex
#
# Installs uv if missing, then `uv tool install`s anime-tools (torch + sam3 included)
# and puts `anime-tools-gui` on PATH.
#
# Options (env vars):
#   $env:ANIME_TOOLS_VERSION = "v0.2.0"     specific tag (default: latest release)
#   $env:TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
#       PyPI's Windows torch wheel is CPU-only; set this for a CUDA build.
$ErrorActionPreference = "Stop"

$Repo    = "sorryhyun/anime_tools"
$Version = $env:ANIME_TOOLS_VERSION
$Torch   = if ($env:TORCH_INDEX) { $env:TORCH_INDEX } else { "https://download.pytorch.org/whl/cu130" }

function Say($m) { Write-Host "==> $m" -ForegroundColor Cyan }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Say "installing uv (https://astral.sh/uv)"
  irm https://astral.sh/uv/install.ps1 | iex
  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv install failed; open a new shell and re-run" }

if (-not $Version) {
  Say "resolving latest release of $Repo"
  $Version = (irm "https://api.github.com/repos/$Repo/releases/latest").tag_name
  if (-not $Version) { throw "could not resolve latest release tag" }
}

Say "installing anime-tools @ $Version (torch from $Torch; may take a while)"
uv tool install --force --python 3.13 --index $Torch "anime-tools @ git+https://github.com/$Repo@$Version"
if ($LASTEXITCODE -ne 0) { throw "uv tool install failed" }
uv tool update-shell | Out-Null

Write-Host ""
Write-Host "✓ anime-tools $Version installed" -ForegroundColor Green
Write-Host @"

Next steps:
  cd <your dataset folder>      # image_dataset\, post_image_dataset\, models\ live here
  anime-tools-gui --open        # web GUI on http://127.0.0.1:8790

Update:   uv tool upgrade anime-tools
Remove:   uv tool uninstall anime-tools
"@
