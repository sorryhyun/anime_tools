# Build the Solid frontend into anime_tools/gui/static/ -- index.html with its script
# and stylesheet inlined, plus the woff2 it points at. Windows twin of
# build_frontend.sh. Needs bun (https://bun.sh) and nothing
# else; users never run this, the built file is committed and CI fails if it drifts.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\frontend")

# $ErrorActionPreference does not cover a native command's exit code, so check each.
function Invoke-Bun { & bun @args; if ($LASTEXITCODE -ne 0) { throw "bun $args failed ($LASTEXITCODE)" } }

Invoke-Bun install --frozen-lockfile
Invoke-Bun run check
Invoke-Bun run build
