# Build the Solid frontend into anime_tools/gui/static/ -- index.html with its script
# and stylesheet inlined, plus the woff2 it points at. Windows twin of
# build_frontend.sh. Needs bun (https://bun.sh) and nothing
# else; users never run this, the built file is committed and CI fails if it drifts.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\frontend")

# $ErrorActionPreference does not cover a native command's exit code, so check each.
function Invoke-Bun { & bun @args; if ($LASTEXITCODE -ne 0) { throw "bun $args failed ($LASTEXITCODE)" } }

# Same bun as the committed bundle (frontend/.bun-version): minified output differs
# between bun releases, and CI diffs the bundle.
$want = (Get-Content .bun-version -Raw).Trim()
$have = (& bun --version).Trim()
if ($have -ne $want) {
    throw "build_frontend: bun $have found, but the committed bundle is built with bun $want (frontend/.bun-version) -- run: bun upgrade --version $want"
}

Invoke-Bun install --frozen-lockfile
Invoke-Bun run check
Invoke-Bun run build
