# Install bun (https://bun.sh) unless it is already on PATH -- Windows twin of ensure_bun.sh.
# bun is the frontend bundler (frontend/build.ts); it is not needed to *run* the GUI,
# only to rebuild anime_tools/gui/static/index.html from frontend/src.
$ErrorActionPreference = "Stop"

$bun = Get-Command bun -ErrorAction SilentlyContinue
if ($bun) {
    Write-Host "bun: $($bun.Source) ($(& $bun.Source --version))"
    exit 0
}

$local = Join-Path $env:USERPROFILE ".bun\bin\bun.exe"
if (Test-Path $local) {
    Write-Host "bun: $local ($(& $local --version)) -- not on PATH"
    Write-Host '     add it with: $env:Path = "$env:USERPROFILE\.bun\bin;$env:Path"'
    exit 0
}

# The pinned bundler version (frontend/.bun-version): the committed bundle is built
# with it and scripts/build_frontend.ps1 refuses any other.
$want = (Get-Content (Join-Path $PSScriptRoot "..\frontend\.bun-version") -Raw).Trim()
Write-Host "bun not found; installing bun $want from https://bun.sh/install.ps1 ..."
& ([scriptblock]::Create((Invoke-RestMethod https://bun.sh/install.ps1))) -Version $want

Write-Host ""
Write-Host "bun installed to $env:USERPROFILE\.bun\bin -- the installer put it on your user"
Write-Host "PATH, so open a new shell before 'make frontend'."
