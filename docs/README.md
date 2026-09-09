# Documentation

Index of the `docs/` tree. Each row is a one-line orientation; read the linked doc before working
on the thing it describes.

- Guidelines — user-facing walkthroughs: start here if you installed the package to curate a
  dataset.
- Stages — one reference per stage family: what it reads and writes, running it from the GUI
  and the CLI, the knobs, the limits.

The architecture notes for people changing the code are `CLAUDE.md` at the repo root (the
invariants and the map), one `CLAUDE.md` per package under `anime_tools/` (that module's
architecture), `frontend/CLAUDE.md` for the browser half, and `.claude/skills/` for procedures;
`examples/` has one runnable script per feature, API beside CLI.

## Guidelines

The four books live in the package (`anime_tools/gui/guidebooks/`), not here: the GUI serves them
behind ☰ → 📖 in the language the panel is in, and only `anime_tools*` ships in the wheel.

| Doc | Description |
|-----|-------------|
| [guidebook.md](../anime_tools/gui/guidebooks/guidebook.md) | The guidebook — install, the curation home, the web panel, every stage in the order you run it, troubleshooting |
| [가이드북.md](../anime_tools/gui/guidebooks/가이드북.md) | 가이드북 (Korean) |
| [ガイドブック.md](../anime_tools/gui/guidebooks/ガイドブック.md) | ガイドブック (Japanese) |
| [指南书.md](../anime_tools/gui/guidebooks/指南书.md) | 指南书 (Chinese) |

## Stages

| Doc | Description |
|-----|-------------|
| [anima_tagger.md](anima_tagger.md) | Anima Tagger — the vocab / threshold / sidecar head over the dbv4 caformer; vocab build, calibration, batch autotag |
| [position_captions.md](position_captions.md) | Position captions — the clause grammar, the four gates and five move rules, SAM3 detection, knobs and skip reasons |
| [multiview_audit.md](multiview_audit.md) | Multiview audit — finding untagged `multiple views` in the caption master |
| [grouping.md](grouping.md) | Near-twin grouping — PE-Spatial features, `groups.json`, the feature cache, custom embedders, decensor match tools |
| [masking.md](masking.md) | Training masks — SAM3 subject masks, merge, where a mask lives |
