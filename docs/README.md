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

| Doc | Description |
|-----|-------------|
| [guidelines/guidebook.md](guidelines/guidebook.md) | The guidebook — install, the curation home, the web panel, every stage in the order you run it, the hand-off to the trainer, troubleshooting |
| [guidelines/가이드북.md](guidelines/가이드북.md) | 가이드북 (Korean) |
| [guidelines/ガイドブック.md](guidelines/ガイドブック.md) | ガイドブック (Japanese) |
| [guidelines/指南书.md](guidelines/指南书.md) | 指南书 (Chinese) |

## Stages

| Doc | Description |
|-----|-------------|
| [anima_tagger.md](anima_tagger.md) | Anima Tagger — the vocab / threshold / sidecar head over the dbv4 caformer; vocab build, calibration, batch autotag |
| [position_captions.md](position_captions.md) | Position captions — the clause grammar, the four gates and five move rules, SAM3 detection, knobs and skip reasons |
| [multiview_audit.md](multiview_audit.md) | Multiview audit — finding untagged `multiple views` in the caption master |
| [grouping.md](grouping.md) | Near-twin grouping — PE-Spatial features, `groups.json`, the feature cache, custom embedders, decensor match tools |
| [masking.md](masking.md) | Training masks — SAM3 subject masks, merge, where a mask lives |
