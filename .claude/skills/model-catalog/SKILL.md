---
name: model-catalog
description: The model catalog (anime_tools/downloads.py) — every checkpoint's repo, files,
destination and installed probe; derived rows that build (ONNX graph, English tag CSV); the rule
that loaders import their paths from here. Load before adding a weight, moving one, changing a
loader's default path, or touching the GUI Models pane.
---

# The model catalog

`anime_tools/downloads.py` (torch-free) is one `Asset` per checkpoint: the tagger + gated dbv4
backbone + the ONNX graph traced from it, SAM3, PE-Spatial, the MIT text net, ComicTextDetector,
the SAM3 subject soft prompt, PP-OCRv6 det/rec, the Danbooru tag KB and its English build. Each
row carries repo, files, destination, `used_by` / `stages`, and an offline `installed` probe.
`python -m anime_tools.downloads [ID…]` fetches in catalog order; the GUI's Models pane runs
exactly that. `tests/test_downloads.py` pins the behaviour below.

## The one rule

**It is the single source of truth for weight locations.** `vision/pe.py`, `masking/mit.py`,
`masking/_sam3.py`'s flag defaults and `ocr/_onnx.py` import theirs from here
(`default_pe_spatial_path`, `default_ctd_onnx_path`, `default_ppocr_*_dir`, …), and
`default_ctd_onnx_path()` has no flag at all. A path you could point elsewhere is a Download
button that writes where the loader doesn't look;
`test_rows_land_where_the_loaders_look` is the guard. Destinations hang off `models_dir()`
(`_env.py`), which follows the curation home.

## Row kinds

- **HF-hub rows** (most): `repo` + `files` (+ `subfolder`, `repo_type`), fetched through `_hf.py`
  under the user's token. `gated="<accept-terms url>"` marks a row whose repo needs a click; the
  GUI shows the link.
- **Plain-HTTPS rows** (`url=`): the soft prompt, the CTD net, the tag KB go through
  `Asset._fetch_http`, with the recovery text on failure.
- **Derived rows** (`derived=(<input row ids>,)` + `build=callable`): the downloads are *inputs*
  that stay in the hub cache, and the probe asks for the file `build` writes, so a hub sweep can't
  turn a built row back to "missing". Two exist: `danbooru_tags_en` builds its CSV from the 45 MB
  Danbooru wiki mirror; `tagger_onnx` traces `dbv4.onnx` beside the tagger checkpoint out of the
  gated backbone (`_export_dbv4_onnx`) — a build and not a download because GPL weights can't be
  redistributed, so every user exports their own. It reads the `tagger` row's `config.json`, so it
  sits **after** it in catalog order, and refuses without a checkpoint.

## Adding a row

1. Add the `Asset(...)` to `catalog()` in dependency order. Pick a stable `id`; the GUI and the
   trainer address rows by it.
2. Put the loader's default path in this module as a `default_*` function and import it from the
   loader; never spell the path in the loader.
3. If the row is derived, write the `build(dest, log)` callable here and list its inputs in
   `derived`.
4. Tests: `test_downloads.py` — `test_rows_land_where_the_loaders_look` gets the new loader
   pair; a derived row gets a build test like
   `test_the_onnx_row_builds_the_graph_beside_the_checkpoint`.
5. `docs/guidelines/guidebook.md` lists what the Models pane installs; the tagger's ONNX
   backend rule is in `anime_tools/tagger/CLAUDE.md`.

The `onnx` + `onnxscript` dependencies exist only because the `tagger_onnx` build needs the
exporter; running a graph needs neither. `onnxruntime` is split by marker (`onnxruntime` on
macOS, `onnxruntime-gpu` elsewhere) because OCR has no fallback without it; the ORT provider
choice is `_onnx.py` (CPU and CUDA only, never CoreML).
