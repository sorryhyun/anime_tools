---
name: model-catalog
description: The model catalog (anime_tools/downloads/) — every checkpoint's repo, files,
destination and installed probe; derived rows that build (the English tag CSV); the rule
that loaders import their paths from here. Load before adding a weight, moving one, changing a
loader's default path, or touching the GUI Models pane.
---

# The model catalog

The package is four halves: `_locations.py` (where each weight lives — the repo, filename and
destination constants the loaders import), `_assets.py` (`Asset` / `Pack` and the fetch engine),
`_catalog.py` (the rows, plus `by_id` / `by_pack` / `expand`) and `_cli.py`
(`python -m anime_tools.downloads`). `__init__.py` re-exports the public surface, so every existing
`from anime_tools.downloads import …` is unchanged.

`anime_tools/downloads/` (torch-free) is one `Asset` per checkpoint: the tagger + gated dbv4
backbone, SAM3, PE-Spatial,
the SAM3 subject soft prompt, the OCR trio (AnimeText detector, PaddleOCR-VL-1.6 base, the manga
SFX reader), the Danbooru tag KB and its English build. Each row carries repo, files, destination,
`used_by` / `stages`, a `pack`, and an offline `installed` probe.
`python -m anime_tools.downloads [ID…]` fetches in catalog order (an ID is a row or a pack); the
GUI's Models pane runs exactly that. `tests/test_downloads.py` pins the behaviour below.

## The one rule

It is the single source of truth for weight locations. `vision/pe.py`,
`masking/_sam3.py`'s flag defaults, `ocr/animetext.py` and `ocr/sfx.py` import theirs from here
(`default_pe_spatial_path`, `default_animetext_dir`, `default_sfx_reader_dir`, …). A path you
could point elsewhere is a Download button that writes where the loader doesn't look;
`test_rows_land_where_the_loaders_look` is the guard. Destinations hang off `models_dir()`
(`_env.py`), which follows the curation home.

## Row kinds

- HF-hub rows (most): `repo` + `files` (+ `subfolder`, `repo_type`), fetched through `_hf.py`
  under the user's token. `gated="<accept-terms url>"` marks a row whose repo needs a click; the
  GUI shows the link.
- Plain-HTTPS rows (`url=`): the soft prompt, the tag KB go through
  `Asset._fetch_http`, with the recovery text on failure.
- Derived rows (`derived=(<input row ids>,)` + `build=callable`): the downloads are inputs
  that stay in the hub cache, and the probe asks for the file `build` writes, so a hub sweep can't
  turn a built row back to "missing". One exists: `danbooru_tags_en` builds its CSV from the 45 MB
  Danbooru wiki mirror. `tagger_onnx` was the other until 2026-09-09, when the tagger's exported
  graph went — see `anime_tools/tagger/CLAUDE.md` for why. A derived row that reads another row's
  product sits after it in catalog order and refuses when the input is absent.

## Packs

`PACKS` (a tuple of `Pack(id, title, description)`, display order) is the vocabulary a Download
button is a button for: `tagger`, `tags`, `masking`, `ocr`, `grouping`. Every
row names exactly one via `Asset.pack` (pinned by
`test_every_row_names_a_pack_and_every_pack_is_known`;
the id list itself is pinned too, because the trainer hides packs by id — its Models panel shows
the Anima half and leaves the curation packs to this GUI). `by_pack()` buckets the catalog in
`PACKS` order, catalog order inside, dropping empty packs; `expand()` turns a mix of row and pack
ids into row ids (catalog order, deduped) and raises `KeyError` naming both vocabularies on a
typo. A name that is both a row and a pack (`tagger`) resolves as the pack. The CLI's `--list`
prints rows under pack headers and accepts pack ids; `/api/models` ships `packs` (only those with
rows) beside `models`, and `/api/models/download` expands a pack before naming the job, so the
job stays row-level. Pack titles and descriptions are server text: the browser renders them as
they arrive.

## Adding a row

1. Add the `Asset(...)` to `catalog()` in dependency order. Pick a stable `id`; the GUI and the
   trainer address rows by it. Give it a `pack=` from `PACKS` — a new kind of weight gets a new
   `Pack` first.
2. Put the loader's default path in this module as a `default_*` function and import it from the
   loader; never spell the path in the loader.
3. If the row is derived, write the `build(dest, log)` callable here and list its inputs in
   `derived`.
4. Tests: `test_downloads.py` — `test_rows_land_where_the_loaders_look` gets the new loader
   pair; a derived row gets a build test that writes its product and asserts the probe flips.
5. `anime_tools/gui/guidebooks/guidebook.md` lists what the Models pane installs.

**Every model here runs on torch.** onnxruntime, `onnx` and `onnxscript` were dropped 2026-09-09:
the tagger's exported graph existed only to beat timm on an Apple CPU, and `mps` beats them both
(`anime_tools/tagger/CLAUDE.md`), while the AnimeText detector moved to the vendored
`anime_tools/vision/yolo12.py` and now fetches the upstream `model.pt` instead of its
`model.onnx`. A new row that would want an ONNX runtime back should say why torch cannot serve
it — the answer so far has always been that torch can.
