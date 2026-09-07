# anime_tools/gui/

The server half of the `anime-tools-gui` web panel, torch-free by test. `frontend/CLAUDE.md`
owns the browser half; this file owns the **seam** — what the server sends, which routes exist,
what is bound where — and nothing about how the browser is built is repeated here.

## `stages.py`: request dataclass → form schema → argv

`stages.py` turns each stage's request dataclass into a form schema and a form payload back
into the request's argv. The registry is `stages/registry.py` (re-exported); `schema()` walks
`_request.args_of(Request)` — the same field list the CLI parser is generated from, so a flag's
kind, default, help, group and drawer reach the form without argparse in between — and
`build_argv()` coerces the payload into a namespace, reads it with `Request.from_namespace` (so the
request's own validation runs server-side, as a 400) and spells it with `to_argv()`. Both happen
in-process: the request modules are torch-free by test, and building all eleven schemas takes
~0.1 s, so there is no child interpreter and no cache. Field binding:

| Map | Bound to | Shown? |
|---|---|---|
| `ROOT_FIELDS` (`--src`, `--dst`, `--image-dir`, …) | dataset roots | hidden |
| `SETTING_FIELDS` (`--path_pattern`, `--tagger_dir`, `--checkpoint`, `--prompt_embed`) | Settings stage defaults | hidden |
| `REPORT_SETTING` / `REPORT_INPUTS` (`--report_dir`, groups' `--out`, Export's `--index`) | `report_root` + the stage's own tail | hidden |
| `MASK_FIELDS` | `mask_root` + tail | hidden |
| `PANEL_FIELDS` (Export's `--out`, `--index`) | as above, but a per-run choice | **shown** |
| `AUTO_FIELDS` (`--device`) | resolved in the child by `_device.resolve_device` | neither shown nor sent |
| `BASIC_FIELDS` | — | shown; everything else in that stage folds under `advanced (n)` |

Report and mask roots are split per-stage (`report_subpath` / `mask_subpath`) so one stage's
`--from_report` can't read another's report, and so the two generators don't overwrite each other. A
blank root means *beside* the relevant dataset root (`_root_beside`). A drawer's gate, a required
field and an already-hidden field are never folded — the server settles that, not the browser.
`PREPROCESS_STAGE` (`resize`) runs as a preflight in front of every stage bound to `dst`;
`NO_PREFLIGHT` names the exceptions.

## Other server pieces

- `dataset.py` joins the trees (`src`/`dst`/`masks`/`master`/`out`) into the sidebar's image→caption
  tree by relative path, reads/writes single captions, renders thumbnails. Only `master` and
  `revised` are writable. An image's captions are a **ladder** (`CAPTION_LADDER`, one `Rung` per
  caption kind), shipped to the browser as per-row dot flags and as `caption_versions`' ordered
  list, where a sidecar rung expands into one entry per caption it holds (`v0`, `v1`,
  `revised@2`…). The sidebar draws one listing in two orderings, `tree` and `groups`;
  `load_groups` reads `<report_root>/<GROUPS_SUBPATH>` and answers **rels only**, so filters and
  pending dots mean the same thing in both modes. A missing groups manifest is not an error; an
  unparseable one is a 400.
- **The browser never splits a caption**: clause structure and every tag's `[start, end)` come from
  `/api/dataset/item` and `/api/dataset/parse` via `position_clauses.tag_spans`, which is what lets
  the editor stay a real `<textarea>` with boxes painted behind it.
- `proposals.py` is `stages/replay.py` seen from the server: `load_report` / `report_rows` /
  `apply_one` are imported; an Undo is `apply_one` with the two texts swapped, and Export branches
  to `revert_export` at the top. `proposals.SHAPES` is `contract.REPLAY_SHAPES`, the same objects
  the three stage CLIs bind as their `REPLAY_SPEC` (importing a stage CLI would pull torch in);
  `tests/test_gui_proposals.py` pins the identity.
- `jobs.py` runs one `python -m` subprocess at a time over SSE. A job is a *sequence* of `Step`s
  sharing one slot, log and stream, because `preprocess_for()` puts `resize` in front of every stage
  bound to the `dst` root; a failing step stops the chain. `masks_merge` and the `NO_PREFLIGHT`
  names sit outside it. A running stage tells the browser nothing but its stdout, so the panel's
  progress bar and log window are read straight off it — `stages/cli/_args.py::make_progress`'s
  `  [done/total] detail` and the `── step i/n: label ──` header this module prints in front of
  each step of a sequence are the two formats parsed there, and a stage printing neither simply
  has no bar.
- `tags.py` merges the two Danbooru KB files (base CSV = taxonomy; optional `.en.csv` replaces only
  the description) for `/api/tags/describe`, cached on both mtimes; a missing KB answers
  `installed: false` rather than erroring.
- `nativepick.py` opens the *host's* file chooser (zenity/kdialog/osascript/PowerShell) as a
  subprocess for `POST /api/pick`. `/api/ls` is the fallback for headless or remote browsers.
- What the panel may **read** is `dataset.dataset_bases()`: the curation home plus any root the
  *saved* settings pin outside it (`reachable()`, lexical). What it may **create** is narrower —
  `owned()`, under the home only — so a typo in an external root is a missing root, not a new empty
  directory.
- ⚙ Settings is **three dialogs, not one tabbed one** (`SETTINGS_PANES` in `settings.py`): roots,
  stage defaults + preflight, models. Only the open pane is mounted, so `SettingsOut` carries
  `null` for the other two. The Models pane runs `python -m anime_tools.downloads` (the
  `model-catalog` skill): `/api/models` answers `{packs: [{id, title, description}], models:
  [Asset.to_dict()…], models_dir}` — `packs` in `PACKS` order, only packs that have rows, each
  model carrying its `pack` — and `POST /api/models/download {ids}` takes row *or* pack ids,
  expanding a pack before the job is named so the job stays `download:<row ids>`.
- The panel's own chrome is translated (`frontend/src/i18n/`), and so is the dock's navigation —
  the panel buttons and the stage names on them, keyed by the registry's own ids. Everything else
  the server owns (a stage's doc and notes, argparse labels and help, the model catalog) ships as it
  arrives.

`static/` is the committed bundle built from `frontend/` by `make frontend`; never edit it by
hand, and CI fails on drift.

Tests: `test_gui`, `test_gui_dataset`, `test_gui_proposals`, `test_gui_nativepick`,
`test_boundary` (`create_app()` without torch).
