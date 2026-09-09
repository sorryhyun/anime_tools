# anime_tools/gui/

The server half of the `anime-tools-gui` web panel, torch-free by test. `frontend/CLAUDE.md`
owns the browser half; this file owns the seam — what the server sends, which routes exist,
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
| `PANEL_FIELDS` (Export's `--out`, `--index`) | as above, but a per-run choice | shown |
| `AUTO_FIELDS` (`--device`) | resolved in the child by `_device.resolve_device` | neither shown nor sent |
| `BASIC_FIELDS` | — | shown; everything else in that stage folds under `advanced (n)` |

Report and mask roots are split per-stage (`report_subpath` / `mask_subpath`) so one stage's
`--from_report` can't read another's report, and so the two generators don't overwrite each other. A
blank root means beside the relevant dataset root (`_root_beside`). A drawer's gate, a required
field and an already-hidden field are never folded — the server settles that, not the browser.
`PREPROCESS_STAGE` (`resize`) runs as a preflight in front of every stage bound to `dst`;
`NO_PREFLIGHT` names the exceptions.

## Other server pieces

- `dataset.py` joins the trees (`src`/`dst`/`masks`/`master`/`out`) into the sidebar's image→caption
  tree by relative path, reads/writes single captions, renders thumbnails. Only `master` and
  `revised` are writable. An image's captions are a ladder (`CAPTION_LADDER`, one `Rung` per
  caption kind), shipped to the browser as per-row dot flags and as `caption_versions`' ordered
  list, where a sidecar rung expands into one entry per caption it holds (`v0`, `v1`,
  `revised@2`…). The sidebar draws one listing in two orderings, `tree` and `groups`;
  `load_groups` reads `<report_root>/<GROUPS_SUBPATH>` and answers rels only, so filters and
  pending dots mean the same thing in both modes. A missing groups manifest is not an error; an
  unparseable one is a 400.
- The browser never splits a caption: clause structure and every tag's `[start, end)` come from
  `/api/dataset/item` and `/api/dataset/parse` via `position_clauses.tag_spans`, which is what lets
  the editor stay a real `<textarea>` with boxes painted behind it.
- `proposals.py` is `stages/replay.py` seen from the server: `load_report` / `report_rows` /
  `apply_one` are imported; an Undo is `apply_one` with the two texts swapped, and Export branches
  to `revert_export` at the top. `proposals.SHAPES` is `contract.REPLAY_SHAPES`, the same objects
  the three stage CLIs bind as their `REPLAY_SPEC` (importing a stage CLI would pull torch in);
  `tests/test_gui_proposals.py` pins the identity.
- `jobs.py` runs one `python -m` subprocess at a time over SSE. A job is a sequence of `Step`s
  sharing one slot, log and stream, because `preprocess_for()` puts `resize` in front of every stage
  bound to the `dst` root; a failing step stops the chain. `masks_merge` and the `NO_PREFLIGHT`
  names sit outside it. A running stage tells the browser nothing but its stdout, so the panel's
  progress bar and log window are read straight off it — `stages/cli/_args.py::make_progress`'s
  `  [done/total] detail` and the `── step i/n: label ──` header this module prints in front of
  each step of a sequence are the two formats parsed there, and a stage printing neither simply
  has no bar.
- The server exits with the window it opened. `main()`'s `--open` implies `--exit-with-window`
  (`--no-exit-with-window` opts out, and the flag alone turns it on for a run that opens nothing),
  which hands `create_app` a `ClientWatch`. The page holds `/api/alive` open for as long as it is on
  screen and the watch stops uvicorn five seconds after the last stream closes, so closing the app
  window reaps the server and the stages it is running (the lifespan's `mgr.shutdown()`) rather than
  leaving them in the terminal. It is a held connection and not a polled heartbeat because a hidden
  tab's timers are throttled to once a minute; the grace is what a reload gets back inside; and
  nothing is armed until the first client attaches, so a `--open` whose browser never appears keeps
  serving.
- `tags.py` merges the two Danbooru KB files (base CSV = taxonomy; optional `.en.csv` replaces only
  the description) for `/api/tags/describe`, cached on both mtimes; a missing KB answers
  `installed: false` rather than erroring.
- `nativepick.py` opens the host's file chooser (zenity/kdialog/osascript/PowerShell) as a
  subprocess for `POST /api/pick`. `/api/ls` is the fallback for headless or remote browsers.
- What the panel may read is `dataset.dataset_bases()`: the curation home plus any root the
  saved settings pin outside it (`reachable()`, lexical). What it may create is narrower —
  `owned()`, under the home only — so a typo in an external root is a missing root, not a new empty
  directory.
- ⚙ Settings is four dialogs, not one tabbed one (`SETTINGS_PANES` in `frontend/src/config.ts`):
  roots, stage defaults + preflight, models, update. Only the open pane is mounted, so
  `SettingsOut` carries `null` for the other three. The Models pane runs
  `python -m anime_tools.downloads` (the `model-catalog` skill): `/api/models` answers
  `{packs: [{id, title, description}], models: [Asset.to_dict()…], models_dir}` — `packs` in
  `PACKS` order, only packs that have rows, each model carrying its `pack` — and
  `POST /api/models/download {ids}` takes row or pack ids, expanding a pack before the job is named
  so the job stays `download:<row ids>`.
- `guidebook.py` + `guidebooks/*.md` are the manual the panel opens (☰ → 📖). One book per UI
  language, keyed by the same locale ids `frontend/src/i18n/` uses, and a locale with no book of
  its own reads the English one rather than 404ing. They sit in the package because
  `packages.find` ships only `anime_tools*`: an installed copy has no `docs/` tree, so a book
  under it would open an empty window everywhere but a checkout. `GET /api/guidebook?lang=` sends
  the markdown unrendered — the server has no markdown library and the page already owns how the
  panel looks — plus `base`, the book's own directory on GitHub, which is what the browser
  resolves a book's relative `../../../docs/x.md` links against so a click leaves for GitHub
  instead of dying inside a `<dialog>`. The `translator` agent owns keeping the four in step.
- `updates.py` is the Update pane's half of `anime_tools/update.py` (which owns what an update
  *is*): `GET /api/update` answers the version row — installed (`/api/info`'s `version`), the latest
  release, their ordering, the notes, and which of the three install shapes this is — and
  `POST /api/update/run {version}` starts it as `update:<tag>`, one `python -m anime_tools.update`
  step in the same single job slot the stages and downloads share. GitHub's answer is cached six
  hours in the settings file (`update_check`), and the network is touched only on `?force=true`
  ("Check now") or when the `auto_update` checkbox is on and the cache has aged out — so an offline
  or opted-out panel still renders what it knows. Only the installer's `uv tool` environment may be
  rewritten; a checkout or a venv install gets the refusal (`refusal()`, the same answer the pane
  greys its button on) as a 409. The route is a plain `def`: FastAPI runs it in the threadpool, so
  the GitHub call cannot stall a streaming job.
- The panel's own chrome is translated (`frontend/src/i18n/`), and so are the two things the
  browser reads *as* prose rather than as data: the dock's navigation (the panel buttons and the
  stage names on them) and what the stage bar's (?) reveals (a stage's doc and its notes), both
  keyed by the registry's own ids and both falling back to what this server sent. So a docstring
  edited here is the English of that (?) and stays the answer in every locale that has no line for
  the stage. Everything else the server owns (argparse labels and help, the model catalog) ships as
  it arrives.

`static/` is the committed bundle built from `frontend/` by `make frontend`; never edit it by
hand, and CI fails on drift.

Tests: `test_gui` (the guidebook route and that every UI language has a book that ships
included), `test_gui_dataset`, `test_gui_proposals`, `test_gui_nativepick`,
`test_gui_updates` (version ordering, the cached check, the refused install shapes — never the
network), `test_boundary` (`create_app()` without torch).
