# frontend/CLAUDE.md

Guidance for Claude Code when working in `frontend/` — the source of the GUI's
single committed bundle. The server half is `anime_tools/gui/`; the root
`CLAUDE.md` describes how the two meet.

## What this is

Solid + TypeScript, bundled by `bun` into **`anime_tools/gui/static/index.html`**
— one file, fonts and CSS inlined, committed to the repo. The bundle is a build
artifact that is nevertheless checked in (the installed package ships no
toolchain), so **every source change here needs `make frontend`**; CI fails on
drift between `frontend/src` and that file.

```bash
make frontend        # rebuild the committed bundle (scripts/build_frontend.sh)
make frontend-dev    # bun dev server with hot reload, /api proxied to `make gui`
cd frontend && bun run check          # tsc --noEmit; strict, and it must stay clean
cd frontend && bun run format:check   # prettier; the pre-commit hook formats staged files
```

`.prettierignore` holds `index.html` (copied verbatim into the bundle), `src/styles.css`
(hand-compacted: one rule per line, so `design/boards` can quote it line-for-line)
and `*.md`. Those three are edited by hand and must stay unformatted.

## Architecture

`App.tsx` is **wiring only** — it creates the state modules, bridges the two of
them that must talk, and hands their signals to components. Nothing derived,
nothing fetched, no business rule lives there. The state is five composables at
`src/`, each a plain function called once inside `App()`:

- **`config.ts`** — what the server says about *itself*: `/api/info`, the dataset
  roots, the weights catalog, the saved settings, and the Settings dialog that
  edits them. `/api/settings` is read exactly **once** (`loaded`, a promise the
  stage forms are seeded from), and the four resources are the single refetch
  points, so a finished download and a saved root land everywhere at once.
- **`layout.ts`** — which panes are open and how tall the dock is. Preferences
  that survive a reload but mean nothing to the server.
- **`dataset.ts`** — the image listing, the selection and the selected row's
  detail. The selection *is* the page's address: it is mirrored into the location
  hash both ways (`#rel|kind`), so a link into the GUI opens on an image.
  `reloadRels` re-stats named rows in place after a job wrote; `onSaved` folds a
  just-saved caption back into the row rather than re-walking the tree.
- **`stages.ts`** — the stage registry and the form over the open one, including
  how the dock's buttons bucket stages into panels. Every field comes from the
  stage CLI's own argparse (`gui/stages.py` dumps it in a child interpreter), so
  **nothing about a flag is ever re-typed here** — a label, a default or a choice
  list in this directory is a bug.
- **`runner.ts`** — the Run → diff → Apply → Undo loop, and the one job the dock
  follows. Run computes proposals and writes only the report; Apply replays *that
  report*, so it can write nothing the diff did not show, which is why it stays
  blocked until a Run has produced one and why the run is dropped the moment the
  form moves on.

Plus **`downloads.ts`** (a weights fetch: the same job slot, but it reports into
the Settings dialog rather than the dock) and **`state.ts`**, the two primitives
that outlive a render: `persisted` and `createJobFollower`.

`api.ts` is the only place a URL is spelled; `types.ts` mirrors the server's
dicts and says which module writes each one.

## Conventions

- **Never split a caption in the browser.** Clause structure comes from the
  server (`/api/dataset/item`, `/api/dataset/parse`) — the grammar has one
  implementation, in `anime_tools/captions/`. No `split(",")`, ever.
- **Props are not destructured.** Solid props are getters; `const { x } = props`
  reads them once and freezes the value. Write `props.x` at the use site, and
  pass callbacks rather than setters where a component should not own state.
- A composable is called **once**, from `App()`, and returns accessors. Anything
  it registers (a listener, an `EventSource`) is torn down in its own
  `onCleanup`, so App never has to remember to.
- Components under `components/` **draw and emit** — they hold view-local state
  (an expanded folder, a draft caption) but never fetch a stage, decide what
  Apply may write, or reach into another component's state.
- Two drag sizes are deliberately *not* `persisted`: the dock height and
  `ItemView`'s `--cap-w` move on every pointermove, so each saves once on
  pointerup instead of at frame rate.
- Comments here explain *why* a thing is shaped the way it is, in prose, above
  the code. Match that; a comment restating the line below it is noise.
