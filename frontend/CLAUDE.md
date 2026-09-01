# frontend/CLAUDE.md

Guidance for Claude Code when working in `frontend/` — the source of the GUI's
single committed bundle. The server half is `anime_tools/gui/`, and the split is
also how these two files divide: the root `CLAUDE.md` owns the **seam** (what
the server sends, which routes exist, what is bound where), and everything about
how the browser half is built is here and not repeated there.

## What this is

Solid + TypeScript, bundled by `bun` into **`anime_tools/gui/static/`** —
`index.html` with script and CSS inlined, and the woff2 it points at beside it
(served from `/assets/<name>`). Both are build artifacts that are nevertheless
checked in (the installed package ships no toolchain), so **every source change
here needs `make frontend`**; CI fails on drift between `frontend/src` and that
directory.

The font is the one thing not folded into the page. Inlined as a base64 `data:`
URL it was 2.25 MB of a 2.45 MB file — 92% of a blob git re-stored for every
one-line CSS edit — while the page itself is ~150 KB. Beside the page it is one
immutable object in history and the diff of a frontend change is the part that
actually moved. Nothing depends on the page being a *single* file: it is served
over HTTP by `gui/server.py`, never opened as `file://`, and the wheel's
`package-data` is `static/*`, which globs the whole (flat) directory.

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
  roots, the weights catalog, the saved settings, and the three Settings dialogs
  that edit them (`settingsPane` says which one is open; they are separate
  windows, not tabs, so each entry point opens one and OK saves only its block).
  `/api/settings` is read exactly **once** (`loaded`, a promise the
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
  list in this directory is a bug. Which fields are `advanced` is that same
  rule: the server marks them, `StageForm`'s `FieldGroup` only folds them, one
  fold per group and on that group's own bottom edge. That fold is local,
  unsaved state — it is a look at one group of one stage, not a preference
  about how you work, which is what `layout.ts`'s `help` is.
- **`runner.ts`** — the Run → versions → Undo loop, and the one job the dock
  follows. A Run *writes* (`--apply`, always): there is no Apply button, because
  the caption ladder is what that gate was standing in for — the text a run
  replaces becomes a version badge beside the caption, so what a run did is read
  after it and on the caption rather than agreed to in a dialog first. The
  report it leaves is still read back, as the diff of what it *did* and the dots
  on the rows it touched, and Undo replays that report backwards.

Plus **`downloads.ts`** (a weights fetch: the same job slot, but it reports into
the Settings dialog rather than the dock) and **`state.ts`**, the two primitives
that outlive a render: `persisted` and `createJobFollower`.

**`i18n/`** is the GUI's own text, one file per language — `en.ts` / `ko.ts` /
`ja.ts` / `zh.ts`, with `index.ts` holding the locale signal, `t()` and
`slots()` and nothing else, so a translator edits one table and touches no
machinery. `en.ts` is the schema (`type Dict = typeof en`, exported from there
and imported by the other three), so every other locale is checked key-for-key
and arity-for-arity by `tsc` — a missing string is a build error, never a blank
label. `t()` reads the locale signal, which is why `t().…` inside JSX re-renders
on a switch and why it is never hoisted into a `const` outside a component. A
message with markup in the middle of it (a `<code>`, a link) stays one string
with `{0}` slots and is rendered through `slots()`, so translators never see a
tag and word order is theirs; every locale must spell the same slots.

`api.ts` is the only place a URL is spelled; `types.ts` mirrors the server's
dicts and says which module writes each one.

## Conventions

- **Every user-facing string comes from `i18n/`.** A literal in a component
  ships as English to four languages. Server-owned text is the exception and is
  *not* re-typed here either: stage titles, field labels, argparse help and the
  model catalog's rows are rendered as they arrive, and captions, tags and paths
  are data.
- **Never split a caption in the browser.** Clause structure comes from the
  server (`/api/dataset/item`, `/api/dataset/parse`) — the grammar has one
  implementation, in `anime_tools/captions/`. No `split(",")`, ever. The boxed
  editor slices the caption, but only at offsets that parse returned as `spans`;
  the one thing done to them here is `alignSpans`, which compares two strings
  and invents no boundary of its own.
- **Props are not destructured.** Solid props are getters; `const { x } = props`
  reads them once and freezes the value. Write `props.x` at the use site, and
  pass callbacks rather than setters where a component should not own state.
- A composable is called **once**, from `App()`, and returns accessors. Anything
  it registers (a listener, an `EventSource`) is torn down in its own
  `onCleanup`, so App never has to remember to.
- Components under `components/` **draw and emit** — they hold view-local state
  (an expanded folder, a draft caption) but never fetch a stage, decide what a
  Run may write, or reach into another component's state.
- Two drag sizes are deliberately *not* `persisted`: the dock height and
  `ItemView`'s `--cap-w` move on every pointermove, so each saves once on
  pointerup instead of at frame rate.
- Comments here explain *why* a thing is shaped the way it is, in prose, above
  the code. Match that; a comment restating the line below it is noise.
