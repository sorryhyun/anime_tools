# design/

The **anime_tools design system** as a Claude Design canvas — six artboards
built to read at a glance: a hero board showing the whole GUI in one annotated
miniature, then Foundations, Controls, and the tree / caption-card / run-bar
assemblies zooming into its parts.

Published at <https://claude.ai/code/artifact/50c1186a-7d09-4416-99ec-e95ac73c7ea3>
(private; share from the page's own menu). Where saving is enabled the canvas is
a live editor: click an element, edit it in the properties panel, **Save**
publishes a new version for everyone. Otherwise it views and exports PNG/PDF.

## The one rule

**Every value on these boards is lifted from `frontend/src/styles.css` and the
components beside it — nothing here is invented.** That is the whole point: the
deck is a *reading* of the stylesheet, so a number that drifts is a bug, not a
design choice. When the GUI's CSS changes, change the board to match; when the
board wants a value the CSS does not have, change the CSS first.

The boards say so in their own words too — "eleven colour tokens and two font
stacks on `:root`", "193 lines of CSS, no framework, no
component library". Keep them honest.

## Layout

| Path | What it is |
| --- | --- |
| `boards/<Name>.html` | one artboard's **body** — the design content, hand-edited |
| `helmet.html` | the `<style>` block every board shares (font, ground, links) |
| `canvas.json` | positions, frame sizes, sticky notes, launch view |
| `fonts/pretendard-latin.woff2` | 14.8 KB Latin subset, inlined at build |
| `build.py` | wraps each board into a Design Component and seeds the canvas |
| `build/` | generated, git-ignored — never hand-edit anything in here |

`boards/Main.html` is the entry artboard (titled *At a glance* on the canvas);
the Design Components format requires that filename. `canvas.json` is the source of
truth for which boards exist — `build.py` fails if the two disagree in either
direction.

## Rebuilding

```sh
python3 design/build.py                    # assemble build/*.dc.html only
python3 design/build.py --seeder <dir>     # ...and seed the publishable page
```

Seeding needs the `/design` skill, which ships the canvas-editor payload and
`seed-canvas.mjs`. Its directory is version-pathed and lives outside the repo,
so run `/design` in Claude Code to extract it and pass the path (or set
`$DESIGN_SKILL_DIR`). Then publish `build/anime-tools-design-system.html` to the
artifact URL above — **the same URL**, or you get a second, orphaned canvas.

Edits made in the published editor do not flow back here. To pick them up, read
the artifact and run `seed-canvas.mjs --extract` into a scratch directory, then
port the changes into `boards/`.

## The font

`fonts/pretendard-latin.woff2` is `frontend/fonts/PretendardVariable.subset.woff2`
subset again to Latin — the GUI's own subset carries Hangul and is 1.7 MB, too
large to inline six times. Each artboard renders in an isolated iframe with no
network egress, so the face has to ride inside the file as a base64
`@font-face`; this mirrors what `frontend/build.ts` does for the shipped GUI.

Pretendard is **SIL Open Font License 1.1**, © 2021 Kil Hyung-jin (길형진).
See `frontend/fonts/README.md` for the full notice and the upstream link.
Regenerate with:

```sh
uvx --from 'fonttools[woff]' pyftsubset frontend/fonts/PretendardVariable.subset.woff2 \
  --unicodes='U+0020-007E,U+00A0,U+2018-201D,U+2013,U+2014,U+2026,U+2190-2193,U+2212,U+2713,U+26A0,U+2699,U+00B7,U+2022,U+25B8,U+25BE' \
  --layout-features='' --flavor=woff2 --output-file=design/fonts/pretendard-latin.woff2
```

## Authoring a board

A board body is plain HTML with **inline `style=` attributes** — that is what
the canvas editor's properties panel binds to, so a stylesheet class is a value
nobody can tweak. Lay sibling groups out with `display:flex`/`grid` and `gap`,
never source whitespace: gap spacing survives drag-reorder and delete, text
nodes do not. Icons are inline SVG, never emoji. `{{...}}` is a template hole in
this format — avoid it in copy.
