# Workspace + Export

Status: **Phases 1, 3 and 5 done**, 2026-08-31; Phase 4's caption ladder — and
then the version history that let it take Apply's job — done 2026-09-01. The
rest of Phase 4 and then Phase 2 are pending, in that order.
Reframes the curation side around a **workspace** the tools own, and an explicit
**Export** that publishes from it.

## The idea

Today every stage writes straight at the trainer's paths: the clause rewrite
into `post_image_dataset/resized/`, the mask generators into
`post_image_dataset/masks/`, and autotag + the multiview audit into the caption
master under `image_dataset/`. There is no moment at which curation is *done* —
an `--apply` is a publish, and the only thing standing between a half-finished
run and the trainer is that you have not started the trainer yet.

The workspace makes that moment explicit:

```
<home>/
  image_dataset/                  INPUT — read-only for the tools from now on
  workspace/                      everything the tools produce
    master/<rel>.txt                revised master  (autotag / audit / GUI master edit)
    resized/<rel>.{png,txt,variants.txt}
    masks/<rel>/{stem}_mask.png
    captions/{autotag,position,resize,multiview_audit}/report.json   <- the diffs
    groups/groups.json
    export/report.json              the export ledger
  post_image_dataset/             OUTPUT — written only by Export
    resized/  masks/
```

The invariant, and the thing worth a test of its own:

> **No stage writes outside `workspace/`. Export is the only thing that touches
> `image_dataset/` or `post_image_dataset/`.**

Three decisions are settled and the rest of the plan assumes them:

| | |
|---|---|
| **Revised master** | Goes to a `workspace/master/` overlay. `image_dataset/` becomes read-only for the tools. |
| **Export destination** | Back onto the existing paths (`image_dataset/`, `post_image_dataset/{resized,masks}`), so `docs/contract.md` stays frozen and the trainer needs no change. |
| **Pixel artifacts** | Always copy. The export tree is independent bytes; a re-export stays cheap because identical files are skipped, not rewritten. |

## Order of work

Execution order is **1 → 3 → 4 → 2**, not phase order. Phases 1 and 3 are
independent of the master overlay and much lower risk, so the workspace and the
Export button land first and are usable on their own; Phase 2 goes last because
the `apply_one` read/write split is where the real care goes.

---

## Phase 1 — the layout ✅

Cheap, and two of the four artifact classes move for free.

> **Landed 2026-08-31.** Two things grew past the sketch below, both because the
> invariant demanded it. First, the **stage CLI defaults moved too**: leaving
> `--dst post_image_dataset/resized` in `cli/_args.py` would have made "no stage
> writes outside the workspace" true only from the GUI, so every `--dst`,
> `--report_dir` and `--out` default is now written in terms of the
> `anime_tools.workspace` constants, along with the dbv4 feature cache, the
> caption index and the GUI's own job logs. Second, `gui/settings.py` was split
> out of `server.py` so the migrate CLI can read the settings file without
> importing FastAPI, rather than retyping the filename.
>
> **The pipeline is mid-move until Phase 3.** The stages now write
> `workspace/resized`, and nothing publishes to `post_image_dataset/` yet, so
> the trainer sees an empty tree. Point ⚙ Settings › Dataset roots at the old
> paths, or pass `--dst post_image_dataset/resized`, until Export lands.

**`anime_tools/_env.py`** gains `workspace_dir()`, in the same three-step shape
as `models_dir()`: `ANIME_TOOLS_WORKSPACE` → `<home>/workspace`.

**`anime_tools/gui/dataset.py`** — `DEFAULT_ROOTS` becomes:

```python
DEFAULT_ROOTS: dict[str, str] = {
    "src": "image_dataset",  # input, read-only
    "master": "workspace/master",  # NEW — revised masters (Phase 2 fills it)
    "dst": "workspace/resized",
    "masks": "workspace/masks",
    "out": "post_image_dataset",  # NEW — the export destination
}
```

`Roots` grows from three fields to five. Keeping the names `src` / `dst` /
`masks` is deliberate: `ROOT_FIELDS`, `root_paths`, `caption_paths`,
`mask_path` and `list_items` keep working untouched and only the *default paths*
change — `master` and `out` are purely additive.

Two things then follow with no further code, because they are already derived
rather than hardcoded:

- **The diffs.** `server.report_root` is
  `PurePosixPath(rel_to_home(roots.dst)).parent` whenever Settings is blank, so
  moving `dst` under `workspace/` puts every stage report at
  `workspace/captions/<stage>/`. "Diffs saved to the workspace" is essentially
  this one default.
- **The group manifest.** `dataset.GROUPS_SUBPATH` hangs off the same
  `report_root`, so `groups.json` lands at `workspace/groups/groups.json` and
  stays pinned to `build_groups --out` by `tests/test_gui_groups.py`.

**Migration.** `python -m anime_tools.workspace.migrate` moves an existing
`post_image_dataset/{resized,masks,captions,groups}` into `workspace/`. An
install that already has explicit roots saved in ⚙ Settings is unaffected either
way — only the defaults move — so the migrate CLI is a convenience, not a gate.

**Docs.** `docs/contract.md` §2's rows do not change (Export writes exactly the
paths the trainer already reads); only the *Producer* column's wording moves to
"Export, from the workspace", plus a line noting the workspace is
curation-private in the same sense as the near-twin feature cache.

### What it touched

- `anime_tools/workspace/__init__.py` — **new**: `DEFAULT_ROOTS`, `OUTPUT_ROOTS`,
  `EXPORT_ROOTS`, `RESIZED` / `MASKS` / `REPORTS` / `GROUPS`, `LEGACY_ROOTS`
- `anime_tools/workspace/migrate.py` — **new**: `plan_moves`, `pinned_roots`
- `anime_tools/_env.py` — `workspace_dir()`
- `anime_tools/gui/settings.py` — **new**, split out of `server.py`
- `anime_tools/gui/dataset.py` — the layout imported, `Roots.items()`, five roots
- `anime_tools/gui/server.py` — `roots_for(**overrides)`, `make_output_dirs`, job logs
- every stage CLI's `--dst` / `--report_dir` / `--out` default, plus
  `tagger/feature_cache.py`, `captions/index.py`, `grouping/cli/_decensored.py`
- `frontend/src/types.ts`, `components/SettingsDialog.tsx` — five roots in ⚙ Settings
- `docs/contract.md`, `CLAUDE.md`
- `tests/test_workspace.py` — **new** — plus `test_gui{,_dataset,_proposals}.py`,
  `test_stage_cli_args.py`

---

## Phase 3 — Export as a stage, not a special case ✅

> **Landed 2026-08-31.** Three decisions differ from the sketch below, all found
> by reading what the run bar actually does:
>
> - **No `--from_report`.** The replay flag exists so Apply cannot re-run a
>   model; Export loads none, so Apply just runs the pass again for real and
>   re-decides every row at write time. That also sidesteps the run bar fetching
>   caption proposals for a report whose rows are file copies —
>   `schema()["replay"]` is false, so it never asks. (`audit_apply` already
>   works this way.)
> - **`NO_PREFLIGHT`.** Export binds `dst`, which would have earned it the
>   resize preflight. It publishes the resized tree rather than consuming it, so
>   quietly resizing first would make a publish do work nobody asked for; an
>   empty tree is a refusal instead.
> - **`proposals.undo` branches at the top** to `revert_export`, rather than the
>   `/api/jobs/{id}/undo` route growing a second entry point. Same answer shape,
>   so the frontend needed no change.
>
> The caption index binds through `REPORT_INPUTS["export"] = "index"` — the
> mechanism `audit_apply` already used for a report it *reads* — so it follows
> `report_root` like everything else and stays off the form.

Built as `anime_tools/stages/export_workspace.py` +
`anime_tools/stages/cli/export_workspace.py` on the existing `cli/_args.py`
scaffolding. That is the lever: it inherits dry-run-by-default, `--apply`,
`--path_pattern` scoping, `--report_dir` and `--from_report` replay, and
therefore shows up in the GUI's stage registry, job runner, SSE log and
per-image scope with almost no new server code.

Registry entry in `gui/stages.py`:

```python
(
    Stage(
        "export",
        "Export workspace",
        "anime_tools.stages.cli.export_workspace",
        "Export",
        "stages",
        report=("report_dir", "report.json"),
    ),
)
```

with every path bound to Settings and nothing on the form:

```python
ROOT_FIELDS["export"] = {
    "src": "master",
    "resized": "dst",
    "masks": "masks",
    "out": "out",
}
```

**Rows are per artifact, not per image** — one image contributes up to five:

```json
{"rel": "…", "kind": "image|mask|master|caption|variants",
 "src": "…", "dst": "…", "status": "…"}
```

Plus one whole-dataset row: `caption_index.json`. Phase 1 moved its default to
`workspace/captions/`, because it is something the tools produce — but it is a
`docs/contract.md` §2 artifact the trainer reads, so Export has to publish it
back to `post_image_dataset/captions/` like everything else in that table.

Statuses reuse the existing vocabulary: `identical` (skip) / `would-copy` /
`copied` / `overwrote` / `missing-source`. Sameness is `(size, mtime_ns)` for
pixels and a text compare for captions — the same cheap stamp
`grouping/cli/match_decensored.py` uses, and enough to make a re-export a walk
rather than a rewrite of the whole resized tree.

**Undo** deletes what the export *created* and restores caption text it
overwrote. A pixel it overwrote reports `overwrote, not undoable` rather than
carrying a backup: the workspace is the source of those bytes and re-export is
idempotent, so there is nothing a snapshot would buy.

**Refusals.** An empty workspace is an error with the "run Resize first" shape,
not a silent zero-row success.

### What it touched

- `anime_tools/stages/export_workspace.py` — **new**: `ExportPaths`, `plan_export`,
  `export_one`, `run_export`, `revert_export`, `rows_from_report`
- `anime_tools/stages/cli/export_workspace.py` — **new**: `--apply`, `--undo`
- `anime_tools/gui/stages.py` — `STAGES`, `ROOT_FIELDS`, `REPORT_INPUTS`, `NO_PREFLIGHT`
- `anime_tools/gui/proposals.py` — `EXPORT_STAGE`, `_undo_export`, the branch in `undo`
- `docs/contract.md`, `CLAUDE.md`
- `tests/test_export.py` — **new**, 17 tests — plus `test_gui.py`, `test_gui_proposals.py`

---

## Phase 4 — GUI

Most of this arrived with Phase 3: registering Export as a stage gave it a dock
panel, a form, the job runner, the SSE log, per-image scope, and a working
Run → Apply → Undo, with no frontend change at all. What is left is the part the
registry cannot infer — telling the user what is *pending*:

- **A per-row export flag** — in workspace / exported / stale — computed from
  the export report plus the mtime stamp above, so the sidebar says what is
  published without opening anything. It belongs beside `resized` and `mask` in
  `_row`'s row *flags*, not on the caption strip: the strip is the caption
  ladder now (see below), and export is a fact about the image.
- **A count in the status line** — "12 images pending export" — so the button
  means something before you press it.

### Touchpoints

- `anime_tools/gui/dataset.py` — the export state on each row (`_row`)
- `anime_tools/gui/server.py` — the pending-export count on `/api/dataset`
- `frontend/src/components/{DatasetTree,StatusLine}.tsx` — the flag and the count

### Done ahead of the rest: the caption ladder ✅

> **Landed 2026-09-01.** The panel used to be one editable card per *tree*
> (master, derived) plus a read-only variants table, which made you answer
> "which tree?" before you could type — a question the workspace/Export split
> had already made the wrong one to ask, since the trees are where the tools
> keep working copies and Export is the moment one becomes training data.
>
> `dataset.CAPTION_LADDER` is now the one declaration of what captions an image
> has: a `Rung` each for `master`, `derived` and the generated `variants`
> sidecar, in order. It reaches the browser twice — as the `captions` flag map
> on every `/api/dataset` row (the dot strip) and as `caption_versions`' ordered
> `versions` on `/api/dataset/item`, with the sidecar rung **expanded** into one
> entry per label (`v0`, `v1`, `r1`…), each already parsed, because a variant is
> a caption and the browser may not split one. `CAPTION_KINDS` — what
> `write_caption` accepts — is derived as the `editable` rungs, and the listing
> ships `ladder` so `DatasetTree` retypes no rung names.
>
> The panel is one editor with a badge per version. It opens on what the last
> run rewrote, else the newest writable caption on disk; a badge is the switch,
> and a run's diff rides under the editor when its rung is the one on screen and
> is a one-line pointer to that badge otherwise.

### Then: the ladder took Apply's job ✅

> **Landed 2026-09-01.** Apply is gone from the run bar. A Run passes `--apply`
> and writes; the bar is **Run / Run batch / Undo**.
>
> Apply existed because a stage run was irreversible in the one way that
> mattered — it overwrote the derived caption and the previous text was simply
> gone — so the write had to be inspected first. The ladder makes that false:
> `captions/history.py` writes a `{stem}.history.txt` sidecar in the variants
> sidecar's shape, every caption write into that rung pushes what it replaced
> onto it (`_caption_io.write_caption(history_by=…)`, threaded through
> `apply_one` and `ReplaySpec.history_by` so a replay files the same version its
> live pass would), and the ladder expands it into `revised@1`, `revised@2` …
> badges. The question Apply was asking is now answered *after* the run, on the
> caption, by clicking one badge to the left.
>
> Two names moved with it. The `dst` rung is **`revised`**, not `derived` — a
> caption there is the master revised by a stage, which is what the badge should
> say — and a `CaptionEntry` now carries the `rung` it came out of, so
> `revised@2` wears the history rung's colour rather than the panel guessing a
> hue from the shape of an id.
>
> **This takes Phase 2's name.** That phase's sketch below calls the master
> overlay `Rung("revised", "master", …)`; `revised` is the `dst` rung now, so
> the overlay needs its own name when it lands. The rest of the sketch stands as
> written.
>
> The report a run leaves is unchanged and still read back — as the diff of what
> it *did*, and as the sidebar dots on the rows it touched — and Undo still
> replays it backwards, itself pushing a version, so an unintended Undo is one
> badge away as well. What is gone from the frontend is `ApplyDialog.tsx`, the
> `applyBlocked` / freshness gating and `formKey`: a diff that no longer gates a
> write does not need to be dropped when the form moves on.

---

## Phase 2 — the master overlay

Last, because it is the one change with blast radius.

`stages/_walk_captions.py` already owns the one-line rule "revised first, master
as the read-only fallback". It gets a twin of exactly the same shape:

```python
def resolve_master(overlay_dir: Path, source_dir: Path, rel: Path) -> Path | None:
    """Overlay first, original as the read-only fallback."""
```

and `resolve_caption`'s master fallback goes *through* it, so a revised master is
visible to the clause rewrite too and the rule stays written once. Autotag and
the multiview audit read through it and write to the overlay.

### The snag: `apply_one` needs a read/write split

`stages/replay.apply_one` takes a single `target` that is both the file it reads
for the drift check and the file it writes. With an overlay those differ: the
first apply reads `image_dataset/<rel>.txt` and writes
`workspace/master/<rel>.txt`, which does not exist yet — so the ladder hits
`missing-caption` and refuses every row.

The fix is to check drift against the *resolved read path* and write to the
overlay:

```python
def apply_one(target: Path, before: str, after: str, *,
              read_from: Path | None = None,   # defaults to target
              apply: bool, newline: bool = False, drop_variants: bool = False) -> str:
```

`read_from` defaulting to `target` keeps every existing call site byte-identical,
which matters because this is the one drift-guarded write in the package:
`replay_rows`, the audit's `apply_findings` / `apply_curated` / `revert_curated`,
and `gui/proposals` all funnel through it. `replay_rows`' `root = src if
spec.target_root == "src" else dst` becomes a lookup on the roots mapping so it
can name `master`.

### Then, mechanically

- `ReplaySpec.target_root`: `"src"` → `"master"` for the `autotag` and `audit`
  specs — in the CLIs **and** in the hand-copies in `gui/proposals.SHAPES`.
  `tests/test_gui_proposals.py` compares them field for field, so the two move
  together or the test says so.
- `proposals.CAPTION_KIND`: `{"src": "master", "dst": "revised"}` →
  `{"master": "<the overlay's name>", "dst": "revised"}`.
- **`dataset.CAPTION_LADDER` gains one row**: the master overlay, editable, in
  the `master` root, between `master` and `history` — **not** under the name
  `revised`, which the `dst` rung took when Apply was removed (see above) —
  with `master` flipped to `editable=False` — `image_dataset/` is no longer
  writable, and `CAPTION_KINDS`, `caption_paths`, the row's `captions` map, the
  panel's badges and the sidebar's dot strip are all derived from that tuple, so
  that line plus a `caption.revised` label and a `.dot.revised` colour is the
  whole GUI half of this phase. The three-state master dot the earlier sketch
  called for is not needed: a revised master is its own rung and its own badge,
  which is what the ladder was for.

### Touchpoints

- `anime_tools/stages/_walk_captions.py` — `resolve_master`, `resolve_caption`
- `anime_tools/stages/replay.py` — `apply_one`, `replay_rows`, `ReplaySpec` docstring
- `anime_tools/stages/{autotag,multiview_audit}.py` and their CLIs
- `anime_tools/gui/{proposals,dataset}.py` — plus `HISTORY_OF`, if the overlay
  rung is to keep versions the way `revised` does (it should: it is the rung
  autotag and the audit will be writing)
- `frontend/src/i18n/*.ts`, `frontend/src/styles.css` — the new rung's label and hue
- `tests/test_gui_proposals.py`, `tests/test_replay.py`, `tests/test_gui_dataset.py`

---

## Tests that move or arrive

| Test | Why |
|---|---|
| `tests/test_gui_dataset.py` | roots grew to five; caption kinds grew to three |
| `tests/test_caption_history.py` | **new** — the sidecar, the write seam, the badges |
| `tests/test_gui_groups.py` | the manifest now resolves under `workspace/` |
| `tests/test_gui_proposals.py` | the two `ReplaySpec` copies flip `target_root` |
| `tests/test_stage_cli_args.py` | the export stage's shared flags |
| `tests/test_export.py` | **new** — plan / apply / idempotence / scope |
| `tests/test_workspace_boundary.py` | **new** — run each caption stage's `--apply` in a tmp home and assert nothing under `src` or `out` was touched |

That last one is the invariant at the top of this document, made enforceable —
the same trick `tests/test_boundary.py` plays on the import graph.

---

## Phase 5 — the resized tree is the one decode substrate ✅

Done 2026-08-31, out of band: it depends on nothing in Phase 2 or 4 and they
depend on nothing in it.

### What was wrong

The pipeline had two input trees, and which one a stage read was historical
rather than decided. The caption stages walk `workspace/resized/` on purpose —
`stages/autotag.py` says why in one line: *"Tagging runs on the resized image
because that is the pixel data training sees."* Masking and grouping walked the
master, for no reason beyond nobody having extended the rule.

That is not merely inelegant. `docs/contract.md`'s mask row has the trainer's
loader NEAREST-resize each mask onto the latent's pixel size, so a mask cut from
master pixels is rescaled onto resized geometry. Free-fit preserves aspect and
leaves a sub-patch crop residual, so that rescale is *nearly* free — except when
the `max_ratio` 4.0 clamp fires, where the crop is real and the mask lands off
the subject. Long strips and banners are exactly the images you would want a
mask on.

### The move

`ROOT_FIELDS` binds `groups` / `masks_sam` / `masks_mit` to `dst` instead of
`src`, and `build_groups --source-dir` defaults to `WS.RESIZED`. That is the
whole change; `preprocess_for` then gives all three the resize preflight for
free, because it keys off the `dst` binding. The masking CLIs keep `--image-dir`
`required=True` — the CLI stays pointable at any tree, the *GUI binding* is the
policy.

`audit_apply` stays on `src`: it rewrites captions the audit already found and
never opens an image. `export` stays in `NO_PREFLIGHT` for the reason Phase 3
gave.

### The two prerequisites, both landed

**The near-twin feature cache needed a staleness stamp.** It is keyed by
parent-dir hash + stem — a *location*, which does not move when `resize`
regenerates the tree at a different tier or crop anchor. Over the master that
was sound by construction (those bytes never change); over `resized/` it would
have served features for pixels that no longer exist, with no symptom. Entries
now carry `(size, mtime_ns)` + `FEATURE_CACHE_VER`, checked by `_load_feature`;
anything wrong means recompute, never an error, so pre-stamp `.npz` files are
just misses. The stamp is captured *before* the decode, not at save time: a file
rewritten mid-run then stores the old stamp against new pixels and re-embeds
next run, which is the safe direction to be wrong in.

**`min_pixels` became a pipeline-wide filter.** `resize` skips sub-0.5MP images
rather than upscaling them. That used to mean "not in training"; it now means
"invisible to masking, grouping and the tagger too". Rather than change the
policy — those images were already never seen by the trainer, so dropping them
is the honest reading — the skip was made *legible*: `ResizeStats.too_small`
records `"<path>: <W>x<H>"` per skip the way `failures` does, the CLI prints
each one, and it lands in `report.json`.

### Pinned by

- `tests/test_workspace.py` — `build_groups`' default is `WS.RESIZED`, and every
  pixel-reading stage is bound to `dst` (`audit_apply` is not).
- `tests/test_gui.py::test_only_resized_tree_stages_get_the_preflight` — the
  preflight set, now seven stages.
- `tests/test_grouping_features.py` — a rewritten source re-embeds; an unchanged
  one is still a cache hit (both mutation-checked); a pre-stamp entry is a miss.
- `tests/test_resize_images.py` — `too_small` names the file and its size, and
  reaches `report.json`.

### Not done, and deliberately

`max_ratio` 4.0 is still the clamp, so a >4:1 image is still cropped before
anything embeds it — which for grouping means a long strip can lose the part
that would have twinned it. That is an argument for revisiting the clamp, not
for keeping grouping on a second tree, and it is now the same crop every other
stage sees rather than a discrepancy between two.
