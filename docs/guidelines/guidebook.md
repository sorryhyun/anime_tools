# anime_tools Guidebook

This is the start-to-finish guide to curating a dataset with **anime_tools**: install, put your
images in place, open the web GUI, run the caption / grouping / mask stages in order, and hand the
result to the [Anima LoRA trainer](https://github.com/sorryhyun/anima_lora). It is written for
someone who installed from the one-line installer and has never opened the code. It explains what
each stage reads and writes and how to take a run back; the knobs themselves are on each stage's
form, generated from its own `--help`, and the design docs linked from each section explain them.

---

## Table of Contents

1. [What this is](#1-what-this-is)
2. [Requirements](#2-requirements)
3. [Install](#3-install)
4. [The curation home and its layout](#4-the-curation-home-and-its-layout)
5. [Hugging Face sign-in and models](#5-hugging-face-sign-in-and-models)
6. [The GUI](#6-the-gui)
7. [The workflow, stage by stage](#7-the-workflow-stage-by-stage)
8. [Handing off to the trainer](#8-handing-off-to-the-trainer)
9. [CLI equivalents](#9-cli-equivalents)
10. [Updating](#10-updating)
11. [Troubleshooting](#11-troubleshooting)
12. [Further reading](#12-further-reading)

---

## 1. What this is

`anime_tools` is the dataset-curation half of the Anima LoRA trainer, split out so it can run
on its own: a smaller install, no DiT or VAE, and a dataset it produces can go to any trainer.
It takes a folder of images with hand-written captions and produces everything the trainer
reads before training — bucket-resized images, corrected and enriched captions, caption
variants, a typed tag index, training masks, and a grouping manifest for thinning duplicates.

The trainer depends on this package; this package never imports the trainer. Everything the
two agree on — file names, the caption grammar, where things land — is written down once in
[`docs/contract.md`](../contract.md).

---

## 2. Requirements

| Item | Note |
|---|---|
| Python | **3.13**. The installer pins it; `uv` fetches one if the machine has none. |
| GPU | Optional but strongly recommended. Every model stage (the tagger, SAM3, PE-Spatial, the text segmenter, OCR) runs on CPU, at a fraction of the speed. Each stage picks CUDA when torch sees it and CPU otherwise. |
| OS | Linux, Windows, macOS. On macOS there is no CUDA and no triton; the package shims SAM3 around both, so the SAM3 stages run on CPU there. |
| Disk | The models the stages fetch on first use (the tagger backbone, SAM3, PE-Spatial, the text nets, OCR, the Danbooru tag KB) plus a resized copy of your dataset under `workspace/`. |

> **Windows torch is CPU-only on PyPI.** The PowerShell installer defaults to the CUDA 13.0
> torch index for that reason (§3). Linux torch from PyPI already bundles CUDA.

---

## 3. Install

### 3.1 One line, no checkout

Linux / macOS:

```bash
curl -fsSL https://github.com/sorryhyun/anime_tools/releases/latest/download/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://github.com/sorryhyun/anime_tools/releases/latest/download/install.ps1 | iex
```

Both install [uv](https://astral.sh/uv) if it is missing, then `uv tool install` the latest
release into its own virtual environment, torch and SAM3 included, and put `anime-tools-gui`
on your PATH. No git checkout and no CUDA toolkit are needed: the torch wheel carries its CUDA
runtime.

Two environment variables steer the installer:

| Variable | Effect |
|---|---|
| `ANIME_TOOLS_VERSION=v0.3.1` | Install that tag instead of the latest release. |
| `TORCH_INDEX=https://download.pytorch.org/whl/cu130` | Extra package index for torch. The PowerShell installer already defaults to this one; on Linux it is only needed for a CPU-only build (`…/whl/cpu`). |

When it finishes, open a **new shell** so the PATH change is seen, then:

```bash
cd <your dataset folder>
anime-tools-gui --open
```

### 3.2 As a dependency of your own project

```bash
uv add "anime-tools @ git+https://github.com/sorryhyun/anime_tools"
```

There is no PyPI package; pin a tag or point `[tool.uv.sources]` at a checkout. One thing to
copy into your own `pyproject.toml`: SAM3 pins `numpy<2`, and that pin is stale. This repo
overrides it, but uv honours `[tool.uv]` only in the workspace root, so a project that
*depends* on anime-tools has to repeat the override or the resolve fails on numpy:

```toml
[tool.uv]
override-dependencies = ["numpy>=2.0"]
```

The installers above pass the same override on the command line.

### 3.3 Running the CLIs from a tool install

`anime-tools-gui` is the only command on your PATH. To run a stage from the terminal, run
Python inside the tool's environment:

```bash
uv tool run --from anime-tools python -m anime_tools.downloads --list
```

---

## 4. The curation home and its layout

Everything is relative to one directory, the **curation home**. It is, in order of precedence,
`ANIME_TOOLS_HOME`, then `ANIMA_HOME` (the trainer's own home, so a tree the trainer uses
works unchanged), then the directory you run from. `anime-tools-gui --home <dir>` sets it
for that server.

```
<home>/
  image_dataset/                  INPUT  — your images + hand-written captions; the tools only read it
  workspace/                      everything the tools produce
    resized/<rel>.png               bucket-resized image — the tree every stage reads
    resized/<rel>.txt               the revised caption (stage output)
    resized/<rel>.history.txt       what that caption used to say, one line per version
    resized/<rel>.variants.txt      shuffle / dropout variants, v0 = pristine
    ocr/<rel>.ocr.txt               words in the picture (OCR); not a caption
    masks_sam/  masks_mit/          each mask generator's own tree
    masks/<rel>/{stem}_mask.png     the merge of the two — what Export publishes
    captions/<stage>/report.json    what each run did, and how to undo it
    groups/groups.json              the grouping manifest
  post_image_dataset/             OUTPUT — written only by Export; the tree the trainer reads
  models/                         model weights (ANIME_TOOLS_MODELS overrides)
```

Three rules explain most of what you will see:

- **`image_dataset/` is read-only for the stages.** Your captions there are the *master*. Every
  stage writes a *revised* caption under `workspace/resized/` instead, falling back to the master
  when no revised caption exists yet. Only two things ever write the master: the caption editor
  in the GUI when you edit that rung by hand, and Export.
- **Every stage that opens an image opens it under `workspace/resized/`** — masking and grouping
  included, so the whole pipeline shares one geometry. An image that exists only in
  `image_dataset/` is invisible to them. That is why the GUI runs Resize automatically before
  any stage that needs it (§7.1).
- **Nothing writes outside `workspace/` except Export** (§7.9). You can delete the workspace and
  regenerate it; you cannot lose a master caption to a stage.

Image files are matched to captions by stem: `chars/alice/001.png` reads `chars/alice/001.txt`.
Subfolders are kept as-is through the whole pipeline (`<rel>` above).

> **Upgrading from a pre-workspace install** (one that wrote `post_image_dataset/resized/`
> directly): `python -m anime_tools.workspace.migrate` prints what it would move and
> `--apply` moves it. It renames directories and never merges: an existing destination is
> reported and skipped. A dataset root you had pinned to the old path in ⚙ Settings is named
> but not rewritten — clear it there yourself.

---

## 5. Hugging Face sign-in and models

Two of the models are **gated** on the Hugging Face Hub: the dbv4 tagger backbone (GPL-3.0,
never vendored — it is fetched under *your* token at load) and SAM 3. Both need a token with
read access, and the account behind it has to have accepted each repo's terms once.

In the GUI: ☰ → **⚙ Settings…** → *Hugging Face* → paste a token. It is handed to
`huggingface_hub`'s login and stored by it, never shown again. The header shows **⚠ no HF
token** until one is set. Gated rows in the Models list carry an *accept the terms* link; open
it signed in as the same account.

Nothing has to be pre-fetched: **every stage fetches what it needs on first use.** The
**Models & weights** dialog (☰ → *Models & weights*) lists one row per checkpoint — what it is
for, whether it is installed, where it lands — with a **Download** button per row and
**Download all N missing**. A download runs as an ordinary job, one at a time, sharing the slot
with the stages. The buttons only move the wait, and any gated-repo refusal, to a moment you
picked.

From the terminal:

```bash
python -m anime_tools.downloads --list        # every row: installed / MISSING, repo, destination
python -m anime_tools.downloads               # fetch every missing one
python -m anime_tools.downloads sam3 tagger   # fetch by id
```

Weights land under `<home>/models/` (`ANIME_TOOLS_MODELS` overrides), except the SAM3 subject
soft prompt, which sits at `networks/calibration/` because that is where the trainer keeps it.

---

## 6. The GUI

```bash
anime-tools-gui --open              # http://127.0.0.1:8790, opens the browser
anime-tools-gui --home ~/datasets/x # a different curation home
anime-tools-gui --host 0.0.0.0      # expose on the LAN (a headless GPU box)
```

If the port is busy the next free one is used and printed. `--host 0.0.0.0` has **no
authentication** — put your own tunnel in front of it. The server itself never loads a model:
every stage runs as a `python -m …` subprocess, one at a time, and its output streams into the
panel.

### 6.1 The sidebar is the dataset

Every image under `image_dataset/`, in its folders, with dots on the row for what exists:
resized, has a mask, and one dot per caption rung. Under each image its captions form a
**ladder**, oldest first:

| Rung | File | Editable |
|---|---|---|
| **master** | `image_dataset/<rel>.txt` — hand-written; the stages only read it | yes |
| **history** (`revised@1`, `revised@2`, …) | `<rel>.history.txt` — what the revised caption used to say, before each run that replaced it | no |
| **revised** | `workspace/resized/<rel>.txt` — the stage output; the next run rewrites it and keeps this text as a version | yes |
| **variants** (`v0`, `v1`, …) | `<rel>.variants.txt` — generated; `v0` is the pristine revised caption | no |

The **filter** box narrows the tree; `↑`/`↓` or `j`/`k` walk the images; `#<rel>|<kind>` in
the URL is a link to one caption. The **tree / groups** toggle draws the same listing in two
orders — the folders, or the near-twin groups the Groups stage found (§7.7).

### 6.2 The caption editor

Selecting an image shows it (source / mask / overlay) beside its captions, one editor with a
badge per version. The tag bag and each position clause are boxed in the text; the boxes come
from the server's own parser, so the browser never guesses at a caption's structure.
Double-click a tag to look it up in the Danbooru tag KB (once it is downloaded).

**Save** (⌘/Ctrl+Enter) writes `master` or `revised`; the other rungs are read-only. Every
write pushes the text it replaced onto the history rung, by hand or by a stage alike. A save
tells you what to do next: the trainer's text-encoder re-encode always, and *re-run Correct*
too when you edited a revised caption that already had a `.variants.txt`, because that
sidecar is now stale.

Below the captions, an image that the OCR stage has read shows the **text in the image** with
each line's confidence and position (§7.6).

### 6.3 The dock is the stage runner

The button strip along the bottom **is the stage list**: Resize sits behind the scenes, and
the buttons are **Autotag · Curate · OCR · Groups · Masks · Export**. Curate holds three stages
(Position / Correct / Audit) and Masks holds three (Subject / Text / Merge), picked inside the
panel. One click opens a stage's form; a second click on the open one folds the dock away.

The form is generated from the stage's own `--help`. It opens on the knobs a run changes its
mind about; the rest fold under **▸ advanced (n)** at the bottom of each group, with a note when
a hidden field is off its default. Dataset roots, the report root and the model paths never
appear on a form — they come from ⚙ Settings.

- **Run** runs the stage on the selected image alone (**just `<rel>`**).
- **Run batch** runs it over every image the Settings `path_pattern` names — `*` is the whole
  dataset.
- **Undo** puts back what the last run of this stage wrote, by replaying its report backwards.
- **Cancel** stops the running job.

**A Run writes for real.** There is no Apply gate in the GUI: what a run replaces becomes a
version badge on the caption (`revised@2`), the run's report is read back as a per-image diff
in the caption panel, and Undo is that report replayed with the two texts swapped. Undo is
guarded: a caption you edited by hand after the run is left alone and counted as skipped.

The newest output line shows in the stage bar; the **log** button opens the whole thing. A
stage that walks images shows a progress bar; a run with a preflight in front of it shows
*step 1/2 · resize* first.

### 6.4 ⚙ Settings is three dialogs

| Dialog | What it holds |
|---|---|
| **Settings** | The curation home and models dir, the five **dataset roots** (`src`, `master`, `dst`, `masks`, `out`), the Hugging Face token. |
| **Advanced settings** | **Stage defaults** filled into every stage that takes them (`path_pattern`, the tagger dir, the SAM3 checkpoint and soft prompt, the report root, the mask root), and the **Preprocess** block — the Resize stage's own form, since Resize has no dock button. |
| **Models & weights** | The model rows of §5. |

Each dialog saves only what it holds. Roots are relative to the curation home; a missing one is
flagged. The panel may *read* any root the saved settings point at, but it will only *create*
directories under the home, so a typo in an external root is a missing root rather than a new
empty one.

The ☰ menu's **Language** row switches the panel between English, Korean, Japanese and
Chinese in place; a first visit follows the browser's own language list. Only the panel's chrome
is translated — a stage's title, its form labels and help come from its `--help` and stay in
English, and captions, tags and paths are data.

---

## 7. The workflow, stage by stage

Run them top to bottom. Each stage reads the resized tree, so an image only ever goes through
Resize once, and each caption stage reads the revised caption the previous one wrote.

### 7.1 Resize (automatic)

**Reads** `image_dataset/`. **Writes** `workspace/resized/<rel>.png`.

Every image lands in the bucket tier that resizes it the least, keeping its native aspect
inside that tier's token band. The geometry is the trainer's own, so whichever side resizes
first, the other finds every image already at its bucket and skips it. Images under the pixel
floor (0.5 MP by default) are **skipped and named** in the report — such an image is never
resized, so no stage sees it; the image panel says so on the pixel-count chip, and the floor
is in ⚙ Advanced settings › Preprocess.

You never click it: the GUI runs it as the first step of every stage that reads the resized
tree, and a run over one image resizes just that image. Already-current images are skipped, so
the preflight is near-free. Always writes; there is no dry run.

> The tiers (`target_res`) must match the trainer's, or each side keeps re-resizing the
> other's output.

### 7.2 Autotag captions

**Reads** each resized image and its caption (revised, else master). **Writes** the revised
caption. **Model**: the Anima Tagger (tagger checkpoint + gated dbv4 backbone).

Predicts an Anima-order tag string — `rating, count, characters, copyrights, @artists,
generals` — per image. Three modes:

| Mode | What it does |
|---|---|
| `missing` (default) | Only images **no caption speaks for**. Nothing existing is touched. |
| `merge` | Appends tags the caption lacks, keeping its position clauses. |
| `overwrite` | Replaces the caption outright. The old text is kept as a history version. |

Only `missing` is non-destructive; the other two are undoable through the history rung and
Undo. Treat the output as a starting point — check names, series and artists before training.
See [`docs/anima_tagger.md`](../anima_tagger.md) for the tagger itself.

### 7.3 Correct + mirror captions

**Reads** the master captions and the Danbooru tag KB. **Writes** the revised caption and,
optionally, `.variants.txt`.

Mirrors each master into a corrected revised caption: tags typed against the KB and ordered
into Anima's buckets, an optional trigger word slotted in, an optional `@no-artist` sentinel,
and optional **drop groups** that strip whole tag families from every mirrored caption without
touching the master. With a variant count set it also writes `<rel>.variants.txt`: `v0` is the
corrected caption, `v1…` are smart-shuffled and dropout draws that the trainer encodes verbatim.

Always writes; there is no dry run and no report, so there is no Undo either — the replaced
text is on the history rung. The KB is the **Danbooru tag KB** row in Models (§5); its
optional English row rewrites the descriptions the tag lookup shows.

> Correct mirrors the **master**. A revised caption's position clauses survive a re-run (they
> are re-attached to the fresh mirror), but tags Autotag `merge` added to the flat bag do not,
> so run Correct before Autotag `merge`, not after.

### 7.4 Position captions

**Reads** resized images and their captions. **Writes** the revised caption. **Models**: SAM 3
(gated), the subject soft prompt, the Anima Tagger.

For an image with two or more subjects, detects each one, orders them in reading order, tags a
mask-blanked crop of each, and rewrites the caption into the position-clause grammar:
`<flat tag bag>. On the left, …. On the right, ….` A tag that belongs to one subject *moves*
out of the flat bag into that subject's clause; nothing the curated caption never said is
invented beyond a small allowance. Single-subject images are skipped and say why in the report.

An apply drops any stale `.variants.txt` beside the captions it rewrote. **Backing it out** is
its own mode, `--flatten`, which merges every clause back into the flat bag with no model
loaded — or Undo. The full grammar, the gates and every knob are in
[`docs/position_captions.md`](../position_captions.md).

### 7.5 Multiview audit

**Reads** the single-subject images the position stage skipped. **Writes** `multiple views`
into the **master caption**, and only that. **Models**: as Position.

Finds `1girl` images that are really several views of one girl and reports each with a
contact sheet under the report directory. The apply is gated to the strong findings by
default; a weak finding has only the geometry behind it, so review its sheet first.

> This is the one stage that writes `image_dataset/` directly. The report holds the
> before-text of every write, and Undo reads it back. See
> [`docs/multiview_audit.md`](../multiview_audit.md).

### 7.6 OCR text

**Reads** resized images. **Writes** `workspace/ocr/<rel>.ocr.txt`. **Model**: PP-OCRv6
detection + recognition (English, Chinese, Japanese; no hangul).

Records the words *in the picture* — dialogue, signs, sound effects — one line each with its
confidence and position. It is **not a caption**: nothing downstream encodes it, no caption is
read or written, and no re-encode is needed afterwards. The image panel shows the lines. By
default ASCII-only lines are dropped (page numbers, URLs) and vertical Japanese columns are
joined into one line per balloon.

### 7.7 Build groups

**Reads** resized images. **Writes** `workspace/groups/groups.json`. **Model**: PE-Spatial.

Clusters near-identical images — duplicates, alternate versions, crops of one picture — per
top-level folder, by visual content rather than filename or caption. The sidebar's **groups**
ordering draws the result: one collapsible header per group, so redundancy sits together and
is easy to thin out. Filters and pending dots mean the same thing in both orderings.

Re-running is cheap: features are cached under `~/.cache/near_twin/` (`NEAR_TWIN_CACHE`
overrides) and stamped with the file's size and mtime, so a re-resize recomputes exactly the
images that changed. Tighten or loosen the clustering with the two match thresholds on the
form. See [`docs/grouping.md`](../grouping.md).

### 7.8 Masks: Subject, Text, Merge

**Read** resized images. **Write** `workspace/masks_sam/`, `workspace/masks_mit/`, and their
merge under `workspace/masks/` — 8-bit `{stem}_mask.png` mirroring the source subfolder.
**Models**: SAM 3 for both generators; the manga text segmenter and the ComicTextDetector gate
for Text.

- **Subject** keeps the subject and masks out the background: by default it grounds SAM3 on the
  learned subject prompt. Prompts to mask *out* (`speech bubble,text`) can be added.
- **Text** masks lettering and balloons with two detectors, each behind its own switch — SAM3 on
  a prompt (a balloon is a shape) and the UNet++ segmenter (a letter is a stroke) — unioned
  before one dilation. Both off is the one form the stage refuses.
- **Merge** takes the pixel-wise minimum of the two trees into `workspace/masks/`, the tree the
  sidebar shows as *mask* / *overlay* and Export publishes. A missing input tree is skipped, so
  running one generator is a valid half.

The three directories are one setting, not three fields, because both generators name a mask
identically and would overwrite each other in a shared tree. Masks always write; regenerate to
change one. See [`docs/masking.md`](../masking.md).

### 7.9 Export workspace

**Reads** the workspace. **Writes** `post_image_dataset/` — and, for a revised master,
`image_dataset/`.

The only stage that writes outside the workspace. It publishes six artifact kinds — resized
image, revised caption, variants sidecar, mask, revised master, caption index — each decided on
its own against its destination: identical files are skipped (byte compare for text, size and
mtime for pixels), so re-exporting an unchanged dataset is a walk and a stat apiece. It always
**copies**, never links, so the export tree survives the workspace being cleared.

From the CLI it is dry-run by default and lists what it would copy. In the GUI **Run** copies,
and **Undo** restores the text it overwrote from the export's own ledger; an overwritten
*pixel* cannot be restored and is reported as such. The trainer reads only what Export wrote —
see [`docs/contract.md`](../contract.md) §2 for every path.

---

## 8. Handing off to the trainer

1. **Export** (§7.9). The trainer reads `post_image_dataset/resized/` for images and revised
   captions, `post_image_dataset/masks/` for masks, and `image_dataset/` for the master.
2. In the trainer, **re-encode the text embeddings** — its `make preprocess-te` — after any
   apply that changed a caption. The trainer's caches are reused as-is and never expire on their
   own, so a changed caption with an old cache trains on the old text.
3. Resize is shared: the trainer's `make preprocess-resize` finds every image already at its
   bucket and skips it, provided the tiers match.

Point both at the same home (`ANIMA_HOME` is honoured here) and nothing moves. From there
the trainer's own guide takes over — the [Anima LoRA Guidebook][trainer-guide].

[trainer-guide]: https://github.com/sorryhyun/anima_lora/blob/main/docs/guidelines/guidebook.md

---

## 9. CLI equivalents

Every stage is a `python -m` module over the same request object the GUI form fills, with one
flag per form field (`--help` lists them). Paths are relative to the curation home. Caption
stages spell flags with underscores (`--path_pattern`), grouping and masking with hyphens
(`--source-dir`); either spelling is accepted everywhere.

| Stage | Module | Writes without `--apply`? |
|---|---|---|
| Resize | `anime_tools.stages.cli.resize_images` | always writes |
| Autotag | `anime_tools.stages.cli.autotag_captions` | dry run → `report.json` |
| Position | `anime_tools.stages.cli.position_captions` | dry run → `report.json` |
| Correct | `anime_tools.stages.cli.correct_captions` | always writes |
| Audit | `anime_tools.stages.cli.audit_multiview` | dry run → `report.json` |
| OCR | `anime_tools.stages.cli.ocr_captions` | dry run → `report.json` |
| Groups | `anime_tools.grouping.cli.build_groups` | always writes `groups.json` |
| Subject masks | `anime_tools.masking.cli.generate_masks` | always writes |
| Text masks | `anime_tools.masking.cli.generate_masks_mit` | always writes |
| Merge masks | `anime_tools.masking.cli.merge_masks` | always writes |
| Export | `anime_tools.stages.cli.export_workspace` | dry run → `report.json` |

The dry-run stages write `report.json` under `workspace/captions/<stage>/` and stop.
`--apply` writes for real. `--from_report <report.json>` replays a dry run's proposals without
loading a model, skipping any caption that changed in between, and writes `apply_report.json`
beside the report it read. The GUI's Undo is the same replay backwards.

```bash
python -m anime_tools.stages.cli.autotag_captions --mode merge            # dry run
python -m anime_tools.stages.cli.autotag_captions --mode merge --apply    # write
python -m anime_tools.stages.cli.export_workspace --apply                 # publish
```

The Python API is the same object: [`examples/`](../../examples/README.md) has one runnable
script per feature, API beside CLI.

---

## 10. Updating

```bash
uv tool upgrade anime-tools          # tool install
uv lock --upgrade-package anime-tools   # as a dependency
```

The GUI is a committed bundle, so an update needs no frontend build. A stage's form is
generated from the installed code, so a new flag appears on the form by itself.

---

## 11. Troubleshooting

**A stage fails with a gated-repo error (401 / 403 from the Hub).** The token is missing, or
the account has not accepted that repo's terms. Set the token in ⚙ Settings, then open the
*accept the terms* link on the row in Models & weights, signed in as the same account. The
Download button reproduces the failure without a full stage run.

**`uv` fails resolving numpy in my own project.** SAM3's stale `numpy<2` pin. Add the
override from §3.2 to your `pyproject.toml`.

**macOS: `import sam3` complains about triton, or a tensor cannot be put on `cuda`.** Both are
handled inside the package's SAM3 loader; if you see them, you are importing sam3 yourself or
running an old release. The SAM3 stages run on CPU there, slowly.

**Windows: torch has no CUDA.** PyPI's Windows torch is CPU-only. Reinstall with the PowerShell
installer, which defaults to the CUDA index, or set `TORCH_INDEX` explicitly.

**A stage sees no images / does nothing for this image.** The image is not under
`workspace/resized/`. Either it sits under the resize floor (the pixel-count chip says so;
lower the floor in ⚙ Advanced settings › Preprocess) or the `src` root does not point at your
dataset.

**"saved — .variants.txt is now stale".** You edited a revised caption that had variants. Re-run
Correct with the same variant count, then the trainer's TE re-encode.

**Undo says it skipped things.** The caption no longer holds what the run wrote — you or a later
run changed it — so that row was left alone. The skipped rows are still in the report.

**A run reads the wrong report, or two stages share one.** Each stage keeps its own directory
under the report root (`captions/autotag`, `captions/position`, …); the root is one Settings
value and moving it moves them all. Leave it blank to keep reports beside the `dst` root.

**The audit wrote my master and I want it back.** `image_dataset/` is not versioned by the
tools; the audit's `report.json` holds the before-text of every write, and Undo replays it.

---

## 12. Further reading

- [`docs/contract.md`](../contract.md) — what the trainer reads: every file, format and seam.
- [`docs/anima_tagger.md`](../anima_tagger.md) — the tagger, its vocab and calibration.
- [`docs/position_captions.md`](../position_captions.md) — the clause grammar, gates and knobs.
- [`docs/multiview_audit.md`](../multiview_audit.md) — the audit's verdicts and sheets.
- [`docs/grouping.md`](../grouping.md) — near-twin grouping.
- [`docs/masking.md`](../masking.md) — subject and text masks.
- [`examples/README.md`](../../examples/README.md) — the Python API, one script per feature.
- [Anima LoRA
  Guidebook](https://github.com/sorryhyun/anima_lora/blob/main/docs/guidelines/guidebook.md)
  — training, from preprocessing to ComfyUI.
