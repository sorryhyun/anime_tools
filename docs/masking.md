# Training masks — subject masks, text masks, and their merge

Three stages under the GUI's **Masks** button (*Subject* / *Text* / *Merge*), three
`python -m` CLIs, three request objects in `anime_tools.masking`. Together they write the
`{stem}_mask.png` files the trainer's masked loss reads, so a speech bubble, a signature or
the background behind the subject stops contributing to the gradient.

## 1. What a mask is here

An 8-bit **L** PNG named `{stem}_mask.png`, at the image's own relative path under the mask
directory (`chars/alice/001.png` → `chars/alice/001_mask.png`). **White (255) = train on this
pixel, black (0) = ignore it in the loss.** The trainer converts the file to L, NEAREST-resizes
it to the latent's pixel size and scales it to `[0, 1]` (`docs/contract.md` §2); an image
with no mask is trained on in full, so generating none is fine.

Every generator writes that polarity through one of two helpers in `_masks.py`: `write_mask`
saves a *keep* array as `keep * 255`, and `write_ignore_mask` saves the **inverse** of a
detection, `detected=1 → alpha=0`. The two are the whole difference between "keep only the
subject" and "mask out the balloons".

All three stages read `workspace/resized/` — the tree `resize` populates and every other
stage opens — so a mask is cut at the same geometry the trainer sees. A mask cut from the
master pixels would land off the subject for a ratio-clamped image, which is why the GUI runs
resize as a preflight in front of both generators.

## 2. Three trees, one Settings value

```
workspace/masks_sam/<rel>/{stem}_mask.png   the subject generator's own tree
workspace/masks_mit/<rel>/{stem}_mask.png   the text generator's own tree
workspace/masks/<rel>/{stem}_mask.png       the merge — what the sidebar shows and Export publishes
```

Both generators name a mask identically at the same relative path, so a shared directory
would have the second run overwrite the first and leave the merge one tree to union. Each
generator therefore has its own `--mask-dir`, and the merge's two positional inputs default
to exactly those two trees (`tests/test_masking_plan.py` pins that the defaults line up).

In the GUI the three directories are **one ⚙ Settings value, `mask_root`**, not three form
fields: each stage keeps its own tail under it (`masks_sam`, `masks_mit`), the merge's input
list moves with them, and a blank root means *beside the `masks` root*. Only the merged
output is the dataset's `masks` root; that is the tree `Export` copies to
`post_image_dataset/masks/`, and the one the trainer's `make mask` lands in.

Export decides a mask by `(size, mtime_ns)` against the destination and overwrites a pixel
file without keeping the old bytes, so a mask it replaced reports `not-undoable` on Undo.

## 3. Subject masks — `generate_masks`

SAM3 grounded on text prompts. Two prompt lists, opposite polarity:

- `--focus-prompts` (default `girl`) — keep **only** these regions; everything outside is
  masked out. A bare run isolates the subject from her background.
- `--prompts` (default none) — mask these **out**. `speech bubble,text` is the usual spelling.

Give both and the focus region survives minus the ignore regions (`focus * (1 - ignore)`).
Pass `none` to either to empty it; both empty is refused before a weight is read. A cleared
field is not the same thing — the GUI omits a blank flag, so a blank prompt box reads back as
its default, which is why `none` is a word rather than an empty string.

**The soft prompt.** By default the word `girl` is not sent through SAM3's text encoder at
all: `--prompt_embed` names a learned soft prompt (`networks/calibration/
sam3_girl_prompt.safetensors`, the catalog's `soft_prompt` row) that *is* what the encoder
would have produced, so the encode is skipped and the three saved tensors go straight into
the grounding call (`_sam3.ground_with_soft_prompt`). It stands in for `girl` and for no other
prompt; everything else in either list stays textual. `--prompt_embed none` uses the plain
text prompt, a missing *default* file warns and falls back to text, and an explicit path that
does not exist is an error. The flag keeps its underscore so ⚙ Settings can fill it, together
with the position stage's and the audit's, from one value.

**What gets written.** Per image, in this order:

| Situation | Written | Progress line |
|---|---|---|
| focus prompts set, subject found | `focus - ignore` as a keep mask | `train 41.2%` (share kept) |
| focus prompts set, subject **not** found | nothing — the image trains in full rather than zeroing its loss | `focus not found` |
| only ignore prompts, something found | the inverse of the detection | `12.3%` (share ignored) |
| only ignore prompts, nothing found | nothing | `skipped` |

Knobs: `--threshold` (SAM3 confidence floor, 0.5), `--dilate` (pixels, 5, `0` = off; applied to
each detection before the two are combined), `--batch-size` (1), `--checkpoint` (SAM3 weights,
`models/sam3/sam3.pt`).

## 4. Text masks — `generate_masks_mit`

Two detectors over one walk, each behind its own switch, unioned before a single dilation.
They answer different questions — a balloon is a shape, a letter is a stroke — so neither
subsumes the other, and both switches off is the one argv the stage refuses.

**`--use-mit` (on by default)** runs a UNet++ text segmenter, the stroke-accurate half and the
only one that finds lettering outside a balloon. `--text-threshold` (0.8) binarises its
probability map. Behind it sits `--ctd-gate` (on): the segmenter's mask is split into connected
components and only those overlapping a ComicTextDetector *text block* survive, which drops
the UNet++ false positives on halos and decorative line art while letting each kept letter
keep its own outline rather than the rectangle around it. `--no-ctd-gate` gives the raw
segmenter masks. The gate's net is the catalog's `ctd_onnx` row at `models/mit/
comictextdetector.pt.onnx` and has no flag; a missing file warns and leaves the masks
ungated. It runs on onnxruntime's CUDA provider when there is one (about 17 ms a forward) and
falls back to `cv2.dnn` on CPU, with a warning, when there is not. `--model-path` points at a
local `model.pth`; without it the weights come from the hub cache (`mit_text`).

**`--use-sam` (off by default)** grounds SAM3 on `--sam-prompts` (default `speech bubble`;
`speech bubble,sign,watermark` is a fuller list) and masks what it finds — the polarity of
`generate_masks --prompts`, everything named is ignored. Off by default because it is a
second set of weights to load and the segmenter answers most of the question alone; turn it
on when balloons matter. `--sam-threshold` (0.5) and `--checkpoint` sit in the same drawer.
There is no soft prompt on this side: every prompt here is textual.

The two masks are OR-ed, then `--dilate` (3, `0` = off) runs **once** over the union — the two
overlap on a lettered balloon, and dilating twice would grow that seam twice. The result is
written as an ignore mask. An image whose union is empty gets no file; the progress line says
`skipped`, or `skipped (ctd-gated)` when the segmenter found strokes and the gate threw every
one of them away, which is the one case that is a knob to reconsider.

In the GUI each switch is a **drawer**: shutting it folds its knobs away and drops them from
the argv, so a request with the drawer shut cannot carry a stale prompt list. The generated
parser exposes the same shape as two argparse groups, *SAM3 prompts* and *MIT text
segmentation*.

## 5. Merge — `merge_masks`

```bash
python -m anime_tools.masking.cli.merge_masks                      # workspace/masks_sam + masks_mit → workspace/masks
python -m anime_tools.masking.cli.merge_masks DIR1 DIR2 --output-dir OUT
```

Inputs are positional, default to the two generators' trees, and a missing directory is
skipped rather than an error — running one generator is a valid half of this. Masks are keyed
by `(relative dir, name)`, so two inputs merge only when the file sits at the same relative
path in both; a mask present in one tree is copied through. Merging is the **pixel-wise
minimum**, i.e. the union of what either input ignores (a second input at another size is
NEAREST-resized to the first). The nested layout is preserved under `--output-dir`. This stage
loads no model and needs no resize preflight.

## 6. Running it

From the GUI: **Masks → Subject**, **Masks → Text**, **Masks → Merge**, in that order. Each
generator's form shows the prompts, thresholds, dilation and `force`; the walk flags are bound
to the dataset roots and hidden, and `--device` is resolved by the child. The sidebar marks an
image that has a merged mask, and selecting it shows the mask beside the source and resized
images. Jobs run one at a time as subprocesses; the `name: what` progress line is the same
text the CLI prints beside its bar.

From a shell, home-anchored (`ANIME_TOOLS_HOME` → `ANIMA_HOME` → the current directory):

```bash
python -m anime_tools.masking.cli.generate_masks --image-dir workspace/resized --recursive
python -m anime_tools.masking.cli.generate_masks_mit --image-dir workspace/resized --recursive --use-sam
python -m anime_tools.masking.cli.merge_masks
```

| Flag | Meaning |
|---|---|
| `--image-dir` | required; the resized tree |
| `--recursive` | walk subfolders; the output mirrors them |
| `--path-pattern` | fnmatch glob (`\|` to OR) on the path relative to `--image-dir`, the training `path_pattern` semantics |
| `--force` | regenerate a mask that already exists; without it an existing file is skipped |
| `--workers` | I/O threads for loading and saving (4) |
| `--device` | `cuda` / `cpu`, default auto |

Flags are hyphenated and take the underscore spelling as an alias
(`--image-dir` / `--image_dir`). The mask stages **always write** — there is no dry run and no
`report.json`; `--force` is the only thing that changes an existing file. The same stem twice
in one folder is refused by the walk (the two would overwrite each other's mask); the same
stem in two folders is fine, since the mirrored layout keeps them apart. Nothing left to do is
a sentence (`No images to process.`), not an error. From Python:

```python
from anime_tools.masking import SamMaskRequest, MitMaskRequest, MergeMasksRequest
from anime_tools.masking import run_sam_masks, run_mit_masks, run_merge_masks

run_sam_masks(SamMaskRequest(image_dir="workspace/resized", recursive=True))
run_mit_masks(
    MitMaskRequest(image_dir="workspace/resized", recursive=True, use_sam=True)
)
run_merge_masks(MergeMasksRequest())
```

`load_sam3` is cached per process on its arguments, so a text pass with `--use-sam` after a
subject pass in one interpreter reuses the model. `examples/masking.py` is this sequence with
the requests printed as their command lines.

**Weights.** SAM3 (`sam3`, gated on the Hub — sign in under ⚙ Settings → Models first), the
soft prompt (`soft_prompt`), the segmenter (`mit_text`) and the gate net (`ctd_onnx`, 95 MB)
are all catalog rows: `python -m anime_tools.downloads sam3 soft_prompt mit_text ctd_onnx`, or
the Models pane's Download buttons. Every loader still fetches on first use; the buttons only
move the wait.

## 7. CPU and macOS

The same model runs without CUDA, slowly. Two shims in `_sam3.py` make that true:
`stub_edt_kernel` pre-seeds the one sam3 module that imports triton (which has no macOS build)
with a stand-in that refuses to run, since that kernel belongs to the video tracker the image
model never calls; and `shim_sam3_for_cpu` redirects the image model's two build-time
`"cuda"` literals, its bf16-only fused linear and `Tensor.pin_memory` to CPU when torch has no
CUDA. Both are inert on a machine with a GPU. The half-precision autocast a SAM3 pass runs
under is simply skipped on CPU. The CTD gate takes its `cv2.dnn` path there.

## 8. Limits

- A subject the focus prompt does not find leaves the image unmasked, silently apart from the
  progress line; a run over a dataset of `1boy` images with the default `girl` focus is a run
  that writes little. Change `--focus-prompts`, or `none` it and use `--prompts` alone.
- The soft prompt is the textual inversion of one phrase. A different focus phrase is a plain
  text prompt, and the shipped embed is not consulted.
- There is no per-image review or Undo for masks: what a run changes its mind about is
  answered by `--force` and a re-run, and Export's copy of a pixel file is not undoable.
- The merge's minimum treats any input as an ignore mask; a keep-only subject mask and an
  ignore-only text mask combine correctly because both are already in the trainer's polarity.

## 9. Code map

| File | Role |
|---|---|
| `anime_tools/masking/requests.py` | `SamMaskRequest` / `MitMaskRequest` / `MergeMasksRequest` — the flags, defaults, drawers and validation |
| `anime_tools/masking/sam.py` | `run_sam_masks`: focus / ignore passes, soft-prompt binding |
| `anime_tools/masking/mit.py` | `run_mit_masks`: UNet++ + CTD gate, optional SAM3, union, one dilation |
| `anime_tools/masking/merge.py` | `run_merge_masks`: `(rel_dir, name)`-keyed pixel-wise minimum |
| `anime_tools/masking/_masks.py` | Layout and polarity: `mask_name`, `mask_path_for`, `plan_mask_jobs`, `write_mask` / `write_ignore_mask`, `iter_masks`, `mask_run` |
| `anime_tools/masking/_sam3.py` | The only SAM3 construction: `load_sam3` (cached), `detect_union`, `ground_with_soft_prompt`, `prompt_list`, the numpy / triton / CPU shims |
| `anime_tools/masking/cli/{generate_masks,generate_masks_mit,merge_masks}.py` | One-line shells over the requests |
| `anime_tools/workspace/__init__.py` | `MASKS_SAM` / `MASKS_MIT` / `MASKS` — the three trees |
| `anime_tools/gui/stages.py` | `MASK_SETTING` / `MASK_FIELDS` / `mask_subpath`: the one Settings root |
| `anime_tools/downloads.py` | `sam3`, `soft_prompt`, `mit_text`, `ctd_onnx` rows; `default_ctd_onnx_path` |
| `tests/test_masking_plan.py`, `tests/test_masking_requests.py` | The pinned shape: layout, polarity, defaults, the two drawers, the refused argv |
