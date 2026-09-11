# Training masks — subject masks and their merge

Two stages under the GUI's Masks button (Subject / Merge), two `python -m` CLIs,
two request objects in `anime_tools.masking`. Together they write the
`{stem}_mask.png` files a masked-loss training run reads, so a speech bubble, a signature or
the background behind the subject stops contributing to the gradient.

## 1. What a mask is here

An 8-bit L PNG named `{stem}_mask.png`, at the image's own relative path under the mask
directory (`chars/alice/001.png` → `chars/alice/001_mask.png`). **White (255) = train on this
pixel, black (0) = ignore it in the loss.** The training side converts the file to L,
NEAREST-resizes it to the latent's pixel size and scales it to `[0, 1]`; an image with no mask
is trained on in full, so generating none is fine.

The generator writes that polarity through one of two helpers in `_masks.py`: `write_mask`
saves a keep array as `keep * 255`, and `write_ignore_mask` saves the inverse of a
detection, `detected=1 → alpha=0`. The two are the whole difference between "keep only the
subject" and "mask out the balloons".

Both stages read `workspace/resized/` — the tree `resize` populates and every other
stage opens — so a mask is cut at the same geometry every other stage sees. A mask cut from the
master pixels would land off the subject for a ratio-clamped image, which is why the GUI runs
resize as a preflight in front of the generator.

## 2. Two trees, one Settings value

```
workspace/masks_sam/<rel>/{stem}_mask.png   the generator's own tree
workspace/masks/<rel>/{stem}_mask.png       the merge — what the sidebar shows and Export publishes
```

The generator writes its own tree rather than the `masks` root: a second tree listed beside
it at merge time (hand-painted masks, another tool's output) names a mask identically at the
same relative path, and a shared directory would have the second run overwrite the first. The
merge's positional input defaults to exactly the generator's tree
(`tests/test_masking_plan.py` pins that the defaults line up).

In the GUI the two directories are one ⚙ Settings value, `mask_root`, not two form
fields: the generator keeps its own tail under it (`masks_sam`), the merge's input list
moves with it, and a blank root means beside the `masks` root. Only the merged
output is the dataset's `masks` root; that is the tree `Export` copies to the
`out` root's `masks/`.

Export decides a mask by `(size, mtime_ns)` against the destination and overwrites a pixel
file without keeping the old bytes, so a mask it replaced reports `not-undoable` on Undo.

## 3. SAM3 masks — `generate_masks`

SAM3 grounded on a list of masks, `--masks`, one `ROLE:KIND:VALUE` entry each (the GUI's
Setup form draws it as rows with + and ×):

- role `keep` — keep only these regions; everything outside their union is masked out.
- role `ignore` — mask these out. `ignore:text:speech bubble` and `ignore:text:text` are the
  usual spellings; balloons and lettering are ordinary ignore prompts here.
- kind `text` — the value is a SAM3 text prompt, run through its text encoder.
- kind `soft` — the value is a learned soft prompt file (`.safetensors`): what the encoder
  would have produced for some phrase, so the encode is skipped and the three saved tensors
  go straight into the grounding call (`_sam3.ground_with_soft_prompt`). Each soft file is
  loaded once per run however many rows name it.

Give both roles and the kept region survives minus the ignored ones (`keep * (1 - ignore)`).
An empty list is refused before a weight is read, and the GUI reads an emptied list back as
the default. `VALUE` is everything after the second colon, so a Windows path keeps its drive
letter.

The default is one row, `keep:soft:networks/calibration/sam3_girl_prompt.safetensors` — the
catalog's `soft_prompt` row, the textual inversion of `girl`. When that default file is not
downloaded it warns and falls back to the text prompt `girl`; any other soft path that does
not exist is an error, and `soft:none` is refused (a text row is how you ask for text). The
mask stage has no `--prompt_embed` of its own any more, so ⚙ Settings' soft prompt reaches the
position stage and the audit only; a different soft prompt for masking is a row.

What gets written. Per image, in this order:

| Situation | Written | Progress line |
|---|---|---|
| keep rows set, something kept found | `keep - ignore` as a keep mask | `train 41.2%` (share kept) |
| keep rows set, nothing kept found | nothing — the image trains in full rather than zeroing its loss | `focus not found` |
| only ignore rows, something found | the inverse of the detection | `12.3%` (share ignored) |
| only ignore rows, nothing found | nothing | `skipped` |

Knobs: `--threshold` (SAM3 confidence floor, 0.5), `--dilate` (pixels, 5, `0` = off; applied to
each detection before the two are combined), `--batch-size` (1), `--checkpoint` (SAM3 weights,
`models/sam3/sam3.pt`).

## 4. Merge — `merge_masks`

```bash
python -m anime_tools.masking.cli.merge_masks                      # workspace/masks_sam → workspace/masks
python -m anime_tools.masking.cli.merge_masks DIR1 DIR2 --output-dir OUT
```

Inputs are positional, default to the generator's tree, and a missing directory is skipped
rather than an error; a hand-made tree is simply listed beside the default. Masks are keyed
by `(relative dir, name)`, so two inputs merge only when the file sits at the same relative
path in both; a mask present in one tree is copied through. Merging is the pixel-wise
minimum, i.e. the union of what either input ignores (a second input at another size is
NEAREST-resized to the first). The nested layout is preserved under `--output-dir`. This stage
loads no model and needs no resize preflight.

## 5. Running it

From the GUI: Masks → Subject, then Masks → Merge. The generator's form shows the prompts,
thresholds, dilation and `force`; the walk flags are bound
to the dataset roots and hidden, and `--device` is resolved by the child. The sidebar marks an
image that has a merged mask, and selecting it shows the mask beside the source and resized
images. Jobs run one at a time as subprocesses; the `name: what` progress line is the same
text the CLI prints beside its bar.

From a shell, home-anchored (`ANIME_TOOLS_HOME` → `ANIMA_HOME` → the current directory):

```bash
python -m anime_tools.masking.cli.generate_masks --image-dir workspace/resized --recursive
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
(`--image-dir` / `--image_dir`). The mask stages always write — there is no dry run and no
`report.json`; `--force` is the only thing that changes an existing file. The same stem twice
in one folder is refused by the walk (the two would overwrite each other's mask); the same
stem in two folders is fine, since the mirrored layout keeps them apart. Nothing left to do is
a sentence (`No images to process.`), not an error. From Python:

```python
from anime_tools.masking import SamMaskRequest, MergeMasksRequest
from anime_tools.masking import run_sam_masks, run_merge_masks

run_sam_masks(SamMaskRequest(image_dir="workspace/resized", recursive=True))
run_merge_masks(MergeMasksRequest())
```

`load_sam3` is cached per process on its arguments, so a second pass in one interpreter (the
position stage, say) reuses the model. `examples/masking.py` is this sequence with the
requests printed as their command lines.

Weights. SAM3 (`sam3`, gated on the Hub — sign in under ⚙ Settings → Models first) and
the soft prompt (`soft_prompt`) are the `masking` pack: `python -m anime_tools.downloads
masking`, or the Models pane's Download buttons. Every loader still fetches on first use;
the buttons only
move the wait.

## 6. CPU and macOS

The same model runs without CUDA, slowly. Two shims in `_sam3.py` make that true:
`stub_edt_kernel` pre-seeds the one sam3 module that imports triton (which has no macOS build)
with a stand-in that refuses to run, since that kernel belongs to the video tracker the image
model never calls; and `shim_sam3_off_cuda` redirects the image model's two build-time
`"cuda"` literals, its bf16-only fused linear and `Tensor.pin_memory` to CPU when torch has no
CUDA. Both are inert on a machine with a GPU. The half-precision autocast a SAM3 pass runs
under is simply skipped on CPU.

## 7. Limits

- A subject no keep row finds leaves the image unmasked, silently apart from the progress
  line; a run over a dataset of `1boy` images with the default `girl` row is a run that writes
  little. Replace the keep row, or drop it and use ignore rows alone.
- A soft prompt is the textual inversion of one phrase. The shipped one is `girl`; any other
  phrase is a text row, or a soft file you trained for it.
- There is no per-image review or Undo for masks: what a run changes its mind about is
  answered by `--force` and a re-run, and Export's copy of a pixel file is not undoable.
- The merge's minimum treats any input as an ignore mask; a keep-only subject mask and a
  hand-painted ignore mask combine correctly when both are in the polarity above.

## 8. Code map

| File | Role |
|---|---|
| `anime_tools/masking/requests.py` | `SamMaskRequest` / `MergeMasksRequest` / `MaskPrompt` — the flags, defaults and validation |
| `anime_tools/masking/sam.py` | `run_sam_masks`: keep / ignore passes, soft-prompt loading |
| `anime_tools/masking/merge.py` | `run_merge_masks`: `(rel_dir, name)`-keyed pixel-wise minimum |
| `anime_tools/masking/_masks.py` | Layout and polarity: `mask_name`, `mask_path_for`, `plan_mask_jobs`, `write_mask` / `write_ignore_mask`, `iter_masks`, `mask_run` |
| `anime_tools/masking/_sam3.py` | The only SAM3 construction: `load_sam3` (cached), `detect_union` (text and soft prompts in one list), `ground_with_soft_prompt`, `prompt_list`, the numpy / triton / CPU shims |
| `anime_tools/masking/cli/{generate_masks,merge_masks}.py` | One-line shells over the requests |
| `anime_tools/workspace/__init__.py` | `MASKS_SAM` / `MASKS` — the two trees |
| `anime_tools/gui/stages.py` | `MASK_SETTING` / `MASK_FIELDS` / `mask_subpath`: the one Settings root |
| `anime_tools/downloads.py` | `sam3`, `soft_prompt` rows (the `masking` pack) |
| `tests/test_masking_plan.py`, `tests/test_masking_requests.py` | The pinned shape: layout, polarity, defaults, the refused argv |
