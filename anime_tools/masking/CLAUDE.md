# anime_tools/masking/

SAM3 subject masks, SAM3/UNet++/ComicTextDetector text masks, merged into 8-bit L
`{stem}_mask.png` mirroring the source subdir. Docs: `docs/masking.md` (the three trees, running
it, CPU/macOS, limits).

## Requests

**The surface is a request object per stage** (`requests.py`, torch-free): `SamMaskRequest`,
`MitMaskRequest`, `MergeMasksRequest`, run by `sam.py::run_sam_masks`, `mit.py::run_mit_masks`,
`merge.py::run_merge_masks`. Each field is one flag of the matching CLI, whose parser is
generated from the class (`Request.parser()`, hyphenated: `FLAG_SEP = "-"`, with the underscore
spelling as an alias); `to_argv()` / `from_namespace()` come from `anime_tools/_request.py` and
are inverses over it (`tests/test_registry_requests.py` round-trips every one, and a default argv
must read back as a default request). Validation lives in `__post_init__`; the CLIs in `cli/` are
shells that parse, build the request, and turn its `ValueError` into `parser.error`. `load_sam3`
caches per process on its arguments, so a text-mask pass after a subject-mask pass reuses the
model. `__init__.py` exposes all six names lazily.

## Two private cores

- `_sam3.py` is the **only** place SAM3 is constructed, the one declaration of `--checkpoint` /
  `--prompt_embed` (defaults imported from `downloads.py`), and the home of
  `ground_with_soft_prompt` (a soft prompt *is* the text encoder's output, so the encode is
  skipped), `prompt_list` (`none`/`off` = no prompts) and `detect_union`. It installs the `np.bool`
  alias sam3 needs as an import side effect, with sam3 imports deferred into functions so
  importing it stays torch-free. Two more shims run inside those functions: `stub_edt_kernel`
  pre-seeds `sam3.model.edt` (the one module that imports triton, which has no macOS build) with a
  stand-in that refuses to run, and `shim_sam3_for_cpu` redirects the image model's two build-time
  `"cuda"` literals to CPU when torch has none. Neither fakes `triton` itself: torch guards its own
  import of it and would take a fake one for real. `tests/test_sam3_import.py` pins the shims.
- `_masks.py` owns the mask layout — `plan_mask_jobs`, `write_mask`/`write_ignore_mask`
  (`detected=1 → alpha=0`), the read side (`mask_name`/`mask_path_for`/`iter_masks`), and
  `mask_run`, the scaffolding both generators wrap their inner loop in (it reads the walk fields
  of `MaskWalkRequest` by attribute). It deliberately does **not** import `_sam3`, because
  `gui/dataset.py` imports `mask_name` and would otherwise drag in that side effect.

## Prompts, drawers, directories

The subject-mask CLI takes prompts, not a config: `--prompts` (masked out) / `--focus-prompts`
(keep only, default `girl`) / `--prompt_embed`. The text-mask CLI is two detectors over one walk,
each behind its own switch, unioned before the single dilation: `--use-sam` grounds SAM3 on
`--sam-prompts`, `--use-mit` runs the UNet++ segmenter (with the ComicTextDetector gate falling
back to `cv2.dnn` when onnxruntime is absent). They answer different questions (a balloon is a
shape, a letter is a stroke), neither subsumes the other, and both being off is the one argv the
stage refuses.

Each switch is a **drawer**: the switch field carries `gate=<its own name>` plus the drawer's
`group` title, and every knob inside it carries `gate=<the switch>` (`MitMaskRequest`). The GUI
folds a shut drawer's knobs away and drops them from the argv; the generated parser also stamps
the argparse group with `contract.GATE_ATTR` for anyone introspecting it directly.
`tests/test_masking_plan.py` pins the shape.

The three mask directories are **one ⚙ Settings value, not three form fields**: both generators
name a mask `{stem}_mask.png` at the same relative path, so a shared directory would have the
second run overwrite the first and leave the merge one tree to union. Each `--mask-dir` is its own
tree, all three hanging off `MASK_SETTING` (`mask_root`) in `gui/stages.py`.

Tests: `test_masking_requests`, `test_masking_plan`, `test_sam3_import`.
