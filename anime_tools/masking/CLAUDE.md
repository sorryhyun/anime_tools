# anime_tools/masking/

SAM3 subject masks, merged into 8-bit L `{stem}_mask.png` mirroring the source subdir. (The
UNet++ / ComicTextDetector text-mask stage was removed in 0.5 — nobody used it; balloons and
lettering are ordinary ignore prompts on the SAM3 stage.) Docs: `docs/masking.md` (the three trees,
running
it, CPU/macOS, limits).

## Requests

The surface is a request object per stage (`requests.py`, torch-free): `SamMaskRequest`,
`MergeMasksRequest`, run by `sam.py::run_sam_masks`, `merge.py::run_merge_masks`.
Each field is one flag of the matching CLI, whose parser is
generated from the class (`Request.parser()`, hyphenated: `FLAG_SEP = "-"`, with the underscore
spelling as an alias); `to_argv()` / `from_namespace()` come from `anime_tools/_request.py` and
are inverses over it (`tests/test_registry_requests.py` round-trips every one, and a default argv
must read back as a default request). Validation lives in `__post_init__`; the CLIs in `cli/` are
shells that parse, build the request, and turn its `ValueError` into `parser.error`. `load_sam3`
caches per process on its arguments, so a second pass in one interpreter reuses the model.
`__init__.py` exposes all four names lazily.

## Two private cores

- `_prompts.py` is what a *prompt* is, for both packages: `SUBJECT_PROMPT` (`girl`), the
  `--checkpoint` / `--prompt_embed` help (defaults imported from `downloads/`), `prompt_list`
  (`none`/`off` = no prompts; the detection stages' lists), and the soft prompt —
  `resolve_prompt_embed` (`none`/`off`/`text`/`""` = the text prompt; a missing *shipped* default
  degrades to it with a warning, any other missing path raises), `SOFT_PROMPT_KEYS`,
  `load_soft_prompt`, `prompt_embed_sha256`. Torch-free *and numpy-free*, which is the point: it is
  split out of `_sam3` so `masking/requests.py` and `stages/requests.py` can read a default without
  the `np.bool` side effect below, and so the seam runs one way — a stage reaches in here, nothing
  here reaches into `stages/` (`tests/test_boundary.py`).
- `_sam3.py` is the only place SAM3 is constructed, and the home of
  `ground_with_soft_prompt` (a soft prompt is the text encoder's output, so the encode is
  skipped) and `detect_union`. It installs the `np.bool`
  alias sam3 needs as an import side effect, with sam3 imports deferred into functions so
  importing it stays torch-free. Two more shims run inside those functions: `stub_edt_kernel`
  pre-seeds `sam3.model.edt` (the one module that imports triton, which has no macOS build) with a
  stand-in that refuses to run, and `shim_sam3_off_cuda` redirects the image model's two build-time
  `"cuda"` literals to CPU when torch has none. Neither fakes `triton` itself: torch guards its own
  import of it and would take a fake one for real. `tests/test_sam3_import.py` pins the shims.
- `_masks.py` owns the mask layout — `plan_mask_jobs`, `write_mask`/`write_ignore_mask`
  (`detected=1 → alpha=0`), the read side (`mask_name`/`mask_path_for`/`iter_masks`), and
  `mask_run`, the scaffolding a generator wraps its inner loop in (it reads the walk fields of
  `MaskWalkRequest` by attribute). It deliberately does not import `_sam3`, because
  `gui/dataset.py` imports `mask_name` and would otherwise drag in that side effect.

## Prompts, drawers, directories

The SAM3 mask CLI takes one list, `--masks ROLE:KIND:VALUE …` (`MaskPrompt`: role `keep` /
`ignore`, kind `text` / `soft`; the value is everything after the second colon). A soft entry is
a prompt file `sam.py::_resolve_prompts` loads once per path, and `detect_union` takes text and
loaded soft prompts in one list. The default is `keep:soft:<the shipped girl embed>`, which falls
back to the text `girl` when the file is missing; an empty list is the one argv the stage
refuses. The field carries `kind="masks"` metadata (`_request.KIND`), so the GUI draws it as a
row editor (`frontend/src/components/MaskList.tsx`) while argparse still sees a plain `nargs="*"`
flag — which is also why the stage takes no `--prompt_embed` and ⚙ Settings' soft prompt no
longer reaches it.

A drawer is a switch field carrying `gate=<its own name>` plus the drawer's `group` title,
with every knob inside it carrying `gate=<the switch>`; the GUI folds a shut drawer's knobs away
and drops them from the argv, and the generated parser stamps the argparse group with
`contract.GATE_ATTR`. No masking request has one any more — Export's Combine OCR is the
carrier `tests/test_masking_plan.py` pins the seam on.

The two mask directories are one ⚙ Settings value, not two form fields: the generator writes
its own tree (`masks_sam`) rather than the `masks` root, so a hand-made tree listed beside it at
merge time cannot overwrite a SAM3 mask at the same relative path. Both hang off `MASK_SETTING`
(`mask_root`) in `gui/stages.py`.

Tests: `test_masking_requests`, `test_masking_plan`, `test_sam3_import`.
