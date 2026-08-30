# Anima Tagger (ComfyUI)

Multi-label image tagger trained on the Anima caption distribution. Drop in an image, get back a comma-separated tag string in exactly Anima's training-time T5 format — `rating, count, characters, copyrights, @artists, generals`, underscores replaced by spaces.

Two nodes in the `anima` category:

| Node | Inputs | Outputs | Use |
|------|--------|---------|-----|
| **Anima Tagger Loader** | `tagger_dir` (STRING) | `tagger` (ANIMA_TAGGER) | Load the checkpoint once; ComfyUI memoizes the output so the tagger persists across graph runs. |
| **Anima Tagger Caption** | `tagger` (ANIMA_TAGGER), `image` (IMAGE) | `caption` (STRING) | Tag an image. Drop the STRING into any text input. |

## What it's for

- **DirectEdit ψ_src.** The `ANIMA_TAGGER` socket plugs straight into [`comfyui-anima-directedit`](https://github.com/sorryhyun/anima_lora/tree/main/custom_nodes/comfyui-anima-directedit). DirectEdit's edit leverage collapses when ψ_src is structurally far from Anima's training-time embedding manifold — Anima Tagger fixes that vs. a generic WD-tagger.
- **Caption pre-fill for LoRA training.** Tag your dataset, paste into `.txt` sidecars.
- **Prompt scaffolding.** Wire the caption STRING into `CLIPTextEncode` to seed a generation from an existing image's tag set.

## Install

The node lives inside the `anime_tools` repo (`comfyui/anima_tagger/`). Clone the
repo once and link (or copy) the node directory into `custom_nodes/`:

```bash
git clone https://github.com/sorryhyun/anime_tools
pip install ./anime_tools                                   # the tagger implementation (timm, transformers)
ln -s "$PWD/anime_tools/comfyui/anima_tagger" ComfyUI/custom_nodes/comfyui-anima-tagger
# Windows / no symlinks: copy anime_tools\comfyui\anima_tagger to ComfyUI\custom_nodes\comfyui-anima-tagger instead
```

Restart ComfyUI; the nodes appear under the `anima` category. The tagger
implementation is the [`anime_tools`](https://github.com/sorryhyun/anime_tools)
package (`anime_tools.tagger.AnimaTagger`) — this directory is only the node
surface, nothing is vendored.

The checkpoint auto-downloads on first use: our data + sidecar head (a few MB) is fetched from [`sorryhyun/anima-tagger`](https://huggingface.co/sorryhyun/anima-tagger) (`dbv4/` subfolder) into `tagger_dir` (default `models/captioners/anima-tagger-dbv4`) when any required file is missing.

The default checkpoint runs on the **dbv4** backend: the trunk is the external `animetimm/caformer_b36.dbv4-full` tagger (GPL-3.0, fetched by `anime_tools` on first use under your Hugging Face token — accept the repo terms first), projected onto Anima's vocab and topped with our sidecar head. No PE vision encoder is involved — `tagger_dir` is the loader's only widget.

## Checkpoint layout

`tagger_dir` should contain (the published `sorryhyun/anima-tagger` checkpoint already does — auto-downloaded if missing):

```
<tagger_dir>/
  config.json              # backend + vocab alignment metadata         (required)
  vocab.json               # tag list with category + median_pos info   (required)
  rules.yaml               # caption-normalization rules snapshot       (required)
  thresholds.safetensors   # per-tag F1-optimal thresholds              (optional, falls back to 0.5)
  groups.yaml              # tag-group taxonomy → softmax argmax mode   (optional)
  sidecar.safetensors      # sidecar head over dbv4 features            (optional)
  sidecar.json             # sidecar head config                        (optional)
```

`config.json`'s `backend` must be `"dbv4"` (the legacy PE-head backend was removed from `anime_tools` 2026-08-30).

Default `tagger_dir` is `models/captioners/anima-tagger-dbv4`, relative to `ANIME_TOOLS_HOME` / `ANIMA_HOME` when set (an `anima_lora` checkout), else the ComfyUI base directory. Absolute paths are used as-is. Build a custom checkpoint with `python -m anime_tools.tagger.cli` (see the `anime_tools` docs).

## Usage

### Caption an image

```
[Load Image] ──┐
               ├─► [Anima Tagger Caption] ──► [Save Text File]
[Anima Tagger Loader] ──┘
       tagger_dir: models/captioners/anima-tagger-dbv4
```

### Drive a normal text-to-image generation from an existing image's tags

```
[Load Image] ──┐
               ├─► [Anima Tagger Caption] ──► caption ──► [CLIPTextEncode] ──► [KSampler] ──► …
[Anima Tagger Loader] ──┘
```

### Plug into DirectEdit (cross-package)

```
[Anima Tagger Loader] ──► tagger ──┐
                                    │
                                    ▼
[Load Image] ─────────────────► [Anima DirectEdit] ──► edited image
                                    ▲
                  edit_text: "double peace"
```

DirectEdit owns its own ψ_tar logic and only needs the `ANIMA_TAGGER` socket — see [`comfyui-anima-directedit`](https://github.com/sorryhyun/anima_lora/tree/main/custom_nodes/comfyui-anima-directedit).

## Files

| File | Role |
|------|------|
| `nodes.py` | `AnimaTaggerLoader` + `AnimaTaggerCaption`. |
| `__init__.py` | Re-exports `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`. |
| `pyproject.toml` | ComfyUI Registry metadata. |

## References

- **AnimaTagger architecture.** [`anime_tools/docs/anima_tagger.md`](https://github.com/sorryhyun/anime_tools/blob/main/docs/anima_tagger.md).
- **DirectEdit integration.** `docs/experimental/directedit_editing_v3.md` in `anima_lora` (why ψ_src manifold-fit matters).
- **Vocab / checkpoint build.** `python -m anime_tools.tagger.cli --mode build_vocab` (`anime_tools`).
