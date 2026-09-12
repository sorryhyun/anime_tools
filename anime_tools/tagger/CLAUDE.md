# anime_tools/tagger/

`AnimaTagger` = vocab/thresholds/optional sidecar head over the external `animetimm/*.dbv4-full`
caformer. Docs: `docs/anima_tagger.md` (architecture, training pipeline, calibration, the
`.env` knobs). The batch stage over a dataset is `stages/autotag.py`; `cli/autotag.py` here is
single-image/stdout only.

## The three torch-free leaves

`tagger.py` is the model and nothing else. What needs no torch sits beside it and is re-exported
from it, so the vocab build and the ComfyUI node do not load a backbone to read a tuple:

| Module | What is in it |
|---|---|
| `dbv4_meta.py` | where a checkpoint's *files* live: repo ids, the required/optional file sets, `DEFAULT_TAGGER_DIR` |
| `schema.py` | what a tag *means* to a checkpoint: `SLOT_ORDER`, `TAG_TYPE_NAMES`, `RATINGS`, `PEOPLE_COUNT_LABELS`, `TagEntry`, `dedupe_count_tags` |
| `fetch.py` | getting one onto disk: `ensure_tagger_checkpoint`, `ensure_tagger_backbone`, `is_dbv4_dir` |

Every value in `schema.py` is baked into `vocab.json` at build time and read back at inference, so
changing one invalidates existing checkpoints. `tests/test_boundary.py` pins all three importable
without torch.

## Checkpoint

Checkpoint dir = `config.json`, `vocab.json`, `rules.yaml`, `groups.json`,
`thresholds.safetensors`, optional `sidecar.safetensors` (the file set is spelled once, in
`contract.py`); GPL backbone weights are fetched at load via `_hf.py` under the user's HF token,
never vendored. `data.py::TaggerCheckpoint.from_dir` is the one read of a checkpoint dir.
`dbv4_meta.py` (the constants the ComfyUI node and the GUI import) must stay importable without
torch.

## The backbone

The backbone runs on timm (`dbv4_backend.py`) and nothing else. It could also run from an
exported ONNX graph until 2026-09-09: that existed because timm on an Apple CPU was 3.7x slower
than onnxruntime (0.49 s/img against 1.80 fp32, 2.39 bf16), and it went when `_device.py` started
answering `mps`, where torch is 5.7x *faster* than the graph ever was — 87 ms/img against 491, at
2.3e-06 on the scores. Gone with it: `dbv4_onnx.py`, `onnx_export.py`, `cli/export_onnx.py`, the
`tagger_onnx` catalog row, `ANIMA_TAGGER_BACKEND`, and `AnimaTagger.dbv4_runtime` (the autotag
report's `runtime` key). The upstream repo's own `model.onnx` was never usable anyway: its
`embedding` output is the 768-d pooled feature, while the sidecar trains on the 3072-d
`mlp_hidden` from inside timm's `MlpHead`.

`default_dtype` picks the dtype per device and is the one place that decides: bf16 on CUDA,
float32 everywhere else — including MPS, where bf16 measured *slower* (102 ms/img against 87) and
1.3e-02 off.

## Derived state

`feature_cache.py` owns the dbv4 hidden-state cache (a cache built for another manifest is
misaligned row-for-row, so every reader checks the stem list in the safetensors metadata).
`derive_groups.derive_from_args` + `write_merged_groups` are the one groups-derivation path,
shared by `--mode derive_groups --apply` and `build_vocab --derive_groups`.

`stages/_models.py::load_anima_tagger` caches the tagger per `(checkpoint dir, device)` for a
process that runs several stages; `stages.release_models()` empties it.

## Tests

`test_tagger_checkpoint`, `test_tagger_dbv4_backend`,
`test_tagger_vocab_precedence`, `test_tagger_sentinel_groups`, `test_tagger_derive_groups`,
`test_tagger_calibrate_thresholds`, `test_tagger_eval_metrics`; CPU-only unless
`ANIMA_TEST_GPU=1`.
