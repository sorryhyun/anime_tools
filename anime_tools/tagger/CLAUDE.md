# anime_tools/tagger/

`AnimaTagger` = vocab/thresholds/optional sidecar head over the external `animetimm/*.dbv4-full`
caformer. Docs: `docs/anima_tagger.md` (architecture, training pipeline, calibration, the
`.env` knobs). The batch stage over a dataset is `stages/autotag.py`; `cli/autotag.py` here is
single-image/stdout only.

## Checkpoint

Checkpoint dir = `config.json`, `vocab.json`, `rules.yaml`, `groups.json`,
`thresholds.safetensors`, optional `sidecar.safetensors` (the file set is spelled once, in
`contract.py`); GPL backbone weights are fetched at load via `_hf.py` under the user's HF token,
never vendored. `data.py::TaggerCheckpoint.from_dir` is the one read of a checkpoint dir.
`dbv4_meta.py` (the constants the ComfyUI node and the GUI import) must stay importable without
torch.

## Backend selection

The backbone runs on timm (`dbv4_backend.py`), or on onnxruntime (`dbv4_onnx.py`) when
`<ckpt_dir>/dbv4.onnx` exists — the `tagger_onnx` catalog row builds that graph as part of
downloading the tagger (`python -m anime_tools.tagger.cli.export_onnx` is the same export with
knobs), and its presence is the whole selection rule (`ANIMA_TAGGER_BACKEND=torch` opts out).
The upstream repo's own `model.onnx` is the wrong graph: its `embedding` output is the 768-d pooled
feature, while the sidecar trains on the 3072-d `mlp_hidden` from inside timm's `MlpHead`, so
`onnx_export.py` exports both outputs itself and the sidecar / feature cache / trainer stay
untouched. Measured end to end on one CPU: 0.49 s/img against torch's 1.80 (fp32) and 2.39
(bf16), 7.6e-06 off the fp32 scores. bf16 was both slower and less accurate there, so
`default_dtype` picks per device. The ORT session comes from `_onnx.py::make_session` (CPU and
CUDA only, never CoreML).

## Derived state

`feature_cache.py` owns the dbv4 hidden-state cache (a cache built for another manifest is
misaligned row-for-row, so every reader checks the stem list in the safetensors metadata).
`derive_groups.derive_from_args` + `write_merged_groups` are the one groups-derivation path,
shared by `--mode derive_groups --apply` and `build_vocab --derive_groups`.

`stages/_models.py::load_anima_tagger` caches the tagger per `(checkpoint dir, device)` for a
process that runs several stages; `stages.release_models()` empties it.

## Tests

`test_tagger_checkpoint`, `test_tagger_dbv4_backend`, `test_tagger_onnx_backend`,
`test_tagger_vocab_precedence`, `test_tagger_sentinel_groups`, `test_tagger_derive_groups`,
`test_tagger_calibrate_thresholds`, `test_tagger_eval_metrics`; CPU-only unless
`ANIMA_TEST_GPU=1`.
