# bench/tagger_external

Moved here from the trainer (`anima_lora/bench/tagger_external/`) in the curation split,
Phase 3b (2026-08-30); the `results/` history came along (gitignored).

`run_bench.py` (in-house PE dual-encoder head vs an external dbv4 tagger) was
archived 2026-08-30 to
`_archive/anima_tagger_training/pe_backend_removed_2026_08_30/bench_tagger_external/`
(now under this repo's gitignored `_archive/`) when the PE tagger backend was removed —
"ours" *is* dbv4-backed now, so the comparison it ran is moot. Results under `results/`
predate that.

Still live: `calibration_check.py` (CPU-only ECE check on the dbv4 sidecar
probs) and `probe_position_rescore.py` (re-scores the trainer's
`bench/position_captions/` probe crops with an external timm tagger — run it from
the trainer checkout, or point `--autocaption_run` / `--binding_run` at those artifacts).

Envelope helper: `bench/_common.py` (`make_run_dir` / `write_result`) — a copy of the
trainer's, never an import.
