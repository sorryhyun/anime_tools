# Anima Tagger on the dbv4 backbone — remaining phases

Status: **backend SHIPPED 2026-08-27; Phase 6 (archive) DONE 2026-08-27; open = Phase 5 (RWR).**

Phases 0–4 landed and their write-up moved to
[`docs/experimental/anima_tagger.md`](../experimental/anima_tagger.md)
(§*dbv4 backend*); the **A3 knob resweep came back green 2026-08-27** with no
knob change, and its write-up moved to
[`docs/experimental/position_captions.md`](../experimental/position_captions.md)
(§*Re-swept on the dbv4 tagger*). The only proposal-level facts worth keeping
here:

- `config.json["backend"]="dbv4"` swaps the in-house PE dual-encoder head for
  `animetimm/caformer_b36.dbv4-full` (GPL-3.0, gated → fetched under the
  user's HF token, never vendored) behind the unchanged `AnimaTagger.predict()`
  contract; a sidecar linear head covers copyright / OC characters / renamed
  generals / people-count. **`@artist` is out of scope by decision** (the 92
  artist tags stay unsupported).
- Gates passed: copyright macro-F1 0.638 → 0.815; people-count 0.885 → 0.943
  (count-tag rule, authoritative); position hair-per-crop 10/10 and
  character-position 6/6 on the hand-GT 12, binding 48/48 (**pass `--images`**
  — the default GT discovery scores the pipeline's own output, see
  `bench/position_captions/README.md`); head-tier ECE 0.019 (card thresholds kept
  — zero mean bias vs val-optimal, do not recalibrate on 791 images);
  readback win-rate 1.000 / AUROC 1.000 (v3-era 0.991 / 0.98).
- Default flipped (`DEFAULT_TAGGER_DIR` → `models/captioners/anima-tagger-dbv4`,
  HF subfolder `dbv4/` holds our files only); v5 moved to
  `_archive/anima_tagger_training/checkpoints/` with Phase 6. Commit `2e519f03`.

## Open phases

| Phase | What | Gate | Cost |
|---|---|---|---|
| **5** | RWR artist LoRA per `tag_readback_reward.md` with the new (stronger) judge — `TagReadback` already runs on dbv4 | that proposal's gates (CMMD non-regression + held-out read-back lift + eyeball); **not** FM-val | as budgeted there |
| **6** | archive the PE training pipeline (below) | **DONE 2026-08-27** — see `_archive/anima_tagger_training/README.md`; `GroupRouter` → `anime_tools/captions/group_router.py`, path helpers → `feature_cache.py` | — |

### Training-time use — "tag adherence boost" (Phase 5 rationale)

Do **not** invent a new objective. `anime_tools/tagger/readback.py::TagReadback`
(Read-It-Back, arXiv 2607.11886) is the validated instrument — mean log σ(tag
prob) over the caption's content tags, **group-relative only** (same caption,
N images; absolute values across captions carry a language-prior term). Its
limits are recorded in `tag_readback_reward.md`: blind to the text/pose axis
(chance on turbo teacher-vs-student), and the FM-val trap
(`project_closed_lines_rollup`: never gate a training change on FM-MSE —
judge by CMMD + held-out read-back + eyeball).

With the judge now on dbv4, the consumers the readback proposal names light
up: `dave_mod_bestofn q_tag`, soup ingredient gating, seed selection, and the
**RWR artist LoRA** (reward-weighted regression — per-sample loss weights from
read-back on the model's own renders, ReST grow/improve over the existing CFM
loop). Estimator and phase discipline are inherited from PR #67.

A direct differentiable tag loss (tagger on the 1-step x₀ estimate) stays a
**later** option, not a phase: VAE decode per step, REPA-adjacent in shape
(feature alignment on turbo was refuted — `project_turbo_repa_phase0_drift`),
and its failure mode (the LoRA learns to please the tagger rather than the
caption) is exactly what the group-relative RWR formulation avoids.

### Archive plan (Phase 6)

Move to `_archive/anima_tagger_training/` (untracked tier):
`scripts/anima_tagger/{train_cached, train_common, caches, embed_tags}.py`,
the `build_features` / `train` / `calibrate` modes of `cli.py`, `make tagger`
/ `make preprocess-tagger`, the training tests
(`test_anima_tagger_{cached_dataset,dual_encoder,label_embed,pe_cache_batching}.py`,
`test_tagger_spatial_headroom.py`, `test_tagger_calibration_and_strokes.py`),
the v2/v3/v5/v6 checkpoints, and `docs/experimental/anima_tagger.md`
§Training pipeline (rewrite the doc around the backend/sidecar split). Reclaim
**158 GB**: `post_image_dataset/anima_tagger/` (42 GB, incl. the dead 32 GB
spatial-L dir) + `anima_tagger_stroked/` (116 GB); the `dbv4/` feature cache
(≈300 MB) stays — the sidecar trainer, readback bench and calibration check
read it. `project_tagger_resident_mmap_ram_budget` becomes moot.

Keep in tree: `anime_tools/captions/{anima_tagger, dbv4_backend, position_clauses,
tag_rules, taxonomy, tag_groups, readback, correction}.py`,
`scripts/anima_tagger/{vocab, derive_groups, predict, autotag, autotag_server,
eval_metrics, build_dbv4_ckpt, train_sidecar, calibrate}.py` (`calibrate.py`'s
`calibrate_thresholds` is what the sidecar trainer and calibration check use —
only its CLI mode goes), `bench/tagger_external/`, and the `GroupRouter`
(moves out of `train_common.py` into `anime_tools/captions/`).

Also close: `project_tagger_dual_hardrouted` / `project_tagger_v5_stroke_aug`
memories become historical; the "spatial-L headroom" line is superseded, not
refuted (its premise — the PE trunk is the ceiling — is confirmed from the
other direction).

## Risks still live

- **Licence.** dbv4 repos are **GPL-3.0** and gated; this repo is MIT.
  Loading GPL weights at runtime does not relicense our code, but the ComfyUI
  node must never *bundle* them and the sidecar ships as a **separate**
  safetensors containing only our head — that is how it is wired; keep it so.
- **Name-space drift.** dbv4 uses danbooru names as of 2025-10; `rules.yaml`
  renames and the booru id-space collision
  (`project_booru_id_space_collision`) both bite when joining vocabularies.
  `anime_tools/tagger/dbv4_backend.py::align_vocab` is the single join point
  (the bench shims onto it) — keep it that way.
- **Over-confidence above 0.5** (conf−acc +0.08…+0.16 on native tags): fine
  for ranking and thresholded emission, but don't use raw sigmoid values as
  sample weights without the group-relative readback normalisation.
- **`convformer_b36`** (the user's first "lighter" candidate) is the same
  MetaFormer family; re-run `bench/tagger_external/run_bench.py
  --external_repo animetimm/convformer_b36.dbv4-full --external_arch
  convformer_b36 --external_img_size 384` when its gate opens — one command,
  and `make tagger-dbv4 --repo … --arch …` builds the checkpoint if it wins.
