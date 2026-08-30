# Soft prompt for SAM3 — textual inversion of the `caption-position` subject prompt

> Moved from the `anima_lora` trainer (`docs/`) in the curation split, Phase 3b (2026-08-30); `bench/…` paths below are this repo's `bench/`.

Status: **SHIPPED 2026-08-27** as the default subject detector of
`caption-position` / `audit_multiview` / `probe_nms_pairs`
(`networks/calibration/sam3_girl_prompt.safetensors`). Phase 0 ran 2026-08-26,
the integration gates (A1/A2) and the default flip (A4) 2026-08-27. The
"more concepts" follow-up (speech bubbles, SFX, face, boy) is the open
proposal [`sam3_soft_prompt_expansion.md`](sam3_soft_prompt_expansion.md).

## 1. Problem

`caption-position` asks SAM3 for `girl`. The multiview audit
([`multiview_audit.md`](multiview_audit.md) §5.3–5.5) swept every text
variant: `girl` finds nothing on **310** of 3008 images (headless / cropped
body views), `anime girl` recovers them (zero-proposal 310 → 18) but inflates
the junk with them — degenerate survivors 2 → 8, whole-canvas phantoms 53 →
66, and the `R = 2.0` NMS tie-break band disappears. One discrete word moving
recall that much says the bottleneck is text↔feature alignment, not the trunk
— so optimise the prompt tensor continuously and leave SAM3 untouched.

## 2. What is learned

SAM3 reduces a text prompt to `language_features (32, 1, 256)` +
`language_mask` after the text tower's `resizer`; nothing downstream sees the
words (`SAM3Image._encode_prompt` concatenates the pair in front of the
geometric prompt). The bench (`bench/sam3_soft_prompt/`) optimises a delta on
that tensor — 4 valid tokens × 256 = **1024 parameters** — with the trunk
under `no_grad` and the fusion encoder / decoder / scoring frozen (the backward
pass traverses them: 0.45 s and 4.6 GB per image at 1008). Init from a real
phrase; Adam 2e-3, cosine, 2000 steps × 4 images (~25 min). The delta ends at
|δ|≈12 against |feats|≈94.

**Targets are SAM3's own outputs**, filtered to the uncontroversial subset
(`build_targets.py`): caption count exactly `1girl` / `2girls`, no `multiple
views`, and `girl`@0.5 leaves exactly that many NMS survivors with box fill
≥ 0.2 → **1526 images** (1374 train / 152 val). Loss = Hungarian-matched DETR
recipe (focal objectness + presence, L1 + GIoU, BCE + dice on the 288² masks).
Everything the prompt is meant to fix — the 313 zero-`girl` images and the 478
`girl` / `anime girl` disagreements — is excluded from training and forms the
eval.

The input-token variant (learn a `<anime_girl>` *word* so it composes —
`<anime_girl>, red hair`) is deliberately out of scope: SAM3 prompts are noun
phrases and the audit showed composition degrades grounding (§5.3). Post-tower
stays.

## 3. Phase 0 — result (2026-08-26)

Disagreement set, 478 held-out images (`ab_sam3_prompt.py --which disagree`;
sheets `post_image_dataset/captions/sam3_soft_prompt/ab_disagree_*/index.html`):

| subject prompt | zero-survivor imgs | survivors | degenerate (fill<0.15) | whole-canvas junk | caption-count match |
|---|---|---|---|---|---|
| `girl` (shipped before) | 296 | 283 | 1 | 0 | 24 / 145 |
| `anime girl` (text) | 1 | 866 | 39 | 27 | 106 / 145 |
| soft, `girl` init | 2 | 783 | 2 | 0 | 107 / 145 |
| **soft, `anime girl` init** | **0** | 706 | **0** | **0** | **121 / 145** |

Full corpus (3008 images, `probe_nms_pairs.py`, dual-floor replay as §5.4/5.5):

| | `girl` | `anime girl` | soft (anime-girl init) |
|---|---|---|---|
| zero-proposal images @0.5 | 310 | 18 | **4** |
| suppressions @0.5 / @0.35 | 28 / 110 | 86 / — | 37 / 74 |
| degenerate survivors @0.5 / @0.35 | 2 / 12 | 8 / 21 | **0 / 0** |
| whole-canvas proposals with fill < 0.10 | 53 | 66 | **0** |

The learned prompt keeps `anime girl`'s recall and removes the junk the audit
refused to ship it for. Eyeballed: headless / cropped body views
(`ama_mitsuki/5828774`, 0 → 3 clean masks at 0.83–0.92) are the recovered
population; masks are tight. **Init matters** — the `girl`-init arm is worse
on every column. Keeper:
`bench/sam3_soft_prompt/results/20260826-2310-animegirl-init/soft_prompt.safetensors`.

## 4. Integration — gates run 2026-08-27

### 4.1 Boys-drift: measured, negligible (A0 retrain closed)

The keeper visibly boxes boys on `pepper0/10243694` (7 survivors incl. 3
boys) and its pool had no boy negatives; the pure-girl count metric cannot
see this. The planned fix was a retrain on tagger-verified pair negatives.
`pair_negatives.py build` measured the defect first: keeper survivors @0.5 on
every image with a boy count tag, each crop mask-blanked and tagged with the
dbv4 tagger (`1boy` ∧ ¬`1girl` = boy box):

| pool | images | crops | girl | boy | both `1boy`+`1girl` |
|---|---|---|---|---|---|
| `1boy,1girl` exactly | 919 | 1118 | 670 | **8** | 440 |
| any boy count tag | 1045 | 1350 | 842 | **11** | 497 |

The "both" crops are the girl with a POV / faceless partner's limbs inside her
box — the corpus's `1boy` is mostly not a drawn figure. The boy crops are real
drawn boys (`pepper0/10243694`, `koh_(minagi_kou)/13408448`, …) and the tagger
labels every one correctly; the two-box pair images (105) are multi-panel
girls, not boys. So the drift is real only where a boy is fully drawn — **~1 %
of boy-tagged images** — a retrain has no pool (≤ 11 negatives) and the
100-image holdout gate cannot exist. `caption-position` already absorbs the
rare boy box through its `expected..expected+boys` count band, and a crop
tagged `1boy` gets its own clause (true content, not corruption). **Closed —
no retrain, no post-filter.** The `boy` concept for the region task is a
Part-B row with its own label source. Manifests:
`post_image_dataset/captions/sam3_soft_prompt/pairs{,_anyboy}/manifest.json`;
`pair_negatives.py eval` (boy-box rate / girl recall on held-out pair rows)
and the trainer's `--extra_manifest` are in place should a boy-heavy pool
ever appear.

### 4.2 Detector A/B inside `caption-position` (A1 — passed)

`ab_position_captions.py --a_flags= --b_flags='--prompt_embed <keeper>'`. The
A/B now builds a second detector on the same loaded SAM3 when the B side's
subject prompt differs (it used to share one detection pass, so a detector
A/B was silently impossible) and reports per-side status counts. Same dbv4
tagger both sides.

| set | candidates | proposed A → B | newly proposed | lost | too-few-instances A → B |
|---|---|---|---|---|---|
| `ama_mitsuki/*\|ie_(raarami)/*` (146) | 83 | 71 → 74 | 5 | 2 | 11 → 7 |
| corpus (3008) | 480 | 433 → 439 | 17 | 11 | 32 → 29 |

The swing is small because `is_candidate` gates on multi-view captions: the
313 zero-proposal images are mostly not candidates. All 17 newly proposed
sheets eyeballed: the recovered population is the headless / ass-focus panel
beside a full-body view (`ama_mitsuki/5828775`, `6359929`, `6040950`,
`ie_(raarami)/7428011`), every clause correct, ≤ 2 weak (`open mouth`,
`cropped legs` on a small inset). The 11 lost are mostly the old prompt's own
junk — one girl boxed twice yielding `feet` / `close-up` / `foot focus`
clauses (`ame_(uten_cancel)/dan_9670924`, `b-ginga/10149009`, `6726185`) —
which the soft prompt correctly no longer produces; 1–2 real losses
(`ama_mitsuki/9760121` drops below `min_instances`). Of 361 both-proposed
images 345 keep identical position words, 14 change clause count (an extra
panel found), 2 change wording (one `top middle` → `middle`) — so
`_EDGE_CLEAR = 0.47` does not flap under the new boxes and stays. Sheets:
`post_image_dataset/captions/position_ab_softprompt_{146,full}/`.

### 4.3 R / audit recalibration (A2 — passed)

R sweep (corpus, both floors): degenerate survivors **0 / 0**; pairs 37 @0.5
/ 74 @0.35; exactly one pair with R ≥ 2.0 (2.55, kept fill 0.274 — clean),
next highest 1.32 / 1.51. The `R = 2.0` swap is inert under the soft prompt
(one benign swap); left as is.

`audit_multiview.py` girl vs soft, same day, same tagger (dry runs,
`multiview_audit_a2_{girl,soft}/`): audited 2510 both; findings 51 → 56,
actionable 5 → 6. Per image: 38 shared (32 `unsure`→`unsure`, 5 `multiple
views` both), 13 only-girl and 18 only-soft — every one weak `unsure`, 2–4
instances, non-actionable. The 18 only-soft split into inset second views of
the same girl (recovered, e.g. `abmayo/*`), low-score body-part second boxes
(0.38–0.47, "read: nothing"), and drawn boys (`asou_(asabu202)/10153824`,
`wakamatsu372/10094155` — §4.1's drift, surfaced as `unsure`). Full-canvas
boxes all contain the girl: **no phantom-body class**. The one verdict flip,
`otokakoto/11809823` `unsure` → `multiple views` (strong), fires on a
0.01-area chibi sticker — dubious, **not applied**; review it by hand if the
audit is ever re-applied.

### 4.4 Tagger side (A3)

Landed independently: the dbv4 caformer backend is `DEFAULT_TAGGER_DIR`, knob
resweep green — [`position_captions.md`](position_captions.md) §*Re-swept on
the dbv4 tagger*.

## 5. Shipped state (A4)

- Keeper copied to **`networks/calibration/sam3_girl_prompt.safetensors`**
  (tracked, 165 KB; sha256 `2b690f78…`) and made the default `--prompt_embed`
  of `position_captions.py`, `audit_multiview.py`, `probe_nms_pairs.py`
  (`DEFAULT_SUBJECT_PROMPT_EMBED` + `resolve_prompt_embed` in
  `anime_tools.stages.instance_detection`). `--prompt_embed none` = the
  plain text prompt; a *missing default file* degrades to text with a warning,
  an explicit missing path raises. Part prompts (`face`, …) stay textual.
- `caption-position`'s `report.json` stamps `prompt`, `prompt_embed` and
  `prompt_embed_sha256` — a soft prompt is a file, so two runs are comparable
  only when the sha matches.
- Corpus re-applied 2026-08-27: flatten 454 → apply **439 written** (rewritten
  436, 3916 tags moved, reuse 0.886) → TE re-encoded (468 caches). Gotcha:
  `make preprocess-te` is not `--queue`-aware (`cache_text_embeddings.py`
  rejects the flag) — run that child through `make daemon-run`.

## 6. Caveats that remain

1. **Self-labelled.** Targets are `girl`'s clean outputs; the gain on hard
   images is extrapolation, verified by eyeballing (Phase 0 sheets, the 17
   A1 sheets, the 19 A2 sheets), not by ground truth.
2. **Drawn boys are boxed** (~1 % of boy-tagged images) and reach the tagger,
   which then labels the crop `1boy` — harmless for clauses, a real
   consideration for the region task's partner masks.
3. **Prompt ≠ word.** Anything that consumes the detector must record which
   file ran (`prompt_embed_sha256`); `sam_mask.yaml` rules will need the same
   once Part B lands per-concept prompt files.

## 7. Part B — text / SFX row: can a soft prompt replace MIT? **NO** (2026-08-27)

The proposal's only row with an *independent* label source (MIT = UNet++
strokes gated by comictextdetector blocks, what `make mask` runs). Gate:
MIT box recall ≥ 0.9 at ≤ 0.1 FP/img on a held-out split.

**Setup.** `build_text_targets.py` ran MIT over the 3 008 resized images:
770 text-positive (2 001 CTD blocks; box = tight stroke bbox, mask = strokes
closed into a glyph blob at 288²), 638 clean negatives, 1 600 ambiguous
(UNet++ or CTD fires alone — excluded). Stable 1/8 hash holdout = 105 pos +
66 neg. `train_soft_prompt.py` gained zero-target rows (negatives push the
presence head to "absent"); `--init text`, defaults (2 000 steps × 4, lr
2e-3), 1 114 train / 123 val, 21 min.

**Phase 0 — plain text prompts are blind.** On the holdout, `text` reaches
box recall 0.02 @0.3 (px recall 0.03); `sound effect text` / `written text`
0.00 at every floor; `handwritten text` 0.03 @0.2. SAM3 simply does not
ground text in this corpus with any phrase.

**Soft prompt — learns presence, not localisation.** Val presence loss
1.09 → 0.16 and objectness 1.21 → 0.40, but box L1 / GIoU and the mask loss
were flat from step 0 (0.013 / 0.13 / 0.25 → 0.012 / 0.11 / 0.25); the
clean-recall proxy plateaued at 0.61 by step 750. Holdout
(`results/20260827-1055-eval-text-soft/`):

| floor | box recall | FP / img | neg-img FP rate | px recall | over-mask × | gate |
|---|---|---|---|---|---|---|
| 0.2 | 0.76 | 3.70 | 0.05 | 0.95 | 4.2 | ✗ |
| 0.3 | 0.67 | 2.08 | 0.05 | 0.90 | 3.5 | ✗ |
| 0.4 | 0.53 | 1.06 | 0.03 | 0.71 | 2.3 | ✗ |
| 0.5 | 0.35 | 0.50 | 0.02 | 0.44 | 1.2 | ✗ |
| 0.6 | 0.15 | 0.16 | 0.02 | 0.20 | 0.4 | ✗ |

The two halves of the gate never meet: ≥ 0.9 pixel recall only at floors
where it emits 2–4 spurious boxes per image and 3–4× MIT's masked area;
≤ 0.1 FP/img only where recall is ≤ 0.2. Negatives are fine (≤ 5 % of
text-free images get any box) — the failure is *on* text images: 38 % of
the FP boxes are extra fragments on images whose targets were all hit, the
rest are non-text regions near the text; small text (< 1 500 MIT px, 20/105
images) recalls only 0.66; 14/105 images are below 0.5 px recall. Sheets
(`sheets/…soft_prompt.safetensors/`) show a typical SFX-heavy frame where the
prompt boxes nothing at 0.5 while MIT catches the four bubbles it does.

**Why it is a capacity verdict, not a label verdict.** The prompt is 1 k
parameters in front of a frozen detector whose 288² mask head and DETR
queries were never trained on glyphs; the loss curve says the prompt found
the direction ("text-like region present") and had nothing left to move for
*where*. The kill condition in the proposal ("no second attempt without a
new label source") does not even apply — the label source is fine. Trunk
LoRA is off the table by the proposal's own rule. **MIT stays the text
backend; `RUN_MIT_MASK=0` is not viable.** Do not re-propose without a
detector that has a text prior (the CTD block head already is one).

Reusable: `build_text_targets.py` / `eval_text_prompt.py` are the generic
"independent-detector label + gate" harness the other Part B rows need
(`--prompts` takes any mix of text and `.safetensors`).

## 8. Usage

```
make daemon-run ARGS="bench/sam3_soft_prompt/build_targets.py"          # pseudo-labels + splits
make daemon-run ARGS="bench/sam3_soft_prompt/train_soft_prompt.py --init 'anime girl' --label x"
make daemon-run ARGS="bench/sam3_soft_prompt/ab_sam3_prompt.py --which disagree --b <soft_prompt.safetensors>"
make daemon-run ARGS="-m anime_tools.masking.cli.probe_nms_pairs --prompt_embed <soft_prompt.safetensors>"
make daemon-run ARGS="-m anime_tools.stages.cli.ab_position_captions --a_flags= --b_flags='--prompt_embed <file>'"
make daemon-run ARGS="bench/sam3_soft_prompt/pair_negatives.py build [--count_filter any_boy]"
make caption-position ARGS="--prompt_embed none"                        # back to the text prompt
```

## 9. Code map

| file | role |
|---|---|
| `bench/sam3_soft_prompt/common.py` | SAM3 plumbing: `encode_image` (no `inference_mode`, so the prompt gradient flows), `encode_text`, `install_prompt`, differentiable `ground`, `proposals` / `nms` mirroring the audit, soft-prompt save/load |
| `bench/sam3_soft_prompt/build_targets.py` | pseudo-labels + `train` / `zero_girl` / `disagree` splits → `post_image_dataset/captions/sam3_soft_prompt/targets/` |
| `bench/sam3_soft_prompt/train_soft_prompt.py` | the 1024-param optimisation; `--extra_manifest` appends train rows from another manifest (pair negatives) |
| `bench/sam3_soft_prompt/ab_sam3_prompt.py` | text-vs-text / text-vs-soft contact sheets + numeric report |
| `bench/sam3_soft_prompt/build_text_targets.py` | MIT (UNet++ + CTD gate) → DETR targets + negatives + 1/8 hash holdout for the text row |
| `bench/sam3_soft_prompt/eval_text_prompt.py` | box recall / FP-per-img / px recall / over-mask vs MIT per floor for any prompt spec; `--sheets` |
| `bench/sam3_soft_prompt/pair_negatives.py` | `build`: tagger-labelled boy / girl boxes on boy-tagged images; `eval`: boy-box rate + girl recall on held-out rows |
| `anime_tools.stages.instance_detection` | `DEFAULT_SUBJECT_PROMPT_EMBED`, `resolve_prompt_embed`, `prompt_embed_sha256` |
| `anime_tools.stages.position_captions::build_detect_fn` | installs the prompt on the subject pass; accepts a loaded `model`/`processor` so the A/B can build a second detector |
| `anime_tools.stages.cli.ab_position_captions` | per-side detector when the B prompt differs; per-side status counts |
