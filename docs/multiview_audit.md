# Multi-view audit — untagged `multiple views` in the caption master

Ships as a stage: it sweeps the captions `caption-position` never looks at,
finds the ones that are several views of one character, and can write the
missing `multiple views` tag into the caption master.

## 1. The blind spot

`is_candidate` (`anime_tools/stages/position_captions.py`) sends a caption to
detection only if it has a layout tag or claims more than one girl. A sheet that
*is* several views of one character but was never tagged `multiple views` claims
`1girl`, so it is skipped as `single-subject` and never looked at. Over the
caption master that is **2551 of 3007 captions**; some unknown fraction of them
are mis-tagged.

The second-order damage matters more than the missing tag. If such an image *did*
reach the clause writer, `is_repeated_subject_layout` would return `False` (no
layout tag), so the `view_invariant` gate would not fire and the writer would bind
the character's name and her view-invariant traits **per view** — asserting one
girl as two. The missing tag is what keeps that gate from working.

Founding example: `ama_mitsuki/5847168` — one office lady drawn twice (bent-over
close-up left, seated full body right), captioned `1girl`, no layout tag.

## 2. The tool

| File | Role |
|---|---|
| `anime_tools/stages/multiview_audit.py` | Orchestration: population filter, detection, verdict |
| `anime_tools/stages/multiview_sheet.py` | Per-finding contact sheet renderer |
| `anime_tools/stages/cli/audit_multiview.py` | Thin CLI |
| `anime_tools/masking/cli/probe_sam_masks.py` | Diagnostic: dump SAM3's raw masks for one image, colour-coded |
| `anime_tools/masking/cli/probe_nms_pairs.py` | Diagnostic: replay NMS over a corpus, record every suppressed pair's scores + mask fills |

## 3. Running it

```bash
python -m anime_tools.stages.cli.audit_multiview              # dry run + sheets
python -m anime_tools.stages.cli.audit_multiview --apply      # write the master
```

Dry-run by default; the report lands in `workspace/captions/multiview_audit/`.

**Population** — exactly the complement of the clause pipeline: every caption
`is_candidate` rejects with reason `single-subject`. Tied to that function's own
reason string so the two cannot drift apart. A girls-count of 0 (scenery, `1boy`,
an uncounted caption) stays in: the subject prompt finding two subjects there is
just as much a caption bug.

**Skip reasons** in the report's counter:

- `handled-by-position-captions:<reason>` — `is_candidate` accepted it, so
  `caption-position` already handles it (e.g. `already-has-clauses`).
- any other `is_candidate` rejection reason, passed through unchanged.
- `count-explained` — not a finding: the caption's own `girls + boys` already
  covers every box. Needed because the subject prompt does not exclude males, so
  a `1girl, 1boy` image lands in this population (girls-count is 1) and detects
  two bodies that are both already named.
- `single-instance` — detection found at most one subject.

**Detection** differs from the clause pipeline in one way on purpose: the
escalation target is forced to `min_instances` (2) rather than the caption's
count. Passing `expected=1` would satisfy the target on the first box and
suppress both the low-threshold retry and the body-part fallback — on the exact
population we are trying to search.

**Evidence model.** Two boxes on a `1girl` caption raises the image; three
signals then argue about what it means, and `--apply` requires two to agree:

1. **Identity agreement** across the per-instance crops (hair / eye / hairstyle,
   plus character name). The only signal that separates "one girl twice" from "a
   second girl" — but it goes silent on a headless crop. A box recovered only by
   the retry escalation (below `score_threshold`) does not vote on identity
   (`reliable=False`), though it still counts as a body.
2. **Whole-image `multiple views` head.** Needs no legible crop; the fallback
   when (1) has nothing. Runs on *every* audited image, so a sheet whose views SAM
   merged into one box still surfaces (`tagger-only`, never above `weak`).
3. **People-count head** saying `1girl` while the geometry sees several bodies.

**Verdicts**: `multiple views` / `extra-character` / `unsure` / `count-explained`.

**Writes**: `--apply` writes the **caption master** (`image_dataset/`), unlike the
clause rewrite which only touches the derived caption — a missing `multiple views`
is a fact about the picture that every later stage should read down from. Append
at the end of the flat bag, via `compose_caption` so trailing clauses survive.
Default `--apply_verdicts` is `multiple views` only, `--apply_confidence` is
`strong` only. **`image_dataset/` is gitignored** — `report.json` holds the
verbatim before-text and is the only undo. Follow any apply with the trainer's
TE re-encode (`make preprocess-te`).

**`--from_report <report.json>`**: replay a dry run's findings instead of
re-auditing — the report already holds `caption_path`, the before-text
(`caption`) and the `proposed` caption, so the write needs **no SAM3 and no
tagger** (the run does not import `torch`; pinned by
`tests/test_stage_replay.py`). The verdict/confidence gate is still applied at
replay time, so one audit pass can be replayed at several tiers:

```bash
python -m anime_tools.stages.cli.audit_multiview            # the model pass, once
python -m anime_tools.stages.cli.audit_multiview --apply \
    --from_report workspace/captions/multiview_audit/report.json \
    --apply_verdicts 'multiple views,extra-character'
```

Same staleness rules as the other two stages (full table in
[`position_captions.md`](position_captions.md)): a report whose recorded
`summary.src`/`dst` disagree with this run, or whose own `applied` is true, is
refused; a master caption edited since the audit is skipped as `skip:drifted`,
never overwritten — the same guard `apply_findings` applies in-process. Output
goes to `apply_report.json` (never over the `report.json` it read), whose
top-level `written[]` lists the relative image paths actually written.

This is the *gate-based* replay. The reviewer-curated workflow — hand-picking
findings across tiers from the contact sheets, with a revert manifest — is
`anime_tools/stages/cli/audit_apply_curated.py`.

**Contact sheets** (`<report_dir>/sheets/`, on by default): one PNG per finding —
boxed original, the crops the tagger actually saw colour-matched to their box, the
identity read off each, the verdict and its witnesses, and the proposed caption.
Filenames are `verdict_confidence_stem.png` so a directory listing sorts by
verdict.

## 4. Shipped detection behaviour

**The subject detector is a learned SAM3 soft prompt**, not a text prompt — it is
the default of this audit and of `caption-position` alike (`--prompt_embed none`
falls back to the plain `girl` text prompt). It has the recall of the best text
variant with the junk profile of `girl`: near-zero whole-canvas empty-mask
proposals, and no degenerate NMS survivors.

**Mask quality decides the survivor of an NMS-matched pair.** Greedy NMS in
shared `dedupe_detections` used to rank on score alone, so a garbage proposal — a
near-whole-canvas box over an almost empty mask — could outscore the clean
duplicate it overlapped by a hair and suppress it, leaving the tagger reading a
crop of nothing. When NMS has already judged two proposals to be **the same
object**, the survivor is now the one that fills more of its own box:
`--dedupe_fill_ratio`, default **2.0**, swaps the pair when the loser's
fill-within-its-own-box is that many times the survivor's; `0` disables it.

2.0 is the default because the corpus measurement found an empty band there —
every pair at ratio ≥ 2.0 had a degenerate survivor and every pair below it a
clean one — so the value sits in a gap rather than on a tuned edge. It is a
*relative* comparison inside a pair NMS has already matched, which is why it does
not run into the settled negative against an *absolute* mask-fill cut (see
[`position_captions.md`](position_captions.md)): it needs no cut-point and it
cannot drop an instance, only swap which of two duplicates represents it. The
rule lives in shared `dedupe_detections`, so `caption-position` gets it too.

A companion guard that would have dropped degenerate *proposals* outright by an
absolute fill threshold was measured and **refuted** — real sparse-subject views
sit in the same fill band as the junk — and does not ship. That failure shape is
handled by the score floor plus audit spot-checking.

The research history behind all of this — the founding investigation, the
per-corpus measurements, the prompt sweeps and the declined alternatives — was
split out of this doc into a local, gitignored archive and is not part of the
published docs.
