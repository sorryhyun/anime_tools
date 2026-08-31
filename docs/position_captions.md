# Position-aware captions — binding attributes to the subject they belong to

SAM3 detects the subjects, they are put in reading order, each mask-blanked crop
is tagged by the Anima Tagger, and the caption is **rewritten** so each attribute
is asserted once, in the clause of the subject it belongs to. The rewrite lands
on the **derived** captions under `workspace/resized/`; the hand-written master
in `image_dataset/` is never written. Dry run is the default.

The measurement logs behind each rule — sweep tables, A/B results, dated
rollouts, refuted alternatives — are in
`_archive/docs/position_captions_history.md` (local-only, gitignored). Read that
before *retuning* a rule; read this to run the stage.

## Why

The flat tag bag cannot say *which* attribute belongs to *which* subject
(`3girls, blonde hair, aqua hair, red hair, …` — who is blonde?). The dataset
already has a hand-written answer, 14 captions carry it, and the base model
already obeys it (a probe scored 48/48 sides correct on counterbalanced
positional prompts). The candidate gate is the number of **detected** instances,
never the girls-count tag: a `1girl, multiple views` sheet is four bindable
subjects and goes through the same machinery as `3girls`.

```
explicit, 3girls, kisaki (blue archive), black hair, white hair, pink hair, …

  ↓ caption-position

explicit, 3girls, kisaki (blue archive), blue archive, @aak. On the left,
kisaki (blue archive), black hair, blue eyes, hair bun, back, double bun, ass,
loli. On the middle, white hair, purple eyes, hair between eyes, underwear
only, black bra, navel, black wings, underwear. On the right, pink hair, blue
eyes, ahoge, halo, loli, heterochromia, standing, black wings.
```

The three hair colors **leave** the flat bag — a bag still listing all three
alongside the clauses is still claiming all three of all three girls. **This is
the convention, not an invention**: across the 14 ground-truth captions every tag
appearing in both a clause and the bag is a **character name**.

## The clause grammar — read this before touching any caption code

The convention delimits clauses with a **period** and tags with **commas**:

```
<flat tag bag>. On the left, akita neru, yellow eyes. On the right, kasane teto.
```

A plain `caption.split(",")` therefore glues the clause header onto the previous
tag (`"white socks. On the left"`), and every consumer keying off
`tag.startswith("On the ")` then sees **no clauses at all** — a live bug once, in
the shuffle and identity-randomize paths, which silently scattered hand-written
clause attributes across the variants.

**Never hand-split a caption.** `anime_tools/captions/position_clauses.py` (pure
stdlib, no torch) is the single grammar:

| Function | Use |
|---|---|
| `parse_caption(text) -> ParsedCaption` | `.flat_tags` + `.clauses`; safe on clause-free captions (round-trips to flat tags alone) |
| `compose_caption(flat_tags, clauses)` | inverse; safe to route every caption through |
| `has_clauses(text)` | "does this already carry clauses?" — the prefilter's leave-it-alone check |
| `flatten_caption(text)` | merge every clause back into the flat bag and drop the clauses |
| `assign_positions(boxes, size)` / `ordered_indices(...)` | position vocabulary + reading order |

It accepts both written forms (period-delimited, and the comma form where the
header is its own token) and case-insensitive `on the left`; emission is always
the canonical capitalized `On the `. `In the …` (a few hand-written scene-region
clauses) parses and round-trips byte-stable but is never emitted.

### Position vocabulary

Boxes are grouped along the axis that actually **separates** them: rows first, by
y-interval overlap (`--row_tol` = the minimum fractional overlap of the narrower
extent), so grid sheets read row-major. When every box lands in one y-group — the
magazine layout, a full-height subject beside a column of stacked panels — the
x-axis is grouped into center-gap lanes instead and the columns read top→bottom.

| N in a row | Words |
|---|---|
| 2 | `left`, `right` |
| 3 | `left`, `middle`, `right` |
| ≥4 | `leftmost`, `second from left`, …, `rightmost` |

A row group prefixes the horizontal word (`On the top left, …`), up to 3 rows
(`top`/`bottom`, or `top`/`middle`/`bottom`; `MAX_ROWS`). A row holding one
subject reads as the bare row word, plus the side it hugs when it leaves the
other clear — a diagonal pair reads `top left` / `bottom right`; on a magazine
layout the stacked column takes `top left` / `bottom left` and the full-height
subject a bare `right`. When nothing separates the boxes, or a grouping outgrows
the row vocabulary, it degrades to plain left→right.

## Pipeline

Per candidate image (`anime_tools/stages/position_captions.py`):

1. **Detect** — SAM3 on the *resized* image (the pixels training actually sees).
   The subject pass runs a **learned soft prompt**: `--prompt_embed`, defaulting
   to `DEFAULT_SUBJECT_PROMPT_EMBED` in `anime_tools/downloads.py`
   (`networks/calibration/sam3_girl_prompt.safetensors`); `--prompt_embed none`
   restores the text prompt `girl`, and a checkout without the artifact degrades
   to it **with a warning** (fetch it with `python -m anime_tools.downloads
   soft_prompt`, or the GUI's Settings → Models row). `report.json` stamps
   `prompt_embed` + `prompt_embed_sha256` — two runs only compare when the sha
   matches. Score floor 0.5, greedy IoU dedupe at 0.65, and a **retry at 0.35**
   when the count undershoots (an unconditional low threshold floods grids with
   part-detections); the retry target is `expected or min_instances`, so
   multi-view sheets attempt it too. `build_detect_fn` builds the processor at
   the *lowest* threshold it may be asked for and memoises raw detections, so the
   retry is a re-filter, not a second encode.
2. **Order** — group by row (or lane), then read within the group.
3. **Crop + blank** — padded bbox crop (6%), every non-instance pixel blanked to
   white using the instance mask. Load-bearing: without blanking a neighbor
   standing inside the padded box contributes their hair to this subject's tags.
4. **Tag** — Anima Tagger per crop, then clause selection (below).
5. **Rewrite** — the tags a clause has earned leave the flat bag and
   `compose_caption(flat_tags, clauses)` is written to the derived caption.

Models are injected as `detect_fn` / `tag_fn` callables, so the orchestration
module imports neither SAM3 nor the tagger and unit-tests with stubs; the CLI
shell (`anime_tools/stages/cli/position_captions.py`) owns argparse + loading.

**Layout tags decouple the girls-count from the bindable-subject count**
(`_LAYOUT_TAGS`): `multiple views` **and** comic pages (`comic`, `silent comic`,
`sequential`, `Nkoma`, `multiple 4koma`) — a `1girl, 2koma` page is two subjects,
so the count check is waived and detection trusted. `Nkoma` restores a
`panels × (girls + boys)` ceiling (`caption_panel_ceiling`) so one girl detected
twice still skips; `page number` is deliberately **not** a layout tag. The
strict-count gate is a **range**, `girls … girls + boys` (`caption_boy_count`;
unknown counts like `multiple boys` drop the upper bound), because the subject
prompt picks males up inconsistently; `6+girls` and friends defer to detection.

## What goes into a clause

Eligibility comes from the tagger's own `groups.yaml`, not substring heuristics,
so the two can't drift. Which groups bind — and every gate below — is
**configuration**: `anime_tools/captions/data/clause_vocabulary.yaml` holds the
sets (`ClauseGroups`) with the rationale inline, validated against the
checkpoint's declared groups at load; `load_clause_groups(path)` →
`load_clause_vocabulary(ckpt, clause_groups=…)` A/Bs a rule set without Python.

**Per-subject groups bind** — hair (color/length/style/accessory), eyes,
expression, body, all clothing, pose/gesture/action (`SUBJECT_GROUPS`) — plus
`framing`, the one group not about the subject (gate 4). **Scene groups never
bind**: lighting, background, medium, `interaction`, `character_relationship`.
**Copyright / artist / metadata / deprecated / count / rating are excluded
outright** on *every* emission path — the check lives in `add()`, not only on the
ranked path, because an excluded tag can also be *grouped* (`light brown hair` is
a deprecated alias still filed under `hair_color`, and it rode the
exclusive-group step into clauses). **Ungrouped tags** — curated compounds like
`pink jacket` — are admitted only when they are both in the flat bag and
attributable (kept on exactly one crop). At most **one member of an exclusive
group** may bind, or a contaminated crop emits `green hair, …, aqua hair`.

Emission order: character name → exclusive-group winners (`hair_color`,
`eye_color`, `hair_length`, `hairstyle`) → everything else ranked, preferring
tags the caption already curated. Cap 8 per clause. A **character name** needs
both floors conjunctively — `--name_confidence` (0.5) *and* membership in the
flat bag, since names are the weakest crop signal (`--allow_unlisted_names`
relaxes the second). One identity per subject.

### Move, don't invent — the flat bag fills a clause first

Candidates are ranked once, then admitted in **two passes**: everything already
in the flat bag, and only then up to `--max_novel_tags` (**1**) tags the caption
never contained. A novel clause tag cannot be a **move** — `plan_bag_removals`
only removes what is in the bag — so it is a pure addition that disambiguates
nothing, and raising the budget buys padding, not bindings. `0` never invents;
`--max_clause_tags` is the bag-blind arm. The report carries `clause_tags` /
`novel_tags` / `reuse_ratio`.

**Refinement collapse is deliberately not built** (`breasts` → `large breasts`):
a safe same-group rule catches almost none of it and the looser rule also
collapses `on stomach` → `stomach`. Doing it properly needs a tag *implication*
table, which neither `taxonomy.py` nor `groups.yaml` has — don't re-propose
without one.

## The four gates — what may enter a clause at all

The gates run **before** the rewrite, so a gated tag can never be moved out of
the bag either.

**1. Only what discriminates.** Any tag *every* crop keeps is suppressed
(`--keep_shared_tags` disables). On a `1girl, multiple views` sheet all views
share the character, hair and eyes; repeating them binds nothing and crowds out
the maid / bunny / bikini that tells the views apart. A shared attribute keeps
its place in the flat bag; when suppression empties every clause the image skips
as `no-discriminative-tags`.

**2. On a repeated-subject layout, only what a view can differ in.** A
`_LAYOUT_TAGS` image is one character drawn several times, so nothing belonging
to *her* can discriminate — a trait that survived gate 1 did so precisely because
some crop *missed* it, and gate 1 then promotes the miss. The multi-view gate
(on; `--bind_view_traits` reverts) suppresses at emission time the **character
name** and **every `_VIEW_INVARIANT_GROUPS` trait** (=
`_CHARACTER_INVARIANT_GROUPS`: hair color/length/style, eyes, face, age, gender,
skin, body shape, species, animal parts). `body_parts` is **not** in that set
(`--gate_view_anatomy` restores it): unlike hair color, what anatomy is *visible*
is a fact about the panel, so a from-behind view takes `ass`/`back` and its front
sibling `breasts` — often the only thing separating them. The residual risk is a
sibling crop that merely *missed* the anatomy, guarded only by
`discriminative_only` and `--attribution_margin`, so spot-check the sheets. What
is left is outfit, pose, expression, framing, visible anatomy; a suppressed trait
stays asserted flat and the freed slot refills from the ranked tail.

**3. The identity gate — the flat bag outranks the crop tagger.** For a group in
`ClauseVocabulary.gated_groups()` a clause may carry a value the caption named,
or nothing. Emitting the crop's winner unconditionally is a noise *amplifier*:
gate 1 suppresses whatever every crop agrees on, so a wrong outlier is exactly
what survives — on a back view the eyes are not visible, the tagger guesses, and
the guess is promoted *because* it disagrees with the front view. With the gate,
invented identity values and contradictory hair/eye colors across views of one
girl go to zero while total clause tags rise, the blocked slot refilling from the
tail. **The gated set is derived, not hand-picked**: `_BAG_GATED_GROUPS`
(`hair_color`, `eye_color`, `hair_length`) plus every exclusive subject group the
checkpoint declares — an exclusive (softmax) group holds one value by
construction, so a crop naming a second is a contradiction, and deriving it from
`groups.yaml` keeps the gate from drifting from the tagger. `hair_length` is the
one non-exclusive member, hence the surviving hand list; `hairstyle` is
deliberately **not** gated, since a crop legitimately reveals a `hair bun` the
booru caption never tagged. An exclusive slot also prefers a **kept tag the
caption already named** over the crop's softmax winner, which otherwise let the
gate reject the winner and emit nothing. The gate is load-bearing for the
rewrite: it keeps a hallucinated hair color from *replacing* the real one in the
bag. `--ungated_identity` reverts it.

**4. Framing — which view a clause is describing.** `framing` is in
`SUBJECT_GROUPS`, so a clause can read `On the left, ass focus, denim,
underwear.` and a sheet of one full body plus a headless hip panel can say which
clause is which (`--no_framing` is the A side). It is the only member describing
the **view** rather than the girl, which costs three exceptions:

- **Exempt from the bag gate** (`_UNGATED_EXCLUSIVE_GROUPS`): gate 3 would
  otherwise derive it, but its premise (a second value contradicts the first) is
  false here — a sheet's bag legitimately says `full body` for one panel and
  `ass focus` for another. Without the exemption the feature is inert on the
  sheets it targets.
- **Three members never bind** (`_PAGE_LEVEL_FRAMING`): `solo focus` is about the
  other characters, `size difference` about a pair, `white border` about the
  canvas — and the rewrite *moves* a bound tag, so binding one would delete a true
  statement about the image and re-assert it about one panel. The check lives in
  `add()`, not only in `is_scene_tag`, because `framing` is a priority group and
  that step reads the winner straight off the tagger.
- **It joins `_PRIORITY_GROUPS`, last** — for the novel budget, not the emission
  order: otherwise a framing tag the caption never named loses the single
  `--max_novel_tags` slot to whatever the crop scored higher.

It does **not** recover a panel kind shared by *every* crop (gate 1 blocks a tag
only when all crops kept it, so a two-backside-panel sheet is the failing case),
and `torso only` / `cropped torso` are not in the tagger's vocabulary at all —
`lower body` is, but ungrouped, so it can only ride the
in-the-bag-and-attributable path.

## Body-part detection fallback (opt-in)

Some sheets are one small full body plus two or three **headless close-up
panels** — a hip, a backside, a crotch — which the subject prompt cannot see at any
threshold, so the image dies on `too-few-instances` with its most attribute-dense
panels never tagged. `--part_prompts buttocks,hips,thighs` runs extra SAM3
prompts as a **second escalation**, under the same undershoot condition as the
low-threshold retry and never on an image the subject prompt already resolved;
the encoding is reused, so each prompt costs a grounding pass, not a re-encode.
Part boxes are typed differently in four ways: **containment suppression is ON**
(0.7) even though it ships off globally (a *part* nested in a subject is never a
second subject); **part crops skip mask-blanking** (the mask *is* the part, so
blanking handed the tagger a bare skin blob); **identity groups are suppressed**
(no head, no evidence); and **part boxes top up to the target and no further**,
since a part prompt fragments. The win is modest, and the residual failure — two
crops of the same kind of panel — is **not** addressed.

## What leaves the flat bag — the five move rules

A tag moves out of the bag into its clause when **all five** hold. Fail any one
and it stays flat *and* stays bound — an additive, less-resolved caption, which is
why nothing here can produce a wrong caption.

| # | Rule | Why |
|---|---|---|
| 1 | **Not a character name** | The cast list stays flat and is bound as well — the hand-written convention (every tag duplicated between bag and clause in the ground truth is a name; none is an attribute). The bag answers *who is in this image* and is how a prompt summons them; the clause answers *which one is where* |
| 2 | **Exactly one clause claims it** | Two clauses claiming a tag means it is shared, and a shared attribute belongs to the bag |
| 3 | **Corroboration**, for a character-invariant group | Hair color, eyes, body shape, species … belong to a *character*, not a view. On a `1girl, multiple views` sheet they hold in every panel, so moving `aqua hair` into one view would claim the others are *not* aqua-haired. Such a tag moves only when the bag names **≥2 values of that group** — i.e. the caption is already enumerating per-subject values |
| 4 | **Exclusive keep** | No *other* crop kept the tag. A crop that reached the tag's own calibrated threshold has the attribute, whatever the clause builder later did with it — kept twice but bound once is a selection artifact (clause budget, gate 1, gate 2), not an attribution. This is the tagger's own per-tag decision answering the question rule 5 can only approximate |
| 5 | **Relative attribution margin** (`--attribution_margin`, 0.25) | The runner-up's probability must fall below `(1 - margin)` of the winner's, so a tag the tagger *nearly* kept on the second subject stays in the bag |

Rule 3's exception: booru tags a **single** character with two hair colors when
the hair is two-toned, so `multicolored hair` / `two-tone hair` / `gradient hair`
/ `heterochromia` in the bag pin that group flat — the "≥2 values" evidence is
explained without a second subject. Those markers are ungrouped, so they are
matched by name. Rule 3 is evidence-based rather than count-based because most
proposals carry no girls-count tag at all, so a `detected == characters` gate
would pin nearly the whole corpus.

**Why the margin is relative.** Rules 4–5 replaced a single **absolute** gap test
(`winner - runner_up ≥ 0.35`), which asked a question the numbers cannot answer:
the tagger's boundaries are per-tag F1 thresholds spanning a wide range, so a
fixed gap is a different — and mostly impossible — test for every tag. The split
takes the *categorical* half from the tagger itself (rule 4, no tuning) and keeps
a *graded* guard (rule 5) scored as `1 - runner_up/winner`, which is scale-free.
`--attribution_margin 0.0` reduces to rule 4 alone; the absolute behaviour is not
recoverable by a flag. The report records `moved[{tag, position, margin}]` and
`pinned{tag: rule}` per image, and `summary.pinned_tags` aggregates the rules.

### Backing it out

The rewrite **moves** tags, never deletes them. `--flatten --apply` merges every
clause back into its flat bag and drops the clauses (`flatten_caption`) — text
only, no models. That is both the undo for an `--apply` run and the way to build
the clause-free control corpus for a training A/B. Tag *order* is not restored
byte-for-byte (a moved tag comes back at the end) and `correct_caption`
re-buckets it anyway. It flattens **hand-written** clauses too — it cannot tell
them apart — a real loss of curation on those 14 captions.

## Running it

```bash
make caption-position                                  # dry run, whole dataset
make caption-position ARGS="--crops --qwen3 models/text_encoders/qwen_3_06b_base.safetensors"
make caption-position ARGS="--path_pattern 'artist_a/*'"   # scope a slice
make caption-position ARGS="--apply"                   # write (after the review)
make preprocess-te                                     # REQUIRED after --apply
make caption-position ARGS="--flatten --apply"         # undo: clauses → flat bag
```

The `make` targets live in the trainer repo; the stage itself is `python -m
anime_tools.stages.cli.position_captions` with the same flags, which is what the
GUI dock runs. It is a GPU job (SAM3 + tagger held resident for the whole sweep),
so under the trainer it is **daemon-routed** — `--queue` detaches, `--inline`
bypasses.

**Dry run is the default and writes nothing.** It emits
`workspace/captions/position/report.json`:

```
summary: {applied, rewrite, src, dst, path_pattern, prompt, prompt_embed,
          prompt_embed_sha256, attribution_margin, seen, candidates, proposed,
          written, rewritten, moved_tags, max_novel_tags, clause_tags,
          novel_tags, reuse_ratio, pinned_tags{rule: n}, skipped{reason: n},
          part_prompts, part_recovered, max_tokens, over_token_budget[]}
images[]: {image, caption_path, status, detected, expected, original, proposed,
           tokens, instances[{position, box, score, tags, crop}],
           moved[{tag, position, margin}], pinned{tag: rule}}
```

`summary.src` / `dst` / `path_pattern` exist so `--from_report` can refuse to
replay a report against a different pair of trees. `--crops` exports the **exact
mask-blanked pixels the tagger saw**, mirroring the dataset layout as
`<stem>_<i>_<position>.png` — the only way to tell a detection miss from a tagging
miss; skipped rows record their `detections` and get a box overlay under
`crops/_skipped/`. `--qwen3 <tokenizer>` adds a token count per proposal and
flags anything past 512, past which the tail truncates **silently** at TE-cache
time, so check `summary.over_token_budget` before applying.

Two read-only review tools sheet the same pass (see the code map):
`ab_position_captions.py` proposes each image **twice** off one detect+tag pass;
`review_position_captions.py` sheets an already-applied run against disk.

### `--from_report` — apply a dry run without re-running the models

`images[].caption_path` is the destination and `images[].proposed` is the exact
text, so the apply needs no pixels at all:

```bash
make caption-position                                      # the model pass, once
make caption-position ARGS="--apply --from_report workspace/captions/position/report.json"
make preprocess-te                                         # still REQUIRED
```

**No model is loaded** on that second line — the run does not even import `torch`
(pinned by `tests/test_stage_replay.py`). Same flag on `caption-autotag` and
`audit-multiview`; the shared implementation is `anime_tools/stages/replay.py`.

| Situation | Result |
|---|---|
| Report's `summary.src`/`dst` ≠ this run's `--src`/`--dst`, or absent | **refused** (`SystemExit`) — the row paths are relative to those roots |
| Report's own `applied` is true | **refused** — its `original` describes the pre-apply world, so every row would read as drifted |
| Caption on disk ≠ the row's `original` | row skipped, `skip:drifted`, counted — **a hand edit between the passes is never overwritten** |
| Caption on disk already == `proposed` | row skipped, `skip:already-applied` — replays are idempotent, so a crashed one can be re-run |
| Caption file gone | row skipped, `skip:missing-caption` |
| Row status ≠ `proposed`, or `--path_pattern` excludes it | counted, not written (the pattern is matched as the live pass matches it) |

Without `--apply` it is a re-play dry run. The replay writes
**`apply_report.json`**, never `report.json` — pointing `--from_report` and
`--report_dir` at the same directory is the normal case, and clobbering the input
would make a re-run impossible. Its shape mirrors the stage's, plus `written[]`
(**the relative image paths actually written** — what a UI reads to reload the
affected items), `summary.from_report`, and `images[]` = `{image, caption_path,
before, after, status}`, one row per candidate that reached the on-disk check. It
carries `applied: true`, so feeding it back in is refused above; `--flatten
--from_report` is rejected because flatten is already text-only.

### Where the rewrite lands, and the one trap left in the ops sequence

The clauses go to the **derived** caption beside the resized image
(`workspace/resized/<rel>.txt`) — the file the caption mirror writes and the TE
step encodes. The master under `image_dataset/` is **never** written; it is only
the read fallback for an image the caption step has not mirrored yet. Three
things make that safe. **The mirror re-attaches them**:
`write_corrected_preprocess_captions` finds clauses on a destination caption
whose master has none and composes them back onto the freshly corrected bag,
minus the tags the rewrite moved, so each attribute stays asserted once and the
next mirror cannot write the clause-free master over the rewrite. **The write
invalidates the TE cache**, since `_cache_is_current` compares the cache mtime
against the caption and its sidecar. And **the stale variant sidecar is
dropped**, because `{stem}.variants.txt` is the encode source of truth when
present (the apply pass unlinks it; the caption step redraws it next run).

The trap that remains: **nothing re-encodes on its own.** After a standalone
`--apply` the caches are correctly stale but training keeps using them until an
explicit `make preprocess-te`, which chains the caption mirror so the sidecars
are regenerated first — and, because the clauses live in the resized tree, forces
that mirror even with correction and variants off.

### Reading the skip reasons

| Reason | What it means |
|---|---|
| `single-subject` | Not a candidate: one subject, no layout tag |
| `already-has-clauses` | Hand-written clauses — left alone |
| `too-few-instances` | Detection undershot even after the retry (and the part fallback, if enabled) |
| `count-mismatch` | Detection disagrees with the girls…girls+boys range, or busts the koma ceiling |
| `too-many-instances` | Past `--max_instances` |
| `no-discriminative-tags` | Every clause emptied — the subjects are indistinguishable to the tagger |

A mismatch is a **skip, not a wrong write**, which is the safe direction; lowering
the detection floor trades `too-few-instances` for `count-mismatch`.

**Two settled negatives — don't re-propose without new evidence.** *Box*
containment suppression ships off (`--containment_threshold 1.01`): it was
measured to break far more proposing rows than it recovers, because a real second
subject — one girl in front of another, an embrace — is exactly as nested as a
group box. Only the inset half is handled automatically, by `--min_area_frac`.
And there is **no automatic gate on fragmentary masks**: mask *fill*, row/column
*gap* and `main_frac` were all measured and none separates a broken mask from a
visually clean crop, so use the per-detection `score` in the report to pick what
to eyeball.

### Mask containment

`--mask_containment_threshold 0.8` suppresses a detection whose **mask** is that
nested inside a kept detection's mask. It is **on** by default for the reason the
box rule is off: two boxes nest identically whether the inner detection is a
fragment of the outer figure or a second girl in front of her, but their masks do
not — a fragment's is a subset, an occluding subject's is disjoint, because SAM3
segments the two separately. Measured nested pairs land near 1.0 (one object) or
near 0.0 (two subjects), so 0.8 is not a tuned edge; `>1.0` disables. **Known
failure mode**: when SAM3 emits one mask spanning *both* girls the individual's
is a subset of it and gets suppressed — both observed regressions are that shape
(2 → 1 → `too-few-instances`), unmitigated because any guard would be tuned on
n=2, and both are skips rather than wrong writes. A pair with no usable mask
falls back to the box rules, so `merge_part_detections` is unaffected.

## Knobs

`ARGS="…"` on the make target, or straight onto the module; every flag has a
`--kebab-case` alias.

| Flag | Default | What it does |
|---|---|---|
| `--apply` | off | Write to the resized captions (else dry run) |
| `--from_report` | — | Replay a dry run's `report.json` instead of re-running SAM3 + the tagger. Writes `apply_report.json`; skips any caption changed since |
| `--path_pattern` | `*` | fnmatch glob (`\|` to OR) relative to the resized dir |
| `--crops` | off | Export the mask-blanked crops next to the report |
| `--prompt_embed` | the shipped `networks/calibration/sam3_girl_prompt.safetensors` | Learned SAM3 soft prompt for the **subject** pass (part prompts stay textual). `none` falls back to `--prompt`; a missing default warns and falls back |
| `--prompt` | `girl` | SAM3 text prompt, used when `--prompt_embed none` (`person` sweeps the rare on-screen-boy images) |
| `--score_threshold` / `--retry_score_threshold` | 0.5 / 0.35 | Detection floor; retry floor when the count undershoots. These are SAM3's **own** confidence floor, not a post-filter |
| `--iou_threshold` / `--pad` | 0.65 / 0.06 | Dedupe IoU; bbox padding fraction |
| `--containment_threshold` | 1.01 (off) | Suppress a box this nested inside a kept one (intersection over the *smaller* box). Measured harmful on this corpus — see above before enabling |
| `--mask_containment_threshold` | 0.8 (**on**) | The same rule on the *masks*, which is what separates a fragment from a second subject. `>1.0` disables |
| `--dedupe_fill_ratio` | 2.0 | Mask-quality tie-break inside an NMS-matched pair; `0` = score-only survivor |
| `--min_area_frac` | 0.005 | Drop detections below this fraction of the image (insets are not subjects) |
| `--no_blank_crops` | — | Skip mask-blanking (diagnostic only — it is what causes cross-subject hair bleed) |
| `--row_tol` | 0.25 | Minimum fractional overlap (of the narrower extent) for two subjects to share a row — and a column, on magazine layouts |
| `--part_prompts` | off | Comma-separated body-part prompts, tried **only** when the subject prompt undershoots — recovers headless close-up panels. Try `buttocks,hips,thighs` |
| `--part_score_threshold` / `--part_containment_threshold` | 0.5 / 0.7 | Part-box confidence floor; drop a part box this nested inside an already-kept box |
| `--min_instances` / `--max_instances` | 2 / 8 | Instance-count window |
| `--no_strict_count` | — | Propose even when detection disagrees with the girls-count |
| `--max_clause_tags` | 8 | Cap per clause |
| `--max_novel_tags` | 1 | How many tags a clause may introduce that the caption never contained — the bag fills it first. `0` never invents; `--max_clause_tags` is the bag-blind A/B arm |
| `--name_confidence` / `--allow_unlisted_names` | 0.5 / off | Character-name floors |
| `--keep_shared_tags` | — | Keep tags every crop agrees on (disables gate 1) |
| `--ungated_identity` | — | Let a clause carry a value the caption never listed for a gated group (disables gate 3) |
| `--bind_view_traits` | — | On a repeated-subject layout, let a clause carry the character's name and view-invariant traits (disables gate 2) |
| `--gate_view_anatomy` | — | On a repeated-subject layout, keep `body_parts` out of every clause (`ass`, `navel`, `breasts`). The A side of the anatomy A/B |
| `--no_framing` | — | Keep `framing` out of every clause, so no clause says whether its view is a close-up or a whole figure. The A side of the framing A/B |
| `--no_rewrite` | — | Additive v1: append the clauses, leave the flat bag untouched (every bound attribute asserted twice). The A/B control arm |
| `--attribution_margin` | 0.25 | How far the winning crop must clear every other **relative to its own probability** (`1 - runner_up/winner`) before a tag may **leave** the bag, on top of the hard rule that no other crop kept it. `0.0` trusts the per-tag thresholds alone. The clause carries the tag either way |
| `--bag_relax` | 0.35 | Multiplier on the tagger's per-tag keep threshold for tags the flat bag already contains — such a tag can only *move*, never be invented, so the caption corroborates it and the crop only has to attribute. Applied to every crop before the attributable/shared census, so a rival crop's borderline residual also blocks a bind. `1.0` = off. It is what recovers pose tags (`lying`, `on back`) whose scores collapse when blanking removes the scene context |
| `--bag_word_relax` | 0.85 | Extra threshold multiplier per word beyond the first, compounding with `--bag_relax` — `black panties` is more specific than `panties`, so a sub-threshold hit is less likely noise. `1.0` = off. It relaxes rival floors too: two-word generics (`high heels`) become easier to false-share |
| `--bag_relax_min_score` | 0.3 | Absolute floor under the relaxation: a relaxed admission still needs this raw probability however far the multipliers drag the threshold, which blocks near-noise fires. **This, not `--bag_relax`, is the binding constraint at the shipped operating point** — move it first. `0.0` = off |
| `--flatten` | off | Inverse pass — merge clauses back into the bag and drop them. Text only (no models). The undo, and the clause-free A/B corpus |
| `--qwen3` / `--max_tokens` | — / 512 | Token-budget column + over-budget flag |

## How clauses behave downstream

- **Caption variants** (`anime_tools/captions/variants.py`) treat **each clause as
  an atomic unit**: kept or dropped whole at `clause_dropout_rate` (defaults to
  `tag_dropout_rate`), tags shuffled inside, header never randomized — per-tag
  dropout inside a clause would leave a half-described position. Clause-free
  captions keep the historical raw split, so v0 stays byte-identical. Because a
  bound attribute is *only* in its clause, dropping a clause drops that subject's
  attributes entirely: truthful, but a stronger perturbation than the same rate
  per-tag, so `clause_dropout_rate = 0.0` is the conservative setting on a
  rewritten corpus.
- **Order correction** (`correct_caption`) splits clauses off before
  bucket-reordering the flat bag: clause tags are position-scoped and already
  ordered left→right, so reordering them across the caption is exactly the
  shredding the grammar fix removed.
- **Auto-tagging** (`caption-autotag --mode merge`) round-trips clauses verbatim
  and counts their bound tags as present, so a merge after `caption-position`
  cannot re-flatten a binding back into the bag.
- **Training** sees no new machinery: the clauses ride the ordinary TE path.

## Turning it on in the trainer's preprocess chain

Off by default in all four surfaces; each runs the stage **with `--apply`** (no
dry run) inline in `make preprocess`, after the VAE cache and before the
caption/TE steps — the same job re-encodes, so the staleness trap is handled for
you. Only the standalone `--apply` path needs a manual `make preprocess-te`.

| Surface | How |
|---|---|
| Config | `caption_position_clauses = true` in `configs/preprocess.toml` (user-owned) |
| CLI | `make preprocess ARGS="--caption_position_clauses"` / `--no_caption_position_clauses` |
| Env | `CAPTION_POSITION_CLAUSES=1` |
| GUI | Preprocessing tab → **캡션 편집 / Caption rewriting** → `위치 절 생성 (다중 인물)` |

Precedence is env → config, with the CLI flag winning over both; the GUI always
exports the env var, so its checkbox **initializes from the config key** and a
CLI-side `true` cannot be silently cancelled by a GUI run exporting `0`. Applying
without a review step rewrites the derived captions in place and there is no undo
button in the GUI — nothing of yours is at risk, since the master is untouched,
but the rewritten text is what trains until you look at it. The pass is
idempotent and reversible from the CLI (`--flatten --apply`), yet a dry run with
`--crops` is still the way to eyeball proposals first.

## Limits / open

- **Hair *length* across crops** — `long hair` vs `medium hair` on two views of one
  character is crop-scale dependent. Gates 1–2 mask most of it, but a scale
  artifact on a real multi-character image will bind.
- **Character names on crops** are the weakest signal, hence the flat-bag floor.
- **Boys / POV** are out of the default sweep — `--prompt_embed none --prompt
  person` sweeps them separately; nothing is hardcoded to `girl`.
- **Under-detection has an irreducible tail.** SAM3 scales every instance
  probability by one global presence score, so on some framings (extreme
  close-up, from-behind, cropped body) *all* boxes sink together.
- **The retry escalation is ungated downstream.** `detect_subjects` targets
  `expected or min_instances`, so a `multiple views` caption (`expected=None`)
  always retries down to `--retry_score_threshold`, and nothing re-checks that the
  survivors cleared the shipped floor — an image whose every detection is a
  sub-threshold fragment can still clear the window and write ungrounded clauses.
- **Deprecated aliases can never bind.** A handful of non-artist tags, several of
  them hair (`silver hair`, `light brown hair`, `light blue hair`, `dark blue
  hair`, `light purple hair`, `french braid`), carry a calibrated threshold above
  1.0 ("never emit") on the dbv4 tagger, so that binding is silently lost.
- **Bag-removal tolerance is the open risk.** The probe validated clause
  *comprehension* (48/48 sides correct) — not that removing a tag from the flat
  bag is safe for a model pretrained on flat bags. **The training A/B (clause
  corpus vs the flattened control) is still owed.**
- **Is the margin in the right place?** The relative one at 0.25 is calibrated
  against one artist slice, not the corpus. The report carries the per-move
  margin on the knob's own scale — retune against a full-corpus spot-check.
- **`sole-value` on non-identity invariants.** `body_shape` / `skin` /
  `face_features` are in the invariant set, so a `2girls` caption naming one
  `large breasts` keeps it flat even when only one girl has it — safe, and the
  class most likely to be over-pinned.

## Code map

| Path | Role |
|---|---|
| `anime_tools/captions/position_clauses.py` | Clause grammar (torch-free) — parse / compose / `flatten_caption` / position vocabulary |
| `anime_tools/captions/data/clause_vocabulary.yaml` | **The clause policy as data** — every group set above (`subject_groups`, `page_level_framing`, `priority_groups`, the gates, `multi_value_markers`) with its rationale inline. Edit here, not in Python |
| `anime_tools/captions/clause_vocabulary.py` | Loads that YAML into `ClauseGroups`; `ClauseVocabulary` = which tags may enter a clause, in what order (`select`). Warns when the policy names a group the checkpoint's `groups.yaml` doesn't declare — a typo would otherwise silently disable a gate |
| `anime_tools/captions/clause_rewrite.py` | The move rules — `plan_bag_removals` + the `RemovalPlan` block reasons |
| `anime_tools/captions/caption_layout.py` | Text-only prefilter — subject/boy counts, `Nkoma` ceiling, layout tags, `is_candidate` |
| `anime_tools/stages/instance_detection.py` | `Detection`, box geometry, NMS + part merge, mask-blanked `crop_instance`, soft-prompt resolution (`resolve_prompt_embed` / `prompt_embed_sha256`) |
| `anime_tools/stages/position_captions.py` | Pipeline orchestration (`propose_for_image`, `flatten_captions`); models injected as `detect_fn` / `tag_fn` |
| `anime_tools/stages/cli/position_captions.py` | CLI shell — argparse + SAM3/tagger loading (`build_options_from_args` is shared with the A/B tool) |
| `anime_tools/stages/cli/ab_position_captions.py` | A/B two flag sets off **one** detect+tag pass. Pass sides as `--a_flags=--foo` — the `=` is required, argparse reads a `-`-leading value as the next option |
| `anime_tools/stages/cli/review_position_captions.py` | Contact sheet for an **applied** run — overlay, crops, master vs derived caption, moved / novel / duplicated marks, `drift` flag |
| `anime_tools/downloads.py` | `DEFAULT_SUBJECT_PROMPT_EMBED` and the `soft_prompt` asset row that fetches it |
| `anime_tools/captions/variants.py` / `correction.py` | Atomic-clause variant generation; clause-aware order correction |
| `tests/test_position_captions.py` | Unit tests (grammar round-trip, ordering, selection, skip paths, the rewrite rules) |

The soft prompt's own training write-up, the dbv4 tagger backend notes and the
label-sharing head experiments belonged to the trainer repo and did not come
across in the split; local copies, where they exist, are under `_archive/docs/`.
