---
name: captions
description: Caption pipeline — position-clause grammar (never hand-split a caption),
make caption-autotag modes, make caption-position (v2 rewrite rules and gates),
and the preprocess-stage wiring for both. Load before parsing/editing captions or caption code,
running either target, or touching the caption preprocess stages.
---

# Caption pipeline: grammar, autotag, position clauses

## Position-clause grammar (`On the left, …`)

A caption may bind attributes to subjects with trailing clauses: `<flat tag bag>.
On the left, akita neru, yellow eyes. On the right, kasane teto.` The **period** delimits clauses,
commas separate tags *inside* one — so a plain `caption.split(",")` glues the header onto the
previous tag (`"white socks. On the left"`) and any `startswith("On the ")` check silently sees no
clauses.

**Never hand-split a caption**: `anime_tools/captions/position_clauses.py` (torch-free) is the
single grammar — `parse_caption` → `ParsedCaption(flat_tags, clauses)`, `compose_caption` back.
Caption variants treat each clause as an **atomic unit** (dropped whole at `clause_dropout_rate`,
shuffled inside, header never randomized); `correct_caption` splits clauses off before
bucket-reordering the flat bag.

Two content rules ride the same parser (both 2026-09-05, for the anima_lora CJK DiT line):
a **quoted line** — `「…」`, `『…』` or `"…"` (`QUOTE_PAIRS`) — is opaque, so a comma or `. On the`
inside a closed pair is content, not a separator; and a **text clause** — `Japanese text reads as
"…", "…"` / `Japanese SFX reads as "…"` (`TEXT_PREFIXES`) — is a clause kind of its own: it parses
to a `PositionClause` with an empty `position` and the quoted lines as tags (`is_text`, build one
with `text_clause(lines)`), `compose_caption` always renders it **last**, after every position
clause, and variants / `correct_caption` / `flatten_caption` pass it through verbatim (reading
order is content). `has_clauses` stays *position*-only (a text sentence binds no subject, so it
must not read as "already rewritten"); `has_text_clauses` is the other question.

## Dropping tag groups (`--caption_drop_groups`, GH #95)

`make preprocess-captions ARGS="--caption_drop_groups artist,lighting,pose"` (or
`CAPTION_DROP_GROUPS=…` / `caption_drop_groups` in `configs/preprocess.toml`) strips whole *kinds*
of tag from every **mirrored** caption — the master under `image_dataset/` is never edited.
Slug table + resolution order in `anime_tools/captions/tag_drop_groups.py`: tag shape (`@`→artist,
count, rating) → danbooru numeric kind → the KB's `[대분류 > 소분류]` path; anything not a slug is a
literal path prefix (`"효과/연출 > 조명"`). Unknown-to-KB tags, ratings, the trigger word and `@no-artist`
never drop; `insert_no_artist` still fires after an `artist` drop (that's the point for style
LoRAs). Applies inside position clauses too (an emptied clause is removed whole).
Setting it alone is enough to enable the correction pass. Note it is KB-faithful, so `thighhighs` is
`accessory`, not `clothing`. CPU-only — no GPU involved.

## Auto-tagging (`make caption-autotag`)

Batch Anima Tagger over the dataset, writing `.txt` sidecars into the **revised** tree
(`workspace/resized/`) — the dataset-wide counterpart to the Dataset tab's per-image autotag button
(`anime_tools/tagger/cli/autotag.py` is single-image + stdout-only and is *not* a batch path).
Orchestration in `anime_tools/stages/autotag.py`, thin CLI at
`anime_tools/stages/cli/autotag_captions.py`. Tags the **resized** image (the pixels training sees)
and writes beside it; the hand-written master is read (`resolve_caption`'s fallback) and never
written. What a write replaces is pushed onto `{stem}.history.txt` under `by=autotag`, so no mode
loses text outright.

Three `--mode`s:

- `missing` (default) — only images no caption speaks for, revised or master.
- `merge` — append only tags the caption lacks. **Position clauses round-trip verbatim and their
  bound tags count as present**, so a merge after `caption-position` can't re-flatten one back into
  the bag, and a second rating is dropped. The revised-first read is what makes this hold: the
  clauses live there.
- `overwrite` — replace outright; the replaced text stays as a history version.

`--min_confidence` is an extra floor on top of the tagger's per-tag F1 thresholds (0 = leave its
calibrated decisions alone; the rating slot ignores it). Dry-run by default (`report.json` with
before/after per image); `ARGS="--apply"` writes and **must** be followed by `make preprocess-te`.

Also a **preprocess stage**: `caption_autotag` (GUI Preprocessing tab → 자동 태깅 box, off by default;
env `CAPTION_AUTOTAG` / `CAPTION_AUTOTAG_MODE` / `CAPTION_AUTOTAG_MIN_CONFIDENCE`,
CLI `--caption_autotag[_mode|_min_confidence]`) runs inline **right after resize** with `--apply` —
first in the chain because it *creates* the captions every later caption stage (position clauses →
correction → TE) reads. Chain order is pinned by a test in `tests/test_preprocess_tasks.py`.

## Position-clause generation (`make caption-position`)

SAM3 `girl` instances → reading order (row-aware, so 2×2 view sheets get `top left`/`bottom right`)
→ mask-blanked crops → Anima Tagger → the **revised** caption rewritten
(`post_image_dataset/resized/<rel>.txt` — the file the mirror writes and TE encodes;
the hand-written master under `image_dataset/` is never written, only read as the fallback for a
not-yet-mirrored image).

**v2 (the default) *moves* a bound tag out of the flat bag into its clause** so each attribute is
asserted exactly once — the hand-written convention. `--no_rewrite` is the additive v1 arm;
`--flatten` merges clauses back into the bag (text-only undo / clause-free A/B corpus).

### Move rules and gates

Five rules bound a move (fail any one and the tag stays flat *and* bound, i.e.
v1 for it — the rules can only under-resolve a caption, never make it wrong):

1. Not a character name.
2. Claimed by exactly one clause.
3. Corroborated for character-invariant groups.
4. **Kept by no other crop.**
5. Clears `--attribution_margin` **relative** to the winner's own probability.

Four gates run **before** the rewrite, on what may enter a clause at all:

- Eligibility from the tagger's `groups.yaml` — per-subject groups bind, scene groups don't;
  copyright/artist/metadata/deprecated filtered on *every* emission path.
- **Only what discriminates** — a tag every crop keeps stays in the bag.
- On a **repeated-subject layout** (`LAYOUT_TAGS` = `multiple views` + comic pages) the clause drops
  the character's name and every view-invariant trait (`--bind_view_traits` reverts).
  **`body_parts` is NOT in that set** (since 2026-08-19; `--gate_view_anatomy` restores it) —
  a clause asserts anatomy that is *visible in this panel*, so a from-behind view takes `ass`/`back`
  and its front sibling `breasts`. Residual risk is a sibling crop that merely *missed* the anatomy;
  `discriminative_only` + `--attribution_margin` are the only guards.
- On a gated group the **flat bag outranks the crop tagger** — the set is *derived*:
  identity trio + every exclusive subject group (`--ungated_identity` reverts), **minus `framing`**.
- **`framing` binds** (`On the left, ass focus, underwear, …`) — the one `subject_groups` member
  describing the *view*, not the girl, so a headless close-up panel says so.
  Three couplings, all load-bearing: exempt from the bag gate (else the bag's `full body` for
  another panel pins every clause and the feature is inert);
  `solo focus`/`size difference`/`white border` blocked in `add()` (page-level, and v2 would *move*
  them out of the bag); and it's in `priority_groups` so a novel framing tag wins the
  `--max_novel_tags` slot. `torso only`/`cropped torso` are NOT in the tagger vocabulary —
  don't try to wire them. `--no_framing` is the A side.

Every group set these gates read is **data**, in `configs/clause_vocabulary.yaml` (loaded into
`ClauseGroups` by `anime_tools/captions/clause_vocabulary.py`, rationale inline, validated against
the checkpoint's `groups.yaml` at load — an undeclared name is warned about, since it would silently
disable its rule). Retune a gate there, not in Python; `load_clause_groups(path)` →
`load_clause_vocabulary(ckpt, clause_groups=…)` runs an alternative set.

Comparing two rule sets: `make daemon-run ARGS="anime_tools/stages/cli/ab_position_captions.py
--path_pattern '<glob>'"` proposes each image twice off **one** detect+tag pass
(`--a_flags`/`--b_flags` take any position_captions flag) and writes contact sheets + `index.html`
to `post_image_dataset/captions/position_ab/`, only for images where the two differ.
When reading the diff, check whether a displaced tag was in the **master** caption or a crop
invention — on the framing A/B all 54 displaced tags were inventions, which flips the verdict.

`--max_novel_tags` (1) admits candidates bag-first, because only a bag tag can *move* —
a novel one is a pure v1-style addition. Layout tags also decouple the girls-count from the
bindable-subject count (a `1girl, 2koma` page is two subjects; `Nkoma` restores a
`panels × (girls+boys)` ceiling; `page number` is **not** a layout tag).
Opt-in `--part_prompts buttocks,hips,thighs` adds a body-part detection fallback (only when the
`girl` prompt undershoots) for sheets built from headless close-up panels;
part boxes skip mask-blanking and carry no identity tags.

### Apply & re-encode

Dry-run by default (`report.json` + `--crops`); `ARGS="--apply"` writes and **must** be followed by
`make preprocess-te` — nothing re-encodes on its own (the write does bump the caption mtime,
so the cache is correctly stale, and the apply pass unlinks the now-stale `.variants.txt` sidecar,
which would otherwise override `{stem}.txt` at encode time).

The mirror (`write_corrected_preprocess_captions`) **reads the revised caption first** and corrects
it in place — the flat bag is reordered around its clauses — reading the master only for an image
with no revised caption yet, so a later `preprocess-captions` cannot mirror the clause-free master
over the rewrite (nor drop the tags autotag merged). Once a revised caption exists a master edit no
longer reaches it: edit the revised one, or delete it to re-mirror.

### Preprocess-stage wiring

Also wired as a **preprocess stage**: `caption_position_clauses` (off by default in all four
surfaces — `configs/preprocess.toml` key, env `CAPTION_POSITION_CLAUSES`,
CLI `--caption_position_clauses`, GUI Preprocessing tab → 캡션 편집 box; CLI flag > env > config,
and the GUI always exports the env var — so its checkbox **initializes from the config key** and an
unchecked box persists as `false`, else a GUI run would export `0` over a CLI opt-in).
With it on, `tasks.py preprocess` runs it inline with `--apply` after the VAE cache and before the
caption/TE steps — no separate `preprocess-te` needed there, since the chain re-encodes anyway.

**`make preprocess-captions` / `preprocess-te` honour the same knob** (both write-or-read the
resized caption, so the stage runs before they touch it — the mirror then re-corrects around the
fresh clauses); with the knob on, `preprocess-te` also **forces the mirror** even with correction
and variants off, since TE must read `resized/` and the images the rewrite didn't touch still need
their master caption mirrored there. The "already ran" mark rides on the shared caption-config dict,
so the full chain still pays the SAM3 load exactly once. Same wiring for `caption_autotag`.

## References

The rules, gates and knob table live in `docs/position_captions.md` — read it before retuning a
rule.
