---
name: captions
description: Caption pipeline — position-clause grammar (never hand-split a caption), text
clauses and quoted lines, autotag modes, position-clause v2 rewrite rules and gates, tag-group
drops, and how the trainer wraps these stages. Load before parsing/editing captions or caption
code, running autotag / position / correct, or touching anime_tools/captions/ or stages/.
---

# Caption pipeline: grammar, autotag, position clauses

## Position-clause grammar (`On the left, …`)

A caption may bind attributes to subjects with trailing clauses: `<flat tag bag>.
On the left, akita neru, yellow eyes. On the right, kasane teto.` The period delimits clauses,
commas separate tags inside one — so a plain `caption.split(",")` glues the header onto the
previous tag (`"white socks. On the left"`) and any `startswith("On the ")` check silently sees no
clauses.

Never hand-split a caption: `anime_tools/captions/position_clauses.py` (torch-free) is the
single grammar — `parse_caption` → `ParsedCaption(flat_tags, clauses)`, `compose_caption` back.
Caption variants treat each clause as an atomic unit (dropped whole at `clause_dropout_rate`,
shuffled inside, header never randomized); `correct_caption` splits clauses off before
bucket-reordering the flat bag.

Two content rules ride the same parser (both 2026-09-05, for the anima_lora CJK DiT line):
a quoted line — `「…」`, `『…』` or `"…"` (`QUOTE_PAIRS`) — is opaque, so a comma or `. On the`
inside a closed pair is content, not a separator; and a text clause — `Japanese text reads as
"…", "…"` / `Japanese SFX reads as "…"` (`TEXT_PREFIXES`) — is a clause kind of its own: it parses
to a `PositionClause` with an empty `position` and the quoted lines as tags (`is_text`, build one
with `text_clause(lines)`), `compose_caption` always renders it last, after every position
clause, and variants / `correct_caption` / `flatten_caption` pass it through verbatim (reading
order is content). `has_clauses` stays position-only (a text sentence binds no subject, so it
must not read as "already rewritten"); `has_text_clauses` is the other question.
The producer of a text clause is Export's `--combine_ocr` (`ocr_sidecar.with_ocr_clause`):
the OCR
stage writes only `workspace/ocr/**/{stem}.ocr.txt`, and the combine attaches those lines to the
published caption and every `.variants.txt` line at export time — the workspace caption never
carries it, an export without the knob takes it back, and `make preprocess-te` must follow either.

## Dropping tag groups (`--caption_drop_groups`, GH #95)

`python -m anime_tools.stages.cli.correct_captions --caption_drop_groups artist,lighting,pose`
(`CorrectRequest.caption_drop_groups`; the trainer spells it `CAPTION_DROP_GROUPS` /
`caption_drop_groups` in its `configs/preprocess.toml`) strips whole kinds of tag from every
revised caption — the master under `image_dataset/` is never edited.
Slug table + resolution order in `anime_tools/captions/tag_drop_groups.py`: tag shape (`@`→artist,
count, rating) → danbooru numeric kind → the KB's `[대분류 > 소분류]` path; anything not a slug is a
literal path prefix (`"효과/연출 > 조명"`). Unknown-to-KB tags, ratings, the trigger word and `@no-artist`
never drop; `insert_no_artist` still fires after an `artist` drop (that's the point for style
LoRAs). Applies inside position clauses too (an emptied clause is removed whole).
Setting it alone is enough to enable the correction pass. Note it is KB-faithful, so `thighhighs` is
`accessory`, not `clothing`. CPU-only — no GPU involved.

## Auto-tagging (`python -m anime_tools.stages.cli.autotag_captions`)

Batch Anima Tagger over the dataset, writing `.txt` sidecars into the revised tree
(`workspace/resized/`). `anime_tools/tagger/cli/autotag.py` is single-image + stdout-only and is
not a batch path. Orchestration in `anime_tools/stages/autotag.py`, request `AutotagRequest`,
thin CLI at `anime_tools/stages/cli/autotag_captions.py`; the trainer's `make caption-autotag`
wraps it. Tags the resized image (the pixels training sees)
and writes beside it; the hand-written master is read (`resolve_caption`'s fallback) and never
written. What a write replaces is pushed onto `{stem}.history.txt` under `by=autotag`, so no mode
loses text outright.

Three `--mode`s:

- `missing` (default) — only images no caption speaks for, revised or master.
- `merge` — append only tags the caption lacks. Position clauses round-trip verbatim and their
  bound tags count as present, so a merge after `caption-position` can't re-flatten one back into
  the bag, and a second rating is dropped. The revised-first read is what makes this hold: the
  clauses live there.
- `overwrite` — replace outright; the replaced text stays as a history version.

`--min_confidence` is an extra floor on top of the tagger's per-tag F1 thresholds (0 = leave its
calibrated decisions alone; the rating slot ignores it). Dry-run by default (`report.json` with
before/after per image); `--apply` writes (the GUI always passes it) and must be followed by
the trainer's `make preprocess-te`.

## Position-clause generation (`python -m anime_tools.stages.cli.position_captions`)

SAM3 `girl` instances → reading order (row-aware, so 2×2 view sheets get `top left`/`bottom right`)
→ mask-blanked crops → Anima Tagger → the revised caption rewritten
(`workspace/resized/<rel>.txt` — the file Export publishes and the trainer's TE encodes;
the hand-written master under `image_dataset/` is never written, only read as the fallback for a
not-yet-mirrored image). Request `PositionRequest`, orchestration in
`anime_tools/stages/position_captions.py`; the trainer's `make caption-position` wraps it.

v2 (the default) *moves* a bound tag out of the flat bag into its clause so each attribute is
asserted exactly once — the hand-written convention. `--no_rewrite` is the additive v1 arm;
`--flatten` merges clauses back into the bag (text-only undo / clause-free A/B corpus).

### Move rules and gates

Five rules bound a move (fail any one and the tag stays flat and bound, i.e.
v1 for it — the rules can only under-resolve a caption, never make it wrong):

1. Not a character name.
2. Claimed by exactly one clause.
3. Corroborated for character-invariant groups.
4. Kept by no other crop.
5. Clears `--attribution_margin` relative to the winner's own probability.

Four gates run before the rewrite, on what may enter a clause at all:

- Eligibility from the tagger's `groups.yaml` — per-subject groups bind, scene groups don't;
  copyright/artist/metadata/deprecated filtered on every emission path.
- Only what discriminates — a tag every crop keeps stays in the bag.
- On a repeated-subject layout (`LAYOUT_TAGS` = `multiple views` + comic pages) the clause drops
  the character's name and every view-invariant trait (`--bind_view_traits` reverts).
  `body_parts` is NOT in that set (since 2026-08-19; `--gate_view_anatomy` restores it) —
  a clause asserts anatomy that is visible in this panel, so a from-behind view takes `ass`/`back`
  and its front sibling `breasts`. Residual risk is a sibling crop that merely missed the anatomy;
  `discriminative_only` + `--attribution_margin` are the only guards.
- On a gated group the flat bag outranks the crop tagger — the set is derived:
  identity trio + every exclusive subject group (`--ungated_identity` reverts), minus `framing`.
- `framing` binds (`On the left, ass focus, underwear, …`) — the one `subject_groups` member
  describing the view, not the girl, so a headless close-up panel says so.
  Three couplings, all load-bearing: exempt from the bag gate (else the bag's `full body` for
  another panel pins every clause and the feature is inert);
  `solo focus`/`size difference`/`white border` blocked in `add()` (page-level, and v2 would move
  them out of the bag); and it's in `priority_groups` so a novel framing tag wins the
  `--max_novel_tags` slot. `torso only`/`cropped torso` are NOT in the tagger vocabulary —
  don't try to wire them. `--no_framing` is the A side.

Every group set these gates read is data, in
`anime_tools/captions/data/clause_vocabulary.yaml` (loaded into `ClauseGroups` by
`anime_tools/captions/clause_vocabulary.py`, rationale inline, validated against the checkpoint's
`groups.yaml` at load — an undeclared name is warned about, since it would silently disable its
rule; the trainer's `configs/clause_vocabulary.yaml` overrides the packaged one). Retune a gate
there, not in Python; `load_clause_groups(path)` → `load_clause_vocabulary(ckpt, clause_groups=…)`
runs an alternative set.

Comparing two rule sets: `python -m anime_tools.stages.cli.ab_position_captions --path_pattern
'<glob>'` proposes each image twice off **one** detect+tag pass (`--a_flags`/`--b_flags` take any
position_captions flag) and writes contact sheets + `index.html` to
`workspace/reports/position_ab/` (`--out`), only for images where the two differ. It reads the
master caption only, on purpose.
When reading the diff, check whether a displaced tag was in the master caption or a crop
invention — on the framing A/B all 54 displaced tags were inventions, which flips the verdict.

`--max_novel_tags` (1) admits candidates bag-first, because only a bag tag can move —
a novel one is a pure v1-style addition. Layout tags also decouple the girls-count from the
bindable-subject count (a `1girl, 2koma` page is two subjects; `Nkoma` restores a
`panels × (girls+boys)` ceiling; `page number` is not a layout tag).
Opt-in `--part_prompts buttocks,hips,thighs` adds a body-part detection fallback (only when the
`girl` prompt undershoots) for sheets built from headless close-up panels;
part boxes skip mask-blanking and carry no identity tags.

### Apply & re-encode

Dry-run by default from the CLI (`report.json` + `--crops`); `--apply` writes and must be
followed by the trainer's `make preprocess-te` — nothing re-encodes on its own (the write does
bump the caption mtime, so the cache is correctly stale, and the apply pass unlinks the now-stale
`.variants.txt` sidecar, which would otherwise override `{stem}.txt` at encode time). The GUI
needs no Apply gate: the replaced text is a history version, and Undo replays the report
backwards through `replay.apply_one`.

The correction pass (`stages/captions.py::write_corrected_preprocess_captions`, `CorrectRequest`)
reads the revised caption first and corrects it in place — the flat bag is reordered around
its clauses — reading the master only for an image with no revised caption yet, so a later
correction run cannot mirror the clause-free master over the rewrite (nor drop the tags autotag
merged). Once a revised caption exists a master edit no longer reaches it: edit the revised one,
or delete it to re-mirror.

### In the trainer

The trainer wraps all three as preprocess stages (`caption_autotag`, off by default, first in
the chain right after resize because it creates the captions every later stage reads;
`caption_position_clauses`, off by default, after the VAE cache and before the caption/TE steps;
correction always). Each is a `configs/preprocess.toml` key, an env var and a CLI flag over the
same `Request`, run in-process with `--apply`; the chain re-encodes TE itself, so no separate
`preprocess-te` there. That wiring, its precedence rules and its tests live in the trainer repo.

## References

The rules, gates and knob table live in `docs/position_captions.md` — read it before retuning a
rule. The module map for `anime_tools/captions/` is `anime_tools/captions/CLAUDE.md`.
