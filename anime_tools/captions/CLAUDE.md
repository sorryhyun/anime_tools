# anime_tools/captions/

Torch-free. The single caption grammar and everything that reads or writes a caption as text.
The grammar itself — clause kinds, quoted lines, the position-clause move rules and gates — is in
the `captions` skill (`.claude/skills/captions/`); **load it before parsing or editing captions
or touching this package**. This file is the module map. Docs: `docs/position_captions.md`.

## Module map

- `position_clauses.py` — the one parser: `<flat tag bag>. On the left, …. In the …, ….` Periods
  delimit clauses, commas delimit tags inside one. Never `split(",")` a caption; go through
  `parse_caption` / `compose_caption`, and `tag_spans` for `[start, end)` offsets (what the GUI
  editor paints boxes from).
- `taxonomy.py` is the one tag-shape vocabulary (pure stdlib): `normalize_tag` is the key every
  "does the caption already say this?" comparison uses, so two danbooru spellings of a tag can never
  read as two tags. `tests/test_tag_taxonomy.py` greps `stages/` for the bare `.lower()` that should
  be this call. Also `is_count_tag`/`count_of`/`exact_count` (the one count regex) and
  `SINGLE_COUNT_NAMES`/`is_solo_names`/`solo_multi_indices` (the `softmax_when_solo` predicate).
- `vocab_io.py` is the one reader of a checkpoint's `vocab.json`.
- `_sidecar.py` is the one tab-delimited sidecar format (`sidecar_header` / `sidecar_path` /
  `read_rows` / `write_rows`) — the multi-dot-stem rule (`with_name`, not `with_suffix`) and
  hand-edit tolerance (blank, `#`, wrong-arity lines skipped) live there. Three sidecars sit on it:
  `variants.py` (`.variants.txt`, `v0` = pristine), `history.py` (`.history.txt`, capped at
  `HISTORY_LIMIT`, sequences never renumbered), `ocr_sidecar.py` (`.ocr.txt`). OCR is not a caption
  — it names words in the picture, so no caption stage reads it and the workspace caption never
  carries it. It reaches the trainer only through Export's `--combine_ocr`
  (`ocr_sidecar.with_ocr_clause`: the lines attached as the trailing `Japanese text reads as
  "…", "…"` clause on the published caption and every variant line; an existing text clause is
  replaced, no lines removes it). Two floors say which lines get there — `DEFAULT_MIN_DET` on the
  detector's box confidence and `DEFAULT_MIN_GLYPH` on `OcrLine.glyph_px` (`sqrt(w*h/len(text))`,
  the em of the line: a box too small for the glyphs read from it, or text too fine to render) —
  and each kind is deduplicated on its own key (`ocr_sfx.dedupe_speech`, exact text;
  `ocr_sfx.dedupe_sfx`, one per sound).
- `correction.py` + `taxonomy.py` / `tag_rules.py` / `tag_groups.py` do Danbooru-KB correction and
  bucket ordering; `tag_drop_groups.py` is `--caption_drop_groups`; `index.py` builds
  `caption_index.json`; `shuffle.py` owns the `@no-artist` sentinel and Anima-prefix shuffle;
  `tokenizers.py` is the length check behind `CorrectRequest`'s randomize tokenizers.
- `clause_rewrite.py` / `clause_vocabulary.py` / `group_router.py` / `caption_layout.py` are the
  position-clause rewrite: which tag may enter which clause, and what moves out of the bag. Gate
  and group sets are data in `data/clause_vocabulary.yaml` (the trainer's
  `configs/clause_vocabulary.yaml` overrides it) — retune there, not in Python.

## Who else reads a caption

Only through this package. `stages/_walk_captions.py` and `stages/_caption_io.py` are the
revised-first read and the trailing-newline write; `grouping/features.py::read_tags` goes through
`parse_caption`; the GUI never splits a caption in the browser (`gui/dataset.py` serves the parse).
