"""Vocab build — caption discovery, tag categorization, frequency cuts.

Writes ``vocab.json`` (the label space: no per-stem data), ``rules.yaml`` (a
snapshot of the source ``tag_rules.yaml``) and ``dataset.json`` (the per-stem
manifest and the sole home of the train/val split) under ``out_dir/``. Every
other CLI mode reads those, never the source corpus.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from anime_tools._json import read_json, write_json
from anime_tools.captions import position_clauses as pc
from anime_tools.captions import tag_groups as tg
from anime_tools.captions import tag_rules as tr
from anime_tools.captions.taxonomy import (
    canonical_rating,
    is_artist_tag,
    is_rating_tag,
    strip_artist_prefix,
)
from anime_tools.tagger.tagger import (
    PEOPLE_COUNT_LABELS,
    RATINGS,
    SLOT_ORDER,
    TAG_TYPE_NAMES,
)

from .constants import (
    classify_people,
    find_image_for_caption,
    is_count_tag,
)

logger = logging.getLogger(__name__)


def find_caption_files(roots: Sequence[Path]) -> list[Path]:
    """Discover all ``.txt`` caption files under the given roots.

    Returns files in **root order** (sorted within each root for determinism);
    a stem appearing under multiple roots is *not* deduped here — that's
    :func:`build_caption_index`'s job, where the earlier one wins.
    """
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            logger.warning("caption root %s does not exist — skipping", root)
            continue
        out.extend(
            sorted(
                p
                for p in root.rglob("*.txt")
                if not any(part.startswith(".") for part in p.parts)
            )
        )
    return out


def build_caption_index(
    paths: Iterable[Path],
    rules: tr.TagRules,
) -> dict[str, tuple[Path, Path | None, list[str]]]:
    """Map ``stem → (caption_path, image_path | None, parsed_tags)``.

    The *first* path for a stem wins, so the caller controls precedence via root
    order. Stems with no sibling image are still indexed, so the coverage scan
    reflects what's *captioned*, not what's *trainable*; the image-required
    filter happens at manifest-build time. Position clauses are dropped: only
    the flat tag bag feeds tag training.
    """
    index: dict[str, tuple[Path, Path | None, list[str]]] = {}
    for path in paths:
        stem = path.stem
        if stem in index:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("non-utf8 caption %s — skipped", path)
            continue
        parsed = pc.parse_caption(content.strip())
        if parsed.has_clauses:
            content = ", ".join(parsed.flat_tags)
        tags = tr.parse_caption(content, rules)
        if not tags:
            continue
        image_path = find_image_for_caption(path)
        index[stem] = (path, image_path, tags)
    return index


def load_tag_cache(path: Path) -> dict[str, str]:
    """Load a tag-taxonomy source and map tag → category name.

    Dispatched by suffix: ``.json`` is gelcrawl's ``{tag: type_id}`` corpus
    cache, ``.csv`` the public ``danbooru_tags_classified.csv`` KB, whose
    ``category`` column carries the same numeric Danbooru type id — so the vocab
    build can run off the downloadable KB with no private-crawl dependency.
    Both normalize underscored keys to the space-separated caption form.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return _load_tag_cache_csv(path)
    raw = read_json(path)
    out: dict[str, str] = {}
    for tag, type_id in raw.items():
        cat = TAG_TYPE_NAMES.get(int(type_id))
        if cat is not None:
            # Cache uses underscored names; canonical caption format uses spaces.
            out[tag.replace("_", " ")] = cat
    return out


def _load_tag_cache_csv(path: Path) -> dict[str, str]:
    """Parse ``danbooru_tags_classified.csv`` into a tag → category-name map."""
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = str(row.get("name") or "").strip()
            raw_cat = str(row.get("category") or "").strip()
            if not name or not raw_cat:
                continue
            try:
                cat = TAG_TYPE_NAMES.get(int(raw_cat))
            except ValueError:
                continue
            if cat is not None:
                out[name.replace("_", " ")] = cat
    return out


# Categories assignable via ``category_overrides:`` in tag_rules.yaml. ``rating``
# (separate corpus field) and ``count`` (regex-matched) are excluded — overriding
# them would only create dead aliases.
_OVERRIDABLE_CATEGORIES = frozenset(TAG_TYPE_NAMES.values())


def categorize(
    tag: str,
    cache: dict[str, str],
    overrides: dict[str, str] | None = None,
) -> str:
    """Return ``rating`` / ``count`` / ``character`` / ``copyright`` /
    ``artist`` / ``general`` / ``metadata`` / ``deprecated`` for ``tag``.

    Resolution order matters:

    1. Rating literals first — ``general`` is *both* a legacy rating value and a
       category name, so rating membership must beat any category lookup.
    2. ``@<non-space>…`` → ``artist``. Emoticons like ``@ @`` (booru ``@_@``)
       fall through to the cache so they get their real category.
    3. Count-tag regex → ``count`` (overrides ``general`` typing for ``1girl``).
    4. ``category_overrides``, which fix booru cache mistypings.
    5. Cache lookup, else ``general``.

    Cache keys drop the ``@``, so lookups use the bare name.
    """
    if is_rating_tag(tag):
        return "rating"
    if is_artist_tag(tag):
        return "artist"
    if is_count_tag(tag):
        return "count"
    bare = strip_artist_prefix(tag)
    if overrides:
        ov = overrides.get(tag) or overrides.get(bare)
        if ov is not None:
            return ov
    cat = cache.get(bare)
    if cat is None:
        return "general"
    return cat


def validate_overrides(overrides: dict[str, str]) -> list[str]:
    """Return human-readable validation errors for overrides; empty = all good.

    Catches typos like ``caracter`` and unsupported categories up front so
    :func:`cmd_build_vocab` fails loudly rather than silently typing tags into a
    slot the trainer doesn't understand.
    """
    errors: list[str] = []
    for tag, cat in overrides.items():
        if cat not in _OVERRIDABLE_CATEGORIES:
            errors.append(
                f"category_overrides[{tag!r}] = {cat!r} — must be one of "
                f"{sorted(_OVERRIDABLE_CATEGORIES)}"
            )
    return errors


def build_vocab(
    caption_index: dict[str, tuple[Path, Path | None, list[str]]],
    tag_cache: dict[str, str],
    min_freq: int,
    category_overrides: dict[str, str] | None = None,
) -> dict:
    """Compute frequencies, categories, median emit positions; cut by min_freq."""
    freq: Counter = Counter()
    sum_pos: dict[str, int] = defaultdict(int)
    pos_counts: dict[str, int] = defaultdict(int)

    rating_freq: Counter = Counter()
    n_with_rating = 0
    people_freq: Counter = Counter()

    for _, _, tags in caption_index.values():
        # Pull rating off the front (Anima puts it first; scan the first few
        # defensively); everything else feeds the multi-label vocab. Legacy
        # booru spellings are folded onto the canonical band so a mixed corpus
        # reports one distribution rather than two.
        rating_seen: str | None = None
        for t in tags[:2]:
            canon = canonical_rating(t)
            if canon is not None:
                rating_seen = canon
                break
        if rating_seen is not None:
            rating_freq[rating_seen] += 1
            n_with_rating += 1

        # People-count distribution is informational; the per-stem label is
        # recomputed at manifest-build time to stay in sync with the rule.
        people_freq[PEOPLE_COUNT_LABELS[classify_people(tags)]] += 1

        for i, tag in enumerate(tags):
            if is_rating_tag(tag):
                continue
            freq[tag] += 1
            sum_pos[tag] += i
            pos_counts[tag] += 1

    kept = sorted(
        (t for t, c in freq.items() if c >= min_freq),
        key=lambda t: (-freq[t], t),
    )
    dropped_lowfreq = sum(1 for c in freq.values() if c < min_freq)

    cat_buckets: Counter = Counter()
    cache_hits = 0
    for tag in kept:
        cat = categorize(tag, tag_cache, category_overrides)
        cat_buckets[cat] += 1
        bare = tag.removeprefix("@")
        if bare in tag_cache:
            cache_hits += 1

    tags_payload: list[dict] = []
    for idx, tag in enumerate(kept):
        cat = categorize(tag, tag_cache, category_overrides)
        median_pos = sum_pos[tag] / max(pos_counts[tag], 1)
        tags_payload.append(
            {
                "name": tag,
                "index": idx,
                "category": cat,
                "freq": freq[tag],
                "median_pos": round(median_pos, 2),
            }
        )

    return {
        "tags": tags_payload,
        "ratings": list(RATINGS),
        "people_count_labels": list(PEOPLE_COUNT_LABELS),
        "slot_order": list(SLOT_ORDER),
        "min_freq": min_freq,
        "n_captions_seen": len(caption_index),
        "n_unique_tags_seen": len(freq),
        "n_tags_kept": len(kept),
        "n_tags_dropped_lowfreq": dropped_lowfreq,
        "category_counts": dict(cat_buckets),
        "cache_hit_rate": round(cache_hits / max(len(kept), 1), 4),
        "rating_distribution": dict(rating_freq),
        "rating_coverage": round(n_with_rating / max(len(caption_index), 1), 4),
        "people_count_distribution": dict(people_freq),
    }


def make_split(
    stems: Sequence[str],
    val_frac: float,
    seed: int,
) -> dict[str, list[str]]:
    """Deterministic random split keyed by ``seed``."""
    rng = random.Random(seed)
    shuffled = list(stems)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_frac))
    return {
        "val": sorted(shuffled[:n_val]),
        "train": sorted(shuffled[n_val:]),
        "seed": seed,
        "val_frac": val_frac,
    }


def build_manifest(
    caption_index: dict[str, tuple[Path, Path | None, list[str]]],
    vocab: dict,
    split: dict,
) -> dict:
    """Compact dataset.json: per-stem image path, multi-hot indices, rating, people-count.

    Stems lacking a sibling image are dropped (vocab statistics still count
    them) and the split is filtered to match. The people-count label is
    recomputed via :func:`classify_people` so the bucketing rule stays the
    single source of truth.
    """
    tag_to_idx: dict[str, int] = {t["name"]: t["index"] for t in vocab["tags"]}
    rating_to_idx: dict[str, int] = {r: i for i, r in enumerate(vocab["ratings"])}

    stems: list[str] = []
    image_paths: list[str] = []
    tag_indices: list[list[int]] = []
    rating_indices: list[int] = []
    people_count_indices: list[int] = []
    n_no_image = 0
    n_no_rating = 0
    n_no_tags = 0

    for stem in sorted(caption_index.keys()):
        _, image_path, tags = caption_index[stem]
        if image_path is None:
            n_no_image += 1
            continue
        rating_idx: int | None = None
        for t in tags[:2]:
            # Raw literal first so a vocab.json built before the safe/nsfw
            # rename still indexes its own spellings; canonical form second so
            # a legacy caption lands in the renamed band.
            hit = rating_to_idx.get(t)
            if hit is None:
                canon = canonical_rating(t)
                hit = rating_to_idx.get(canon) if canon is not None else None
            if hit is not None:
                rating_idx = hit
                break
        if rating_idx is None:
            n_no_rating += 1
            continue
        idxs = sorted(
            tag_to_idx[t] for t in tags if t in tag_to_idx and not is_rating_tag(t)
        )
        if not idxs:
            n_no_tags += 1
            continue
        stems.append(stem)
        image_paths.append(str(image_path.resolve()))
        tag_indices.append(idxs)
        rating_indices.append(rating_idx)
        people_count_indices.append(classify_people(tags))

    kept = set(stems)
    filtered_split = {
        "val": [s for s in split["val"] if s in kept],
        "train": [s for s in split["train"] if s in kept],
        "seed": split["seed"],
        "val_frac": split["val_frac"],
    }

    return {
        "stems": stems,
        "image_paths": image_paths,
        "tag_indices": tag_indices,
        "rating_indices": rating_indices,
        "people_count_indices": people_count_indices,
        "split": filtered_split,
        "n_tags": len(vocab["tags"]),
        "n_ratings": len(vocab["ratings"]),
        "n_people_counts": len(PEOPLE_COUNT_LABELS),
        "dropped_no_image": n_no_image,
        "dropped_no_rating": n_no_rating,
        "dropped_no_invocab_tags": n_no_tags,
    }


def scan_cache_coverage(
    caption_index: dict[str, tuple[Path, Path | None, list[str]]],
    tag_cache: dict[str, str],
    category_overrides: dict[str, str] | None = None,
    coverage_ignore: tuple[str, ...] | None = None,
) -> dict:
    """How many caption tags lack a category in the tag cache?

    A high miss rate means ``categorize()`` falls back to ``general`` too often
    and the gelbooru API fill-in pass should run first; <5 % is safe. Tags in
    ``category_overrides`` count as covered (typed by the curator instead), and
    tags containing a ``coverage_ignore`` substring are skipped from both
    tallies — noisy general descriptors the booru cache doesn't track.
    """
    overrides = category_overrides or {}
    ignore_subs = tuple(coverage_ignore or ())
    seen: Counter = Counter()
    missing: Counter = Counter()
    for _, _, tags in caption_index.values():
        for tag in tags:
            if is_rating_tag(tag):
                continue
            if ignore_subs and any(sub in tag for sub in ignore_subs):
                continue
            seen[tag] += 1
            is_artist = len(tag) >= 2 and tag[0] == "@" and not tag[1].isspace()
            bare = tag[1:] if is_artist else tag
            if (
                is_artist
                or is_count_tag(tag)
                or bare in tag_cache
                or tag in overrides
                or bare in overrides
            ):
                continue
            missing[tag] += 1
    return {
        "n_unique_tags": len(seen),
        "n_unique_missing": len(missing),
        "n_total_tag_occurrences": sum(seen.values()),
        "n_missing_occurrences": sum(missing.values()),
        "missing_top20": missing.most_common(20),
    }


def cmd_build_vocab(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rules_src = Path(args.rules)
    rules = tr.load_rules(rules_src)
    logger.info(
        "rules: %d replacements, %d remove, %d dedup base tags, %d category overrides",
        len(rules.replacements),
        len(rules.remove),
        len(rules.dedup),
        len(rules.category_overrides),
    )
    override_errors = validate_overrides(rules.category_overrides)
    if override_errors:
        for e in override_errors:
            logger.error(e)
        raise SystemExit(
            f"category_overrides has {len(override_errors)} invalid entry(ies) "
            f"in {rules_src} — fix and re-run"
        )
    if rules.category_overrides:
        target_counts: Counter = Counter(rules.category_overrides.values())
        logger.info(
            "category_overrides targets: %s",
            {k: target_counts[k] for k in sorted(target_counts)},
        )

    roots = [Path(r) for r in args.caption_roots]
    cap_paths = find_caption_files(roots)
    logger.info("scanning %d caption files across %d roots", len(cap_paths), len(roots))
    index = build_caption_index(cap_paths, rules)
    logger.info("kept %d unique stems with non-empty captions", len(index))

    tag_cache = load_tag_cache(Path(args.tag_cache))
    logger.info("loaded tag cache with %d entries", len(tag_cache))

    coverage = scan_cache_coverage(
        index,
        tag_cache,
        rules.category_overrides,
        rules.coverage_ignore,
    )
    miss_rate = coverage["n_missing_occurrences"] / max(
        coverage["n_total_tag_occurrences"], 1
    )
    logger.info(
        "cache coverage: %d/%d unique tags categorized (%.2f%% of occurrences missing)",
        coverage["n_unique_tags"] - coverage["n_unique_missing"],
        coverage["n_unique_tags"],
        100 * miss_rate,
    )
    if coverage["missing_top20"]:
        logger.info("top-20 uncategorized tags (will fall back to 'general'):")
        for tag, n in coverage["missing_top20"]:
            logger.info("  %5d × %s", n, tag)

    vocab = build_vocab(
        index,
        tag_cache,
        min_freq=args.min_freq,
        category_overrides=rules.category_overrides,
    )
    vocab["caption_roots"] = [str(r.resolve()) for r in roots]
    vocab["tag_cache_path"] = str(Path(args.tag_cache).resolve())
    vocab["rules_source_path"] = str(rules_src.resolve())
    vocab["coverage"] = coverage

    # Derive tag-groups from the danbooru taxonomy + the scanned captions, merged
    # onto any curated --groups (preserved verbatim), and use the result as the
    # groups source — so one build_vocab call replaces a separate derive_groups.
    groups_src = Path(args.groups) if args.groups else None
    # The derive write reads its own product back to validate it; that load is
    # the one this function needs, so it is carried here instead of parsing the
    # file we just wrote a second time.
    merged: tg.TagGroups | None = None
    if getattr(args, "derive_groups", False):
        from anime_tools.captions.correction import (
            find_tag_csv,
            load_tag_knowledge_base,
        )

        from .derive_groups import (
            derive_from_args,
            solo_sets_from_index,
            write_merged_groups,
        )

        csv_path = (
            Path(args.tag_cache)
            if str(args.tag_cache).lower().endswith(".csv")
            else find_tag_csv()
        )
        if csv_path is None or not Path(csv_path).exists():
            logger.warning(
                "derive_groups on but danbooru_tags_classified.csv KB not found "
                "(set --tag_cache to it or place it under models/); skipping "
                "derivation — using --groups as-is"
            )
        else:
            rows, _unmatched, _n_general = derive_from_args(
                args,
                vocab,
                load_tag_knowledge_base(csv_path),
                rules,
                solo_sets_from_index(index),
                # The scan above, not a second pass over the corpus — this build
                # already holds the index derive_groups would have to rebuild.
                source="this build's caption scan",
            )
            derived_path = out_dir / "groups.yaml"
            merged = write_merged_groups(
                rows,
                derived_path,
                # Preserve --groups verbatim if it exists; else merge onto nothing.
                groups_src if (groups_src and groups_src.exists()) else None,
                min_group_size=args.min_group_size,
            )
            groups_src = derived_path
    if groups_src is not None and groups_src.exists():
        groups = merged if merged is not None else tg.load_groups(groups_src)
        name_to_cat = {
            t["name"]: str(t.get("category", "general")) for t in vocab["tags"]
        }
        # ``$category:<cat>`` member markers expand against the kept vocab —
        # must happen before resolve AND before the snapshot write below, so
        # the shipped groups.yaml carries concrete names for inference.
        groups = tg.expand_category_members(groups, name_to_cat)
        # One synthetic "<none:group>" tag slot per sentinel group, appended
        # after every real tag so existing indices stay stable. Category inherits
        # the member majority; freq=0 — the slot never appears in captions, BCE
        # never supervises it and decode never emits it.
        n_sentinels = 0
        for g in groups.groups:
            if not (g.sentinel and g.mode in ("softmax", "softmax_when_solo")):
                continue
            member_cats = Counter(name_to_cat[t] for t in g.tags if t in name_to_cat)
            vocab["tags"].append(
                {
                    "name": tg.sentinel_tag_name(g.name),
                    "index": len(vocab["tags"]),
                    "category": (
                        member_cats.most_common(1)[0][0] if member_cats else "general"
                    ),
                    "freq": 0,
                    "median_pos": 999.0,
                    "sentinel_for": g.name,
                }
            )
            n_sentinels += 1
        if n_sentinels:
            logger.info(
                "appended %d sentinel tag slot(s) for typed groups", n_sentinels
            )
        tag_to_idx = {t["name"]: t["index"] for t in vocab["tags"]}
        resolved, dropped = tg.resolve_groups(groups, tag_to_idx)
        vocab["groups"] = tg.resolved_to_dict(resolved)
        vocab["groups_source_path"] = str(groups_src.resolve())
        logger.info(
            "groups: %d typed groups, %d tag/escape names dropped (not_in_vocab)",
            len(resolved),
            len(dropped),
        )
        for g in resolved:
            n_drop = len(g.tag_names) < sum(1 for _ in groups.by_name(g.name).tags)
            logger.info(
                "  %-14s mode=%-18s n_tags=%3d n_escape=%2d%s",
                g.name,
                g.mode,
                len(g.tag_indices),
                len(g.escape_indices),
                "  (some tags dropped)" if n_drop else "",
            )
        if dropped:
            sample = list(dropped)[:10]
            logger.info(
                "first %d dropped: %s%s",
                len(sample),
                sample,
                " …" if len(dropped) > len(sample) else "",
            )
    else:
        vocab["groups"] = []
        vocab["groups_source_path"] = None
        if args.groups:
            logger.warning(
                "--groups=%s does not exist — building flat-vocab checkpoint",
                args.groups,
            )
        else:
            logger.info(
                "no --groups given — building flat-vocab checkpoint "
                "(pure BCE on every tag)",
            )

    # The split is a property of the corpus, not the label space: it lives only
    # in dataset.json, never in the shipped vocab.json.
    split = make_split(
        sorted(index.keys()),
        val_frac=args.val_frac,
        seed=args.seed,
    )

    vocab_path = write_json(out_dir / "vocab.json", vocab)
    logger.info("wrote %s", vocab_path)

    # Snapshot groups + rules into the checkpoint dir so the inference wrapper
    # has zero runtime dependency on the source corpus.
    if groups_src is not None and groups_src.exists():
        groups_snap = out_dir / "groups.yaml"
        with open(groups_snap, "w") as f:
            import yaml as _yaml

            _yaml.safe_dump(groups.to_dict(), f, sort_keys=False)
        logger.info("wrote %s", groups_snap)

    snap_path = out_dir / "rules.yaml"
    with open(snap_path, "w") as f:
        import yaml as _yaml

        _yaml.safe_dump(rules.to_dict(), f, sort_keys=False)
    logger.info("wrote %s", snap_path)

    manifest = build_manifest(index, vocab, split)
    manifest_path = write_json(out_dir / "dataset.json", manifest)
    logger.info(
        "wrote %s — %d trainable samples (dropped %d no_image, %d no_rating, "
        "%d no_invocab_tags)",
        manifest_path,
        len(manifest["stems"]),
        manifest["dropped_no_image"],
        manifest["dropped_no_rating"],
        manifest["dropped_no_invocab_tags"],
    )

    print()
    print(f"  caption stems indexed:  {vocab['n_captions_seen']}")
    print(f"  unique tags seen:       {vocab['n_unique_tags_seen']}")
    print(f"  vocab size (≥{args.min_freq}):       {vocab['n_tags_kept']}")
    print(f"  dropped (low-freq):     {vocab['n_tags_dropped_lowfreq']}")
    print(f"  cache hit rate:         {vocab['cache_hit_rate']}")
    print("  category counts:")
    for cat, n in sorted(vocab["category_counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {cat:<12} {n}")
    print(f"  rating coverage:        {vocab['rating_coverage']}")
    print(f"  rating distribution:    {vocab['rating_distribution']}")
    print(f"  people distribution:    {vocab['people_count_distribution']}")
    print(
        f"  split:                  {len(split['train'])} train / {len(split['val'])} val"
    )
    print(f"  cache miss rate:        {miss_rate:.2%}")
    print(f"  trainable samples:      {len(manifest['stems'])}")
    print(
        f"    (dropped {manifest['dropped_no_image']} no-image, "
        f"{manifest['dropped_no_rating']} no-rating, "
        f"{manifest['dropped_no_invocab_tags']} no-invocab-tags)"
    )
