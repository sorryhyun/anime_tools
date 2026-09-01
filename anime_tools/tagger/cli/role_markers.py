"""Role-marker scan — find character-typed tags that behave like copyrights.

Ranks every ``category=='character'`` tag in a checkpoint's ``vocab.json`` +
``dataset.json`` by its conditional co-occurrence with *another* character tag
on **solo** samples, then buckets each candidate by a prefix heuristic over the
partner list:

* **A_costume** — candidate and a top partner share the name prefix before the
  first ``(``; the longer-parenthetical name is the variant. Curate via
  ``tag_rules.yaml`` dedup blocks so the variant wins whenever both fire.
* **D_role** — broad partner pool: an affiliation marker (``sensei (blue
  archive)``). Curate via ``tag_rules.yaml`` ``remove:``.
* **C_pair** — narrow pool dominated by one partner: a genuine couple/sibling
  pair. Leave alone; the data is correct.
* **B_review** — everything else. Likely aliases or noisy edge cases.

Read-only: prints a table and optionally writes a pasteable YAML stub.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from anime_tools.captions.taxonomy import solo_multi_indices as _solo_index_sets
from anime_tools.tagger.data import TaggerCheckpoint

logger = logging.getLogger(__name__)


def _name_prefix(name: str) -> str:
    """Everything before the first ``(``, stripped — costume detection.

    ``"toki (bunny) (blue archive)"`` → ``"toki"``.
    """
    return name.split("(", 1)[0].strip()


def _name_prefix_no_first_token(name: str) -> str:
    """Like :func:`_name_prefix` but also strips the first space token.

    Catches "adjective + base" costume patterns the bare prefix misses
    (``"cool mita (miside)"`` → ``"mita"``). Empty when the prefix has only one
    token — callers treat that as "no match".
    """
    pre = _name_prefix(name)
    parts = pre.split()
    if len(parts) < 2:
        return ""
    return " ".join(parts[1:])


def classify(
    name: str,
    partners_full: list[tuple[str, int]],
    n_co: int,
    min_role_partners: int,
    pair_dominance: float,
) -> tuple[str, str]:
    """Bucket a candidate on its full partner distribution.

    Returns ``(bucket, base_or_top_partner)``: the prefix-matching base for A,
    the top partner for C and B, an empty string for D. Order of checks
    matters — D is tested before A so a popular character with both a costume
    variant and many partners lands under ``remove:`` rather than a dedup row.
    """
    if not partners_full or n_co == 0:
        return "B_review", ""

    n_distinct = len(partners_full)
    top_name, top_count = partners_full[0]
    top_share = top_count / n_co

    # D — role marker: broad partner pool wins over A.
    if n_distinct >= min_role_partners:
        return "D_role", ""

    # A — costume/version variant: prefix-before-paren or drop-first-token
    # matches a partner, which is the *base* when the candidate is the more
    # specific one; otherwise the row is informational (A_base).
    cand_prefix = _name_prefix(name)
    cand_prefix_drop1 = _name_prefix_no_first_token(name)
    matched_partner = None
    if cand_prefix:
        for pname, _ in partners_full:
            if pname == name:
                continue
            if _name_prefix(pname) == cand_prefix:
                matched_partner = pname
                break
    if matched_partner is None and cand_prefix_drop1:
        for pname, _ in partners_full:
            if pname == name:
                continue
            if _name_prefix(pname) == cand_prefix_drop1:
                matched_partner = pname
                break
    if matched_partner is not None:
        if _is_more_specific(name, matched_partner):
            return "A_costume", matched_partner
        return "A_base", matched_partner

    # C — couple/sibling pair: narrow pool dominated by the top partner, but only
    # at ≥2 distinct partners; n_distinct==1 is ambiguous (alias vs. couple).
    if top_share >= pair_dominance and n_distinct >= 2:
        return "C_pair", top_name

    return "B_review", top_name


def _is_more_specific(a: str, b: str) -> bool:
    """True if ``a`` is the more-specific (variant) form of ``b``.

    More parenthetical groups wins, longer string breaks the tie; this picks
    the dedup direction, so we emit ``base: [variant]``.
    """
    a_paren = a.count("(")
    b_paren = b.count("(")
    if a_paren != b_paren:
        return a_paren > b_paren
    return len(a) > len(b)


def scan(
    vocab: dict,
    manifest: dict,
    min_solo: int,
    min_ratio: float,
    top_partners: int,
    min_role_partners: int = 5,
    pair_dominance: float = 0.6,
) -> list[dict]:
    """Return a list of candidate role markers ranked by co-occurrence ratio.

    Each entry is::

        {
          "name": "sensei (blue archive)",
          "index": 620,
          "freq": 130,
          "n_solo": 142,                  # solo samples where this tag fires
          "n_co": 138,                    # of those, how many have another char
          "ratio": 0.971,
          "partners": [(name, count), ...],  # top-K partner chars by count
          "n_distinct_partners": 47,
          "bucket": "D_role",
          "base": "",                      # the variant base (A) or top pair-mate
        }

    Sorted by descending ratio, then descending ``n_solo``.
    """
    tags = vocab["tags"]
    idx2name: dict[int, str] = {int(t["index"]): t["name"] for t in tags}
    char_idx: set[int] = {int(t["index"]) for t in tags if t["category"] == "character"}
    single_idx, multi_idx = _solo_index_sets(tags)

    n_solo: dict[int, int] = {i: 0 for i in char_idx}
    n_co: dict[int, int] = {i: 0 for i in char_idx}
    partner: dict[int, dict[int, int]] = {i: {} for i in char_idx}

    for tags_list in manifest["tag_indices"]:
        s = set(tags_list)
        is_solo = bool(s & single_idx) and not (s & multi_idx)
        if not is_solo:
            continue
        chars_here = s & char_idx
        if not chars_here:
            continue
        is_multi = len(chars_here) > 1
        for c in chars_here:
            n_solo[c] += 1
            if is_multi:
                n_co[c] += 1
                for p in chars_here:
                    if p == c:
                        continue
                    partner[c][p] = partner[c].get(p, 0) + 1

    rows: list[dict] = []
    name_to_freq = {t["name"]: int(t["freq"]) for t in tags}
    for c in char_idx:
        ns = n_solo[c]
        if ns < min_solo:
            continue
        ratio = n_co[c] / ns
        if ratio < min_ratio:
            continue
        partners_full_sorted = sorted(partner[c].items(), key=lambda kv: -kv[1])
        partners_named = [(idx2name[p], n) for p, n in partners_full_sorted]
        bucket, base = classify(
            idx2name[c],
            partners_named,
            n_co[c],
            min_role_partners=min_role_partners,
            pair_dominance=pair_dominance,
        )
        rows.append(
            {
                "name": idx2name[c],
                "index": c,
                "freq": name_to_freq.get(idx2name[c], -1),
                "n_solo": ns,
                "n_co": n_co[c],
                "ratio": ratio,
                "partners": partners_named[:top_partners],
                "n_distinct_partners": len(partners_named),
                "bucket": bucket,
                "base": base,
            }
        )
    rows.sort(key=lambda r: (-r["ratio"], -r["n_solo"]))
    return rows


def _yaml_safe(s: str) -> str:
    """Return ``s`` in a YAML-safe form for use as a sequence item or key.

    Quotes only when needed: a ``": "`` would make `trailblazer (honkai: star
    rail)` parse as a mapping, and a leading reserved indicator or internal
    apostrophe is likewise ambiguous. Apostrophes are doubled inside the quotes
    per YAML 1.2 § 7.3.2.
    """
    needs_quote = (
        ": " in s
        or s.endswith(":")
        or "'" in s
        or (s and s[0] in "@#&*!|>'\"%`-?,[]{}")
    )
    if not needs_quote:
        return s
    return "'" + s.replace("'", "''") + "'"


def _format_table(rows: list[dict], limit: int) -> str:
    if not rows:
        return "(no candidates above threshold)"
    head = (
        f"{'bucket':<10}  {'ratio':>5}  {'n_solo':>6}  {'n_co':>5}  "
        f"{'np':>3}  {'freq':>5}  {'tag':<40}  partners (count)"
    )
    sep = "-" * len(head)
    lines = [head, sep]
    for r in rows[:limit]:
        partners_str = ", ".join(f"{n}×{name}" for name, n in r["partners"])
        lines.append(
            f"{r['bucket']:<10}  {r['ratio']:>5.2f}  {r['n_solo']:>6d}  "
            f"{r['n_co']:>5d}  {r['n_distinct_partners']:>3d}  "
            f"{r['freq']:>5d}  {r['name']:<40}  {partners_str}"
        )
    if len(rows) > limit:
        lines.append(f"... ({len(rows) - limit} more)")
    return "\n".join(lines)


def _emit_yaml_stub(rows: list[dict], min_solo: int, min_ratio: float) -> str:
    """Build a yaml-shaped string with pasteable per-bucket sections.

    Not a single valid YAML document: a working file the curator copies
    snippets out of — A_costume as dedup blocks, D_role as ``remove:`` items,
    B_review/C_pair as commented triage hints.
    """
    by_bucket: dict[str, list[dict]] = {}
    for r in rows:
        by_bucket.setdefault(r["bucket"], []).append(r)

    lines: list[str] = []
    lines.append("# Auto-classified role-marker scan output.")
    lines.append(
        f"# Threshold: n_solo>={min_solo}, ratio>={min_ratio:.2f}. "
        f"{len(rows)} candidate(s)."
    )
    lines.append("# Sections below are ready to paste into tag_rules.yaml — see")
    lines.append("# headers for the target location.")
    lines.append("")

    a_rows = by_bucket.get("A_costume", [])
    lines.append("# ╔══════════════════════════════════════════════════════════════╗")
    lines.append("# ║ A_costume — paste these as top-level dedup blocks in         ║")
    lines.append("# ║ tag_rules.yaml (when any variant fires, the base is dropped) ║")
    lines.append("# ╚══════════════════════════════════════════════════════════════╝")
    if a_rows:
        base_to_variants: dict[str, list[tuple[str, dict]]] = {}
        for r in a_rows:
            base = r["base"]
            base_to_variants.setdefault(base, []).append((r["name"], r))
        for base in sorted(base_to_variants):
            variants = base_to_variants[base]
            lines.append(f"{_yaml_safe(base)}:")
            for vname, vr in sorted(variants, key=lambda kv: kv[0]):
                lines.append(
                    f"  - {_yaml_safe(vname)}  # "
                    f"ratio={vr['ratio']:.2f} n_solo={vr['n_solo']}"
                )
        lines.append("")
    else:
        lines.append("# (no A_costume candidates)")
        lines.append("")

    d_rows = by_bucket.get("D_role", [])
    lines.append("# ╔══════════════════════════════════════════════════════════════╗")
    lines.append("# ║ D_role — paste these under `remove:` in tag_rules.yaml.      ║")
    lines.append("# ║ These are class/affiliation markers (broad partner pool).    ║")
    lines.append("# ║ Removing strips them from training; alternatively keep the   ║")
    lines.append("# ║ tag and recategorize via a future force_general: override.   ║")
    lines.append("# ╚══════════════════════════════════════════════════════════════╝")
    lines.append("remove:")
    if d_rows:
        for r in d_rows:
            lines.append(
                f"  - {_yaml_safe(r['name'])}  # "
                f"ratio={r['ratio']:.2f} n_solo={r['n_solo']} "
                f"n_partners={r['n_distinct_partners']}"
            )
    else:
        lines.append("  # (no D_role candidates)")
    lines.append("")

    b_rows = by_bucket.get("B_review", [])
    lines.append("# ╔══════════════════════════════════════════════════════════════╗")
    lines.append("# ║ B_review — eyeball each. Likely aliases (use `replacements:`)║")
    lines.append("# ║ or genuine pair-mates.                                       ║")
    lines.append("# ╚══════════════════════════════════════════════════════════════╝")
    if b_rows:
        for r in b_rows:
            top = r["partners"][0] if r["partners"] else ("", 0)
            lines.append(
                f"#  - {r['name']:<40}  top: {top[1]}×{top[0]} "
                f"(ratio={r['ratio']:.2f}, n_solo={r['n_solo']}, "
                f"n_partners={r['n_distinct_partners']})"
            )
    else:
        lines.append("# (none)")
    lines.append("")

    c_rows = by_bucket.get("C_pair", [])
    lines.append("# ╔══════════════════════════════════════════════════════════════╗")
    lines.append("# ║ C_pair — leave these alone. Genuine couple/sibling tag pairs;║")
    lines.append("# ║ the data is correct.                                         ║")
    lines.append("# ╚══════════════════════════════════════════════════════════════╝")
    if c_rows:
        for r in c_rows:
            top = r["partners"][0] if r["partners"] else ("", 0)
            lines.append(
                f"#  - {r['name']:<40}  pair: {top[0]} ({top[1]}×, "
                f"ratio={r['ratio']:.2f})"
            )
    else:
        lines.append("# (none)")
    lines.append("")

    return "\n".join(lines)


def cmd_scan_role_markers(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    ckpt = TaggerCheckpoint.from_dir(out_dir, require=("vocab", "dataset"))
    vocab, manifest = ckpt.vocab, ckpt.dataset

    rows = scan(
        vocab,
        manifest,
        min_solo=args.min_solo,
        min_ratio=args.min_ratio,
        top_partners=args.top_partners,
        min_role_partners=args.min_role_partners,
        pair_dominance=args.pair_dominance,
    )
    bucket_counts: dict[str, int] = {}
    for r in rows:
        bucket_counts[r["bucket"]] = bucket_counts.get(r["bucket"], 0) + 1
    logger.info(
        "scanned %d trainable samples, %d character tags in vocab — "
        "%d candidates with n_solo≥%d and ratio≥%.2f",
        len(manifest["tag_indices"]),
        sum(1 for t in vocab["tags"] if t["category"] == "character"),
        len(rows),
        args.min_solo,
        args.min_ratio,
    )
    logger.info(
        "  buckets: A_costume=%d, A_base=%d, D_role=%d, C_pair=%d, B_review=%d",
        bucket_counts.get("A_costume", 0),
        bucket_counts.get("A_base", 0),
        bucket_counts.get("D_role", 0),
        bucket_counts.get("C_pair", 0),
        bucket_counts.get("B_review", 0),
    )
    print(_format_table(rows, args.limit))

    if args.out_yaml:
        out_path = Path(args.out_yaml)
        out_path.write_text(
            _emit_yaml_stub(rows, args.min_solo, args.min_ratio), encoding="utf-8"
        )
        logger.info("wrote bucketed stub → %s", out_path)
