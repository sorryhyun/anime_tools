"""Apply a reviewer-curated accept list of multiview-audit findings — with undo.

The gate-based ``audit_multiview.py --apply`` writes every finding a verdict
tier admits; this is the other workflow: a human reads the contact sheets,
hand-picks findings across tiers, and applies exactly that set. Accepted
``multiple views``/``unsure`` rows get the ``multiple views`` splice; accepted
``extra-character`` rows get the ``{instances}girls`` count replacement.

Every run writes a manifest (verbatim before/after per caption) next to the
report — ``image_dataset/`` is gitignored, so the manifest is the only undo.
``--revert <manifest>`` restores it, refusing any caption edited since.

Dry-run by default; ``--apply`` writes. After a real apply or revert, run
``make preprocess-te`` — the caption master changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from anime_tools._env import resolve_path  # noqa: E402
from anime_tools.stages.multiview_audit import (  # noqa: E402
    apply_curated,
    revert_curated,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--report",
        default="post_image_dataset/captions/multiview_audit_full/report.json",
        help="Audit report.json the accept list refers to",
    )
    p.add_argument(
        "--accept",
        default=None,
        help="Text file of accepted images (one per line, path relative to the "
        "resized dir as printed in the report; # comments allowed)",
    )
    p.add_argument(
        "--revert",
        default=None,
        help="Manifest json from a previous run — undo it instead of applying",
    )
    p.add_argument("--source", default="image_dataset", help="Caption master dir")
    p.add_argument(
        "--manifest",
        default=None,
        help="Where to write the manifest (default: curated_manifest.json "
        "next to the report)",
    )
    p.add_argument("--apply", action="store_true", help="Write (default: dry run)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = resolve_path(args.source)

    if args.revert:
        manifest_path = resolve_path(args.revert)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        results = revert_curated(
            manifest["entries"], source_dir=source_dir, apply=args.apply
        )
        print(json.dumps(dict(Counter(r["status"] for r in results)), indent=2))
        for r in results:
            if r["status"] in ("drifted", "missing-caption"):
                print(f"  !! {r['status']}: {r['image']}")
        if not args.apply:
            print("\nDry run — re-run with --apply to revert.")
        else:
            print("\nReverted. Run `make preprocess-te` to re-encode.")
        return

    if not args.accept:
        print("either --accept <list> or --revert <manifest> is required")
        sys.exit(2)
    report_path = resolve_path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = (
        report["images"] if isinstance(report, dict) and "images" in report else report
    )
    accepted = {
        line.strip()
        for line in resolve_path(args.accept).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    manifest, unmatched = apply_curated(
        rows, accepted, source_dir=source_dir, apply=args.apply
    )
    print(json.dumps(dict(Counter(e["status"] for e in manifest)), indent=2))
    for e in manifest:
        mark = {"written": "->", "would-write": ".."}.get(e["status"], "!!")
        print(f"  {mark} [{e['verdict']}/{e['confidence']}] +{e['tag']}  {e['image']}")
    for image in unmatched:
        print(f"  !! not-in-report: {image}")

    manifest_path = (
        resolve_path(args.manifest)
        if args.manifest
        else report_path.parent / "curated_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            {"report": str(report_path), "applied": args.apply, "entries": manifest},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nmanifest: {manifest_path}")
    if not args.apply:
        print("Dry run — re-run with --apply to write.")
    else:
        print(
            "Written to the caption master. Run `make preprocess-te` to re-encode; "
            f"revert with: --revert {manifest_path} --apply"
        )


if __name__ == "__main__":
    main()
