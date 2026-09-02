"""Near-twin grouping on PE-Spatial features → workspace/groups/groups.json.

    python examples/grouping.py --home ~/data
    python examples/grouping.py --home ~/data --match-frac-min 0.4   # tighter

CLI: ``python -m anime_tools.grouping.cli.build_groups --source-dir workspace/resized``
(hyphenated flags; the underscore spelling is an alias). A curation tool, not a
training step: the GUI sidebar's *groups* ordering reads the manifest. Features
are cached under ``$NEAR_TWIN_CACHE`` (default ``~/.cache/near_twin/``) stamped
with each file's ``(size, mtime_ns)``, so re-tuning the thresholds is only the
matching pass. Weights: ``python -m anime_tools.downloads pe_spatial``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--home")
    p.add_argument("--source-dir", default=None, help="default: workspace/resized")
    p.add_argument("--match-frac-min", type=float, default=None)
    p.add_argument("--dry", action="store_true", help="print the request; run nothing")
    args = p.parse_args()
    if args.home:
        os.environ["ANIME_TOOLS_HOME"] = str(Path(args.home).expanduser().resolve())

    from anime_tools.grouping import GroupRequest

    kw = {}
    if args.source_dir:
        kw["source_dir"] = args.source_dir
    if args.match_frac_min is not None:
        kw["match_frac_min"] = args.match_frac_min
    # Two images group when >= match_frac_min of their pooled grid cells find a
    # distinctive mutual nearest neighbour above cell_match_min; components are
    # computed per top-level folder (artist). min_size=1 keeps singletons.
    req = GroupRequest(min_size=2, **kw)
    print("$ python -m anime_tools.grouping.cli.build_groups", *req.to_argv())

    # A different embedder is a dotted factory ``module:callable(device=...)``
    # returning anything that satisfies ``grouping.features.Embedder``:
    # ``batch [B,3,512,512] in [-1,1] → (cls [B,D] f32 L2-normed, grid16 [B,16,16,D] f16)``.
    # Switch embedders with a fresh $NEAR_TWIN_CACHE — the cache does not record
    # which one wrote it.
    print("custom embedder:", GroupRequest(embedder="my_pkg.embed:factory").to_argv())
    if args.dry:
        return

    from anime_tools._env import resolve_path
    from anime_tools.grouping import run_groups

    manifest = run_groups(req)  # also written to req.out

    # --- the manifest (MANIFEST_VERSION = 2) ----------------------------------
    # groups: [{id, artist, size, mean_cosine, members: [rel-posix, …]}, …]
    data = json.loads(resolve_path(req.out).read_text(encoding="utf-8"))
    assert data["version"] == manifest["version"]
    for g in data["groups"][:5]:
        print(
            f"group {g['id']} ({g['artist'] or '.'}, {g['size']} images, cos {g['mean_cosine']:.3f}):"
        )
        for rel in g["members"]:
            print("   ", rel)


if __name__ == "__main__":
    main()
