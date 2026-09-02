"""The caption stages as request objects: resize → correct → export.

    python examples/stages.py                 # on a throwaway sandbox dataset
    python examples/stages.py --home ~/data   # on your own curation home
    python examples/stages.py --autotag       # also run the Anima Tagger stage

Every stage is a frozen dataclass in ``anime_tools.stages`` (``ResizeRequest``,
``AutotagRequest``, ``PositionRequest``, ``CorrectRequest``, ``OcrRequest``,
``AuditRequest``, ``ExportRequest``) run by ``run_<stage>(req)``. The CLI in
``anime_tools.stages.cli.<stage>`` is a shell over the same object: its parser
is generated from the class, so ``req.to_argv()`` is the exact command line and
``Request.from_argv()`` reads one back. Stages that propose (autotag, position,
audit, OCR, export) are **dry-run by default** and write ``report.json``;
``apply=True`` writes for real. Resize and correct always write.

Where things land (``anime_tools/workspace/__init__.py``)::

    image_dataset/          the hand-written master — read-only for the stages
    workspace/resized/      every stage reads this tree; the revised caption is written here
    workspace/captions/     one report.json per stage
    post_image_dataset/     what Export publishes — the only write outside workspace/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from _sandbox import home_from_args


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--home", help="curation home (default: a fresh sandbox)")
    p.add_argument("--autotag", action="store_true", help="also run the tagger stage")
    args = p.parse_args()
    home = home_from_args(args.home)
    print(f"curation home: {home}\n")

    # Imports after the home is set only for tidiness: requests hold relative
    # paths and every run_* resolves them against the home at call time.
    from anime_tools.stages import (
        AutotagRequest,
        CorrectRequest,
        ExportRequest,
        ResizeRequest,
        run_correct,
        run_export,
        run_resize,
    )

    # --- 1. resize: the master → workspace/resized/ ------------------------
    # Every other stage opens the resized tree, so this runs first (the GUI
    # runs it as an automatic preflight). Free-fit buckets match the trainer's.
    resize = ResizeRequest(target_res=(1024,), workers=1)
    print("$ python -m anime_tools.stages.cli.resize_images", *resize.to_argv())
    run_resize(resize)

    # --- 2. correct: mirror + bucket-order the caption, write sidecars -----
    # Reads the master caption, writes the revised one beside the resized
    # image, plus {stem}.variants.txt when asked. Needs the Danbooru KB.
    correct = CorrectRequest(
        src="image_dataset",
        dst="workspace/resized",
        recursive=True,
        caption_insert_no_artist=True,
        caption_drop_groups="lighting",
        caption_shuffle_variants=3,
        caption_tag_dropout_rate=0.1,
    )
    print("\n$ python -m anime_tools.stages.cli.correct_captions", *correct.to_argv())
    run_correct(correct)
    for txt in sorted((home / "workspace/resized").rglob("*.txt")):
        if ".variants" not in txt.name:
            print(
                f"  {txt.relative_to(home)}: {txt.read_text(encoding='utf-8').strip()}"
            )

    # --- 3. autotag (optional: loads the tagger) -----------------------------
    # ``mode``: missing (default, only images no caption speaks for) / merge
    # (append novel tags, clauses kept) / overwrite (replaced text → history).
    autotag = AutotagRequest(mode="merge", min_confidence=0.35, apply=True)
    print("\n$ python -m anime_tools.stages.cli.autotag_captions", *autotag.to_argv())
    if args.autotag:
        from anime_tools.stages import run_autotag

        _rows, stats = run_autotag(autotag)
        print(f"  autotag: {stats.proposed} proposed, {stats.written} written")
    else:
        print("  (skipped; pass --autotag to load the tagger)")

    # --- 4. export: dry run, read the report, then apply ---------------------
    export = ExportRequest()
    print("\n$ python -m anime_tools.stages.cli.export_workspace", *export.to_argv())
    _rows, _stats = run_export(export)  # apply=False: nothing copied
    report = json.loads((home / "workspace/captions/export/report.json").read_text())
    kinds = sorted({r["kind"] for r in report["rows"]})
    print(f"  dry run planned {len(report['rows'])} rows of kinds {kinds}")

    # The same request with apply=True copies. Frozen dataclass → replace().
    from dataclasses import replace

    run_export(replace(export, apply=True))
    published = sorted(
        p.relative_to(home) for p in (home / "post_image_dataset").rglob("*.txt")
    )
    print("  published captions:", [str(p) for p in published])

    # --- 5. the same stage as a subprocess ------------------------------------
    # to_argv() names only what differs from the defaults, so a saved command
    # line stays short. This is how the GUI and the trainer's daemon run stages.
    cmd = [
        sys.executable,
        "-m",
        "anime_tools.stages.cli.export_workspace",
        *export.to_argv(),
    ]
    print("\n$", " ".join(cmd))
    out = subprocess.run(
        cmd, capture_output=True, text=True, cwd=home, check=True
    ).stdout
    print("  " + out.strip().splitlines()[-1])

    # And back: a CLI argv is the request it names, validation included.
    print(
        "\nfrom_argv:",
        ExportRequest.from_argv(argv=["--apply", "--path_pattern", "char_a/*"]),
    )
    print(f"\nreports: {home / 'workspace/captions'}")


if __name__ == "__main__":
    main()
