"""Position clauses and the multiview audit: SAM3 + Anima Tagger over the resized tree.

    python examples/position_clauses.py --home ~/data            # dry run → report.json
    python examples/position_clauses.py --home ~/data --apply    # write the revised captions
    python examples/position_clauses.py --home ~/data --flatten  # text-only inverse pass

Both stages share one ``DetectionRequest`` (the SAM3 knobs) nested in their
request, so the detector is declared once. The position stage rewrites a
multi-subject caption so each attribute is asserted once, in its subject's
clause; the audit sweeps ``1girl`` captions for images that are really several
views of one girl. Rules, gates and knobs: ``docs/position_captions.md``,
``docs/multiview_audit.md``.

Needs the resized tree (``examples/stages.py`` or the Resize stage), the tagger
checkpoint, ``models/sam3/sam3.pt`` and the subject soft prompt
(``python -m anime_tools.downloads sam3 soft_prompt tagger``).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from anime_tools.stages import AuditRequest, DetectionRequest, PositionRequest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--home", help="curation home (default: env / CWD)")
    p.add_argument(
        "--path_pattern", default="*", help="glob relative to workspace/resized"
    )
    p.add_argument("--apply", action="store_true")
    p.add_argument("--flatten", action="store_true", help="inverse pass, no models")
    p.add_argument(
        "--audit", action="store_true", help="run the multiview audit instead"
    )
    p.add_argument(
        "--dry", action="store_true", help="only print the requests; run nothing"
    )
    args = p.parse_args()
    if args.home:
        import os

        os.environ["ANIME_TOOLS_HOME"] = str(Path(args.home).expanduser().resolve())

    # --- the detector both stages share --------------------------------------
    detection = DetectionRequest(
        score_threshold=0.5,
        # Tried only when the subject prompt undershoots: recovers headless
        # close-up panels (a hip / backside crop beside one full body).
        part_prompts=("buttocks", "hips", "thighs"),
    )

    # --- the position stage -------------------------------------------------
    position = PositionRequest(
        path_pattern=args.path_pattern,
        detection=detection,
        crops=True,  # export the mask-blanked crops beside the report (review aid)
        max_novel_tags=1,  # a clause may introduce one tag the caption never had
        apply=args.apply,
        flatten=args.flatten,
    )
    print("$ python -m anime_tools.stages.cli.position_captions", *position.to_argv())
    # options() builds the library's PositionCaptionOptions field by field from
    # the request, so a knob with no flag is an error rather than a silent default.
    opts = position.options()
    print(
        f"  min_instances={opts.min_instances} rewrite={opts.rewrite} bind_framing={opts.bind_framing}"
    )

    # --- the audit -------------------------------------------------------------
    audit = AuditRequest(
        path_pattern=args.path_pattern,
        detection=detection,
        apply_confidence=("strong",),  # a weak finding has only geometry behind it
        apply=args.apply,
    )
    print("$ python -m anime_tools.stages.cli.audit_multiview", *audit.to_argv())
    if args.dry:
        return

    from anime_tools.stages import run_audit, run_position

    if args.audit:
        run_audit(audit)
        return

    _rows, _stats = run_position(position)

    # --- what a dry run leaves behind, and replaying it -------------------------
    # report.json carries the before/after text per image. A later run with
    # from_report= writes exactly those proposals and loads no model; a caption
    # that changed in between is skipped as ``drifted`` rather than clobbered.
    if not (args.apply or args.flatten):
        from anime_tools._env import resolve_path

        report = resolve_path(position.report_dir) / "report.json"
        data = json.loads(report.read_text(encoding="utf-8"))
        proposed = [r for r in data["rows"] if r.get("proposed")]
        print(f"\n{len(proposed)} proposal(s) in {report}")
        for r in proposed[:3]:
            print(f"  {r['image']}\n    - {r['before']}\n    + {r['after']}")
        replay = replace(position, from_report=str(report), apply=True, crops=False)
        print(
            "\nto write them later:\n$ python -m anime_tools.stages.cli.position_captions",
            *replay.to_argv(),
        )
        print(
            "then re-encode the text embeddings (the trainer's `make preprocess-te`)."
        )


if __name__ == "__main__":
    main()
