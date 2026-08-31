"""Contact sheet for an applied `caption-position` run: image / SAM segment / caption before-after.

What an already-landed run did to the captions, and whether the detector saw
what you think it saw. One row per image: the detection overlay, the
mask-blanked crops, and the master caption next to the derived one now on disk,
every clause tag marked moved / novel / duplicated.

The **after** side is read from disk (`--dst`), not re-proposed, so the sheet
shows what actually trains. Detection is re-run only to recover the (unpersisted)
crops and boxes, proposing from the master under `--src` — after one `--apply`
the derived caption carries clauses and `is_candidate` would reject the corpus.
A fresh proposal that disagrees with disk flags the row `drift`.

Read-only: writes only under `--out`, never a caption.

    make daemon-run ARGS="anime_tools/stages/cli/review_position_captions.py \\
        --path_pattern 'ama_mitsuki/*|ie_(raarami)/*'"
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw

from anime_tools import workspace as WS
from anime_tools._device import resolve_device
from anime_tools._env import resolve_path
from anime_tools._json import write_json
from anime_tools._walk import walk_images
from anime_tools.captions.position_clauses import parse_caption
from anime_tools.stages.cli._models import load_tagger
from anime_tools.stages.cli.position_captions import options_from_flag_string
from anime_tools.stages.multiview_sheet import BOX_COLORS, _fit, _font
from anime_tools.stages.position_captions import is_candidate, propose_for_image

PREVIEW = 900
CROP = 320


class PositionPalette:
    """One color per position phrase, allocated per row in first-seen order.

    Keyed by the *position name*, not an index: the overlay numbers boxes in
    detection order while the clauses sit in caption order. A drifted
    after-caption whose position no longer exists allocates the next color and
    visibly matches nothing, which is the honest rendering.
    """

    def __init__(self) -> None:
        self._map: dict[str, tuple[int, int, int]] = {}

    def __call__(self, position: str) -> tuple[int, int, int]:
        if position not in self._map:
            self._map[position] = BOX_COLORS[len(self._map) % len(BOX_COLORS)]
        return self._map[position]

    def known(self, position: str) -> tuple[int, int, int] | None:
        """The color already allocated to ``position``, without allocating."""
        return self._map.get(position)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="image_dataset", help="Caption master (before)")
    p.add_argument(
        "--dst",
        default=WS.RESIZED,
        help="Resized tree — images, and the derived caption (after) as applied",
    )
    p.add_argument("--path_pattern", dest="path_pattern", default="*")
    p.add_argument("--out", default=f"{WS.REPORTS}/position_review")
    # GOTCHA: pass as `--flags=--foo`, not `--flags --foo` — argparse reads a
    # `-`-leading value as the next option and dies on "expected one argument".
    p.add_argument(
        "--flags",
        default="",
        help="position_captions.py flags to re-derive the crops with (empty = "
        "shipped defaults, i.e. what the preprocess stage runs). Must be "
        "written --flags=--foo",
    )
    p.add_argument("--limit", type=int, default=0, help="0 = no cap")
    p.add_argument(
        "--skips",
        action="store_true",
        help="Also render pre-detection skips (single-subject etc.) as rows. "
        "Off by default — nothing was proposed and there is nothing to see",
    )
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    args = p.parse_args()
    args.device = resolve_device(args.device)
    return args


def flat_key_set(caption: str) -> set[str]:
    return {t.strip().lower() for t in parse_caption(caption).flat_tags if t.strip()}


def classify(tag: str, before_bag: set[str], after_bag: set[str]) -> str:
    """How a clause tag got there — read off the two captions, not the proposal.

    On-disk text rather than ``proposal.moved`` keeps the verdict true when the
    re-derivation drifts: ``moved`` = the bag gave it up, ``novel`` = a crop
    invention the master never carried, ``duplicated`` = still flat as well.
    """
    key = tag.strip().lower()
    if key not in before_bag:
        return "novel"
    return "duplicated" if key in after_bag else "moved"


def render_overlay(
    image: Image.Image,
    boxes: list[list[int]],
    colors: list[tuple[int, int, int]],
) -> Image.Image:
    """The full image with each detection numbered in its crop card's color."""
    preview = _fit(image.convert("RGB"), PREVIEW)
    scale = preview.width / image.width
    draw = ImageDraw.Draw(preview)
    font = _font(22, bold=True)
    for index, (box, color) in enumerate(zip(boxes, colors)):
        draw.rectangle([v * scale for v in box], outline=color, width=4)
        x1, y1 = box[0] * scale, box[1] * scale
        draw.rectangle([x1, y1, x1 + 30, y1 + 30], fill=color)
        draw.text((x1 + 9, y1 + 4), f"{index + 1}", font=font, fill=(20, 20, 24))
    return preview


def caption_html(
    caption: str,
    before_bag: set[str],
    after_bag: set[str],
    palette: PositionPalette,
    fallback_colors: list[tuple[int, int, int]] | None = None,
) -> str:
    """Render one caption, marking what each clause tag did to the flat bag.

    Clause colors are the box colors, matched by position name; when the palette
    does not know it, the i-th clause takes the i-th box's color, since a drifted
    caption still lists clauses in the reading order of the run that wrote it.
    """
    if not caption:
        return "<span class=dim>(empty)</span>"
    parsed = parse_caption(caption)
    parts = [f"<span class=bag>{html.escape(', '.join(parsed.flat_tags))}</span>"]
    for index, clause in enumerate(parsed.clauses):
        color = palette.known(clause.position)
        if color is None and fallback_colors and index < len(fallback_colors):
            color = fallback_colors[index]
        if color is None:
            color = palette(clause.position)
        tags = " ".join(
            f"<span class='t {classify(t, before_bag, after_bag)}'>{html.escape(t)}</span>"
            for t in clause.tags
        )
        swatch = "rgb({},{},{})".format(*color)
        tint = "rgba({},{},{},.09)".format(*color)
        parts.append(
            f"<div class=clause style='border-color:{swatch};background:{tint}'>"
            f"<b style='color:{swatch}'>{html.escape(clause.position)}</b> {tags}</div>"
        )
    return "".join(parts)


def write_index(out_dir: Path, rows: list[dict], summary: dict, pattern: str) -> Path:
    parts = [
        "<meta charset='utf-8'><title>position captions — review</title>",
        """<style>
body{background:#18181c;color:#e8e8ec;font:15px/1.55 system-ui,sans-serif;margin:0;padding:24px}
h1{font-size:21px;margin:0 0 4px}
h2{font-size:16px;margin:0 0 10px;font-weight:600}
section{border-top:1px solid #33333c;padding:22px 0}
.dim{color:#8a8a92;font-size:13px}
.sum{color:#8a8a92;font-size:13px;margin-bottom:8px}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;margin-left:8px;
 background:#2c2c34;color:#b8b8c0}
.pill.proposed{background:#1d3a2a;color:#7ad79c}
.pill.drift{background:#4a2a16;color:#ffb570}
.pill.skip{background:#3a2020;color:#ff9a9a}
.shots{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap;margin-bottom:14px}
.shots img{border-radius:5px;display:block}
.full img{max-width:min(560px,48vw);height:auto}
.crops{display:flex;gap:10px;flex-wrap:wrap}
figure{margin:0}
figcaption{font-size:12px;color:#b8b8c0;margin-top:4px}
.crops img{max-height:260px;width:auto;border:2px solid #33333c}
.caps{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.caps h3{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:#8a8a92;margin:0 0 6px}
.box{background:#1f1f25;border-radius:6px;padding:12px;font-size:13.5px}
.bag{color:#c8c8d0}
.clause{border-left:3px solid;padding:3px 8px;margin-top:8px;border-radius:0 4px 4px 0}
.t{display:inline-block;margin-right:6px}
.t.moved{color:#7ad79c}
.t.novel{color:#ffb570;text-decoration:underline dotted}
.t.duplicated{color:#8a8a92}
.legend span{margin-right:14px}
@media(max-width:1100px){.caps{grid-template-columns:1fr}}
</style>""",
        "<h1>position captions — review</h1>",
        f"<p class=sum><code>{html.escape(pattern)}</code> · "
        + " · ".join(f"{k} {v}" for k, v in summary.items())
        + "</p>",
        (
            "<p class='sum legend'>clause tag: "
            "<span class='t moved'>moved out of the bag (v2)</span>"
            "<span class='t novel'>novel — crop invention</span>"
            "<span class='t duplicated'>still flat too (v1 shape)</span></p>"
        ),
    ]

    def crop_card(i: int, c: dict) -> str:
        img_style = cap_style = ""
        if c.get("color"):
            swatch = "rgb({},{},{})".format(*c["color"])
            img_style = f" style='border-color:{swatch}'"
            cap_style = f" style='color:{swatch}'"
        return (
            f"<figure><img src='{html.escape(c['path'])}' loading=lazy{img_style}>"
            f"<figcaption{cap_style}>{i + 1} · {html.escape(c['position'])}"
            "</figcaption></figure>"
        )

    for row in rows:
        crops = "".join(crop_card(i, c) for i, c in enumerate(row["crops"]))
        overlay = (
            f"<figure class=full><img src='{html.escape(row['overlay'])}' loading=lazy>"
            f"<figcaption>{row['detected']} detection(s)</figcaption></figure>"
            if row.get("overlay")
            else ""
        )
        pill = f"<span class='pill {row['pill']}'>{html.escape(row['status'])}</span>"
        drift = (
            "<span class='pill drift'>drift vs re-derivation</span>"
            if row.get("drift")
            else ""
        )
        parts.append(
            f"<section><h2>{html.escape(row['image'])}{pill}{drift}</h2>"
            f"<div class=shots>{overlay}<div class=crops>{crops}</div></div>"
            "<div class=caps>"
            f"<div><h3>before — master</h3><div class=box>{row['before']}</div></div>"
            f"<div><h3>after — resized, on disk</h3><div class=box>{row['after']}</div></div>"
            "</div></section>"
        )
    path = out_dir / "index.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    src, dst = resolve_path(args.src), resolve_path(args.dst)
    out_dir = resolve_path(args.out)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    options, detect_args = options_from_flag_string(args.flags)
    detect_args.device = args.device

    from anime_tools.stages.cli.position_captions import build_detect_fn

    detect_fn, part_detect_fn, _model, _proc = build_detect_fn(detect_args)

    tagger, vocabulary, _ckpt = load_tagger(detect_args, quiet=True)

    images = walk_images(dst, recursive=True, pattern=args.path_pattern)
    if args.limit:
        images = images[: args.limit]
    print(f"{len(images)} image(s) under {args.path_pattern}")

    rows: list[dict] = []
    summary = {
        "seen": 0,
        "with clauses": 0,
        "proposed": 0,
        "drift": 0,
        "skipped": 0,
        "no caption": 0,
    }
    for index, image_path in enumerate(images, 1):
        summary["seen"] += 1
        rel = image_path.relative_to(dst)
        stem = rel.with_suffix("").as_posix().replace("/", "__")
        before_path = (src / rel).with_suffix(".txt")
        after_path = (dst / rel).with_suffix(".txt")
        if not before_path.exists():
            summary["no caption"] += 1
            continue
        before = before_path.read_text(encoding="utf-8").strip()
        after = (
            after_path.read_text(encoding="utf-8").strip()
            if after_path.exists()
            else ""
        )
        before_bag, after_bag = flat_key_set(before), flat_key_set(after)
        has_clauses = bool(parse_caption(after).clauses)
        if has_clauses:
            summary["with clauses"] += 1

        ok, reason = is_candidate(before)
        if not ok and not args.skips:
            summary["skipped"] += 1
            continue

        image = Image.open(image_path).convert("RGB")
        crops: list[dict] = []
        proposal = None
        if ok:

            def sink(
                i: int,
                position: str,
                crop: Image.Image,
                _stem=stem,
                _crops=crops,
            ) -> str:
                name = f"crops/{_stem}_{i}_{position.replace(' ', '-')}.png"
                _fit(crop, CROP).save(out_dir / name)
                _crops.append({"path": name, "position": position})
                return name

            proposal = propose_for_image(
                image,
                before,
                detect_fn=detect_fn,
                tag_fn=tagger.predict,
                vocabulary=vocabulary,
                options=options,
                part_detect_fn=part_detect_fn,
                crop_sink=sink,
            )
            status = proposal.status
        else:
            status = f"skip:{reason}"

        # Instances (reading order, matching the crop sink) carry the positions;
        # a gate-rejected image has none, so its raw detections take synthetic
        # keys that clause coloring can never collide with.
        palette = PositionPalette()
        if proposal is not None and proposal.instances:
            boxes = [inst.box for inst in proposal.instances]
            box_colors = [palette(inst.position) for inst in proposal.instances]
            for inst, crop in zip(proposal.instances, crops):
                crop["color"] = palette(inst.position)
        else:
            boxes = [d["box"] for d in proposal.detections] if proposal else []
            box_colors = [palette(f"#{i}") for i in range(len(boxes))]

        overlay_rel = ""
        if boxes:
            overlay_rel = f"overlays/{stem}.png"
            render_overlay(image, boxes, box_colors).save(out_dir / overlay_rel)

        drift = bool(
            proposal and proposal.proposed and after and proposal.proposed != after
        )
        if status == "proposed":
            summary["proposed"] += 1
        else:
            summary["skipped"] += 1
        if drift:
            summary["drift"] += 1

        rows.append(
            {
                "image": rel.as_posix(),
                "status": status,
                "pill": "proposed" if status == "proposed" else "skip",
                "detected": len(boxes),
                "overlay": overlay_rel,
                "crops": crops,
                "drift": drift,
                "has_clauses": has_clauses,
                # After first: its clause positions match the fresh instances,
                # so they reuse the box colors.
                "after": caption_html(
                    after, before_bag, after_bag, palette, box_colors
                ),
                "before": caption_html(
                    before, before_bag, after_bag, palette, box_colors
                ),
                "before_text": before,
                "after_text": after,
                "proposed_text": proposal.proposed if proposal else None,
            }
        )
        print(f"  [{index}/{len(images)}] {rel}  {status}{'  DRIFT' if drift else ''}")

    # Clause-carrying rows first: they are the ones with something to adjudicate.
    rows.sort(key=lambda r: (not r["has_clauses"], not r["drift"], r["image"]))
    write_json(
        out_dir / "report.json",
        {
            "pattern": args.path_pattern,
            "flags": args.flags,
            "summary": summary,
            "rows": [
                {k: v for k, v in r.items() if k not in {"before", "after"}}
                for r in rows
            ],
        },
    )
    index_path = write_index(out_dir, rows, summary, args.path_pattern)
    print(json.dumps(summary, indent=2))
    print(f"\nindex:  {index_path}\nreport: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
