"""Eyeball page for one or more `eval_text_prompt.py` runs.

    python bench/sam3_soft_prompt/sheet_html.py results/<run> [results/<run2> ...] \
        [--out results/compare.html] [--floor 0.3]

Writes one HTML page — the summary table across floors, then every prompt's
sheet per holdout image with its hit / FP counts at ``--floor``. Sheet images
are referenced by relative path, so the page has to stay beside the runs.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


def short(spec: str) -> str:
    return spec.split("/")[-2] if "/" in spec else spec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default=None, help="default: <first run>/../compare.html")
    p.add_argument("--floor", default="0.3")
    a = p.parse_args()

    runs = [Path(r).resolve() for r in a.runs]
    out = Path(a.out).resolve() if a.out else runs[0].parent / "compare.html"
    base = out.parent

    cols: list[
        tuple[str, str, dict, dict, Path]
    ] = []  # (run, spec, floors, per_img, sheetdir)
    for run in runs:
        res = json.loads((run / "result.json").read_text())["metrics"]["prompts"]
        per = json.loads((run / "per_image.json").read_text())
        for spec, floors in res.items():
            sdir = run / "sheets" / spec.replace("/", "_").replace(" ", "_")
            pi = {r["image"]: r for r in per.get(spec, [])}
            cols.append((run.name, spec, floors, pi, sdir))

    floors = list(cols[0][2].keys())
    images: dict[str, None] = {}
    for _, _, _, _, sdir in cols:
        if sdir.is_dir():
            for f in sorted(sdir.glob("*.jpg")):
                images[f.name[: -len(".jpg")].replace("__", "/", 1)] = None

    def rel(pth: Path) -> str:
        return html.escape(os.path.relpath(pth, base))

    h = [
        "<!doctype html><meta charset=utf-8><title>SAM3 soft prompt — text row sheets</title>",
        (
            "<style>body{font:14px system-ui;margin:16px;background:#111;color:#ddd}"
            "table{border-collapse:collapse}td,th{border:1px solid #333;padding:3px 8px;text-align:right}"
            "th{background:#222}td.l,th.l{text-align:left}.ok{color:#6f6}.bad{color:#f66}"
            ".row{display:flex;gap:8px;margin:18px 0;align-items:flex-start;flex-wrap:wrap}"
            ".cell{flex:1 1 45%;min-width:600px}.cell img{width:100%;display:block}"
            ".cap{font:12px monospace;color:#aaa;margin:2px 0}.miss{color:#666;padding:20px;border:1px dashed #333}"
            "h2{position:sticky;top:0;background:#111;padding:6px 0;margin:0;font-size:15px}</style>"
        ),
        "<h1>SAM3 soft prompt — text/SFX holdout</h1>",
        (
            "<p>Sheets: <b style='color:#f66'>left = MIT text mask (label)</b>, "
            "<b style='color:#6af'>right = soft-prompt mask</b>. Gate: px recall ≥ 0.9 @ ≤ 0.1 FP/img.</p>"
        ),
        (
            "<table><tr><th class=l>prompt</th><th>floor</th><th>box rec</th><th>px rec</th>"
            "<th>FP/img</th><th>neg FP rate</th><th>overmask</th><th>gate</th></tr>"
        ),
    ]
    for run, spec, fl, _, _ in cols:
        for f, m in fl.items():
            g = (
                "<span class=ok>PASS</span>"
                if m["gate"]
                else "<span class=bad>fail</span>"
            )
            h.append(
                f"<tr><td class=l>{html.escape(short(spec))}</td><td>{f}</td>"
                f"<td>{m['box_recall']:.3f}</td><td>{m['px_recall']:.3f}</td><td>{m['fp_per_img']:.2f}</td>"
                f"<td>{m['neg_img_fp_rate']:.3f}</td><td>{m['overmask_ratio']:.2f}</td><td>{g}</td></tr>"
            )
    h.append("</table>")
    h.append(
        f"<p>{len(images)} images (union of each prompt's worst-recall sheets); per-image counts at floor {a.floor}.</p>"
    )

    for img in images:
        h.append(f"<h2>{html.escape(img)}</h2><div class=row>")
        for run, spec, _, pi, sdir in cols:
            f = sdir / (img.replace("/", "__") + ".jpg")
            st = pi.get(img, {}).get(a.floor)
            cap = (
                f"targets {st['targets']} hit {st['hit']} fp {st['fp']} | px {st['hit_px']}/{st['mit_px']}"
                f" ({st['hit_px'] / max(1, st['mit_px']):.2f}) sam_px {st['sam_px']}"
                if st
                else "no per-image record"
            )
            h.append(
                f"<div class=cell><div class=cap>{html.escape(short(spec))} — {cap}</div>"
            )
            if f.exists():
                h.append(f"<img loading=lazy src='{rel(f)}'>")
            else:
                h.append("<div class=miss>not among this prompt's worst sheets</div>")
            h.append("</div>")
        h.append("</div>")

    out.write_text("\n".join(h), encoding="utf-8")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
