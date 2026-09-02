"""OCR: what the picture *says*, as a sidecar beside nothing — its own tree.

    python examples/ocr.py --home ~/data              # dry run: report.json only
    python examples/ocr.py --home ~/data --apply      # write workspace/ocr/**/{stem}.ocr.txt
    python examples/ocr.py --image page.png           # one image, straight through the engine

PP-OCRv6 (``python -m anime_tools.downloads ppocr_det ppocr_rec``) runs on
onnxruntime, not torch. OCR is not a caption: nothing downstream encodes it, so
the stage reads and writes no caption and needs no re-encode afterwards.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--home")
    p.add_argument(
        "--image", help="read one image directly instead of running the stage"
    )
    p.add_argument("--apply", action="store_true")
    p.add_argument("--keep_en", action="store_true", help="keep ASCII-only lines too")
    args = p.parse_args()
    if args.home:
        os.environ["ANIME_TOOLS_HOME"] = str(Path(args.home).expanduser().resolve())

    if args.image:
        # --- the engine by itself ------------------------------------------
        from anime_tools.ocr import load_ocr, resolve_onnx_device

        engine = load_ocr(device=resolve_onnx_device(), skip_en=not args.keep_en)
        for line in engine.read(Path(args.image)):  # OcrLine, in reading order
            x0, y0, x1, y1 = line.box
            print(
                f"{line.seq:3d} ({x0},{y0})-({x1},{y1}) {line.score:.2f}  {line.text}"
            )
        return

    # --- the stage ----------------------------------------------------------
    from anime_tools.stages import OcrRequest, run_ocr

    req = OcrRequest(
        min_score=0.6, min_chars=3, skip_en=not args.keep_en, apply=args.apply
    )
    print("$ python -m anime_tools.stages.cli.ocr_captions", *req.to_argv())
    _rows, _stats = run_ocr(req)

    # --- reading the sidecars back -------------------------------------------
    # workspace/ocr/<rel>/{stem}.ocr.txt mirrors the resized tree; each row is
    # ``seq ⇥ x0,y0,x1,y1 ⇥ score ⇥ text``. read_ocr tolerates hand edits.
    if args.apply:
        from anime_tools._env import resolve_path
        from anime_tools.captions.ocr_sidecar import read_ocr

        for sidecar in sorted(resolve_path(req.ocr_dir).rglob("*.ocr.txt"))[:5]:
            print(sidecar.name, [line.text for line in read_ocr(sidecar)])


if __name__ == "__main__":
    main()
