"""Export a checkpoint's dbv4 backbone to ``<ckpt_dir>/dbv4.onnx``.

    uv sync --group export     # onnx + onnxscript, the exporter's own dependencies
    python -m anime_tools.tagger.cli.export_onnx

Once the file is there every :class:`~anime_tools.tagger.tagger.AnimaTagger` on that
checkpoint runs on onnxruntime instead of timm — autotag, position clauses, the
multiview audit, the GUI, the ComfyUI node. ``ANIMA_TAGGER_BACKEND=torch`` turns it
back off without deleting the graph.

``--verify`` runs one image through both runtimes and prints the largest
disagreement, which is the only check that matters after an export: the sidecar and
every threshold were fitted against the torch numbers.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from anime_tools._env import resolve_path
from anime_tools.tagger.dbv4_meta import DEFAULT_TAGGER_DIR, dbv4_onnx_path
from anime_tools.tagger.onnx_export import DEFAULT_OPSET, export_for_checkpoint

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--ckpt_dir",
        default=DEFAULT_TAGGER_DIR,
        help="tagger checkpoint dir; the graph is written into it as dbv4.onnx",
    )
    p.add_argument(
        "--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset version"
    )
    p.add_argument(
        "--overwrite", action="store_true", help="rebuild an existing dbv4.onnx"
    )
    p.add_argument(
        "--verify",
        metavar="IMAGE",
        default=None,
        help="after the export, run this image through both runtimes and report "
        "the largest per-tag score difference",
    )
    return p.parse_args()


def _verify(ckpt_dir: Path, image_path: Path) -> None:
    """Same image through both runtimes: speed, and the two disagreements.

    The graph is a float32 export, so *float32 torch is its reference* — that
    difference is the export's own error and should be ~1e-5. The difference
    against the torch backend's default dtype is a separate number, and on a GPU
    (bfloat16) it is the dtype's error, not the graph's.
    """
    from PIL import Image

    from anime_tools.tagger.tagger import AnimaTagger

    image = Image.open(image_path).convert("RGB")
    runs: dict[str, dict] = {}

    def run(label: str, **kwargs) -> AnimaTagger:
        tagger = AnimaTagger(ckpt_dir, device="cpu", **kwargs)
        tagger.predict(image)  # warm up: the first call loads the weights
        t0 = time.time()
        runs[label] = out = tagger.predict(image)
        logger.info(
            "  %-9s %5.2f s/img  %3d tags kept  rating=%s",
            label,
            time.time() - t0,
            len(out["kept"]),
            out["rating"],
        )
        return tagger

    torch_run = run("torch", backend="torch")
    if torch_run.dtype is torch.float32:
        # Already the reference — on CPU that is the default now, and a second
        # identical pass would only cost 1.5 s to print the same numbers.
        runs["torch/f32"] = runs["torch"]
    else:
        run("torch/f32", backend="torch", dtype=torch.float32)
    run("onnx", backend="onnx")

    def worst(a: str, b: str) -> tuple[float, str]:
        sa, sb = runs[a]["scores"], runs[b]["scores"]
        return max(((abs(sa[k] - sb[k]), k) for k in sa), default=(0.0, "-"))

    d, tag = worst("torch/f32", "onnx")
    logger.info("  export error   (onnx vs torch/f32): %.2e on %r", d, tag)
    d, tag = worst("torch", "onnx")
    logger.info("  dtype error    (onnx vs torch):     %.2e on %r", d, tag)
    kept = {label: set(out["kept"]) for label, out in runs.items()}
    if kept["onnx"] != kept["torch/f32"]:
        logger.warning(
            "  emitted tags differ: onnx-only %s, torch/f32-only %s",
            sorted(kept["onnx"] - kept["torch/f32"]),
            sorted(kept["torch/f32"] - kept["onnx"]),
        )


# The exporter's optimiser logs every rewritten node and folded initializer at INFO
# — several thousand lines for one caformer, around the four that say what happened.
NOISY_LOGGERS = ("onnxscript", "onnx_ir", "onnx")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    args = parse_args()
    ckpt_dir = Path(resolve_path(args.ckpt_dir))
    if not (ckpt_dir / "config.json").is_file():
        raise SystemExit(
            f"no tagger checkpoint at {ckpt_dir} — "
            "`python -m anime_tools.downloads tagger`"
        )
    out = dbv4_onnx_path(ckpt_dir)
    # Idempotent: an existing graph is reported, not an error, so `--verify` on
    # yesterday's export is one command and not a rebuild.
    if out.is_file() and not args.overwrite:
        logger.info(
            "%s already exported (%.0f MB) — --overwrite to rebuild it",
            out,
            out.stat().st_size / 1e6,
        )
    else:
        try:
            t0 = time.time()
            out = export_for_checkpoint(ckpt_dir, opset=args.opset, overwrite=True)
        except ImportError as exc:
            raise SystemExit(
                f"{exc}\nThe exporter needs onnx + onnxscript, which are not in the "
                "default sync (running the exported graph does not need them): "
                "`uv sync --group export`"
            ) from exc
        logger.info(
            "wrote %s (%.0f MB) in %.0f s",
            out,
            out.stat().st_size / 1e6,
            time.time() - t0,
        )
    if args.verify:
        _verify(ckpt_dir, Path(resolve_path(args.verify)))
    logger.info(
        "%s now runs on onnxruntime; ANIMA_TAGGER_BACKEND=torch to opt out",
        dbv4_onnx_path(ckpt_dir).parent,
    )


if __name__ == "__main__":
    main()
