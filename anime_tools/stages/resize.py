"""Mirror the caption master into bucket-resolution resized images.

The first stage of the pipeline and the one every later stage assumes has run:
``autotag`` / ``position`` / ``correct`` all walk ``workspace/resized/``
and tag the *resized* pixels, because that is the pixel data training sees. An
image that exists only under ``image_dataset/`` is invisible to all of them.

Ported from the trainer's ``library/preprocess/images.py`` (the walk → min-pixel
filter → parallel resize+crop → caption-mirror loop) against the free-fit
geometry in :mod:`anime_tools.buckets`. The port is deliberately faithful in the
two places that are *interop*, not taste:

* the chosen ``(W, H)`` — same tier, same band, same solver, so the trainer's
  ``make preprocess-resize`` finds every PNG already at its target bucket and
  skips it instead of re-encoding the dataset;
* the ``anima_resize_*`` PNG text keys (:func:`_metadata_signature`), which are
  what the trainer's size-aware skip compares when a non-default crop anchor or
  margin is in play.

Dropped on the way over, with no curation consumer: the vestigial
``resolution`` / ``min_bucket_reso`` / ``max_bucket_reso`` / ``bucket_reso_steps``
knobs (the discrete-bucket path they fed is gone), the ``fit_mode`` switch
(free-fit is the only mode), the ``bucket_resos`` allow-list (documented in the
trainer as accepted-but-never-branching), and ``curation_decisions`` (a trainer
GUI artifact).

Torch-free — PIL only.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

from anime_tools.buckets import (
    DEFAULT_FREEFIT_MAX_RATIO,
    DEFAULT_TARGET_RES,
    choose_edge,
    freefit_band_for_edge,
    freefit_bucket,
)

CAPTION_EXTENSIONS = (".txt", ".caption")

DEFAULT_MIN_PIXELS = 500_000
"""0.5MP. Below this an image cannot fill a 1024 tier without visible upscale."""

DEFAULT_CROP_ANCHOR = "center"
CROP_ANCHORS: dict[str, tuple[float, float]] = {
    "top_left": (0.0, 0.0),
    "top": (0.5, 0.0),
    "top_right": (1.0, 0.0),
    "left": (0.0, 0.5),
    "center": (0.5, 0.5),
    "right": (1.0, 0.5),
    "bottom_left": (0.0, 1.0),
    "bottom": (0.5, 1.0),
    "bottom_right": (1.0, 1.0),
}

MARGIN_SIDES = ("top", "right", "bottom", "left")

# PNG text keys, verbatim from the trainer — its skip check reads these.
_ANCHOR_KEY = "anima_resize_crop_anchor"
_BUCKET_RESOS_KEY = "anima_resize_bucket_resos"
_MARGINS_KEY = "anima_resize_crop_margins"


def normalize_target_res(target_res: Iterable[int] | int | str | None) -> list[int]:
    """Normalize a config/CLI ``target_res`` into a non-empty tier list."""
    if target_res is None:
        return list(DEFAULT_TARGET_RES)
    if isinstance(target_res, int):
        return [target_res]
    if isinstance(target_res, str):
        raw = target_res.strip()
        if not raw:
            return list(DEFAULT_TARGET_RES)
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    values = [int(value) for value in target_res]
    return values or list(DEFAULT_TARGET_RES)


def normalize_crop_anchor(crop_anchor: str | None) -> str:
    value = str(crop_anchor or DEFAULT_CROP_ANCHOR).strip().lower()
    return value if value in CROP_ANCHORS else DEFAULT_CROP_ANCHOR


def normalize_crop_margins(raw) -> tuple[float, float, float, float]:
    """Percent margins as ``(top, right, bottom, left)``, clamped to sane sums.

    Accepts a dict, a 4-sequence, a comma string or ``None``. Opposing margins
    that would eat 95% or more of an axis are scaled back to 95% together, so a
    fat-fingered ``--resize_crop_margins 50 0 60 0`` still leaves a strip to
    resize rather than raising.
    """
    if raw is None:
        margins: dict = {}
    elif isinstance(raw, dict):
        margins = raw
    elif isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        margins = dict(zip(MARGIN_SIDES, parts, strict=False))
    elif isinstance(raw, (list, tuple)):
        margins = dict(zip(MARGIN_SIDES, raw, strict=False))
    else:
        margins = {}

    out: dict[str, float] = {}
    for side in MARGIN_SIDES:
        try:
            out[side] = max(0.0, float(margins.get(side, 0.0)))
        except (TypeError, ValueError):
            out[side] = 0.0
    for a, b in (("left", "right"), ("top", "bottom")):
        total = out[a] + out[b]
        if total >= 95.0:
            out[a] *= 95.0 / total
            out[b] *= 95.0 / total
    return tuple(out[side] for side in MARGIN_SIDES)  # type: ignore[return-value]


def margin_box(
    width: int, height: int, crop_margins: tuple[float, float, float, float]
) -> tuple[int, int, int, int]:
    """The ``(left, top, right, bottom)`` pixel box left after percent margins."""
    top, right, bottom, left = crop_margins
    x0 = round(width * left / 100.0)
    y0 = round(height * top / 100.0)
    x1 = round(width - width * right / 100.0)
    y1 = round(height - height * bottom / 100.0)
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def select_bucket(
    width: int,
    height: int,
    target_res: Iterable[int] | int | str | None = None,
    *,
    max_ratio: float = DEFAULT_FREEFIT_MAX_RATIO,
) -> tuple[int, tuple[int, int]]:
    """``(tier_edge, (W, H))`` for a source size — the whole geometry decision."""
    edge = choose_edge(width, height, normalize_target_res(target_res))
    return edge, freefit_bucket(
        width, height, freefit_band_for_edge(edge), max_ratio=max_ratio
    )


def resize_to_bucket(
    img: Image.Image,
    bucket: tuple[int, int],
    *,
    crop_anchor: str = DEFAULT_CROP_ANCHOR,
) -> Image.Image:
    """Cover-scale ``img`` to ``bucket`` (LANCZOS) then anchor-crop to it.

    ``img`` is the already-transposed, margin-cropped working region; aspect is
    read from ``img.size``. Under free-fit the crop is sub-patch unless the
    ratio clamp fired, so the anchor rarely matters — but it is the trainer's
    exact pixel geometry, which is what makes the two outputs interchangeable.
    """
    bw, bh = bucket
    anchor_x, anchor_y = CROP_ANCHORS[normalize_crop_anchor(crop_anchor)]
    w, h = img.size
    if w / h > bw / bh:
        new_h, new_w = bh, round(bh * w / h)
    else:
        new_w, new_h = bw, round(bw * h / w)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = round((new_w - bw) * anchor_x)
    top = round((new_h - bh) * anchor_y)
    return img.crop((left, top, left + bw, top + bh))


@dataclass(frozen=True)
class ResizeOptions:
    """Geometry knobs for one resize pass — picklable, so workers take it whole."""

    target_res: tuple[int, ...] = DEFAULT_TARGET_RES
    crop_anchor: str = DEFAULT_CROP_ANCHOR
    crop_margins: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    max_ratio: float = DEFAULT_FREEFIT_MAX_RATIO

    @classmethod
    def build(
        cls,
        *,
        target_res=None,
        crop_anchor: str | None = None,
        crop_margins=None,
        max_ratio: float = DEFAULT_FREEFIT_MAX_RATIO,
    ) -> ResizeOptions:
        """Normalize raw CLI/config values into the frozen option set."""
        return cls(
            target_res=tuple(normalize_target_res(target_res)),
            crop_anchor=normalize_crop_anchor(crop_anchor),
            crop_margins=normalize_crop_margins(crop_margins),
            max_ratio=float(max_ratio),
        )


@dataclass
class ResizeStats:
    seen: int = 0
    """Images enumerated under the source root, before any filter."""
    written: int = 0
    skipped_small: int = 0
    """Below ``min_pixels``."""
    skipped_current: int = 0
    """A resized PNG already at the target bucket (pass ``overwrite`` to force)."""
    failed: int = 0
    failures: list[str] = field(default_factory=list)
    """``"<path>: <error>"`` per failure, so a bad file is reported, not silent."""
    buckets: dict[str, int] = field(default_factory=dict)
    """``"WxH" → count`` over every image that reached the resize step."""


def _collect_metadata(src: Image.Image) -> dict:
    """Save kwargs carrying through what ``convert("RGB")`` + ``save()`` drops.

    ICC profile, raw EXIF and PNG text chunks (where ComfyUI / A1111 stash the
    generation prompt). Best-effort per field so a malformed chunk cannot kill a
    worker.
    """
    out: dict = {}
    if icc := src.info.get("icc_profile"):
        out["icc_profile"] = icc
    if exif := src.info.get("exif"):
        out["exif"] = exif
    if text_chunks := getattr(src, "text", None):
        pnginfo = PngInfo()
        for key, value in text_chunks.items():
            try:
                pnginfo.add_text(key, str(value))
            except Exception:  # noqa: BLE001,S112 — passenger metadata, never fatal
                continue
        out["pnginfo"] = pnginfo
    return out


def _metadata_signature(options: ResizeOptions) -> dict[str, str]:
    """The ``anima_resize_*`` keys stamped into the PNG, trainer-compatible.

    Empty for the default geometry — which is what the trainer writes too, so a
    default-geometry PNG from either side is byte-comparable and its skip check
    falls back to the size alone. A non-default anchor or margin stamps the same
    three keys the trainer stamps, in its format, so changing either re-resizes
    on both sides.
    """
    if options.crop_anchor == DEFAULT_CROP_ANCHOR and not any(options.crop_margins):
        return {}
    return {
        _ANCHOR_KEY: options.crop_anchor,
        # Always empty: the bucket_resos allow-list is not ported. The trainer
        # writes "" here too whenever it is unset, which is always in practice.
        _BUCKET_RESOS_KEY: "",
        _MARGINS_KEY: ",".join(f"{m:g}" for m in options.crop_margins),
    }


def _is_current(
    out_path: Path, bucket: tuple[int, int], signature: dict[str, str]
) -> bool:
    """Is ``out_path`` already this image's resized PNG at the right geometry?"""
    try:
        with Image.open(out_path) as existing:
            if existing.size != bucket:
                return False
            text = getattr(existing, "text", {}) or {}
            return all(text.get(k) == v for k, v in signature.items())
    except Exception:  # noqa: BLE001 — a missing/corrupt output is just "not current"
        return False


def process_image(
    image_path: Path,
    out_dir: Path,
    options: ResizeOptions,
    rel_dir: str = "",
    copy_captions: bool = False,
    overwrite: bool = False,
) -> tuple[str, tuple[int, int], bool]:
    """Resize one image into ``out_dir / rel_dir / {stem}.png``.

    Module-level and taking only picklable arguments so it stays a
    ``ProcessPoolExecutor`` worker. Returns ``(name, bucket, skipped)``; unless
    ``overwrite`` is set an output already at the target bucket is left alone,
    so a re-run is near-free and a tier change re-resizes only what moved.
    """
    target_dir = out_dir / rel_dir if rel_dir else out_dir
    out_path = target_dir / f"{image_path.stem}.png"
    signature = _metadata_signature(options)

    with Image.open(image_path) as src:
        save_kwargs = _collect_metadata(src)
        img = ImageOps.exif_transpose(src)
        box = margin_box(*img.size, options.crop_margins)
        work_w, work_h = box[2] - box[0], box[3] - box[1]
        _, bucket = select_bucket(
            work_w, work_h, options.target_res, max_ratio=options.max_ratio
        )

        if not overwrite and _is_current(out_path, bucket, signature):
            return out_path.name, bucket, True

        if signature:
            pnginfo = save_kwargs.setdefault("pnginfo", PngInfo())
            for key, value in signature.items():
                pnginfo.add_text(key, value)

        out_img = resize_to_bucket(
            img.convert("RGB").crop(box), bucket, crop_anchor=options.crop_anchor
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    # compress_level=1: resized PNGs are an intermediate the VAE latent step
    # re-reads, so trade slightly larger files for a much faster zlib encode.
    out_img.save(out_path, format="PNG", compress_level=1, **save_kwargs)

    if copy_captions:
        for ext in CAPTION_EXTENSIONS:
            sidecar = image_path.with_suffix(ext)
            if sidecar.exists():
                shutil.copy2(sidecar, target_dir / f"{image_path.stem}{ext}")

    return out_path.name, bucket, False


def _rel_dir_of(image_path: Path, src: Path) -> str:
    try:
        rel = str(image_path.parent.relative_to(src))
    except ValueError:
        return ""
    return "" if rel == "." else rel


def run_resize_images(
    *,
    src: Path,
    dst: Path,
    options: ResizeOptions | None = None,
    path_pattern: str | None = None,
    recursive: bool = True,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    copy_captions: bool = False,
    overwrite: bool = False,
    workers: int = 4,
    progress: Callable[[int, int, str], None] | None = None,
) -> ResizeStats:
    """Resize every image under ``src`` into ``dst``, mirroring the subdir layout.

    ``path_pattern`` is the usual ``|``-OR fnmatch glob on the path relative to
    ``src``. Images below ``min_pixels`` are skipped rather than upscaled. This
    stage always writes (there is no dry run): it is idempotent, and an
    up-to-date output is skipped without a re-decode.
    """
    from anime_tools._walk import walk_images

    options = options or ResizeOptions()
    stats = ResizeStats()

    images = walk_images(src, recursive=recursive, pattern=path_pattern)
    stats.seen = len(images)

    pending: list[Path] = []
    for image_path in images:
        if min_pixels <= 0:
            pending.append(image_path)
            continue
        try:
            with Image.open(image_path) as im:
                width, height = im.size
        except Exception as exc:  # noqa: BLE001 — one bad file must not abort the pass
            stats.failed += 1
            stats.failures.append(f"{image_path}: {exc}")
            continue
        if width * height < min_pixels:
            stats.skipped_small += 1
            continue
        pending.append(image_path)

    dst.mkdir(parents=True, exist_ok=True)
    total = len(pending)

    def _record(index: int, result: tuple[str, tuple[int, int], bool]) -> None:
        name, bucket, skipped = result
        key = f"{bucket[0]}x{bucket[1]}"
        stats.buckets[key] = stats.buckets.get(key, 0) + 1
        if skipped:
            stats.skipped_current += 1
        else:
            stats.written += 1
        if progress is not None:
            progress(index, total, f"{name} {'skip' if skipped else '→ ' + key}")

    args = [
        (p, dst, options, _rel_dir_of(p, src), copy_captions, overwrite)
        for p in pending
    ]

    if workers <= 1:
        # Inline: one image, or a caller that does not want the ~1s spawn cost
        # of a pool it would use once (the GUI's per-image Run takes this path).
        for index, call in enumerate(args, 1):
            try:
                _record(index, process_image(*call))
            except Exception as exc:  # noqa: BLE001 — see above
                stats.failed += 1
                stats.failures.append(f"{call[0]}: {exc}")
        return stats

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_image, *call): call[0] for call in args}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                _record(index, future.result())
            except Exception as exc:  # noqa: BLE001 — see above
                stats.failed += 1
                stats.failures.append(f"{futures[future]}: {exc}")
    return stats
