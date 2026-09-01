"""Mirror the caption master into bucket-resolution resized images.

Every later stage walks ``workspace/resized/``, so an image that exists only
under the master is invisible to all of them.

Two things are interop with the trainer's resize pass: the chosen ``(W, H)``
(same tier, band and solver) and the ``anima_resize_*`` PNG text keys
(:func:`_metadata_signature`), which its size-aware skip compares. Diverge on
either and each side re-encodes the other's PNGs.

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


def below_min_pixels(size: tuple[int, int], min_pixels: int) -> bool:
    """Would this image be skipped for being too small? ``min_pixels`` 0 = never.

    The floor is on the *source* pixels, and a skip means nothing lands in the
    resized tree — invisible to every other stage, not merely left out of
    training.
    """
    return min_pixels > 0 and size[0] * size[1] < min_pixels


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
    fat-fingered value still leaves a strip to resize rather than raising.
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

    ``img`` is the already-transposed, margin-cropped working region. This is
    the trainer's exact pixel geometry.
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
    too_small: list[str] = field(default_factory=list)
    """``"<path>: <W>x<H>"`` per skip — named rather than counted, since a
    skipped image is invisible to every downstream stage."""
    skipped_current: int = 0
    """A resized PNG already at the target bucket (pass ``overwrite`` to force)."""
    failed: int = 0
    failures: list[str] = field(default_factory=list)
    """``"<path>: <error>"`` per failure, so a bad file is reported, not silent."""
    buckets: dict[str, int] = field(default_factory=dict)
    """``"WxH" → count`` over every image that reached the resize step."""


def _collect_metadata(src: Image.Image) -> dict:
    """Save kwargs carrying through what ``convert("RGB")`` + ``save()`` drops:
    ICC profile, raw EXIF and PNG text chunks (the ComfyUI / A1111 generation
    prompt). Best-effort per field so a malformed chunk cannot kill a worker."""
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

    Empty for the default geometry, as the trainer writes it too, so its skip
    check falls back to the size alone.
    """
    if options.crop_anchor == DEFAULT_CROP_ANCHOR and not any(options.crop_margins):
        return {}
    return {
        _ANCHOR_KEY: options.crop_anchor,
        # Always empty: the trainer writes "" here too whenever it is unset.
        _BUCKET_RESOS_KEY: "",
        _MARGINS_KEY: ",".join(f"{m:g}" for m in options.crop_margins),
    }


def _oriented_size(src: Image.Image) -> tuple[int, int]:
    """``(W, H)`` as :func:`PIL.ImageOps.exif_transpose` would leave it, from the
    header alone — which is what lets an already-resized image be skipped
    without a decode.

    ``getexif()`` is consulted only when the file carries an EXIF block, because
    ``PngImageFile.getexif`` *loads the image* to look for one.
    """
    w, h = src.size
    if "exif" not in src.info:
        return w, h
    try:
        orientation = src.getexif().get(0x0112)  # ExifTags.Base.Orientation
    except Exception:  # noqa: BLE001 — malformed EXIF is just "no orientation"
        return w, h
    return (h, w) if orientation in (5, 6, 7, 8) else (w, h)


def _bucket_for_size(size: tuple[int, int], options: ResizeOptions) -> tuple[int, int]:
    """Target ``(W, H)`` for an already-oriented source size."""
    box = margin_box(size[0], size[1], options.crop_margins)
    _, bucket = select_bucket(
        box[2] - box[0],
        box[3] - box[1],
        options.target_res,
        max_ratio=options.max_ratio,
    )
    return bucket


def _is_current(
    out_path: Path, bucket: tuple[int, int], signature: dict[str, str]
) -> bool:
    """Is ``out_path`` already this image's resized PNG at the right geometry?"""
    try:
        with Image.open(out_path) as existing:
            if existing.size != bucket:
                return False
            if not signature:
                # Default geometry stamps no keys, so the size is the whole
                # answer — and reading ``.text`` off a PNG decodes it (tEXt/zTXt
                # may sit after IDAT).
                return True
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
    size_hint: tuple[int, int] | None = None,
) -> tuple[str, tuple[int, int], bool]:
    """Resize one image into ``out_dir / rel_dir / {stem}.png``.

    Module-level and taking only picklable arguments so it stays a
    ``ProcessPoolExecutor`` worker. Returns ``(name, bucket, skipped)``; unless
    ``overwrite`` is set an output already at the target bucket is left alone.

    Nothing above that skip may decode — not the source (``exif_transpose``
    copies, and a copy loads) nor the output (``.text`` on a PNG loads) — hence
    the header-only geometry via :func:`_oriented_size`, with the metadata read
    and the transpose *after* the skip. ``size_hint`` is that oriented size when
    the caller already read it.
    """
    target_dir = out_dir / rel_dir if rel_dir else out_dir
    out_path = target_dir / f"{image_path.stem}.png"
    signature = _metadata_signature(options)

    if not overwrite:
        if size_hint is None:
            try:
                with Image.open(image_path) as probe:
                    size_hint = _oriented_size(probe)
            except Exception:  # noqa: BLE001 — let the real open below report it
                size_hint = None
        if size_hint is not None:
            bucket = _bucket_for_size(size_hint, options)
            if _is_current(out_path, bucket, signature):
                return out_path.name, bucket, True

    with Image.open(image_path) as src:
        save_kwargs = _collect_metadata(src)
        img = ImageOps.exif_transpose(src)
        box = margin_box(*img.size, options.crop_margins)
        work_w, work_h = box[2] - box[0], box[3] - box[1]
        _, bucket = select_bucket(
            work_w, work_h, options.target_res, max_ratio=options.max_ratio
        )

        # Re-checked against the decoded size: the header estimate above can
        # only differ for a source whose orientation lives in XMP, not EXIF.
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


def _probe(
    image_path: Path,
) -> tuple[Path, tuple[int, int] | None, tuple[int, int] | None, str]:
    """``(path, raw size, oriented size, error)`` from the header alone.

    Module-level and picklable so :func:`_probe_sizes` can run it in the pool.
    """
    try:
        with Image.open(image_path) as im:
            return image_path, im.size, _oriented_size(im), ""
    except Exception as exc:  # noqa: BLE001 — one bad file must not abort the pass
        return image_path, None, None, str(exc)


def _probe_sizes(
    images: list[Path], workers: int
) -> list[tuple[Path, tuple[int, int] | None, tuple[int, int] | None, str]]:
    """Read every source header, in the pool — source order preserved.

    ``Image.open`` on a webp demuxes the whole file (~2 ms each), and it is pure
    per-file CPU with the GIL held inside the decoder, so only a process pool
    helps. ``map`` keeps input order.
    """
    if workers <= 1 or len(images) < 2:
        return [_probe(p) for p in images]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_probe, images, chunksize=16))


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

    Images below ``min_pixels`` are skipped rather than upscaled. This stage
    always writes (there is no dry run): it is idempotent, and an up-to-date
    output is skipped without a re-decode.
    """
    from anime_tools._walk import walk_images

    options = options or ResizeOptions()
    stats = ResizeStats()

    images = walk_images(src, recursive=recursive, pattern=path_pattern)
    stats.seen = len(images)

    # ``(path, oriented size)``: one header read per image, carried into
    # ``process_image`` so it need not re-open the file.
    pending: list[tuple[Path, tuple[int, int] | None]] = []
    for image_path, raw, size, error in _probe_sizes(images, workers):
        if error or raw is None:
            stats.failed += 1
            stats.failures.append(f"{image_path}: {error}")
            continue
        if below_min_pixels(raw, min_pixels):
            stats.skipped_small += 1
            stats.too_small.append(f"{image_path}: {raw[0]}x{raw[1]}")
            continue
        pending.append((image_path, size))

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
        (p, dst, options, _rel_dir_of(p, src), copy_captions, overwrite, size)
        for p, size in pending
    ]

    if workers <= 1:
        # Inline: one image, or a caller avoiding the ~1s pool spawn cost.
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
