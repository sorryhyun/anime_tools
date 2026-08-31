"""Resize stage: free-fit geometry, the mirror layout, and the idempotent skip.

The geometry assertions are the interop contract with the trainer's
``make preprocess-resize`` — if a number here moves, resized PNGs written by one
side stop being skipped by the other and the whole dataset re-encodes.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from PIL import Image

from anime_tools.buckets import (
    ALLOWED_TARGET_RES,
    EDGE_TOKEN_BANDS,
    choose_edge,
    freefit_band_for_edge,
    freefit_bucket,
)
from anime_tools.stages.resize import (
    ResizeOptions,
    margin_box,
    normalize_crop_margins,
    normalize_target_res,
    resize_to_bucket,
    run_resize_images,
    select_bucket,
)


def _write_image(path, size=(1600, 900), color=(200, 120, 60)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


# --------------------------------------------------------------------------- #
# geometry


def test_exact_grid_aspect_lands_with_zero_crop():
    # 1008x1024 is exactly 63x64 patches = 4032 tokens, the 1024 band's floor.
    assert select_bucket(1008, 1024, [1024]) == (1024, (1008, 1024))


def test_bucket_preserves_aspect_to_within_one_patch():
    for width, height in ((1600, 900), (900, 1600), (704, 396), (1440, 2560)):
        _, (bw, bh) = select_bucket(width, height, [1024])
        assert abs(bw / bh - width / height) < 16 / min(bw, bh)


def test_bucket_token_count_stays_inside_the_tier_band():
    for edge in ALLOWED_TARGET_RES:
        lo, hi = freefit_band_for_edge(edge)
        bw, bh = freefit_bucket(1600, 900, (lo, hi))
        assert lo <= (bw // 16) * (bh // 16) <= hi
        # The band only ever widens the tier's natural range, never narrows it.
        natural_lo, natural_hi = EDGE_TOKEN_BANDS[edge]
        assert lo <= natural_lo and hi >= natural_hi


def test_bucket_dims_are_patch_multiples_under_the_rope_cap():
    bw, bh = freefit_bucket(3000, 1000, freefit_band_for_edge(1536))
    assert bw % 16 == 0 and bh % 16 == 0
    assert max(bw // 16, bh // 16) <= 256


def test_choose_edge_picks_the_tier_that_resizes_least():
    # ~0.95MP stays at 1024 (a small upscale) rather than dropping to 768.
    assert choose_edge(1200, 800, [768, 1024]) == 1024
    assert choose_edge(700, 500, [768, 1024]) == 768
    # Single-tier is a no-op whatever the size.
    assert choose_edge(8000, 10, [1024]) == 1024


def test_extreme_aspect_is_clamped_by_max_ratio():
    _, (bw, bh) = select_bucket(6000, 500, [1024], max_ratio=4.0)
    assert bw / bh == pytest.approx(4.0, abs=0.05)


def test_resize_to_bucket_returns_exactly_the_bucket_size():
    out = resize_to_bucket(Image.new("RGB", (1600, 900)), (1360, 768))
    assert out.size == (1360, 768)


def test_crop_anchor_moves_the_kept_region():
    # A 6:1 source is clamped to 4:1, so there is a real horizontal crop to
    # anchor. The left edge column identifies which slice survived.
    src = Image.new("RGB", (1200, 200))
    for x in range(1200):
        for y in range(0, 200, 40):
            src.putpixel((x, y), (x % 256, 0, 0))
    left = resize_to_bucket(src, (800, 200), crop_anchor="left")
    right = resize_to_bucket(src, (800, 200), crop_anchor="right")
    assert left.getpixel((0, 0)) != right.getpixel((0, 0))


def test_normalize_target_res_accepts_config_shapes():
    assert normalize_target_res("768, 1024") == [768, 1024]
    assert normalize_target_res(1024) == [1024]
    assert normalize_target_res([896, 1024]) == [896, 1024]
    assert normalize_target_res(None) == [1024]
    assert normalize_target_res([]) == [1024]


def test_unknown_tier_is_a_clear_error():
    with pytest.raises(ValueError, match="not in allowed tiers"):
        select_bucket(1000, 1000, [999])


def test_crop_margins_normalize_and_clamp():
    assert normalize_crop_margins([10, 5, 10, 5]) == (10.0, 5.0, 10.0, 5.0)
    assert normalize_crop_margins("10,5,10,5") == (10.0, 5.0, 10.0, 5.0)
    assert normalize_crop_margins({"top": 10}) == (10.0, 0.0, 0.0, 0.0)
    assert normalize_crop_margins(None) == (0.0, 0.0, 0.0, 0.0)
    assert normalize_crop_margins("nonsense") == (0.0, 0.0, 0.0, 0.0)
    # Opposing margins that would eat the axis are scaled back together to 95%.
    top, _, bottom, _ = normalize_crop_margins([50, 0, 60, 0])
    assert top + bottom == pytest.approx(95.0)


def test_margin_box_cuts_the_requested_percentages():
    assert margin_box(1000, 1000, (10.0, 20.0, 10.0, 0.0)) == (0, 100, 800, 900)
    # A degenerate box still leaves at least one pixel rather than raising.
    x0, y0, x1, y1 = margin_box(100, 100, (95.0, 0.0, 0.0, 0.0))
    assert x1 > x0 and y1 > y0


# --------------------------------------------------------------------------- #
# the pass


def test_mirrors_subdirs_and_writes_png_at_the_bucket_size(tmp_path):
    src, dst = tmp_path / "master", tmp_path / "resized"
    _write_image(src / "char_aki" / "0001.png", (1600, 900))
    _write_image(src / "flat.webp", (1600, 900))

    stats = run_resize_images(src=src, dst=dst, workers=1)

    assert stats.seen == 2 and stats.written == 2 and stats.failed == 0
    out = dst / "char_aki" / "0001.png"
    assert out.is_file()
    # Always PNG, whatever the source format.
    assert (dst / "flat.png").is_file()
    with Image.open(out) as im:
        assert im.size == select_bucket(1600, 900, [1024])[1]


def test_rerun_skips_outputs_already_at_their_bucket(tmp_path):
    src, dst = tmp_path / "master", tmp_path / "resized"
    _write_image(src / "0001.png", (1600, 900))

    run_resize_images(src=src, dst=dst, workers=1)
    stamp = (dst / "0001.png").stat().st_mtime_ns
    again = run_resize_images(src=src, dst=dst, workers=1)

    assert again.written == 0 and again.skipped_current == 1
    assert (dst / "0001.png").stat().st_mtime_ns == stamp
    # ...and the bucket histogram still counts it.
    assert sum(again.buckets.values()) == 1


def test_a_settled_rerun_never_decodes_the_source(tmp_path, monkeypatch):
    """The skip is decided from headers alone — nothing above it may decode.

    This is the load-bearing half of "a re-run is near-free": the GUI runs this
    pass as a preflight in front of every stage that opens an image, so on a
    settled dataset almost every call ends at the skip. Deciding *not* to write
    used to cost a full decode of the source (``exif_transpose`` copies, and a
    copy loads), which is ~100 ms an image — a minute and a half of nothing on a
    3k-image master, before every batch run.
    """
    from anime_tools.stages import resize as R

    src, dst = tmp_path / "master", tmp_path / "resized"
    _write_image(src / "0001.png", (1600, 900))
    run_resize_images(src=src, dst=dst, workers=1)

    def _explode(_img):
        raise AssertionError("the skip path decoded the source image")

    monkeypatch.setattr(R.ImageOps, "exif_transpose", _explode)
    stats = run_resize_images(src=src, dst=dst, workers=1)

    assert (stats.written, stats.skipped_current) == (0, 1)


def test_a_changed_source_still_re_resizes(tmp_path):
    # The header-only skip carries a *size hint* into the worker; a source
    # rewritten under the same name must still land on its new bucket.
    src, dst = tmp_path / "master", tmp_path / "resized"
    _write_image(src / "0001.png", (1600, 900))
    run_resize_images(src=src, dst=dst, workers=1)

    _write_image(src / "0001.png", (900, 1600))
    stats = run_resize_images(src=src, dst=dst, workers=1)

    assert stats.written == 1
    with Image.open(dst / "0001.png") as im:
        assert im.size == select_bucket(900, 1600)[1]


def test_exif_rotation_is_read_off_the_header(tmp_path):
    # The one thing orientation does to a size is swap it, so the skip check
    # can read it from the header — but it has to actually do so, or a rotated
    # image would be measured on the wrong aspect and land on the wrong bucket.
    from anime_tools.stages.resize import _oriented_size

    src, dst = tmp_path / "master", tmp_path / "resized"
    src.mkdir(parents=True)
    img = Image.new("RGB", (1600, 900), (30, 60, 90))
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90° CW → the displayed image is 900x1600
    img.save(src / "0001.jpg", exif=exif)

    with Image.open(src / "0001.jpg") as raw:
        assert _oriented_size(raw) == (900, 1600)

    stats = run_resize_images(src=src, dst=dst, workers=1)

    assert stats.written == 1
    with Image.open(dst / "0001.png") as out:
        assert out.size == select_bucket(900, 1600)[1]
    # ...and it is skipped on the next pass, i.e. the header estimate agreed
    # with the geometry the decoded path chose.
    assert run_resize_images(src=src, dst=dst, workers=1).skipped_current == 1


def test_overwrite_forces_a_rewrite(tmp_path):
    src, dst = tmp_path / "master", tmp_path / "resized"
    _write_image(src / "0001.png", (1600, 900))
    run_resize_images(src=src, dst=dst, workers=1)

    stats = run_resize_images(src=src, dst=dst, workers=1, overwrite=True)

    assert stats.written == 1 and stats.skipped_current == 0


def test_a_tier_change_re_resizes_only_what_moved(tmp_path):
    src, dst = tmp_path / "master", tmp_path / "resized"
    _write_image(src / "0001.png", (1600, 900))
    run_resize_images(src=src, dst=dst, workers=1)

    stats = run_resize_images(
        src=src,
        dst=dst,
        options=ResizeOptions.build(target_res=[1536]),
        workers=1,
    )

    assert stats.written == 1
    with Image.open(dst / "0001.png") as im:
        assert im.size == select_bucket(1600, 900, [1536])[1]


def test_min_pixels_skips_small_images_instead_of_upscaling(tmp_path):
    src, dst = tmp_path / "master", tmp_path / "resized"
    _write_image(src / "small.png", (200, 200))
    _write_image(src / "big.png", (1600, 900))

    stats = run_resize_images(src=src, dst=dst, workers=1)

    assert stats.written == 1 and stats.skipped_small == 1
    assert not (dst / "small.png").exists()
    # A skip here is invisible to *every* stage, not just to training: the
    # resized tree is what masking, grouping and the tagger all walk. So the
    # dropped image is named, the way a failure is, not merely counted.
    assert len(stats.too_small) == 1
    assert "small.png" in stats.too_small[0] and "200x200" in stats.too_small[0]
    assert run_resize_images(src=src, dst=dst, workers=1, min_pixels=0).written == 1


def test_path_pattern_narrows_to_one_image(tmp_path):
    src, dst = tmp_path / "master", tmp_path / "resized"
    _write_image(src / "001.webp", (1600, 900))
    _write_image(src / "002.png", (1600, 900))

    stats = run_resize_images(src=src, dst=dst, workers=1, path_pattern="001.*")

    assert stats.seen == 1 and stats.written == 1
    assert (dst / "001.png").is_file() and not (dst / "002.png").exists()


def test_captions_are_not_copied_unless_asked(tmp_path):
    src, dst = tmp_path / "master", tmp_path / "resized"
    _write_image(src / "0001.png", (1600, 900))
    (src / "0001.txt").write_text("1girl, solo", encoding="utf-8")

    run_resize_images(src=src, dst=dst, workers=1)
    assert not (dst / "0001.txt").exists()

    run_resize_images(src=src, dst=dst, workers=1, copy_captions=True, overwrite=True)
    assert (dst / "0001.txt").read_text(encoding="utf-8") == "1girl, solo"


def test_non_recursive_ignores_subdirs(tmp_path):
    src, dst = tmp_path / "master", tmp_path / "resized"
    _write_image(src / "top.png", (1600, 900))
    _write_image(src / "sub" / "nested.png", (1600, 900))

    stats = run_resize_images(src=src, dst=dst, workers=1, recursive=False)

    assert stats.seen == 1 and (dst / "top.png").is_file()
    assert not (dst / "sub" / "nested.png").exists()


def test_a_corrupt_file_is_reported_not_fatal(tmp_path):
    src, dst = tmp_path / "master", tmp_path / "resized"
    _write_image(src / "good.png", (1600, 900))
    (src / "broken.png").write_bytes(b"not a png")

    stats = run_resize_images(src=src, dst=dst, workers=1)

    assert stats.written == 1 and stats.failed == 1
    assert any("broken.png" in line for line in stats.failures)


def test_non_default_geometry_stamps_the_trainer_metadata_keys(tmp_path):
    src, dst = tmp_path / "master", tmp_path / "resized"
    _write_image(src / "0001.png", (1600, 900))

    run_resize_images(
        src=src,
        dst=dst,
        options=ResizeOptions.build(crop_anchor="left", crop_margins=[5, 0, 5, 0]),
        workers=1,
    )

    with Image.open(dst / "0001.png") as im:
        text = im.text
    assert text["anima_resize_crop_anchor"] == "left"
    assert text["anima_resize_crop_margins"] == "5,0,5,0"
    assert text["anima_resize_bucket_resos"] == ""

    # Changing the anchor invalidates the stamp, so the image re-resizes.
    stats = run_resize_images(
        src=src,
        dst=dst,
        options=ResizeOptions.build(crop_anchor="right", crop_margins=[5, 0, 5, 0]),
        workers=1,
    )
    assert stats.written == 1


def test_default_geometry_stamps_nothing(tmp_path):
    # Parity with the trainer, whose default-path signature is empty too: a
    # default PNG from either side is skipped by the other on size alone.
    src, dst = tmp_path / "master", tmp_path / "resized"
    _write_image(src / "0001.png", (1600, 900))
    run_resize_images(src=src, dst=dst, workers=1)

    with Image.open(dst / "0001.png") as im:
        assert not any(k.startswith("anima_resize_") for k in (im.text or {}))


def test_source_png_metadata_survives_the_resize(tmp_path):
    from PIL.PngImagePlugin import PngInfo

    src, dst = tmp_path / "master", tmp_path / "resized"
    src.mkdir(parents=True)
    info = PngInfo()
    info.add_text("parameters", "a comfy prompt")
    Image.new("RGB", (1600, 900)).save(src / "0001.png", pnginfo=info)

    run_resize_images(src=src, dst=dst, workers=1)

    with Image.open(dst / "0001.png") as im:
        assert im.text["parameters"] == "a comfy prompt"


def test_worker_pool_agrees_with_the_inline_path(tmp_path):
    src = tmp_path / "master"
    for i in range(3):
        _write_image(src / f"{i:04d}.png", (1600 + i * 8, 900))

    inline = run_resize_images(src=src, dst=tmp_path / "a", workers=1)
    pooled = run_resize_images(src=src, dst=tmp_path / "b", workers=2)

    assert (inline.written, inline.buckets) == (pooled.written, pooled.buckets)
    for name in ("0000.png", "0001.png", "0002.png"):
        assert (tmp_path / "a" / name).read_bytes() == (
            tmp_path / "b" / name
        ).read_bytes()


# --------------------------------------------------------------------------- #
# CLI


def test_cli_writes_the_tree_and_a_report(tmp_path):
    src, dst = tmp_path / "master", tmp_path / "resized"
    _write_image(src / "char_aki" / "0001.png", (1600, 900))
    _write_image(src / "small.png", (100, 100))
    report_dir = tmp_path / "report"

    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "anime_tools.stages.cli.resize_images",
            "--src",
            str(src),
            "--dst",
            str(dst),
            "--report_dir",
            str(report_dir),
            "--workers",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert r.returncode == 0, r.stderr
    assert (dst / "char_aki" / "0001.png").is_file()
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert report["stats"] == {
        "seen": 2,
        "written": 1,
        "skipped_current": 0,
        "skipped_small": 1,
        "failed": 0,
    }
    assert report["target_res"] == [1024]
    assert sum(report["buckets"].values()) == 1
    assert len(report["too_small"]) == 1


def test_cli_rejects_an_unknown_tier(tmp_path):
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "anime_tools.stages.cli.resize_images",
            "--src",
            str(tmp_path),
            "--target_res",
            "999",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode != 0
    assert "not in allowed tiers" in r.stderr


def test_resize_stage_is_torch_free():
    """The GUI collects this parser in a child; it must not drag torch in."""
    code = (
        "import sys, anime_tools.stages.cli.resize_images as m; "
        "m.build_parser(); assert 'torch' not in sys.modules"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr
