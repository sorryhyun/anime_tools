"""The model download catalog: what it claims, and that the loaders agree — a
row has to point where the loader actually looks."""

from __future__ import annotations

from dataclasses import replace

import pytest

from anime_tools import downloads as DL


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIME_TOOLS_HOME", str(tmp_path))
    monkeypatch.delenv("ANIME_TOOLS_MODELS", raising=False)
    return tmp_path


def test_catalog_is_torch_free():
    """The GUI process imports this module, so it must stay torch-free."""
    import subprocess
    import sys

    code = (
        "import sys, anime_tools.downloads as D; D.catalog(); "
        "assert 'torch' not in sys.modules, 'torch imported'"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr


def test_every_row_is_addressable_and_serializable(home):
    ids = [a.id for a in DL.catalog()]
    assert len(set(ids)) == len(ids)
    assert set(DL.by_id()) == set(ids)
    for a in DL.catalog():
        d = a.to_dict()
        assert d["id"] and d["title"] and d["repo"] and d["files"] and d["used_by"]
        assert d["installed"] == (not d["missing"])
        if a.dest is not None:
            # A fresh home holds nothing. (Hub-cache rows answer for the machine,
            # not the home, so they are legitimately "installed" here.)
            assert d["installed"] is False
            # A derived row probes for its product, not its inputs.
            assert d["missing"] == list(a.derived or a.files)


def test_destinations_follow_the_curation_home(home):
    by = DL.by_id()
    assert by["tagger"].dest == home / "models" / "captioners" / "anima-tagger-dbv4"
    assert by["sam3"].dest == home / "models" / "sam3"
    assert by["pe_spatial"].dest == home / "models" / "pe"
    # Not under models/: the soft prompt lands on the path --prompt_embed resolves.
    assert by["soft_prompt"].dest == home / "networks" / "calibration"
    # Hub-cache assets have no path under models/ to keep in sync.
    assert by["tagger_backbone"].dest is None and by["mit_text"].dest is None
    assert by["mit_text"].location == "Hugging Face cache"


def test_a_present_file_flips_the_row_to_installed(home):
    sam3 = DL.by_id()["sam3"]
    assert sam3.missing() == [DL.SAM3_FILENAME]
    (home / "models" / "sam3").mkdir(parents=True)
    (home / "models" / "sam3" / DL.SAM3_FILENAME).write_bytes(b"x")
    assert DL.by_id()["sam3"].installed


def test_rows_land_where_the_loaders_look():
    """Each row's destination is where its loader looks."""
    pytest.importorskip("torch")
    from anime_tools.tagger import dbv4_meta
    from anime_tools.vision import pe

    by = DL.by_id()
    assert by["tagger"].repo == dbv4_meta.TAGGER_HF_REPO
    assert by["tagger"].subfolder == dbv4_meta.TAGGER_HF_SUBFOLDER
    assert by["tagger"].dest == DL.resolve_path(dbv4_meta.DEFAULT_TAGGER_DIR)
    assert by["tagger_backbone"].files == dbv4_meta.DBV4_BACKBONE_FILES
    assert by["pe_spatial"].repo == pe.PE_SPATIAL_REPO
    assert (
        by["pe_spatial"].dest / pe.PE_SPATIAL_FILENAME == pe.default_pe_spatial_path()
    )

    # The SAM3 CLIs' --checkpoint default has to be inside the row's dest.
    from anime_tools.stages.cli import position_captions as pc

    default = pc.build_parser().parse_args([]).checkpoint
    assert DL.resolve_path(default).parent == by["sam3"].dest
    assert DL.resolve_path(default).name == DL.SAM3_FILENAME

    # Same for the soft prompt: the --prompt_embed default is the row's file.
    embed = DL.resolve_path(pc.build_parser().parse_args([]).prompt_embed)
    assert embed == by["soft_prompt"].dest / DL.SOFT_PROMPT_FILENAME

    # The CTD gate has no flag: the row and the loader are the same call.
    from anime_tools.masking.cli import generate_masks_mit as mit

    assert mit.default_ctd_onnx_path is DL.default_ctd_onnx_path
    assert "ctd_onnx" not in {a.dest for a in mit.build_parser()._actions}
    assert DL.default_ctd_onnx_path() == by["ctd_onnx"].dest / DL.CTD_ONNX_FILENAME


def test_the_tag_kb_lands_where_correction_looks_for_it(home):
    """The KB row is data, not weights: it pairs with ``find_tag_csv``."""
    from anime_tools.captions.correction import (
        TAG_CSV_EN_NAME,
        TAG_CSV_NAME,
        find_tag_csv,
    )

    by = DL.by_id()
    assert by["danbooru_tags"].dest == home / "models"
    assert by["danbooru_tags"].files == (TAG_CSV_NAME,)
    csv = home / "models" / TAG_CSV_NAME
    csv.parent.mkdir(parents=True)
    csv.write_text("name,category,post_count,description\n", encoding="utf-8")
    assert find_tag_csv(home) == csv
    assert DL.by_id()["danbooru_tags"].installed

    # The English row builds its file from the wiki mirror, so its probe asks for
    # the product; the parquet stays an input in the hub cache.
    en = by["danbooru_tags_en"]
    assert en.derived == (TAG_CSV_EN_NAME,) and en.build is not None
    assert en.missing() == [TAG_CSV_EN_NAME]
    (home / "models" / TAG_CSV_EN_NAME).write_text("name\n", encoding="utf-8")
    assert DL.by_id()["danbooru_tags_en"].installed


def test_a_built_row_runs_its_build_after_fetching(home, monkeypatch):
    """``fetch`` on a derived row is download-then-build, with the download
    landing in the hub cache rather than in ``dest``."""
    seen: dict[str, object] = {}

    def fake_download(**kwargs):
        seen.update(kwargs)
        path = home / "cache" / "wiki.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return str(path)

    monkeypatch.setattr("anime_tools._hf.hf_download", fake_download)
    monkeypatch.setattr(
        DL, "_build_english_tag_csv", lambda dest, log: seen.update(built=dest)
    )
    row = DL.by_id()["danbooru_tags_en"]
    # by_id() captured the module-level function; rebuild the row with the patched one.
    row = replace(row, build=DL._build_english_tag_csv)
    row.fetch(log=lambda _msg: None)

    assert seen["repo_type"] == "dataset" and seen["repo_id"] == DL.DANBOORU_WIKI_REPO
    assert "local_dir" not in seen  # inputs stay in the hub cache
    assert seen["built"] == home / "models"


def test_gated_rows_carry_their_accept_terms_url(home):
    gated = {a.id for a in DL.catalog() if a.gated}
    assert gated == {"tagger_backbone", "sam3"}
    for a in DL.catalog():
        if a.gated:
            assert a.gated == f"https://huggingface.co/{a.repo}"


def test_cli_rejects_an_unknown_id(home, capsys):
    assert DL.main(["nope"]) == 2
    assert "unknown model id: nope" in capsys.readouterr().err


def test_cli_list_reports_every_row(home, capsys):
    assert DL.main(["--list"]) == 0
    out = capsys.readouterr().out
    for a in DL.catalog():
        assert a.id in out and a.repo in out


def test_cli_downloads_only_what_is_missing(home, monkeypatch, capsys):
    """No id = every missing row, and a failure does not abort the rest."""
    fetched: list[str] = []

    def fake_fetch(self, log=DL._say):
        fetched.append(self.id)
        if self.id == "sam3":
            raise FileNotFoundError("gated")

    monkeypatch.setattr(DL.Asset, "fetch", fake_fetch)
    (home / "models" / "pe").mkdir(parents=True)
    (home / "models" / "pe" / DL.PE_SPATIAL_FILENAME).write_bytes(b"x")

    assert DL.main([]) == 1  # sam3 failed
    assert "pe_spatial" not in fetched and "sam3" in fetched
    assert "gated" in capsys.readouterr().err

    fetched.clear()
    assert DL.main(["pe_spatial"]) == 0  # an explicit id re-fetches regardless
    assert fetched == ["pe_spatial"]


def test_a_url_row_downloads_over_plain_https(home, monkeypatch, capsys):
    """A URL row fetches over https and lands flat in the dest ``missing()`` probes."""
    import urllib.request

    asked: list[str] = []

    class Fake:
        def __init__(self, url):
            asked.append(url)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"soft-prompt-bytes"

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None: Fake(url))

    row = DL.by_id()["soft_prompt"]
    assert not row.installed
    row.fetch()

    assert asked == [f"{DL.SOFT_PROMPT_URL}/{DL.SOFT_PROMPT_FILENAME}"]
    landed = row.dest / DL.SOFT_PROMPT_FILENAME
    assert landed.read_bytes() == b"soft-prompt-bytes"
    assert list(row.dest.iterdir()) == [landed]  # no .part left behind
    assert DL.by_id()["soft_prompt"].installed
    assert str(landed) in capsys.readouterr().out


def test_a_url_row_reports_a_failure_with_the_recovery(home, monkeypatch):
    import urllib.error
    import urllib.request

    def boom(url, timeout=None):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    row = DL.by_id()["soft_prompt"]
    with pytest.raises(FileNotFoundError) as e:
        row.fetch()
    assert "python -m anime_tools.downloads soft_prompt" in str(e.value)
    assert not (row.dest / DL.SOFT_PROMPT_FILENAME).exists()
    assert not list(row.dest.iterdir())  # the partial file is cleaned up
