"""Dataset browsing API: the image/caption tree, the caption writer, thumbs.

Mirrors the sidebar's actual traffic — list the tree, open one item, save a
caption, then read it back.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

pytest.importorskip("fastapi")


def _png(path: Path, size: tuple[int, int] = (8, 8), colour: int = 128) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (colour, colour, colour)).save(path)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A miniature curation home: two images, one fully derived."""
    monkeypatch.setenv("ANIME_TOOLS_HOME", str(tmp_path))
    src = tmp_path / "image_dataset"
    dst = tmp_path / "post_image_dataset" / "resized"
    masks = tmp_path / "post_image_dataset" / "masks"

    _png(src / "a.png")
    (src / "a.txt").write_text("1girl, solo. On the left, cat.", encoding="utf-8")
    _png(src / "sub" / "b.jpg")
    (src / "sub" / "b.txt").write_text("1boy, night", encoding="utf-8")

    _png(dst / "sub" / "b.png")  # the resize step re-encoded jpg -> png
    (dst / "sub" / "b.txt").write_text("1boy, solo, night", encoding="utf-8")
    (dst / "sub" / "b.variants.txt").write_text(
        "# anima caption variants — auto-generated, do not hand-edit\n"
        "v0\t1boy, solo, night\nv1\tnight, 1boy, solo\n",
        encoding="utf-8",
    )
    _png(masks / "a_mask.png")  # legacy flat layout
    return tmp_path


@pytest.fixture
def client(home):
    from fastapi.testclient import TestClient

    from anime_tools.gui.jobs import JobManager
    from anime_tools.gui.server import create_app

    app = create_app(jobs=JobManager(log_dir=home / "logs"), schemas={})
    with TestClient(app) as c:
        yield c, home


def test_listing_joins_the_three_trees(client):
    c, _ = client
    body = c.get("/api/dataset").json()
    assert body["root"] == "image_dataset" and body["total"] == 2
    by_rel = {i["rel"]: i for i in body["items"]}
    assert by_rel["a.png"] == {
        "rel": "a.png",
        "dir": "",
        "name": "a.png",
        "stem": "a",
        "master": True,
        "derived": False,
        "variants": False,
        "mask": True,  # flat masks/{stem}_mask.png is the legacy fallback
    }
    b = by_rel["sub/b.jpg"]
    assert b["dir"] == "sub" and b["derived"] and b["variants"] and not b["mask"]


def test_listing_filters(client):
    c, _ = client
    assert c.get("/api/dataset", params={"q": "sub"}).json()["total"] == 1
    assert c.get("/api/dataset", params={"pattern": "sub/*"}).json()["total"] == 1
    assert c.get("/api/dataset", params={"q": "nope"}).json()["total"] == 0
    trimmed = c.get("/api/dataset", params={"limit": 1}).json()
    assert trimmed["truncated"] and len(trimmed["items"]) == 1 and trimmed["total"] == 2


def test_missing_source_root_is_reported_not_raised(client, home):
    c, _ = client
    body = c.get("/api/dataset", params={"src": "nowhere"}).json()
    assert body["missing"] and body["items"] == [] and body["root"] == "nowhere"


def test_item_detail_parses_the_caption_grammar(client):
    c, _ = client
    it = c.get("/api/dataset/item", params={"rel": "a.png"}).json()
    assert it["image"]["width"] == 8 and it["mask"]["path"].endswith("a_mask.png")
    assert it["resized"] is None
    master, derived = it["captions"]
    assert master["kind"] == "master" and master["exists"]
    # Clauses come parsed: the browser never splits a caption on commas.
    assert master["parsed"] == {
        "flat_tags": ["1girl", "solo"],
        "clauses": [
            {
                "header": "On the left",
                "prefix": "On the ",
                "position": "left",
                "tags": ["cat"],
            }
        ],
    }
    assert derived["kind"] == "derived" and not derived["exists"]
    assert derived["text"] == "" and derived["parsed"] is None


def test_item_detail_matches_a_re_encoded_resized_image(client):
    """The resize step may change the extension, so dst is matched on stem."""
    c, _ = client
    it = c.get("/api/dataset/item", params={"rel": "sub/b.jpg"}).json()
    assert it["resized"]["path"] == "post_image_dataset/resized/sub/b.png"
    assert [r["label"] for r in it["variants"]["rows"]] == ["v0", "v1"]


def test_item_detail_rejects_unknown_and_escaping_paths(client):
    c, _ = client
    assert c.get("/api/dataset/item", params={"rel": "ghost.png"}).status_code == 404
    assert c.get("/api/dataset/item", params={"rel": "../x.png"}).status_code == 404
    assert (
        c.get("/api/dataset/item", params={"rel": "/etc/hostname"}).status_code == 404
    )


def test_writing_a_caption_round_trips(client, home):
    c, _ = client
    r = c.put(
        "/api/dataset/item",
        json={"rel": "a.png", "kind": "master", "text": "1girl, solo, smile"},
    )
    assert r.status_code == 200, r.text
    assert (home / "image_dataset" / "a.txt").read_text(
        encoding="utf-8"
    ) == "1girl, solo, smile"
    assert r.json()["parsed"]["flat_tags"] == ["1girl", "solo", "smile"]
    assert r.json()["variants_stale"] is False


def test_writing_a_derived_caption_flags_the_stale_sidecar(client, home):
    c, _ = client
    r = c.put(
        "/api/dataset/item",
        json={"rel": "sub/b.jpg", "kind": "derived", "text": "1boy, solo, night, rain"},
    )
    assert r.json()["variants_stale"] is True
    # ...and the sidecar itself is left alone; regenerating it is the stage's job.
    assert (home / "post_image_dataset/resized/sub/b.variants.txt").exists()


def test_writing_creates_the_destination_folder(client, home):
    c, _ = client
    c.put(
        "/api/dataset/item", json={"rel": "a.png", "kind": "derived", "text": "1girl"}
    )
    assert (home / "post_image_dataset" / "resized" / "a.txt").read_text() == "1girl"


def test_caption_writes_are_confined_and_validated(client):
    c, _ = client
    bad = [
        {"rel": "a.png", "kind": "variants", "text": "x"},  # generated, not editable
        {"rel": "a.png", "kind": "master", "text": "   "},  # empty is not a delete
        {"rel": "../a.png", "kind": "master", "text": "x"},
        {"rel": "ghost.png", "kind": "master", "text": "x"},
    ]
    assert [c.put("/api/dataset/item", json=b).status_code for b in bad] == [400] * 4


def test_a_caption_is_stored_as_one_line(client, home):
    """The trainer reads captions line-wise; a pasted newline must not survive."""
    c, _ = client
    c.put(
        "/api/dataset/item",
        json={"rel": "a.png", "kind": "master", "text": " 1girl,\n solo \n"},
    )
    assert (home / "image_dataset" / "a.txt").read_text(
        encoding="utf-8"
    ) == "1girl, solo"


def test_parse_endpoint_is_the_only_caption_splitter(client):
    c, _ = client
    parsed = c.post(
        "/api/dataset/parse", json={"text": "a, b. In the background, c"}
    ).json()
    assert parsed["flat_tags"] == ["a", "b"]
    assert parsed["clauses"][0]["header"] == "In the background"


def test_roots_are_settable_and_persisted(client, home):
    c, _ = client
    assert c.get("/api/dataset/roots").json()["roots"]["src"]["path"] == "image_dataset"
    r = c.put("/api/dataset/roots", json={"src": "sub", "dst": "", "masks": ""})
    assert r.status_code == 200
    roots = r.json()["roots"]
    assert roots["src"] == {"path": "sub", "exists": True}
    # A blank field means "default", not "the home directory".
    assert roots["dst"]["path"] == "post_image_dataset/resized"
    assert c.get("/api/dataset/roots").json()["roots"]["src"]["path"] == "sub"
    assert c.put("/api/dataset/roots", json={"src": "/etc"}).status_code == 400


def test_saving_roots_creates_the_missing_directories(client, home):
    """Settings is the explicit 'these are my roots' gesture, so it makes them."""
    c, _ = client
    assert not (home / "shots").exists()
    r = c.put(
        "/api/dataset/roots",
        json={"src": "shots", "dst": "out/resized", "masks": "out/masks"},
    )
    assert r.status_code == 200, r.text
    assert sorted(r.json()["created"]) == ["dst", "masks", "src"]
    for rel in ("shots", "out/resized", "out/masks"):
        assert (home / rel).is_dir()
    assert all(v["exists"] for v in r.json()["roots"].values())
    # Second save has nothing left to create.
    r2 = c.put(
        "/api/dataset/roots",
        json={"src": "shots", "dst": "out/resized", "masks": "out/masks"},
    )
    assert r2.json()["created"] == []


def test_a_rejected_root_is_never_created(client, home):
    """The containment check runs first: no mkdir outside the curation home."""
    c, _ = client
    assert c.put("/api/dataset/roots", json={"src": "../escape"}).status_code == 400
    assert not (home.parent / "escape").exists()


def test_reading_never_creates_a_root(client, home):
    """resolve_roots is on every read request; a typo must stay reported missing."""
    c, _ = client
    body = c.get("/api/dataset", params={"src": "typo"}).json()
    assert body["missing"] and body["items"] == [] and not (home / "typo").exists()
    # Roots saved by some other route are reported, not conjured, on a GET.
    c.put("/api/settings", json={"dataset": {"src": "ghost", "dst": "", "masks": ""}})
    roots = c.get("/api/dataset/roots").json()["roots"]
    assert roots["src"] == {"path": "ghost", "exists": False}
    assert not (home / "ghost").exists()


def test_thumbnails_are_webp_and_confined(client):
    from PIL import Image

    c, _ = client
    r = c.get("/api/thumb", params={"path": "image_dataset/a.png", "size": 32})
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    with Image.open(io.BytesIO(r.content)) as im:
        assert max(im.size) <= 32
    assert c.get("/api/thumb", params={"path": "/etc/hostname"}).status_code == 404
    assert (
        c.get("/api/thumb", params={"path": "image_dataset/a.txt"}).status_code == 415
    )


def test_item_pattern_selects_exactly_the_one_image():
    """How the GUI runs a stage on the selected image: it narrows the
    ``--path_pattern`` the stage already takes, so one image and a batch share
    the same code path."""
    from anime_tools.gui import dataset as D
    from anime_tools.path_filter import filter_paths_by_glob

    # The extension is a wildcard: the resize step may have re-encoded it.
    assert D.item_pattern("char/a.jpg") == "char/a.*"
    paths = ["/r/char/a.png", "/r/char/ab.png", "/r/other/a.png"]
    assert filter_paths_by_glob(paths, "/r", D.item_pattern("char/a.jpg")) == [
        True,
        False,
        False,
    ]

    # fnmatch metacharacters in the name stay literal…
    pat = D.item_pattern("char/b[1].png")
    assert filter_paths_by_glob(
        ["/r/char/b[1].webp", "/r/char/b1.webp"], "/r", pat
    ) == [True, False]

    # …but '|' is the pattern's own separator and cannot be escaped, so a name
    # carrying one is refused instead of silently matching nothing.
    with pytest.raises(D.DatasetError, match=r"\|"):
        D.item_pattern("char/a|b.png")
    with pytest.raises(D.DatasetError):
        D.item_pattern("../escape.png")


# ---- group view: the grouping manifest, read back as the sidebar's second
# ordering of the same rows ------------------------------------------------


def _manifest(home: Path, groups: list[dict], **over) -> Path:
    from anime_tools._json import write_json
    from anime_tools.grouping.groups import MANIFEST_VERSION

    p = home / "post_image_dataset" / "groups" / "groups.json"
    write_json(
        p,
        {
            "version": MANIFEST_VERSION,
            "source_dir": str(home / "image_dataset"),
            "groups": groups,
            **over,
        },
    )
    return p


def test_groups_subpath_is_the_stage_report_tail():
    """The view has to read the file the Groups stage writes, wherever Settings
    points ``report_root`` — same split the stage's own report field gets."""
    from anime_tools.grouping.cli.build_groups import build_parser
    from anime_tools.gui.dataset import GROUPS_SUBPATH
    from anime_tools.gui.stages import report_subpath

    out = next(a for a in build_parser()._actions if a.dest == "out")
    assert report_subpath(out.default) == GROUPS_SUBPATH


def test_groups_are_rels_the_listing_can_join(client):
    c, home = client
    _manifest(
        home,
        [
            {
                "id": 0,
                "artist": "sub",
                "size": 2,
                "mean_cosine": 0.9123,
                "members": ["sub/b.jpg", "a.png"],
            }
        ],
    )
    body = c.get("/api/dataset/groups").json()
    assert not body["missing"] and not body["stale"]
    assert body["path"] == "post_image_dataset/groups/groups.json"
    assert body["source_dir"] == "image_dataset"
    assert body["groups"] == [
        {
            "id": 0,
            "artist": "sub",
            "mean_cosine": 0.9123,
            "members": ["sub/b.jpg", "a.png"],
        }
    ]
    # Every member is a rel the listing keys its rows by, which is the whole
    # join: group view draws the same rows, in a different order.
    rels = {i["rel"] for i in c.get("/api/dataset").json()["items"]}
    assert set(body["groups"][0]["members"]) <= rels


def test_groups_follow_the_report_root_setting(client, home):
    c, _ = client
    c.put("/api/settings", json={"stage_defaults": {"report_root": "elsewhere"}})
    from anime_tools._json import write_json

    write_json(home / "elsewhere/groups/groups.json", {"version": 2, "groups": []})
    body = c.get("/api/dataset/groups").json()
    assert body["path"] == "elsewhere/groups/groups.json" and not body["missing"]


def test_a_missing_manifest_is_not_an_error(client):
    """The Groups stage may simply never have run — the panel says so."""
    c, _ = client
    body = c.get("/api/dataset/groups").json()
    assert body["missing"] and body["groups"] == []
    assert body["path"] == "post_image_dataset/groups/groups.json"


def test_an_older_manifest_still_lists_its_components(client, home):
    c, _ = client
    _manifest(home, [{"id": 0, "artist": "", "members": ["a.png"]}], version=1)
    body = c.get("/api/dataset/groups").json()
    assert body["stale"] and len(body["groups"]) == 1


def test_an_unreadable_manifest_is_a_400(client, home):
    c, _ = client
    p = home / "post_image_dataset" / "groups" / "groups.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    assert c.get("/api/dataset/groups").status_code == 400
