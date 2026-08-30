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
    assert roots["src"] == {"path": "sub", "exists": False}
    # A blank field means "default", not "the home directory".
    assert roots["dst"]["path"] == "post_image_dataset/resized"
    assert c.get("/api/dataset/roots").json()["roots"]["src"]["path"] == "sub"
    assert c.put("/api/dataset/roots", json={"src": "/etc"}).status_code == 400


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
