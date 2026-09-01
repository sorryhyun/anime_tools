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
    dst = tmp_path / "workspace" / "resized"
    masks = tmp_path / "workspace" / "masks"

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
        # One flag per ladder rung, keyed by rung: the sidebar's dot strip.
        "captions": {
            "master": True,
            "history": False,
            "revised": False,
            "variants": False,
        },
        "resized": False,  # resize has not run over this one
        "mask": True,  # flat masks/{stem}_mask.png is the legacy fallback
    }
    b = by_rel["sub/b.jpg"]
    assert b["dir"] == "sub" and b["captions"]["revised"] and b["captions"]["variants"]
    assert not b["mask"]
    # …and `resized` is matched on stem, so the jpg -> png re-encode still counts.
    assert b["resized"]


def test_the_listing_carries_the_caption_ladder(client):
    """The dot strip is drawn from the server's rungs, not a copy of their names.

    Same declaration as the row's ``captions`` map and the panel's badges, which
    is what makes Phase 2's master overlay one line rather than three.
    """
    from anime_tools.gui import dataset as D

    c, _ = client
    body = c.get("/api/dataset").json()
    assert body["ladder"] == [
        {"kind": "master", "editable": True},
        {"kind": "history", "editable": False},
        {"kind": "revised", "editable": True},
        {"kind": "variants", "editable": False},
    ]
    assert [r["kind"] for r in body["ladder"]] == list(body["items"][0]["captions"])
    assert D.CAPTION_KINDS == ("master", "revised")
    # A missing root still says what the rungs are, so the strip has a shape.
    assert c.get("/api/dataset", params={"src": "nowhere"}).json()["ladder"]


def test_listing_filters(client):
    c, _ = client
    assert c.get("/api/dataset", params={"q": "sub"}).json()["total"] == 1
    assert c.get("/api/dataset", params={"pattern": "sub/*"}).json()["total"] == 1
    assert c.get("/api/dataset", params={"q": "nope"}).json()["total"] == 0
    trimmed = c.get("/api/dataset", params={"limit": 1}).json()
    assert trimmed["truncated"] and len(trimmed["items"]) == 1 and trimmed["total"] == 2


def test_the_listing_default_is_the_cap_not_a_smaller_number():
    """A listing shows the whole dataset; only ``MAX_ITEMS`` truncates it.

    The default used to be 2000 with nothing overriding it, so the 20000 cap was
    unreachable and any real dataset came back quietly short — in the group
    ordering too, since both orderings draw this one listing. Pinned in both
    spellings, because the route and the function each carry the default.
    """
    import inspect

    from anime_tools.gui import dataset as D
    from anime_tools.gui import server as SRV

    assert inspect.signature(D.list_items).parameters["limit"].default == D.MAX_ITEMS
    route = next(
        r for r in SRV.create_app().routes if getattr(r, "path", "") == "/api/dataset"
    )
    assert inspect.signature(route.endpoint).parameters["limit"].default == D.MAX_ITEMS


def test_missing_source_root_is_reported_not_raised(client, home):
    c, _ = client
    body = c.get("/api/dataset", params={"src": "nowhere"}).json()
    assert body["missing"] and body["items"] == [] and body["root"] == "nowhere"


def test_item_detail_parses_the_caption_grammar(client):
    c, _ = client
    it = c.get("/api/dataset/item", params={"rel": "a.png"}).json()
    assert it["image"]["width"] == 8 and it["mask"]["path"].endswith("a_mask.png")
    assert it["resized"] is None
    master, history, revised, variants = it["versions"]
    assert master["kind"] == "master" and master["exists"] and master["editable"]
    # Clauses come parsed: the browser never splits a caption on commas.
    assert master["parsed"] == {
        "flat_tags": ["1girl", "solo"],
        # Offsets too: the boxed caption editor draws them, and they must be
        # slices of the very text this entry carries.
        "spans": [
            {"start": 0, "end": 5, "kind": "tag", "clause": -1},
            {"start": 7, "end": 11, "kind": "tag", "clause": -1},
            {"start": 13, "end": 24, "kind": "header", "clause": 0},
            {"start": 26, "end": 29, "kind": "tag", "clause": 0},
        ],
        "clauses": [
            {
                "header": "On the left",
                "prefix": "On the ",
                "position": "left",
                "tags": ["cat"],
            }
        ],
    }
    assert revised["kind"] == "revised" and not revised["exists"]
    assert (
        revised["text"] == "" and revised["parsed"] is None and revised["mtime"] is None
    )
    # Never written, so nothing it used to be: the history rung is hollow too.
    assert history["kind"] == "history" and not history["exists"]
    # No sidecar, but the rung it would fill keeps its place on the badge row.
    assert variants["kind"] == "variants"
    assert not variants["exists"] and not variants["editable"]


def test_item_detail_matches_a_re_encoded_resized_image(client):
    """The resize step may change the extension, so dst is matched on stem."""
    c, _ = client
    it = c.get("/api/dataset/item", params={"rel": "sub/b.jpg"}).json()
    assert it["resized"]["path"] == "workspace/resized/sub/b.png"
    # The sidecar rung expands into one badge per label, each already parsed:
    # a variant is a caption, so the browser does not split it either.
    assert [v["kind"] for v in it["versions"]] == [
        "master",
        "history",
        "revised",
        "v0",
        "v1",
    ]
    v0 = it["versions"][3]
    assert v0["exists"] and not v0["editable"]
    assert v0["path"].endswith("sub/b.variants.txt")
    assert v0["parsed"]["flat_tags"] == ["1boy", "solo", "night"]


def test_item_detail_flags_an_image_under_the_resize_floor(client):
    """The panel says why a stage over this image would do nothing at all.

    Below ``min_pixels`` the resize preflight skips the image, so it never lands
    in ``workspace/resized`` -- the tree every stage walks. A run then reports
    zero images and writes nothing, which reads as a broken stage unless the
    size line says the size is the reason.
    """
    from anime_tools.stages.resize import DEFAULT_MIN_PIXELS

    c, _ = client
    it = c.get("/api/dataset/item", params={"rel": "a.png"}).json()
    assert it["min_pixels"] == DEFAULT_MIN_PIXELS
    assert it["image"]["pixels"] == 64 and it["image"]["too_small"] is True
    # Only the source is measured against the floor: the mask and the resized
    # copy are *outputs* of that decision, so a verdict on them answers nothing.
    assert it["mask"]["too_small"] is None


def test_no_resize_floor_means_no_verdict(client, home):
    """``min_pixels`` 0 turns the floor off, and then there is nothing to say --
    ``None``, not ``False``: the panel must not claim an image passed a test
    nobody ran."""
    from anime_tools.gui import dataset as D
    from anime_tools.gui.server import roots_for

    it = D.item_detail(roots_for({}), "a.png", min_pixels=0)
    assert it["image"]["pixels"] == 64 and it["image"]["too_small"] is None


def test_the_resize_floor_comes_from_the_preprocess_settings():
    """The floor the item route measures against is the one the preflight runs
    at -- the Settings *Preprocess* block, falling back to the stage's own
    constant rather than to a second copy of the number."""
    from anime_tools.gui.server import preprocess_min_pixels
    from anime_tools.gui.stages import PREPROCESS_SETTINGS_KEY
    from anime_tools.stages.resize import DEFAULT_MIN_PIXELS

    key = PREPROCESS_SETTINGS_KEY
    assert preprocess_min_pixels({}) == DEFAULT_MIN_PIXELS
    # An emptied field means "the CLI's own default", as everywhere else.
    assert preprocess_min_pixels({key: {"min_pixels": ""}}) == DEFAULT_MIN_PIXELS
    assert preprocess_min_pixels({key: {"min_pixels": "4096"}}) == 4096
    assert preprocess_min_pixels({key: {"min_pixels": 0}}) == 0


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


def test_writing_a_revised_caption_flags_the_stale_sidecar(client, home):
    c, _ = client
    r = c.put(
        "/api/dataset/item",
        json={
            "rel": "sub/b.jpg",
            "kind": "revised",
            "text": "1boy, solo, night, rain",
        },
    )
    assert r.json()["variants_stale"] is True
    # ...and the sidecar itself is left alone; regenerating it is the stage's job.
    assert (home / "workspace/resized/sub/b.variants.txt").exists()


def test_writing_creates_the_destination_folder(client, home):
    c, _ = client
    c.put(
        "/api/dataset/item", json={"rel": "a.png", "kind": "revised", "text": "1girl"}
    )
    assert (home / "workspace" / "resized" / "a.txt").read_text() == "1girl"


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
    assert roots["dst"]["path"] == "workspace/resized"
    assert c.get("/api/dataset/roots").json()["roots"]["src"]["path"] == "sub"


def test_saving_roots_creates_the_missing_directories(client, home):
    """Settings is the explicit 'these are my roots' gesture, so it makes them."""
    from anime_tools.gui import dataset as D

    c, _ = client
    assert not (home / "shots").exists()
    r = c.put(
        "/api/dataset/roots",
        json={"src": "shots", "dst": "out/resized", "masks": "out/masks"},
    )
    assert r.status_code == 200, r.text
    # Every root but `out`: the export tree is Export's to create.
    assert sorted(r.json()["created"]) == ["dst", "masks", "master", "src"]
    for rel in ("shots", "out/resized", "out/masks"):
        assert (home / rel).is_dir()
    made = {k: v for k, v in r.json()["roots"].items() if k not in D.EXPORT_ROOTS}
    assert all(v["exists"] for v in made.values())
    # Second save has nothing left to create.
    r2 = c.put(
        "/api/dataset/roots",
        json={"src": "shots", "dst": "out/resized", "masks": "out/masks"},
    )
    assert r2.json()["created"] == []


def test_a_root_can_be_a_tree_beside_the_home(client, home):
    """The ordinary layout: ``anime_tools/`` checked out next to ``anima_lora/``,
    so ``src`` is ``../anima_lora/image_dataset`` and no home holds both."""
    from anime_tools.gui import dataset as D

    c, _ = client
    sibling = home.parent / "anima_lora" / "image_dataset"
    _png(sibling / "z.png")
    (home.parent / "not-mine.txt").write_text("private", encoding="utf-8")

    r = c.put("/api/dataset/roots", json={"src": "../anima_lora/image_dataset"})
    assert r.status_code == 200, r.text
    # Outside the home, so it comes back by the name it was given rather than
    # with the home's own prefix repeated back at the reader -- and `lexical`
    # collapses that `..` again, so what goes out is what comes back in.
    assert r.json()["roots"]["src"] == {
        "path": "../anima_lora/image_dataset",
        "exists": True,
    }
    assert D.lexical("../anima_lora/image_dataset") == sibling
    # The listing and the pixels follow the root out of the home...
    assert [i["rel"] for i in c.get("/api/dataset").json()["items"]] == ["z.png"]
    assert c.get("/api/thumb", params={"path": f"{sibling}/z.png"}).status_code == 200
    # ...but only that tree: what is merely *near* it is still nobody's business.
    outsider = home.parent / "not-mine.txt"
    assert c.get("/api/files", params={"path": str(outsider)}).status_code == 404


def test_a_root_outside_the_home_is_never_created(client, home):
    """Pointing at a tree beside the home is the point; conjuring one is not."""
    c, _ = client
    r = c.put("/api/dataset/roots", json={"src": "../escape"})
    assert r.status_code == 200, r.text
    assert "src" not in r.json()["created"]
    assert r.json()["roots"]["src"]["exists"] is False
    assert not (home.parent / "escape").exists()


def test_a_query_param_cannot_point_the_listing_anywhere(client, home):
    """A saved root widens what may be read; a request's own override never
    does -- only the Settings save that means it gets to move that line."""
    c, _ = client
    assert c.get("/api/dataset", params={"src": "../elsewhere"}).status_code == 400


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

    p = home / "workspace" / "groups" / "groups.json"
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
    assert body["path"] == "workspace/groups/groups.json"
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


def test_a_one_component_dst_puts_the_reports_in_the_workspace(client):
    """Beside ``dst`` stops at the home: a ``dst`` with no parent inside it —
    a settings file written before the workspace still pins the pre-workspace
    ``post_image_dataset`` — would otherwise strew ``captions/`` and ``groups/``
    across the project root."""
    c, _ = client
    c.put(
        "/api/dataset/roots",
        json={"src": "image_dataset", "dst": "post_image_dataset"},
    )
    assert c.get("/api/dataset/roots").json()["report_root"] == "workspace"
    assert c.get("/api/dataset/groups").json()["path"] == "workspace/groups/groups.json"


def test_the_mask_root_sits_beside_the_masks_root_by_default(client):
    """Each generator's own tree is an intermediate the merge unions into the
    ``masks`` root, so it belongs next to it — and moves with it. Blank is the
    default; the same stop-at-the-home rule as ``report_root``."""
    c, _ = client
    assert c.get("/api/dataset/roots").json()["mask_root"] == "workspace"
    # A one-component masks root has the home for a parent; the workspace is
    # the answer rather than the project root.
    c.put(
        "/api/dataset/roots",
        json={"src": "image_dataset", "dst": "workspace/resized", "masks": "masks"},
    )
    assert c.get("/api/dataset/roots").json()["mask_root"] == "workspace"


def test_the_mask_root_setting_moves_both_generators_and_the_merge(client):
    """One value, three CLIs: the two ``--mask-dir`` defaults and the merge's
    two inputs all hang off it, each keeping its own tail."""
    from anime_tools.gui import server as SV
    from anime_tools.gui import stages as S

    c, _ = client
    c.put("/api/settings", json={"stage_defaults": {"mask_root": "elsewhere"}})
    settings = c.get("/api/settings").json()
    roots = SV.roots_for(settings)
    assert SV.mask_root(settings, roots) == "elsewhere"

    got = {}
    for sid in ("masks_sam", "masks_mit", "masks_merge"):
        fields = S.schema(S.BY_ID[sid])["fields"]
        argv = S.build_argv(
            fields, {}, roots=SV.root_paths(roots), mask_root="elsewhere"
        )
        got[sid] = [a for a in argv if a.startswith("elsewhere")]
    assert got["masks_sam"] == ["elsewhere/masks_sam"]
    assert got["masks_mit"] == ["elsewhere/masks_mit"]
    assert got["masks_merge"] == ["elsewhere/masks_sam", "elsewhere/masks_mit"]


def test_a_missing_manifest_is_not_an_error(client):
    """The Groups stage may simply never have run — the panel says so."""
    c, _ = client
    body = c.get("/api/dataset/groups").json()
    assert body["missing"] and body["groups"] == []
    assert body["path"] == "workspace/groups/groups.json"


def test_an_older_manifest_still_lists_its_components(client, home):
    c, _ = client
    _manifest(home, [{"id": 0, "artist": "", "members": ["a.png"]}], version=1)
    body = c.get("/api/dataset/groups").json()
    assert body["stale"] and len(body["groups"]) == 1


def test_an_unreadable_manifest_is_a_400(client, home):
    c, _ = client
    p = home / "workspace" / "groups" / "groups.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    assert c.get("/api/dataset/groups").status_code == 400
