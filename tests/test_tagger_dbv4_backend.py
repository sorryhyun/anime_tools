"""dbv4 backend behind the AnimaTagger contract — offline invariants.

The backend is stubbed with a fake ``Dbv4Backend`` returning canned
probabilities, so no HF download or timm weights are needed.

1. ``align_vocab`` joins by space-normalised name, recovers ``rules.yaml``
   renames, and reports unmatched tags per category.
2. A dbv4-backed dir loads with no ``model.safetensors``; ``predict`` returns
   ``rating`` / ``rating_scores`` / ``scores`` / ``kept`` / ``thresholds`` /
   ``people_count``.
3. Unsupported tags never fire: their logit is ``UNSUPPORTED_LOGIT``, and a
   ``softmax`` group of only unsupported tags emits nothing.
4. People-count comes from the count-tag rule; a sidecar adds
   ``people_count_scores`` and makes its BCE rows emittable.
5. ``SidecarHead.save`` / ``load`` round-trips weights + metadata.
"""

from __future__ import annotations

import json

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file as st_save

from anime_tools.captions import tag_rules as tr
from anime_tools.tagger import dbv4_backend as db
from anime_tools.tagger import tagger as at

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

VOCAB_TAGS = [
    ("1girl", "count"),
    ("1boy", "count"),
    ("solo", "general"),
    ("black hair", "general"),
    ("blonde hair", "general"),
    ("black shoes", "general"),  # renamed upstream → unmatched
    ("@someone", "artist"),  # never in dbv4
    ("@other", "artist"),
    ("original", "copyright"),  # no copyright category upstream
    ("touhou", "copyright"),
    ("shiro (mignon)", "character"),  # dataset OC
    ("hakurei reimu", "character"),
]
N = len(VOCAB_TAGS)
IDX = {n: i for i, (n, _) in enumerate(VOCAB_TAGS)}

DBV4_NAMES = [
    "general",
    "sensitive",
    "questionable",
    "explicit",
    "1girl",
    "1boy",
    "solo",
    "black_hair",
    "blonde_hair",
    "hakurei_reimu",
    "black_footwear",
]


def _card() -> db.Dbv4Card:
    rows = []
    for j, n in enumerate(DBV4_NAMES):
        cat = 9 if j < 4 else (4 if n == "hakurei_reimu" else 0)
        rows.append({"name": n, "category": str(cat), "best_threshold": "0.4"})
    card = db.Dbv4Card(repo="fake/dbv4", rows=rows, model_args={})
    for j, r in enumerate(rows):
        if int(r["category"]) == 9:
            card.rating_cols[db.DBV4_RATING_MAP[r["name"]]] = j
        else:
            card.name_to_col[r["name"].replace("_", " ")] = j
    return card


class FakeBackend:
    """Stands in for Dbv4Backend: canned probs, 8-d hidden."""

    repo, arch = "fake/dbv4", "fake"
    d_hidden = 8

    def __init__(self, probs_by_name: dict[str, float]):
        self.card = _card()
        self.probs_by_name = probs_by_name

    def forward(self, images):
        p = torch.zeros(1, len(DBV4_NAMES))
        for n, v in self.probs_by_name.items():
            p[0, DBV4_NAMES.index(n)] = v
        return db.Dbv4Output(probs=p, hidden=torch.ones(1, self.d_hidden))


def _write_ckpt(tmp_path, groups: bool = True):
    vocab = {
        "tags": [
            {"name": n, "index": i, "category": c, "freq": 10, "median_pos": float(i)}
            for i, (n, c) in enumerate(VOCAB_TAGS)
        ],
        "ratings": list(at.RATINGS),
        "people_count_labels": list(at.PEOPLE_COUNT_LABELS),
    }
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "rules.yaml").write_text(
        "replacements:\n  black footwear: black shoes\nremove: []\n"
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"backend": "dbv4", "dbv4": {"repo": "fake/dbv4", "arch": "fake"}})
    )
    thr = torch.full((N,), 1.01)
    for n in ("1girl", "1boy", "solo", "black hair", "blonde hair", "hakurei reimu"):
        thr[IDX[n]] = 0.4
    st_save({"thresholds": thr}, str(tmp_path / "thresholds.safetensors"))
    if groups:
        (tmp_path / "groups.yaml").write_text(
            "version: 1\n"
            "hair_color:\n  mode: softmax_when_solo\n  tags: [black hair, blonde hair]\n"
            "artist:\n  mode: softmax\n  tags: ['@someone', '@other']\n"
        )
    return tmp_path


def _tagger(tmp_path, probs, monkeypatch, sidecar=None):
    monkeypatch.setattr(at, "Dbv4Backend", lambda **kw: FakeBackend(probs))
    if sidecar is not None:
        sidecar.save(tmp_path)
    return at.AnimaTagger(tmp_path, device="cpu")


IMG = Image.new("RGB", (32, 48), (200, 100, 50))


# --------------------------------------------------------------------------- #
# 1. alignment
# --------------------------------------------------------------------------- #


def test_align_vocab_recovers_renames_and_reports_unmatched():
    card = _card()
    # rules.yaml replacements run on the space-form caption string.
    rules = tr.from_dict({"replacements": {"black footwear": "black shoes"}})
    vocab_tags = [
        {"name": n, "index": i, "category": c} for i, (n, c) in enumerate(VOCAB_TAGS)
    ]
    a = db.align_vocab(vocab_tags, card, db.rename_recovery_from_rules(rules))
    supported = {VOCAB_TAGS[i][0] for i in a.ours_idx.tolist()}
    assert supported == {
        "1girl",
        "1boy",
        "solo",
        "black hair",
        "blonde hair",
        "hakurei reimu",
        "black shoes",
    }
    assert a.unmatched_by_category == {"artist": 2, "copyright": 2, "character": 1}
    assert a.supported_mask(N).sum() == 7


def test_align_without_recovery_leaves_rename_unmatched():
    a = db.align_vocab(
        [{"name": n, "index": i, "category": c} for i, (n, c) in enumerate(VOCAB_TAGS)],
        _card(),
    )
    assert a.unmatched_by_category["general"] == 1


# --------------------------------------------------------------------------- #
# 2–4. predict contract
# --------------------------------------------------------------------------- #


def test_dbv4_dir_loads_without_model_weights(tmp_path, monkeypatch):
    _write_ckpt(tmp_path)
    assert not (tmp_path / "model.safetensors").exists()
    # No checkpoint fetch; the backbone preflight is stubbed as "cached".
    monkeypatch.setattr("anime_tools.tagger.dbv4_meta.backbone_cached", lambda _r: True)
    assert at.ensure_tagger_checkpoint(tmp_path) == tmp_path
    t = _tagger(tmp_path, {"1girl": 0.95, "solo": 0.9, "explicit": 0.9}, monkeypatch)
    assert t.backend_kind == "dbv4"
    assert not hasattr(t, "model")  # PE head is gone
    assert t.cfg.n_tags == N
    out = t.predict(IMG)
    for k in ("rating", "rating_scores", "scores", "kept", "thresholds"):
        assert k in out
    assert out["rating"] == "explicit"
    assert set(out["kept"]) == {"1girl", "solo"}
    assert out["thresholds"] is t.threshold_map
    assert t.tag_logits(IMG).shape == (N,)


def test_unsupported_tags_never_fire_and_empty_softmax_group_emits_nothing(
    tmp_path, monkeypatch
):
    _write_ckpt(tmp_path)
    t = _tagger(tmp_path, {"1girl": 0.95, "solo": 0.9, "black_hair": 0.8}, monkeypatch)
    logits = t.tag_logits(IMG)
    for n in ("@someone", "@other", "original", "touhou", "shiro (mignon)"):
        assert float(logits[IDX[n]]) == db.UNSUPPORTED_LOGIT
    out = t.predict(IMG)
    assert not any(n.startswith("@") for n in out["kept"])
    assert out["groups"]["artist"] is None
    # softmax_when_solo group still resolves on supported members
    assert out["groups"]["hair_color"] == "black hair"


def test_people_count_rule_without_sidecar(tmp_path, monkeypatch):
    _write_ckpt(tmp_path)
    t = _tagger(tmp_path, {"1girl": 0.9, "1boy": 0.9}, monkeypatch)
    out = t.predict(IMG)
    assert out["people_count"] == "1girl_1boy"
    assert out["people_count_source"] == "count-tag-rule"


def test_sidecar_rows_become_emittable_and_people_head_wins(tmp_path, monkeypatch):
    _write_ckpt(tmp_path)
    bce_rows = [IDX["touhou"], IDX["shiro (mignon)"]]
    head = db.SidecarHead(
        d_in=8, bce_indices=bce_rows, people_count_labels=list(at.PEOPLE_COUNT_LABELS)
    )
    with torch.no_grad():
        head.fc.weight.zero_()
        head.fc.bias.zero_()
        head.fc.bias[0] = 4.0  # touhou → σ(4) ≈ 0.98
        head.fc.bias[1] = -4.0  # shiro → ≈ 0.02
        head.fc.bias[2 + at.PEOPLE_COUNT_LABELS.index("2girls")] = 5.0
    # thresholds for sidecar rows are written by the trainer
    t = _tagger(tmp_path, {"1girl": 0.9, "1boy": 0.9}, monkeypatch, sidecar=head)
    t.thresholds[IDX["touhou"]] = 0.5
    t.thresholds[IDX["shiro (mignon)"]] = 0.5
    t.thresholds_dev = t.thresholds.to(t.device)
    out = t.predict(IMG)
    assert "touhou" in out["kept"]
    assert "shiro (mignon)" not in out["kept"]  # σ(-4) < threshold
    # the count-tag rule stays authoritative; the head only adds scores.
    assert out["people_count"] == "1girl_1boy"
    assert out["people_count_source"] == "count-tag-rule"
    assert out["people_count_scores"]["2girls"] > 0.9


def test_oc_character_survives_original_without_artist(tmp_path, monkeypatch):
    """dbv4 emits no @artist, so the artist-consistency rule cannot drop an OC."""
    _write_ckpt(tmp_path)
    bce_rows = [IDX["original"], IDX["shiro (mignon)"]]
    head = db.SidecarHead(d_in=8, bce_indices=bce_rows)
    with torch.no_grad():
        head.fc.weight.zero_()
        head.fc.bias[:] = 4.0  # both ≈ 0.98
    t = _tagger(tmp_path, {"1girl": 0.9}, monkeypatch, sidecar=head)
    for n in ("original", "shiro (mignon)"):
        t.thresholds[IDX[n]] = 0.5
    t.thresholds_dev = t.thresholds.to(t.device)
    out = t.predict(IMG)
    assert "original" in out["kept"]
    assert "shiro (mignon)" in out["kept"]


# --------------------------------------------------------------------------- #
# 5. sidecar persistence
# --------------------------------------------------------------------------- #


def test_sidecar_round_trip(tmp_path):
    head = db.SidecarHead(d_in=8, bce_indices=[3, 7], people_count_labels=["a", "b"])
    head.save(tmp_path, extra_meta={"note": "x"})
    back = db.SidecarHead.load(tmp_path)
    assert back is not None
    assert back.bce_indices == (3, 7)
    assert back.people_count_labels == ("a", "b")
    assert torch.equal(back.fc.weight, head.fc.weight)
    assert json.loads((tmp_path / "sidecar.json").read_text())["note"] == "x"
    assert db.SidecarHead.load(tmp_path / "nope") is None


def test_unknown_backend_rejected(tmp_path):
    _write_ckpt(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"backend": "wat"}))
    with pytest.raises(ValueError, match="unsupported tagger backend"):
        at.AnimaTagger(tmp_path, device="cpu")


def test_legacy_pe_backend_rejected(tmp_path):
    """A legacy checkpoint (``backend`` absent or ``"pe"``) fails loudly at load."""
    _write_ckpt(tmp_path)
    for cfg in ({}, {"backend": "pe"}):
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        with pytest.raises(ValueError, match="'pe' dual-encoder head was removed"):
            at.AnimaTagger(tmp_path, device="cpu")


# --------------------------------------------------------------------------- #
# gated backbone preflight
# --------------------------------------------------------------------------- #


def test_backbone_preflight_skips_fetch_when_cached(tmp_path, monkeypatch):
    _write_ckpt(tmp_path)
    monkeypatch.setattr("anime_tools.tagger.dbv4_meta.backbone_cached", lambda _r: True)
    monkeypatch.setattr(
        "anime_tools._hf.hf_download",
        lambda **_k: (_ for _ in ()).throw(AssertionError("must not fetch")),
    )
    from anime_tools.tagger.dbv4_meta import backbone_repo_for

    assert at.ensure_tagger_backbone(tmp_path) == backbone_repo_for(tmp_path)


def test_backbone_preflight_fetches_every_file_when_uncached(tmp_path, monkeypatch):
    from anime_tools.tagger.dbv4_meta import DBV4_BACKBONE_FILES

    _write_ckpt(tmp_path)
    monkeypatch.delenv("ANIMA_TAGGER_NO_AUTOFETCH", raising=False)
    monkeypatch.setattr(
        "anime_tools.tagger.dbv4_meta.backbone_cached", lambda _r: False
    )
    calls = []
    monkeypatch.setattr(
        "anime_tools._hf.hf_download",
        lambda **kw: calls.append(kw["filename"]) or "/x",
    )
    at.ensure_tagger_backbone(tmp_path)
    assert calls == list(DBV4_BACKBONE_FILES)


def test_backbone_preflight_respects_no_autofetch(tmp_path, monkeypatch):
    _write_ckpt(tmp_path)
    monkeypatch.setenv("ANIMA_TAGGER_NO_AUTOFETCH", "1")
    monkeypatch.setattr(
        "anime_tools.tagger.dbv4_meta.backbone_cached", lambda _r: False
    )
    with pytest.raises(
        FileNotFoundError, match=r"anime_tools\.downloads tagger_backbone"
    ):
        at.ensure_tagger_backbone(tmp_path)


def test_hf_download_translates_gated_repo_error(monkeypatch):
    """A gated 401/403 fails fast and names the recovery."""
    import httpx
    import huggingface_hub
    from huggingface_hub.utils import GatedRepoError

    from anime_tools._hf import hf_download

    def _boom(**_k):
        resp = httpx.Response(401, request=httpx.Request("GET", "https://hf.co/r"))
        raise GatedRepoError(
            "401 Client Error: Cannot access gated repo", response=resp
        )

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom)
    with pytest.raises(FileNotFoundError, match="accept the terms"):
        hf_download(
            what="x",
            hint="hf auth login, then accept the terms at https://hf.co/r",
            repo_id="r",
            filename="model.safetensors",
        )
