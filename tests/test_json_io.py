"""``_json.write_json`` / ``read_json``: non-ASCII escaping, read encoding, parent mkdir."""

from __future__ import annotations

import json

import pytest

from anime_tools._json import read_json, write_json


def test_non_ascii_stays_readable(tmp_path):
    """Non-ASCII dataset paths are written unescaped."""
    p = write_json(tmp_path / "r.json", {"path": "이미지/소녀.webp", "tag": "1girl"})
    assert "이미지/소녀.webp" in p.read_text(encoding="utf-8")
    assert "\\u" not in p.read_text(encoding="utf-8")


def test_round_trips_and_returns_the_path(tmp_path):
    payload = {"a": [1, 2, {"b": None}], "c": "김"}
    p = write_json(tmp_path / "deep" / "r.json", payload)
    assert p == tmp_path / "deep" / "r.json"
    assert read_json(p) == payload


def test_creates_the_parent_directory(tmp_path):
    write_json(tmp_path / "a" / "b" / "c.json", {})
    assert (tmp_path / "a" / "b" / "c.json").is_file()


def test_reads_utf8_regardless_of_the_platform_locale(tmp_path):
    """Reads are UTF-8, not the locale codepage."""
    p = tmp_path / "r.json"
    p.write_bytes(json.dumps({"k": "값"}, ensure_ascii=False).encode("utf-8"))
    assert read_json(p) == {"k": "값"}


def test_indent_is_two_by_default_and_overridable(tmp_path):
    p = write_json(tmp_path / "a.json", {"k": 1})
    assert p.read_text(encoding="utf-8") == '{\n  "k": 1\n}'
    q = write_json(tmp_path / "b.json", {"k": 1}, indent=None)
    assert q.read_text(encoding="utf-8") == '{"k": 1}'


def test_no_trailing_newline(tmp_path):
    """No trailing newline: skip checks compare bytes."""
    p = write_json(tmp_path / "a.json", [1])
    assert not p.read_text(encoding="utf-8").endswith("\n")


def test_read_json_propagates_a_malformed_file(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{oops", encoding="utf-8")
    with pytest.raises(ValueError):
        read_json(p)
