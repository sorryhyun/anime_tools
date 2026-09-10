"""Self-update: version ordering, the install-shape probe, and the cached check.

Nothing here goes to the network — every test replaces
``anime_tools.update.latest_release``, and the one job that is actually started
refuses itself in the child (the checkout this repo is, not a uv tool), so no
test can rewrite the environment it is running in.
"""

from __future__ import annotations

import time

import pytest

from anime_tools import update as U
from anime_tools.gui import updates as UP


@pytest.mark.parametrize(
    "text, want",
    [
        ("v0.6.0", (0, 6, 0)),
        ("0.6.0", (0, 6, 0)),
        ("V1.2", (1, 2)),
        ("0+unknown", None),  # a checkout with nothing installed
        ("v0.6.0rc1", None),  # a pre-release is not something to order
        ("", None),
    ],
)
def test_parse_version(text, want):
    assert U.parse_version(text) == want


@pytest.mark.parametrize(
    "current, latest, want",
    [
        ("0.5.1", "v0.6.0", U.AVAILABLE),
        ("0.5.1", "v0.5.2", U.AVAILABLE),
        ("0.5.1", "v0.5.1", U.CURRENT),
        ("0.6.0", "v0.5.1", U.AHEAD),
        ("0+unknown", "v0.6.0", U.UNKNOWN),
        ("0.5.1", "nightly", U.UNKNOWN),
    ],
)
def test_compare_versions(current, latest, want):
    assert U.compare_versions(current, latest) == want


def test_install_kind_reads_the_uv_receipt(tmp_path, monkeypatch):
    """A uv tool environment is the one shape this may rewrite, and
    ``uv-receipt.toml`` beside its ``pyvenv.cfg`` is what says so."""
    monkeypatch.setattr(U.sys, "prefix", str(tmp_path))
    assert U.install_kind() in ("checkout", "other")  # no receipt yet
    (tmp_path / "uv-receipt.toml").write_text("[tool]\n", encoding="utf-8")
    assert U.install_kind() == "uv-tool"
    assert U.can_update()
    assert UP.refusal() is None


def test_checkout_is_refused_with_its_own_hint(monkeypatch):
    """This repo is a checkout, so the panel's button is off and the route 409s
    with the sentence the pane shows."""
    monkeypatch.setattr(U, "install_kind", lambda: "checkout")
    assert not U.can_update()
    assert UP.refusal() == U.INSTALL_HINTS["checkout"]
    assert U.run_update("v9.9.9") == 2


def test_update_argv_is_the_installers_command():
    argv = U.update_argv("v0.6.0", overrides="/tmp/o.txt", index="https://cpu")
    assert argv[:4] == ["uv", "tool", "install", "--force"]
    assert argv[argv.index("--overrides") + 1] == "/tmp/o.txt"
    assert argv[argv.index("--index") + 1] == "https://cpu"
    assert argv[-1] == f"{U.PACKAGE} @ git+{U.REPO_URL}@v0.6.0"


def test_windows_update_defaults_to_the_installers_torch_index():
    """`uv tool install` enables no dependency group, so the cu132 binding the
    checkout gets from `cuda-windows` has to ride the argv on Windows -- the
    same `--index` install.ps1 passes. Elsewhere torch comes from PyPI."""
    assert U.default_index("win32") == U.WINDOWS_TORCH_INDEX
    assert U.default_index("linux") is None
    assert U.default_index("darwin") is None


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A curation home of its own: the check writes its cache into the settings
    file beside it, and the checkout's own must not be touched."""
    monkeypatch.setenv("ANIME_TOOLS_HOME", str(tmp_path))
    return tmp_path


def _release(tag: str = "v9.9.9"):
    def fake(*_a, **_kw):
        return U.Release(tag=tag, notes="what is new", url=f"{U.REPO_URL}/x")

    return fake


def _explode(*_a, **_kw):
    raise OSError("no network here")


def test_check_caches_the_answer(home, monkeypatch):
    monkeypatch.setattr(U, "latest_release", _release())
    first = UP.check()
    assert first["checked"] and first["latest"] == "v9.9.9"
    assert first["status"] == U.compare_versions(first["current"], "v9.9.9")
    assert first["notes"] == "what is new"

    # Inside the TTL nothing goes out again -- a fetch here would raise.
    monkeypatch.setattr(U, "latest_release", _explode)
    again = UP.check()
    assert not again["checked"] and again["latest"] == "v9.9.9"
    assert not again["error"]

    # …and "Check now" is the way past it -- the failure is reported, never raised.
    assert UP.check(force=True)["error"] == "no network here"


def test_stale_cache_is_refreshed(home, monkeypatch):
    monkeypatch.setattr(U, "latest_release", _release("v1.0.0"))
    UP.check()
    settings = UP.load_settings()
    settings[UP.CACHE_KEY]["checked_at"] = int(time.time()) - UP.CACHE_TTL - 1
    UP.save_settings(settings)

    monkeypatch.setattr(U, "latest_release", _release("v2.0.0"))
    assert UP.check()["latest"] == "v2.0.0"


def test_a_failed_check_keeps_the_last_answer(home, monkeypatch):
    monkeypatch.setattr(U, "latest_release", _release("v1.0.0"))
    UP.check()
    monkeypatch.setattr(U, "latest_release", _explode)
    out = UP.check(ttl=0)
    assert out["error"] == "no network here"
    assert out["latest"] == "v1.0.0"  # yesterday's tag beats a blank row


def test_auto_off_never_reaches_github(home, monkeypatch):
    """The checkbox is the whole rate limit: with it off the panel renders what
    it already knows and asks nothing."""
    monkeypatch.setattr(U, "latest_release", _explode)
    settings = UP.load_settings()
    settings[UP.AUTO_KEY] = False
    UP.save_settings(settings)

    out = UP.check()
    assert out["auto_check"] is False
    assert out["latest"] == "" and out["status"] == U.UNKNOWN and not out["error"]

    # …but the button still can.
    assert UP.check(force=True)["error"] == "no network here"


def test_steps_run_the_update_module():
    (step,) = UP.steps("v0.6.0")
    assert step.command()[1:] == ["-m", "anime_tools.update", "--version", "v0.6.0"]
    (latest,) = UP.steps()
    assert latest.command()[1:] == ["-m", "anime_tools.update"]


# -- the routes -----------------------------------------------------------


@pytest.fixture
def client(home, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from anime_tools.gui.jobs import JobManager
    from anime_tools.gui.server import create_app

    app = create_app(jobs=JobManager(log_dir=home / "logs"), schemas={})
    with TestClient(app) as c:
        yield c


def test_info_carries_the_installed_version(client):
    assert client.get("/api/info").json()["version"] == U.current_version()


def test_update_route_answers_the_cached_check(client, monkeypatch):
    monkeypatch.setattr(U, "latest_release", _release("v9.9.9"))
    body = client.get("/api/update").json()
    assert body["latest"] == "v9.9.9" and body["checked"]
    assert body["install"] == U.install_kind()

    monkeypatch.setattr(U, "latest_release", _explode)
    assert not client.get("/api/update").json()["checked"]


def test_run_is_refused_for_an_install_uv_does_not_own(client, monkeypatch):
    monkeypatch.setattr(U, "install_kind", lambda: "checkout")
    r = client.post("/api/update/run", json={})
    assert r.status_code == 409 and r.json()["detail"] == U.INSTALL_HINTS["checkout"]


def test_run_starts_one_update_job(client, monkeypatch):
    """The job is `python -m anime_tools.update --version …`. The child is this
    checkout, so it refuses itself and exits 2 — which is the point: nothing is
    installed, and the argv is still the one the pane asked for."""
    monkeypatch.setattr(UP, "refusal", lambda: None)
    r = client.post("/api/update/run", json={"version": "v9.9.9"})
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["stage"] == "update:v9.9.9"
    assert job["argv"][1:] == ["-m", "anime_tools.update", "--version", "v9.9.9"]

    for _ in range(200):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["state"] != "running":
            break
        time.sleep(0.05)
    assert job["state"] == "failed" and job["exit_code"] == 2
