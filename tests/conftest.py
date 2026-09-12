from __future__ import annotations

import os
from pathlib import Path

# CPU-only unless opted in.
if os.environ.get("ANIMA_TEST_GPU") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session", autouse=True)
def _chdir_repo_root(repo_root: Path):
    """``curation_home()`` falls back to the CWD — pin it to the checkout."""
    prev = os.getcwd()
    os.chdir(repo_root)
    try:
        yield
    finally:
        os.chdir(prev)


def write_png(path: Path, size: tuple[int, int] = (8, 8), colour: int = 128) -> None:
    """One small solid PNG, parents created — what a test means by "an image".

    ``colour`` is the grey level, which is the only pixel content any test
    cares about: the near-twin feature cache keys on it, and everything else
    just needs a file the walkers see. Lives here, and is imported rather than
    injected, because half its callers are module-level helpers rather than
    tests.
    """
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (colour, colour, colour)).save(path)


@pytest.fixture
def png():
    """:func:`write_png` as a fixture, for a test that would rather ask."""
    return write_png


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An empty curation home with ``ANIME_TOOLS_HOME`` pinned at it, and the
    models-dir override cleared so the catalog answers ``<home>/models``.

    One level *under* ``tmp_path``, not ``tmp_path`` itself, so that a test
    about a root beside the home writes its sibling tree inside the temp
    directory pytest cleans up rather than into the basetemp parent.

    A module that wants a *populated* home overrides this fixture and requests
    this one by the same name, which is how pytest spells "the next one up".
    """
    root = tmp_path / "curation_home"
    root.mkdir()
    monkeypatch.setenv("ANIME_TOOLS_HOME", str(root))
    monkeypatch.delenv("ANIME_TOOLS_MODELS", raising=False)
    return root


def make_gui_app(home: Path, **kwargs):
    """``create_app`` with a job manager under ``home`` — how every GUI test
    boots the server. Imported rather than injected for the same reason as
    :func:`write_png`; :func:`gui_app` is the fixture over it."""
    from anime_tools.gui.jobs import JobManager
    from anime_tools.gui.server import create_app

    kwargs.setdefault("jobs", JobManager(log_dir=Path(home) / "logs"))
    kwargs.setdefault("schemas", {})
    return create_app(**kwargs)


@pytest.fixture
def gui_app():
    """:func:`make_gui_app` as a fixture."""
    return make_gui_app


@pytest.fixture
def gui_client():
    """A ``TestClient`` factory whose clients close with the test, so a caller
    needs no ``with``. ``client=(host, port)`` reaches ``TestClient``;
    everything else reaches ``create_app``."""
    from contextlib import ExitStack

    with ExitStack() as stack:

        def make(home: Path, *, client=None, **kwargs):
            from fastapi.testclient import TestClient

            app = make_gui_app(home, **kwargs)
            args = {} if client is None else {"client": client}
            return stack.enter_context(TestClient(app, **args))

        yield make
