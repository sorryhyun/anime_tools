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
