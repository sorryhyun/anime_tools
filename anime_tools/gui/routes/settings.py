"""Settings, model weights, the guidebook and the self-update.

What the ☰ menu's dialogs read and write, none of it touching the dataset or a
job's output: the settings file itself, the download catalog, the manual, and
the two update routes. The two that install something (weights, a release) start
a normal job, so a large pull or an upgrade cannot land under a running Run —
they share the one slot with the stages.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from anime_tools import downloads as DL
from anime_tools._env import curation_home, models_dir
from anime_tools.gui import guidebook as GB
from anime_tools.gui import updates as UP
from anime_tools.gui.jobs import Step
from anime_tools.gui.settings import edit_settings, load_settings

router = APIRouter()


@router.get("/api/settings")
def get_settings() -> dict[str, Any]:
    return load_settings()


@router.put("/api/settings")
async def put_settings(request: Request) -> dict[str, Any]:
    body = await request.json()
    token = body.pop("hf_token", None)
    with edit_settings() as data:
        data.update(body)
    if token:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    return data


# ---- model weights (the Settings dialog's download rows) -------------


@router.get("/api/models")
def list_models() -> dict[str, Any]:
    """The download catalog, re-probed per request: a job may have just
    installed one. ``packs`` is the display grouping (:data:`DL.PACKS`
    order, only packs with rows); each model names its pack."""
    rows = DL.catalog()
    packed = DL.by_pack(rows)
    return {
        "packs": [
            {"id": p.id, "title": p.title, "description": p.description}
            for p in DL.PACKS
            if p.id in packed
        ],
        "models": [a.to_dict() for a in rows],
        "models_dir": str(models_dir()),
    }


@router.post("/api/models/download")
async def download_models(request: Request) -> dict[str, Any]:
    """Fetch weights as a normal job, so a large pull cannot run under a
    stage. Empty ``ids`` means every missing model; a pack id stands for
    its rows, expanded here so the job name stays row-level."""
    body = await request.json()
    try:
        ids = DL.expand([str(i) for i in (body.get("ids") or [])])
    except KeyError as e:
        raise HTTPException(404, f"unknown model: {e.args[0]}") from e
    try:
        job = request.app.state.jobs.start(
            f"download:{','.join(ids) or 'missing'}",
            [Step(DL.__name__, ids, label="download")],
            home=curation_home(),
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return job.to_dict()


@router.get("/api/guidebook")
def guidebook(lang: str = "en") -> dict[str, str]:
    """The manual the menu opens, as markdown; the browser renders it.

    Unrendered on purpose: the server has no markdown library and the page
    already owns how the panel looks, so a book stays one plain file that
    GitHub and the modal both read.
    """
    return GB.load(lang)


# ---- self-update (the Settings dialog's Update pane) -----------------


@router.get("/api/update")
def update_status(force: bool = False) -> dict[str, Any]:
    """Installed vs latest, cached six hours (:mod:`gui.updates`).

    A plain ``def`` on purpose: FastAPI runs it in the threadpool, so the
    GitHub call cannot stall the event loop while a job is streaming.
    """
    return UP.check(force=force)


@router.post("/api/update/run")
async def run_update(request: Request) -> dict[str, Any]:
    """Install a release as a normal job, so it shares the one slot with the
    stages -- an upgrade must not land under a running Run.

    A blank ``version`` lets the child resolve the latest tag itself.
    """
    body = await request.json()
    tag = str(body.get("version") or "").strip()
    why = UP.refusal()
    if why is not None:
        raise HTTPException(409, why)
    try:
        job = request.app.state.jobs.start(
            f"{UP.JOB_PREFIX}{tag or 'latest'}",
            UP.steps(tag or None),
            home=curation_home(),
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return job.to_dict()
