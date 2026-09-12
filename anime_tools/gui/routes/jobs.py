"""Stages and the runs over them: the form schemas, starting a job, watching one,
and reading a finished one back as a per-image diff.

``/api/alive`` sits here too — it is the stream the page holds open so the server
can tell it is still on screen (:class:`~anime_tools.gui.server.ClientWatch`),
which is the same kind of long-lived connection as a job's log.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from anime_tools._env import curation_home, resolve_path
from anime_tools._json import read_json
from anime_tools.gui import proposals as P
from anime_tools.gui import stages as S
from anime_tools.gui._context import RunContext, make_output_dirs, preprocess_steps
from anime_tools.gui.jobs import Step
from anime_tools.gui.settings import edit_settings

router = APIRouter()


def _job(request: Request, job_id: str):
    job = request.app.state.jobs.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


@router.get("/api/stages")
def list_stages(request: Request) -> list[dict[str, Any]]:
    """Every stage's form schema, in registry order.

    The schemas are built once at startup; the only settings-dependent
    part is the default of a field the panel may override
    (``S.PANEL_FIELDS``), which opens on what Settings says a Run would use.
    """
    schemas = request.app.state.schemas.get()
    ctx = RunContext.load()
    return [
        S.resolved_schema(schemas[s.id], **ctx.bindings())
        for s in S.STAGES
        if s.id in schemas
    ]


@router.post("/api/jobs")
async def start_job(request: Request) -> dict[str, Any]:
    """Start a stage.

    ``rel`` scopes the run to one dataset image by narrowing the stage's
    ``--path_pattern``; without it the run uses the Settings pattern.
    """
    store = request.app.state.schemas
    body = await request.json()
    stage = S.BY_ID.get(body.get("stage", ""))
    sc = store.get().get(stage.id) if stage else None
    if stage is None or sc is None:
        raise HTTPException(404, "unknown stage")
    if not sc["available"]:
        raise HTTPException(400, f"stage unavailable: {sc['error']}")
    values = S.form_values(sc["fields"], body.get("values") or {})
    apply = bool(body.get("apply"))
    rel = str(body.get("rel") or "").strip()
    # One read for the whole request: the roots, the stage defaults and the
    # preflight's knobs all come out of it.
    ctx = RunContext.load()
    if rel:
        # Refuse rather than silently run the batch: a stage with no
        # --path_pattern has nothing to narrow.
        if not sc.get("scoped"):
            raise HTTPException(400, f"{stage.id} cannot be scoped to one image")
        ctx = ctx.scoped_to(rel)
    try:
        argv = S.build_argv(sc, values, apply=apply, **ctx.bindings())
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    report = S.report_path(stage, sc["fields"], values, ctx.report_root)
    make_output_dirs(stage, report, ctx.roots)
    steps = [
        *preprocess_steps(stage, ctx, schemas=store.get()),
        Step(stage.module, argv, stage.id),
    ]
    try:
        job = request.app.state.jobs.start(
            stage.id,
            steps,
            home=curation_home(),
            report_path=report,
            values=values,
            apply=apply,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    # Read *and* write under the settings lock: a settings PUT could have
    # landed while the job started, and merging into this request's snapshot
    # would roll it back.
    with edit_settings() as data:
        data.setdefault("values", {})[stage.id] = values
    return job.to_dict()


@router.get("/api/jobs")
def list_jobs(request: Request) -> list[dict[str, Any]]:
    jobs = request.app.state.jobs.jobs
    return [j.to_dict() for j in sorted(jobs.values(), key=lambda j: j.started)]


@router.get("/api/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> dict[str, Any]:
    return _job(request, job_id).to_dict()


@router.post("/api/jobs/{job_id}/cancel")
def cancel_job(request: Request, job_id: str) -> dict[str, Any]:
    _job(request, job_id)
    return {"cancelled": request.app.state.jobs.cancel(job_id)}


@router.get("/api/jobs/{job_id}/log")
def job_log(request: Request, job_id: str, offset: int = 0) -> StreamingResponse:
    job = _job(request, job_id)

    def gen():
        # Absolute line numbers, which is what ``wait_lines`` returns: the
        # buffer is a sliding window on a long run, so an index into it
        # would point past the trimmed prefix and drop lines silently.
        i = offset
        while True:
            new, i = job.wait_lines(i)
            for line in new:
                yield f"data: {json.dumps(line)}\n\n"
            if job.exit_code is not None and i >= job.total_lines:
                yield f"event: done\ndata: {json.dumps(job.to_dict())}\n\n"
                return

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/api/alive")
async def alive(request: Request) -> StreamingResponse:
    """The stream the page holds open so the server can tell it is still
    there (:class:`~anime_tools.gui.server.ClientWatch`). It carries nothing but
    keep-alive comments; without a watch it is an idle connection the browser is
    free to keep."""
    watch = getattr(request.app.state, "watch", None)

    async def gen():
        if watch is not None:
            watch.attach()
        try:
            yield b"retry: 1000\n\n"
            while not await request.is_disconnected():
                await asyncio.sleep(15)
                yield b": ping\n\n"
        finally:
            if watch is not None:
                watch.detach()

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/api/jobs/{job_id}/report")
def job_report(request: Request, job_id: str) -> Response:
    job = _job(request, job_id)
    if not job.report_path:
        raise HTTPException(404, "stage has no report")
    p = resolve_path(job.report_path)
    if not p.exists():
        raise HTTPException(404, f"report not found: {p}")
    return JSONResponse({"path": str(p), "report": read_json(p)})


# ---- proposals: a finished Run, read as a per-image diff -------------


def _report_of(request: Request, job_id: str) -> tuple[Any, Path]:
    job = _job(request, job_id)
    if not job.report_path:
        raise HTTPException(404, "stage writes no report")
    return job, resolve_path(job.report_path)


def _proposals(request: Request, job_id: str) -> tuple[Any, dict[str, P.Proposal]]:
    """A finished run and what it proposes, keyed by dataset rel. ``P.read``
    is report-mtime cached, so a second ask is a dict lookup."""
    job, path = _report_of(request, job_id)
    return job, P.read(path, RunContext.load().roots, job.stage)


@router.get("/api/jobs/{job_id}/proposals")
def job_proposals(request: Request, job_id: str) -> dict[str, Any]:
    """Which dataset images this run wants to change — the index only.

    The before/after text of one image comes from ``/proposal`` as the
    selection lands on it, so a large batch's diff stays off the wire.
    """
    job, found = _proposals(request, job_id)
    return {
        "stage": job.stage,
        "apply": job.apply,
        "kind": P.CAPTION_KIND[P.SHAPES[job.stage].target_root],
        "total": len(found),
        "rels": sorted(found),
    }


@router.get("/api/jobs/{job_id}/proposal")
def job_proposal(request: Request, job_id: str, rel: str) -> dict[str, Any]:
    """One image's pending change, both texts already parsed."""
    _, found = _proposals(request, job_id)
    got = found.get(rel)
    if got is None:
        raise HTTPException(404, f"no proposal for {rel}")
    return got.to_dict()


@router.post("/api/jobs/{job_id}/undo")
def job_undo(request: Request, job_id: str) -> dict[str, Any]:
    """Put back the captions this run wrote. A dry run is refused: it wrote
    nothing, so an undo could only touch some other run's writes."""
    job, path = _report_of(request, job_id)
    if not job.apply:
        raise HTTPException(400, "that run wrote nothing — nothing to undo")
    try:
        return P.undo(path, RunContext.load().roots, job.stage)
    except OSError as e:
        raise HTTPException(500, f"undo failed: {e}") from e
