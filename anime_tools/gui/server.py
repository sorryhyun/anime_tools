"""FastAPI app behind ``anime-tools-gui``.

Serves the static frontend and a small JSON/SSE API over the stage registry
(:mod:`anime_tools.gui.stages`) and the subprocess runner
(:mod:`anime_tools.gui.jobs`). Binds to localhost by default; ``--host 0.0.0.0``
opts into remote use (there is no auth — put it behind your own tunnel).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import os
import threading
import webbrowser
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from anime_tools import downloads as DL
from anime_tools._env import curation_home, models_dir, resolve_path, workspace_dir
from anime_tools._json import read_json
from anime_tools.gui import dataset as D
from anime_tools.gui import nativepick as NP
from anime_tools.gui import proposals as P
from anime_tools.gui import stages as S
from anime_tools.gui import tags as T
from anime_tools.gui.jobs import JobManager, Step
from anime_tools.gui.settings import load_settings, save_settings

STATIC = Path(__file__).parent / "static"


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"})
"""Who may open a window on this desktop: only the machine it is drawn on."""


def _is_loopback(request: Request) -> bool:
    client = request.client
    return client is not None and client.host in LOOPBACK_HOSTS


def _within(p: Path) -> bool:
    """Is this path one the panel may show a stranger's browser?"""
    try:
        D.reachable(p)
    except D.DatasetError:
        return False
    return True


def _start_dir(path: str) -> Path | None:
    """Where the host's chooser should open, given whatever the field holds.

    The field may be relative, may name a file, and may not exist yet, so walk
    up to the first directory that *is* there; ``None`` lets the desktop decide.
    """
    try:
        p = resolve_path(path) if path.strip() else curation_home()
    except (OSError, ValueError):
        return None
    for cand in (p, *p.parents):
        if cand.is_dir():
            return cand
    return None


def _hf_token_present() -> bool:
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:  # noqa: BLE001 - best effort probe
        return False


# ---- settings-derived values, as pure functions ---------------------------- #
# Each takes the settings mapping rather than reading it, so one request reads
# the file once and everything it derives agrees. ``D.DatasetError`` propagates
# to the app-wide 400 handler in ``create_app``.


def roots_for(settings: Mapping[str, Any], **overrides: str) -> D.Roots:
    """The dataset roots for this request: overrides win, blanks fall back to
    the saved roots, then to :data:`D.DEFAULT_ROOTS`."""
    saved = settings.get(D.SETTINGS_KEY) or {}
    merged = {**saved, **{k: v for k, v in overrides.items() if v}}
    return D.resolve_roots(merged)


def stage_defaults(settings: Mapping[str, Any]) -> dict[str, str]:
    """The Settings dialog's stage defaults (``S.SETTING_FIELDS``). Blanks are
    dropped, so an emptied field means "the CLI's own default"."""
    got = settings.get(S.SETTINGS_KEY) or {}
    return {
        k: str(got[k]).strip()
        for k in S.SETTING_FIELDS.values()
        if str(got.get(k) or "").strip()
    }


def report_root(settings: Mapping[str, Any], roots: D.Roots) -> str:
    """Where every stage's report lands, home-relative — the root only; each
    stage appends its own tail (``S.Field.report``), so no two share a
    ``--report_dir``. Blank means *beside the* ``dst`` *root*, so reports follow
    the resized tree they describe wherever the dataset moves."""
    got = str((settings.get(S.SETTINGS_KEY) or {}).get(S.REPORT_SETTING) or "").strip()
    return got or PurePosixPath(D.rel_to_home(roots.dst)).parent.as_posix()


def root_paths(roots: D.Roots) -> dict[str, str]:
    """The dataset roots, home-relative, for the fields bound to them
    (``S.ROOT_FIELDS``)."""
    return {k: v["path"] for k, v in roots.as_dict().items()}


def make_output_dirs(stage: S.Stage, report: str | None, roots: D.Roots) -> None:
    """Create the directories this run *writes* to, so a fresh home does not
    need a mkdir tour before the first job.

    Only the outputs the GUI itself chose: the workspace roots this stage binds
    (``D.OUTPUT_ROOTS`` ∩ ``S.ROOT_FIELDS``) and the report directory. Never
    ``src``, and never a free-text path off the stage form — an empty tree
    conjured behind a mistyped ``--source`` hides the typo. Never ``out``
    either, so an export tree that exists means an export happened.
    """
    for name in set(S.ROOT_FIELDS.get(stage.id, {}).values()) & D.OUTPUT_ROOTS:
        D.ensure_output_dir(getattr(roots, name))
    if report:
        D.ensure_output_dir(Path(report).parent)


def preprocess_steps(
    stage: S.Stage,
    *,
    defaults: dict[str, str],
    roots: D.Roots,
    settings: Mapping[str, Any],
    schemas: Mapping[str, Any],
) -> list[Step]:
    """The resize preflight for ``stage``, or nothing.

    It runs with the *same* ``defaults`` the stage got, so a per-image Apply
    resizes exactly that image and a batch resizes the batch; its own knobs come
    from the Settings ``preprocess`` block. A stage whose preflight is
    unavailable runs alone — resize is a convenience, not a gate.
    """
    pre_id = S.preprocess_for(stage.id)
    pre = S.BY_ID.get(pre_id or "")
    sc = schemas.get(pre_id or "")
    if pre is None or sc is None or not sc["available"]:
        return []
    saved = settings.get(S.PREPROCESS_SETTINGS_KEY) or {}
    values = S.form_values(sc["fields"], saved)
    reports = report_root(settings, roots)
    argv = S.build_argv(
        sc["fields"],
        values,
        roots=root_paths(roots),
        settings=defaults,
        report_root=reports,
    )
    make_output_dirs(pre, S.report_path(pre, sc["fields"], values, reports), roots)
    return [Step(pre.module, argv, pre.id)]


class Schemas:
    """The stage form schemas, kept off the startup path.

    Collecting them means a child interpreter (this process must stay
    torch-free — ``test_boundary`` pins that), which costs seconds on a cache
    miss, so the loader runs on a background thread and only the endpoints that
    need a schema block on it. A failing dump is a 500 there, not a dead server.
    """

    TIMEOUT = 60.0

    def __init__(self, value: dict[str, Any] | None = None) -> None:
        self._value = value
        self._error: BaseException | None = None
        self._ready = threading.Event()
        if value is not None:  # tests / callers that already have them
            self._ready.set()
            return
        threading.Thread(target=self._load, name="gui-schemas", daemon=True).start()

    def _load(self) -> None:
        try:
            self._value = S.load_schemas()
        except Exception as e:  # noqa: BLE001 - reported on /api/stages
            self._error = e
        finally:
            self._ready.set()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def get(self, timeout: float | None = None) -> dict[str, Any]:
        """Block until the dump lands. 503 while it is still running (the
        frontend shows that as a spinner), 500 if it failed."""
        if not self._ready.wait(self.TIMEOUT if timeout is None else timeout):
            raise HTTPException(503, "stage schemas are still loading")
        if self._error is not None:
            raise HTTPException(500, f"stage schema dump failed: {self._error}")
        return self._value or {}


def create_app(
    *, jobs: JobManager | None = None, schemas: dict[str, Any] | None = None
) -> FastAPI:
    # Job logs are curation output like everything else, so they live in the
    # workspace rather than in the tree Export publishes to.
    mgr = jobs or JobManager(log_dir=workspace_dir() / "gui_logs")
    # Passing schemas in short-circuits the background loader (see Schemas).
    store = Schemas(schemas)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        mgr.shutdown()

    app = FastAPI(title="anime_tools", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.jobs = mgr
    app.state.schemas = store

    # A refused path, root or report is a bad request, not a crash. The routes
    # that owe a *404* instead say so at the one call that can produce theirs.
    @app.exception_handler(D.DatasetError)
    @app.exception_handler(P.ProposalError)
    async def _bad_request(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        return {
            "home": str(curation_home()),
            "models_dir": str(models_dir()),
            "hf_token": _hf_token_present(),
            "running": mgr.running.id if mgr.running else None,
            "schemas_ready": store.ready,
        }

    @app.get("/api/stages")
    def list_stages() -> list[dict[str, Any]]:
        schemas = store.get()
        return [schemas[s.id] for s in S.STAGES if s.id in schemas]

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        return load_settings()

    @app.put("/api/settings")
    async def put_settings(request: Request) -> dict[str, Any]:
        body = await request.json()
        data = load_settings()
        token = body.pop("hf_token", None)
        data.update(body)
        save_settings(data)
        if token:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)
        return data

    # ---- model weights (the Settings dialog's download rows) -------------

    @app.get("/api/models")
    def list_models() -> dict[str, Any]:
        """The download catalog, re-probed per request: a job may have just
        installed one."""
        return {
            "models": [a.to_dict() for a in DL.catalog()],
            "models_dir": str(models_dir()),
        }

    @app.post("/api/models/download")
    async def download_models(request: Request) -> dict[str, Any]:
        """Fetch weights as a normal job, so a 3 GB pull cannot run under a
        stage. Empty ``ids`` means every missing model."""
        body = await request.json()
        ids = [str(i) for i in (body.get("ids") or [])]
        unknown = [i for i in ids if i not in DL.by_id()]
        if unknown:
            raise HTTPException(404, f"unknown model: {', '.join(unknown)}")
        try:
            job = mgr.start(
                f"download:{','.join(ids) or 'missing'}",
                [Step(DL.__name__, ids, label="download")],
                home=curation_home(),
            )
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
        return job.to_dict()

    @app.post("/api/jobs")
    async def start_job(request: Request) -> dict[str, Any]:
        """Start a stage.

        ``rel`` scopes the run to one dataset image by narrowing the stage's
        ``--path_pattern``; without it the run uses the Settings pattern.
        """
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
        # preflight's knobs all come out of it and must agree.
        settings = load_settings()
        roots = roots_for(settings)
        defaults = stage_defaults(settings)
        reports = report_root(settings, roots)
        if rel:
            # Refuse rather than silently run the batch: a stage with no
            # --path_pattern has nothing to narrow, and "apply to this image"
            # quietly meaning "apply to everything" is the worst outcome here.
            if not sc.get("scoped"):
                raise HTTPException(400, f"{stage.id} cannot be scoped to one image")
            defaults = {**defaults, S.SCOPE_FIELD: D.item_pattern(rel)}
        try:
            argv = S.build_argv(
                sc["fields"],
                values,
                apply=apply,
                roots=root_paths(roots),
                settings=defaults,
                report_root=reports,
            )
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        report = S.report_path(stage, sc["fields"], values, reports)
        make_output_dirs(stage, report, roots)
        steps = [
            *preprocess_steps(
                stage,
                defaults=defaults,
                roots=roots,
                settings=settings,
                schemas=store.get(),
            ),
            Step(stage.module, argv, stage.id),
        ]
        try:
            job = mgr.start(
                stage.id,
                steps,
                home=curation_home(),
                report_path=report,
                values=values,
                apply=apply,
            )
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
        # Re-read before writing: the job started, and a settings PUT could
        # have landed while it did — merging into the request's own snapshot
        # would silently roll that back.
        data = load_settings()
        data.setdefault("values", {})[stage.id] = values
        save_settings(data)
        return job.to_dict()

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return [j.to_dict() for j in sorted(mgr.jobs.values(), key=lambda j: j.started)]

    def _job(job_id: str):
        job = mgr.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return job

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return _job(job_id).to_dict()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        _job(job_id)
        return {"cancelled": mgr.cancel(job_id)}

    @app.get("/api/jobs/{job_id}/log")
    def job_log(job_id: str, offset: int = 0) -> StreamingResponse:
        job = _job(job_id)

        def gen():
            i = offset
            while True:
                new = job.wait_lines(i)
                for line in new:
                    yield f"data: {json.dumps(line)}\n\n"
                i += len(new)
                if job.exit_code is not None and i >= len(job.lines):
                    yield f"event: done\ndata: {json.dumps(job.to_dict())}\n\n"
                    return

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/jobs/{job_id}/report")
    def job_report(job_id: str) -> Response:
        job = _job(job_id)
        if not job.report_path:
            raise HTTPException(404, "stage has no report")
        p = resolve_path(job.report_path)
        if not p.exists():
            raise HTTPException(404, f"report not found: {p}")
        return JSONResponse({"path": str(p), "report": read_json(p)})

    # ---- proposals: a finished Run, read as a per-image diff -------------

    def _report_of(job_id: str) -> tuple[Any, Path]:
        job = _job(job_id)
        if not job.report_path:
            raise HTTPException(404, "stage writes no report")
        return job, resolve_path(job.report_path)

    def _proposals(job_id: str) -> tuple[Any, dict[str, P.Proposal]]:
        """A finished run and what it proposes, keyed by dataset rel.
        ``P.read`` is report-mtime cached, so a second ask is a dict lookup."""
        job, path = _report_of(job_id)
        return job, P.read(path, roots_for(load_settings()), job.stage)

    @app.get("/api/jobs/{job_id}/proposals")
    def job_proposals(job_id: str) -> dict[str, Any]:
        """Which dataset images this run wants to change — the index only.

        The before/after text of one image comes from ``/proposal`` as the
        selection lands on it, so opening a 2000-image batch's diff does not put
        2000 captions on the wire.
        """
        job, found = _proposals(job_id)
        return {
            "stage": job.stage,
            "apply": job.apply,
            "kind": P.CAPTION_KIND[P.SHAPES[job.stage].target_root],
            "total": len(found),
            "rels": sorted(found),
        }

    @app.get("/api/jobs/{job_id}/proposal")
    def job_proposal(job_id: str, rel: str) -> dict[str, Any]:
        """One image's pending change, both texts already parsed."""
        _, found = _proposals(job_id)
        got = found.get(rel)
        if got is None:
            raise HTTPException(404, f"no proposal for {rel}")
        return got.to_dict()

    @app.post("/api/jobs/{job_id}/undo")
    def job_undo(job_id: str) -> dict[str, Any]:
        """Put back the captions this run wrote.

        Refuses a dry run outright — it wrote nothing, so "undo" could only mean
        undoing some *other* run sharing the report shape.
        """
        job, path = _report_of(job_id)
        if not job.apply:
            raise HTTPException(400, "that run wrote nothing — nothing to undo")
        try:
            return P.undo(path, roots_for(load_settings()), job.stage)
        except OSError as e:
            raise HTTPException(500, f"undo failed: {e}") from e

    # ---- dataset browsing (the sidebar's image/caption tree) -------------

    @app.get("/api/dataset/roots")
    def dataset_roots() -> dict[str, Any]:
        settings = load_settings()
        return {
            "roots": roots_for(settings).as_dict(),
            "defaults": D.DEFAULT_ROOTS,
            # What a blank `report_root` resolves to, as Settings' placeholder.
            "report_root": report_root({}, roots_for(settings)),
        }

    @app.put("/api/dataset/roots")
    async def put_dataset_roots(request: Request) -> dict[str, Any]:
        body = await request.json()
        picked = {k: str(body.get(k) or "").strip() for k in D.DEFAULT_ROOTS}
        roots = D.resolve_roots(picked, trusted=True)
        # The only ``trusted`` resolve: saving Settings is what *defines* the
        # trees the panel may read (``dataset_bases``), so a root outside the
        # home is set here or nowhere.
        try:
            created = D.ensure_roots(roots)
        except OSError as e:
            raise HTTPException(500, f"cannot create root: {e}") from e
        data = load_settings()
        data[D.SETTINGS_KEY] = picked
        save_settings(data)
        return {
            "roots": roots.as_dict(),
            "defaults": D.DEFAULT_ROOTS,
            "report_root": report_root({}, roots),
            "created": created,
        }

    @app.get("/api/dataset")
    def dataset_list(
        src: str = "",
        dst: str = "",
        masks: str = "",
        pattern: str = "",
        q: str = "",
        limit: int = D.MAX_ITEMS,
    ) -> dict[str, Any]:
        return D.list_items(
            roots_for(load_settings(), src=src, dst=dst, masks=masks),
            pattern=pattern or None,
            query=q,
            limit=limit,
        )

    @app.get("/api/dataset/groups")
    def dataset_groups() -> dict[str, Any]:
        """The near-twin components the **Groups** stage wrote, rels only.

        The client joins them onto the ``/api/dataset`` rows it already has, so
        one filter and one truncation serve both sidebar orderings. The path is
        derived exactly like a stage's report, so the view reads the file the
        Groups panel writes wherever Settings points it.
        """
        settings = load_settings()
        return D.load_groups(report_root(settings, roots_for(settings)))

    @app.post("/api/dataset/items")
    async def dataset_items(request: Request) -> dict[str, Any]:
        """Refresh named sidebar rows: re-stat what a job wrote rather than
        re-walking the source root."""
        body = await request.json()
        rels = [str(r) for r in (body.get("rels") or [])][: D.MAX_ITEMS]
        roots = roots_for(
            load_settings(), **{k: str(body.get(k) or "") for k in D.DEFAULT_ROOTS}
        )
        return {"items": D.item_rows(roots, rels)}

    @app.get("/api/dataset/item")
    def dataset_item(
        rel: str, src: str = "", dst: str = "", masks: str = ""
    ) -> dict[str, Any]:
        roots = roots_for(load_settings(), src=src, dst=dst, masks=masks)
        try:
            return D.item_detail(roots, rel)
        except D.DatasetError as e:
            # 404, not the app-wide 400: the roots resolved, this image is
            # simply not in the dataset.
            raise HTTPException(404, str(e)) from e

    @app.put("/api/dataset/item")
    async def put_dataset_item(request: Request) -> dict[str, Any]:
        body = await request.json()
        roots = roots_for(
            load_settings(), **{k: str(body.get(k) or "") for k in D.DEFAULT_ROOTS}
        )
        try:
            return D.write_caption(
                roots,
                str(body.get("rel") or ""),
                str(body.get("kind") or ""),
                body.get("text") or "",
            )
        except OSError as e:
            raise HTTPException(500, f"write failed: {e}") from e

    @app.post("/api/dataset/parse")
    async def dataset_parse(request: Request) -> dict[str, Any]:
        """Parse an *unsaved* caption for the editor's live clause preview.

        The grammar has exactly one implementation (``position_clauses``) — the
        browser must never hand-split a caption on commas, so it asks instead.
        """
        body = await request.json()
        return D.parsed_caption(str(body.get("text") or ""))

    @app.get("/api/tags/describe")
    def describe_tag(tag: str) -> dict[str, Any]:
        """What one Danbooru tag means — the caption panel's click-a-tag panel.

        Answers even when the KB is not downloaded (``installed: false``), so
        the panel can point at Settings > Models instead of erroring.
        """
        tag = tag.strip()
        if not tag:
            raise HTTPException(400, "tag is required")
        return T.describe(tag)

    @app.get("/api/thumb")
    def thumb(path: str, size: int = 192) -> Response:
        try:
            data = D.thumbnail(path, size)
        except D.DatasetError as e:
            raise HTTPException(404, str(e)) from e
        except Exception as e:  # unreadable / corrupt image
            raise HTTPException(415, f"cannot thumbnail: {e}") from e
        return Response(
            data,
            media_type="image/webp",
            headers={"cache-control": "max-age=3600"},
        )

    @app.get("/api/files")
    def files(path: str) -> FileResponse:
        # reachable collapses ".." before the containment test — resolve_path +
        # is_relative_to alone is purely textual and lets traversal through.
        try:
            p = D.reachable(path)
        except D.DatasetError as e:
            raise HTTPException(404, "not found") from e
        if not p.is_file():
            raise HTTPException(404, "not found")
        mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        return FileResponse(p, media_type=mime)

    @app.post("/api/pick")
    async def pick_path(request: Request) -> dict[str, Any]:
        """Open the *host's* folder/file chooser and answer with what it got.

        The dialog opens on the machine running this server, so it is offered
        only to a browser on that same machine: from anywhere else it would be
        a window nobody can see, holding the request until it times out. That
        refusal and a host with no chooser both come back as
        ``available: false``, the panel's cue to fall back to ``/api/ls``.

        The answer is home-relative when it is under the home, absolute when it
        is not. A root outside the home is still refused, but by the save that
        means it, not by the browse.
        """
        if not _is_loopback(request):
            return {"available": False, "path": None}
        body = await request.json()
        kind = "dir" if str(body.get("kind") or "dir") != "file" else "file"
        # A dialog is as slow as the person in front of it, so it waits on a
        # thread; the loop keeps serving the panel behind it.
        res = await asyncio.to_thread(
            NP.pick,
            kind,
            _start_dir(str(body.get("path") or "")),
            title=str(body.get("title") or ""),
        )
        return {
            "available": res.available,
            "path": D.rel_to_home(Path(res.path)) if res.path else None,
        }

    @app.get("/api/ls")
    def ls(request: Request, path: str = "") -> dict[str, Any]:
        """Directory listing for the fallback path browser.

        The fallback has to reach the same places the host's own chooser can,
        or a host without one could never point a root at a sibling tree — so
        for a browser on *this* machine it walks anywhere, and ``parent`` is how
        it goes up (the client joins names but never takes a path apart). From
        anywhere else it stays inside ``D.dataset_bases``, the same tree the
        file and thumbnail routes serve.
        """
        try:
            p = (
                (D.lexical(path) if _is_loopback(request) else D.reachable(path))
                if path
                else curation_home()
            )
        except D.DatasetError as e:
            raise HTTPException(404, "not found") from e
        if not p.is_dir():
            raise HTTPException(404, "not found")
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        parent = p.parent
        return {
            # Home-relative inside the home, absolute outside it, "" at the home
            # itself -- the picker joins names onto this and hands it back.
            "path": "" if p == curation_home() else D.rel_to_home(p),
            # None at the filesystem root, and at the edge of what this client
            # may see: the ".." the picker draws is exactly this field.
            "parent": (
                D.rel_to_home(parent)
                if parent != p and (_is_loopback(request) or _within(parent))
                else None
            ),
            "entries": [
                {"name": e.name, "dir": e.is_dir()}
                for e in entries
                if not e.name.startswith(".")
            ][:500],
        }

    return app


def pick_port(host: str, preferred: int, *, tries: int = 50) -> int:
    """``preferred`` if bindable, else the first free port above it (``0`` = OS-chosen)."""
    import socket

    candidates = [0] if preferred == 0 else range(preferred, preferred + tries)
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise SystemExit(f"no free port in {preferred}..{preferred + tries - 1} on {host}")


_CHROMIUM_BINARIES = {
    "darwin": (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Vivaldi.app/Contents/MacOS/Vivaldi",
    ),
    "win32": (
        "chrome.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        "msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ),
    "linux": (
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "brave-browser",
        "microsoft-edge",
    ),
}


def _chromium_binary() -> str | None:
    """First Chromium-family browser on this machine, or ``None`` — only
    Chromium understands ``--app=URL``."""
    import shutil
    import sys

    for cand in _CHROMIUM_BINARIES.get(sys.platform, _CHROMIUM_BINARIES["linux"]):
        if os.path.isabs(cand):
            if os.path.exists(cand):
                return cand
        elif (found := shutil.which(cand)) is not None:
            return found
    return None


def _open_app_window(url: str) -> None:
    """Open ``url`` as a chromeless app window, falling back to a browser tab.

    ``--app=`` reuses the running browser's default profile, so there is no
    second session and no extra login.
    """
    import subprocess

    binary = _chromium_binary()
    if binary is not None:
        try:
            subprocess.Popen(
                [binary, f"--app={url}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except OSError:
            pass
    webbrowser.open(url)


def _open_when_ready(host: str, port: int, url: str, *, timeout: float = 60.0) -> None:
    """Open ``url`` once the server accepts connections: a fixed delay races
    startup and lands the browser on an error only a refresh clears."""
    import socket
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)
    else:
        return
    _open_app_window(url)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="anime_tools web GUI")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to expose on the LAN")
    p.add_argument(
        "--port",
        type=int,
        default=8790,
        help="Preferred port; if busy, the next free one above it is used (0 = let the OS pick)",
    )
    p.add_argument(
        "--home",
        default=None,
        help="Curation home (image_dataset/, workspace/, models/ live "
        "here). Default: $ANIME_TOOLS_HOME, $ANIMA_HOME, or the CWD",
    )
    p.add_argument(
        "--open",
        action="store_true",
        help="Open the GUI on start, in a chromeless Chromium app window if there is one",
    )
    args = p.parse_args(argv)
    if args.home:
        os.environ["ANIME_TOOLS_HOME"] = str(Path(args.home).expanduser().resolve())

    import uvicorn

    port = pick_port(args.host, args.port)
    if port != args.port and args.port != 0:
        print(f"port {args.port} is in use; using {port}", flush=True)
    connect_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{connect_host}:{port}"
    print(f"anime_tools GUI → {url}   (home: {curation_home()})", flush=True)
    app = create_app()
    if args.open:
        threading.Thread(
            target=_open_when_ready, args=(connect_host, port, url), daemon=True
        ).start()
    uvicorn.run(app, host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
