"""FastAPI app behind ``anime-tools-gui``.

Serves the static frontend and a small JSON/SSE API over the stage registry
(:mod:`anime_tools.gui.stages`) and the subprocess runner
(:mod:`anime_tools.gui.jobs`). Binds to localhost by default; ``--host 0.0.0.0``
opts into remote use (there is no auth — put it behind your own tunnel).
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from anime_tools._env import curation_home, models_dir, resolve_path
from anime_tools.gui import stages as S
from anime_tools.gui.jobs import JobManager

STATIC = Path(__file__).parent / "static"
SETTINGS_NAME = ".anime_tools_gui.json"


def _settings_path() -> Path:
    return curation_home() / SETTINGS_NAME


def load_settings() -> dict[str, Any]:
    p = _settings_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}


def save_settings(data: dict[str, Any]) -> None:
    _settings_path().write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _hf_token_present() -> bool:
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:  # noqa: BLE001 - best effort probe
        return False


def create_app(
    *, jobs: JobManager | None = None, schemas: dict[str, Any] | None = None
) -> FastAPI:
    mgr = jobs or JobManager(
        log_dir=curation_home() / "post_image_dataset" / "gui_logs"
    )
    # Schemas come from a child interpreter: one stage CLI imports torch at
    # module level, and this process must stay light (test_boundary pins it).
    schemas = S.load_schemas() if schemas is None else schemas

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        mgr.shutdown()

    app = FastAPI(title="anime_tools", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.jobs = mgr
    app.state.schemas = schemas

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
        }

    @app.get("/api/stages")
    def list_stages() -> list[dict[str, Any]]:
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

    @app.post("/api/jobs")
    async def start_job(request: Request) -> dict[str, Any]:
        body = await request.json()
        stage = S.BY_ID.get(body.get("stage", ""))
        sc = schemas.get(stage.id) if stage else None
        if stage is None or sc is None:
            raise HTTPException(404, "unknown stage")
        if not sc["available"]:
            raise HTTPException(400, f"stage unavailable: {sc['error']}")
        values = body.get("values") or {}
        apply = bool(body.get("apply"))
        try:
            argv = S.build_argv(sc["fields"], values, apply=apply)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        try:
            job = mgr.start(
                stage.id,
                stage.module,
                argv,
                home=curation_home(),
                report_path=S.report_path(stage, sc["fields"], values),
                values=values,
                apply=apply,
            )
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
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
        return JSONResponse(
            {"path": str(p), "report": json.loads(p.read_text(encoding="utf-8"))}
        )

    @app.get("/api/files")
    def files(path: str) -> FileResponse:
        p = resolve_path(path)
        home = curation_home()
        if not p.is_relative_to(home) or not p.is_file():
            raise HTTPException(404, "not found")
        mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        return FileResponse(p, media_type=mime)

    @app.get("/api/ls")
    def ls(path: str = "") -> dict[str, Any]:
        """Directory listing for the path picker, restricted to the home tree."""
        home = curation_home()
        p = resolve_path(path) if path else home
        if not p.is_relative_to(home) or not p.is_dir():
            raise HTTPException(404, "not found")
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        return {
            "path": str(p.relative_to(home)) if p != home else "",
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
        help="Curation home (image_dataset/, post_image_dataset/, models/ live "
        "here). Default: $ANIME_TOOLS_HOME, $ANIMA_HOME, or the CWD",
    )
    p.add_argument("--open", action="store_true", help="Open the browser on start")
    args = p.parse_args(argv)
    if args.home:
        os.environ["ANIME_TOOLS_HOME"] = str(Path(args.home).expanduser().resolve())

    import uvicorn

    port = pick_port(args.host, args.port)
    if port != args.port and args.port != 0:
        print(f"port {args.port} is in use; using {port}", flush=True)
    url = f"http://{args.host if args.host != '0.0.0.0' else '127.0.0.1'}:{port}"
    print(f"anime_tools GUI → {url}   (home: {curation_home()})", flush=True)
    if args.open:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(), host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
