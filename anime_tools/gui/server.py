"""The FastAPI app behind ``anime-tools-gui``: the factory, and nothing else.

Serves the static frontend and mounts the API (:mod:`anime_tools.gui.routes`,
one router per area) over the stage registry (:mod:`anime_tools.gui.stages`) and
the subprocess runner (:mod:`anime_tools.gui.jobs`). What a route needs beyond a
request lives on ``app.state`` — the job manager, the schema loader, the client
watch — which is what lets the routers live outside this function.

Binding and the app window are :mod:`anime_tools.gui.launch`'s; it binds to
localhost by default, and ``--host 0.0.0.0`` opts into remote use (there is no
auth — put it behind your own tunnel).
"""

from __future__ import annotations

import mimetypes
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from anime_tools import __version__
from anime_tools._env import curation_home, models_dir, workspace_dir
from anime_tools.gui import dataset as D
from anime_tools.gui import nativepick as NP
from anime_tools.gui import proposals as P
from anime_tools.gui import stages as S
from anime_tools.gui._context import NO_CACHE, is_loopback
from anime_tools.gui.jobs import JobManager
from anime_tools.gui.routes import ROUTERS

# `mimetypes` has no built-in font rows -- it learns them from /etc/mime.types,
# which Windows has no equivalent of (it reads the registry, where woff2 is
# absent), so `/assets/<the bundled font>` goes out as application/octet-stream
# and the browser drops the face. Register the two we serve rather than let the
# answer depend on what the host happens to have installed.
mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("font/woff", ".woff")

STATIC = Path(__file__).parent / "static"


def _hf_token_present() -> bool:
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:  # noqa: BLE001 - best effort probe
        return False


class Schemas:
    """The stage form schemas, kept off the startup path.

    Building them imports every request module (torch-free, but numpy and PIL
    come along), so the loader runs on a background thread and only the
    endpoints that need a schema block on it. A failing build is a 500 there,
    not a dead server.
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


class ClientWatch:
    """Exit with the app window.

    The page holds one ``/api/alive`` stream open for as long as it is on screen,
    and the server stops itself once the last one has been gone for ``grace``
    seconds — closing the window reaps the server the way closing the trainer's
    window reaps what it started. A reload reconnects well inside the grace, and
    an open stream also survives a backgrounded tab, which a polled heartbeat
    would not: a hidden tab's timers are throttled to once a minute.

    Nothing is armed until the first client attaches, so a ``--open`` whose
    browser never appears leaves the server running rather than exiting behind
    the user's back.
    """

    def __init__(self, stop: Callable[[], None], *, grace: float = 5.0) -> None:
        self._stop = stop
        self._grace = grace
        self._lock = threading.Lock()
        self._clients = 0
        self._timer: threading.Timer | None = None

    def attach(self) -> None:
        with self._lock:
            self._clients += 1
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def detach(self) -> None:
        with self._lock:
            self._clients = max(0, self._clients - 1)
            if self._clients:
                return
            self._timer = threading.Timer(self._grace, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        with self._lock:
            if self._clients:  # a reload got back in under the grace
                return
        self._stop()


def create_app(
    *,
    jobs: JobManager | None = None,
    schemas: dict[str, Any] | None = None,
    watch: ClientWatch | None = None,
) -> FastAPI:
    # Job logs are curation output, so they live in the workspace.
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
    app.state.watch = watch

    # A refused path, root or report is a bad request, not a crash. Routes that
    # owe a 404 instead say so at the call that produces theirs.
    @app.exception_handler(D.DatasetError)
    @app.exception_handler(P.ProposalError)
    async def _bad_request(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers=NO_CACHE)

    @app.get("/assets/{name}")
    def asset(name: str) -> FileResponse:
        """Serve a sibling of index.html -- today only the bundled woff2.

        The bundle inlines its script and stylesheet but not the 1.7 MB font.
        A path param never spans "/", so `name` cannot climb out of STATIC; the
        checks below are for a name that simply is not an asset.
        """
        p = STATIC / name
        if name == "index.html" or name.startswith(".") or not p.is_file():
            raise HTTPException(404, "not found")
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        return FileResponse(p, media_type=mime, headers=NO_CACHE)

    @app.get("/api/info")
    def info(request: Request) -> dict[str, Any]:
        return {
            "home": str(curation_home()),
            "models_dir": str(models_dir()),
            "version": __version__,
            "hf_token": _hf_token_present(),
            "running": mgr.running.id if mgr.running else None,
            "schemas_ready": store.ready,
            # Whether ``POST /api/reveal`` would do anything for *this* client:
            # a desktop to open, and a browser on the machine holding it. The
            # panel draws no reveal button when it is false, which is the whole
            # of what it is for.
            "can_reveal": is_loopback(request) and NP.can_reveal(),
        }

    for router in ROUTERS:
        app.include_router(router)
    return app
