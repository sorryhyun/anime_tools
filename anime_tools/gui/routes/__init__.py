"""The API, one router per area. Every route reads its state off
``request.app.state`` (``jobs`` / ``schemas`` / ``watch``), which is what lets
them live outside ``create_app``.

:data:`ROUTERS` is the include order; ``server.py`` mounts them and owns nothing
but the app, the static files and ``/api/info``.
"""

from __future__ import annotations

from fastapi import APIRouter

from anime_tools.gui.routes.dataset import router as dataset_router
from anime_tools.gui.routes.desktop import router as desktop_router
from anime_tools.gui.routes.jobs import router as jobs_router
from anime_tools.gui.routes.settings import router as settings_router

__all__ = ["ROUTERS"]

ROUTERS: tuple[APIRouter, ...] = (
    settings_router,
    jobs_router,
    dataset_router,
    desktop_router,
)
