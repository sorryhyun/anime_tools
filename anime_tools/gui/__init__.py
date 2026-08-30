"""Small standalone web GUI over the curation stages (``anime-tools-gui``).

Needs the ``gui`` extra (FastAPI + uvicorn). The server process never imports
torch; stages run as ``python -m`` subprocesses. See ``docs/gui_plan.md``.
"""


def main() -> None:  # console-script entry
    from anime_tools.gui.server import main as _main

    _main()
