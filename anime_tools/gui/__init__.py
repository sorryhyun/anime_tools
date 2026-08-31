"""Small standalone web GUI over the curation stages (``anime-tools-gui``).

FastAPI + uvicorn are plain dependencies. The server process never imports
torch; stages run as ``python -m`` subprocesses.
"""


def main() -> None:  # console-script entry
    from anime_tools.gui.server import main as _main

    _main()
