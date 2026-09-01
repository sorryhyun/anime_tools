"""Web GUI over the curation stages (``anime-tools-gui``).

The server process never imports torch; stages run as ``python -m`` subprocesses.
"""


def main() -> None:  # console-script entry
    from anime_tools.gui.server import main as _main

    _main()
