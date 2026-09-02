"""The model catalog, the stage registry, and driving stages from outside.

Torch-free — ``python examples/models_registry_gui.py``.

- ``anime_tools.downloads`` is the single source of truth for weight locations:
  one ``Asset`` per checkpoint with an offline ``installed`` probe and a fetch.
  ``python -m anime_tools.downloads --list`` / ``python -m anime_tools.downloads [ID…]``.
- ``anime_tools.stages.registry`` lists every stage with its request class and
  ``python -m`` module, resolved lazily, so a host can enumerate them (and build
  each one's ``--help``) without importing a model.
- The web GUI (``anime-tools-gui --open``) is a FastAPI app over the same
  objects; stages run as one ``python -m`` subprocess at a time.
"""

from __future__ import annotations

import sys


def main() -> None:
    # --- the catalog ---------------------------------------------------------
    from anime_tools import downloads

    print("models:")
    for asset in downloads.catalog():
        mark = "ok " if asset.installed else "-- "
        gated = "  (gated)" if asset.gated else ""
        print(f"  {mark} {asset.id:<16} {asset.used_by:<28} → {asset.location}{gated}")
    missing = [a.id for a in downloads.catalog() if not a.installed]
    if missing:
        print("fetch with: python -m anime_tools.downloads", *missing)
    # In-process: downloads.by_id()["tagger"].fetch(); the CLI's main() is
    # downloads.main(["--list"]) / downloads.main(["sam3", "soft_prompt"]).

    # --- the registry ----------------------------------------------------------
    from anime_tools._request import args_of
    from anime_tools.stages.registry import STAGES

    print("\nstages:")
    for stage in STAGES:
        cls = stage.request_class()  # imports the request module, not the stage
        n = len(args_of(cls))  # the field list the CLI parser and the GUI form share
        hidden = "  [preflight]" if stage.hidden else ""
        print(
            f"  {stage.id:<12} {stage.panel:<8} {n:2d} flags  python -m {stage.module}{hidden}"
        )

    # A stage's parser is generated from its request class, so --help and the
    # form agree by construction. Try any: ``stage.request_class().parser()``.
    ocr = next(s for s in STAGES if s.id == "ocr").request_class()
    parser = ocr.parser(
        prog=f"python -m {next(s for s in STAGES if s.id == 'ocr').module}"
    )
    print("\n" + parser.format_usage().strip())

    # argv ⇄ request is a round trip; a default reads back as a default.
    req = ocr.from_argv(parser, ["--keep_en", "--min_score", "0.7"])
    print("request:", req.skip_en, req.min_score, "→ argv:", req.to_argv())
    assert ocr.from_argv(parser, req.to_argv()) == req

    # --- running a stage from another program ----------------------------------
    # This is all the GUI's job runner and the trainer's daemon do: spawn
    # ``python -m <module> <argv>`` in the curation home and read stdout. A
    # stage prints ``  [done/total] detail`` lines the GUI turns into a bar; with
    # ANIMA_DAEMON_JOB_DIR set it also streams them to <job_dir>/progress.jsonl.
    cmd = [sys.executable, "-m", "anime_tools.downloads", "--list"]
    print("\n$", " ".join(cmd))

    # --- the GUI -------------------------------------------------------------------
    # anime-tools-gui --open [--home DIR] [--host 0.0.0.0] [--port 8790]
    # or in-process:
    #     from anime_tools.gui.server import create_app
    #     import uvicorn; uvicorn.run(create_app(), host="127.0.0.1", port=8790)
    # The form for each stage is /api/stages/<id>; the server never loads torch.
    try:
        from anime_tools.gui.server import create_app

        app = create_app()
        routes = sorted(r.path for r in app.routes if r.path.startswith("/api"))
        print("\nGUI routes:", len(routes), "e.g.", routes[:6])
    except ImportError:
        print("\n(fastapi not installed: the GUI half is skipped)")
    assert "torch" not in sys.modules, "nothing here should have imported torch"


if __name__ == "__main__":
    main()
