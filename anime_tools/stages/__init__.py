"""Caption-master stages: resize, autotag, position clauses, correction +
variants, OCR, multiview audit, export.

The surface is one request object per stage (:mod:`requests`, torch-free) and
the function that runs it — ``run_autotag(AutotagRequest(...))``, which lives in
the stage's own module — with the CLIs in ``cli/`` as shells over them. Both
halves are exposed lazily (PEP 562) so the GUI server can name a request
without importing a stage. ``release_models`` drops the per-process model caches
a chain of in-process runs leaves resident (the tagger, SAM3) so a driver can
hand the GPU on.
"""

__all__ = [
    "AuditRequest",
    "AutotagRequest",
    "CorrectRequest",
    "DetectionRequest",
    "ExportRequest",
    "MultiviewRequest",
    "OcrRequest",
    "PositionRequest",
    "ResizeRequest",
    "release_models",
    "run_audit",
    "run_autotag",
    "run_correct",
    "run_export",
    "run_ocr",
    "run_position",
    "run_resize",
]

_RUNNERS = {
    "release_models": "_models",
    "run_audit": "multiview_audit",
    "run_autotag": "autotag",
    "run_correct": "captions",
    "run_export": "export_workspace",
    "run_ocr": "ocr",
    "run_position": "position_captions",
    "run_resize": "resize",
}
"""Which module each runner lives in — its stage's own, since a runner is that
stage's ``main`` minus the parsing. ``registry.py`` names the same pairs as
``module:function`` strings for a driver that starts from a stage id."""

_HOME = {
    name: ("requests" if name.endswith("Request") else _RUNNERS[name])
    for name in __all__
}


def __getattr__(name: str):
    module = _HOME.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f"{__name__}.{module}"), name)
