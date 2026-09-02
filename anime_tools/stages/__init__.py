"""Caption-master stages: resize, autotag, position clauses, correction +
variants, OCR, multiview audit, export.

The surface is one request object per stage (:mod:`requests`, torch-free) and
the function that runs it — ``run_autotag(AutotagRequest(...))`` — with the
CLIs in ``cli/`` as shells over them. Both halves are exposed lazily (PEP 562)
so the GUI server can name a request without importing a stage.
"""

__all__ = [
    "AuditRequest",
    "AutotagRequest",
    "CorrectRequest",
    "DetectionRequest",
    "ExportRequest",
    "OcrRequest",
    "PositionRequest",
    "ResizeRequest",
    "run_audit",
    "run_autotag",
    "run_correct",
    "run_export",
    "run_ocr",
    "run_position",
    "run_resize",
]

_HOME = {name: ("requests" if name.endswith("Request") else "run") for name in __all__}


def __getattr__(name: str):
    module = _HOME.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f"{__name__}.{module}"), name)
