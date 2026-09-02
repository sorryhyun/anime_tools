"""Training-mask generation: SAM3 subject masks,
MIT / ComicTextDetector text masks, and their merge.

The surface is three request objects (:mod:`requests`, torch-free) and the
function that runs each — ``run_sam_masks(SamMaskRequest(...))`` — with the
CLIs in ``cli/`` as shells over them. Two private cores underneath:
:mod:`_sam3` constructs SAM3 (and installs the ``np.bool`` compat alias),
:mod:`_masks` owns the ``{stem}_mask.png`` layout.

Everything is exposed lazily (PEP 562): ``gui/dataset.py`` imports ``_masks``
through this package and must not pick up ``_sam3``'s side effect or the models.
"""

__all__ = [
    "MergeMasksRequest",
    "MitMaskRequest",
    "SamMaskRequest",
    "run_merge_masks",
    "run_mit_masks",
    "run_sam_masks",
]

_HOME = {
    "MergeMasksRequest": "requests",
    "MitMaskRequest": "requests",
    "SamMaskRequest": "requests",
    "run_merge_masks": "merge",
    "run_mit_masks": "mit",
    "run_sam_masks": "sam",
}


def __getattr__(name: str):
    module = _HOME.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f"{__name__}.{module}"), name)
