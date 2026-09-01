"""Training-mask generation: SAM3 subject masks,
MIT / ComicTextDetector text masks, and their merge.

Two private cores under the CLIs in ``cli/``: :mod:`_sam3` constructs SAM3 (and installs
the ``np.bool`` compat alias), :mod:`_masks` owns the ``{stem}_mask.png`` layout.
"""
