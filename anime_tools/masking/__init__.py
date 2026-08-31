"""Training-mask generation: SAM3 subject masks,
MIT / ComicTextDetector text masks, and their merge.

Two private cores under the CLIs in ``cli/``: :mod:`_sam3` is the one place
SAM3 is constructed (and the one place the ``np.bool`` compat alias sam3 needs
is installed), :mod:`_masks` owns the ``{stem}_mask.png`` layout — the flag
block, the walk-mirror-skip plan, the invert-and-save tail, and the read side
the merge step and the GUI look masks up through.
"""
