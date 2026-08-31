# Pure-stdlib path filtering, so torch-free callers (notably the GUI process)
# can use it without paying a torch/cv2 import.

import fnmatch
import os


def filter_paths_by_glob(
    img_paths: list[str],
    image_dir: str | None,
    pattern: str | None,
) -> list[bool]:
    """Return a per-path boolean mask: True keeps the file, False drops it.

    The pattern is matched against each file's path relative to ``image_dir``
    (with forward slashes, no leading "./") via ``fnmatch``. ``|`` separates
    alternatives — ``char_a/*|char_b/*`` keeps anything under either folder.
    Default ``*``, empty, or None all keep everything. Returns a mask rather
    than a filtered list so callers can keep parallel arrays (sizes,
    captions) aligned.
    """
    if not pattern:
        return [True] * len(img_paths)
    alternatives = [alt.strip() for alt in pattern.split("|")]
    alternatives = [alt for alt in alternatives if alt]
    if not alternatives or any(alt == "*" for alt in alternatives):
        return [True] * len(img_paths)
    base = os.path.abspath(image_dir) if image_dir else None
    keep: list[bool] = []
    for p in img_paths:
        if base is not None:
            try:
                rel = os.path.relpath(p, base)
            except ValueError:
                rel = os.path.basename(p)
        else:
            rel = os.path.basename(p)
        rel = rel.replace(os.sep, "/")
        keep.append(any(fnmatch.fnmatchcase(rel, alt) for alt in alternatives))
    return keep
