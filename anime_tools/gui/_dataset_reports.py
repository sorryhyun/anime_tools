"""What the stages left under the Settings ``report_root`` that the sidebar and
the caption panel read back: the grouping manifest and the position stage's
per-image analysis. Re-exported by :mod:`anime_tools.gui.dataset`.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from anime_tools import workspace as WS
from anime_tools._json import read_json
from anime_tools.grouping.groups import MANIFEST_VERSION
from anime_tools.gui._dataset_roots import (
    DatasetError,
    _rel_key,
    reachable,
    rel_to_home,
)
from anime_tools.stages._analysis import ANALYSIS_SUBDIR, analysis_paths

GROUPS_SUBPATH = "groups/groups.json"
"""The grouping manifest's tail under the Settings ``report_root``; the same split
``stages.report_subpath`` makes of ``build_groups``' own ``--out`` default."""

POSITION_SUBPATH = f"{WS.REPORTS_SUBDIR}/position"
"""The position stage's report tail under the Settings ``report_root`` (the same
split ``stages.report_subpath`` makes of its ``--report_dir`` default). Its
per-image analysis hangs off it, and its audit phase's under ``audit/``."""

ANALYSIS_KINDS: dict[str, str] = {
    # what said it → its analysis tree under :data:`POSITION_SUBPATH`
    "position": ANALYSIS_SUBDIR,
    "audit": f"audit/{ANALYSIS_SUBDIR}",
}


def load_analysis(report_root: str, rel: str) -> dict[str, Any]:
    """What the position stage and its audit phase last said about one image:
    per kind, the stage's row (``record``) and its instance label map as a PNG
    ``data:`` URL (``mask``), or ``None`` for a kind with nothing on file.

    The files are the stage's own (:mod:`anime_tools.stages._analysis`), kept
    per image rather than per run, so a scoped Run on another image does not
    take this one's away. An unreadable record is a 400 like an unreadable
    groups manifest, never a silent blank.
    """
    key = _rel_key(rel).as_posix()
    out: dict[str, Any] = {}
    for kind, sub in ANALYSIS_KINDS.items():
        root = reachable(f"{report_root}/{POSITION_SUBPATH}/{sub}")
        record_path, mask_path = analysis_paths(root, key)
        if not record_path.is_file():
            out[kind] = None
            continue
        try:
            record = read_json(record_path)
        except (OSError, ValueError) as e:
            raise DatasetError(
                f"unreadable analysis {rel_to_home(record_path)}: {e}"
            ) from e
        mask = None
        if mask_path.is_file():
            mask = "data:image/png;base64," + base64.b64encode(
                mask_path.read_bytes()
            ).decode("ascii")
        out[kind] = {"record": record, "mask": mask}
    return out


def load_groups(report_root: str) -> dict[str, Any]:
    """The grouping manifest under ``report_root``, as the sidebar's group view.

    Rels only: the group view joins them against the one ``/api/dataset`` listing
    the tree view already has. A missing manifest is not an error, but one built
    against another source tree joins onto nothing, so ``source_dir`` rides along.
    """
    path = reachable(f"{report_root}/{GROUPS_SUBPATH}")
    out: dict[str, Any] = {
        "path": rel_to_home(path),
        "missing": True,
        "stale": False,
        "source_dir": "",
        "groups": [],
    }
    if not path.is_file():
        return out
    try:
        data = read_json(path)
    except (OSError, ValueError) as e:
        raise DatasetError(f"unreadable grouping manifest {out['path']}: {e}") from e
    if not isinstance(data, dict):
        raise DatasetError(f"not a grouping manifest: {out['path']}")
    src = str(data.get("source_dir") or "")
    out.update(
        missing=False,
        # A v1 manifest still lists usable components, so this is a "rebuild me"
        # note rather than a reason to show nothing.
        stale=data.get("version") != MANIFEST_VERSION,
        source_dir=rel_to_home(Path(src)) if src else "",
        groups=[
            {
                "id": int(g.get("id", i)),
                "artist": str(g.get("artist") or ""),
                "mean_cosine": g.get("mean_cosine"),
                "members": [str(m) for m in (g.get("members") or [])],
            }
            for i, g in enumerate(data.get("groups") or [])
            if isinstance(g, dict)
        ],
    )
    return out
