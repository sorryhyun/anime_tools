# Examples

One runnable script per feature area. Each one shows the **Python API** (the request object or
library call) beside the **CLI** that wraps it, and names the GUI panel that runs the same thing.
Run them from a checkout with `uv run python examples/<name>.py`; every script prints the
`python -m …` command line it is equivalent to.

The first three need no model and no dataset. The rest walk a curation home: pass `--home <dir>`
(the directory holding `image_dataset/`, `workspace/`, `models/`), or let `stages.py` build a
throwaway one (`_sandbox.py`). Weights are fetched on first use; to fetch them up front,
`python -m anime_tools.downloads --list` shows what is missing and `python -m anime_tools.downloads
[ID…]` gets it.

| Script | Feature | API | CLI | GUI |
|---|---|---|---|---|
| `caption_grammar.py` | The caption grammar: parse / compose / spans / normalize / shuffle | `anime_tools.captions.parse_caption`, `compose_caption`, `position_clauses.tag_spans`, `taxonomy.normalize_tag`, `shuffle.*` | — | caption editor (parses server-side) |
| `caption_correction.py` | Danbooru-KB correction, `--caption_drop_groups`, `.variants.txt`, `.history.txt`, `caption_index.json` | `captions.correction.correct_caption`, `tag_drop_groups.parse_drop_groups`, `variants.generate_caption_variants`, `history.push_history`, `index.build_index` | `python -m anime_tools.captions.index` | — |
| `models_registry_gui.py` | Model catalog, stage registry, argv ⇄ request, the web GUI | `downloads.catalog()`, `stages.registry.STAGES`, `Request.parser()` / `from_argv()` / `to_argv()`, `gui.server.create_app()` | `python -m anime_tools.downloads`, `anime-tools-gui` | ☰ → Models |
| `stages.py` | Resize → correct → (autotag) → export, dry run / report / apply, running a stage as a subprocess | `stages.ResizeRequest` … `ExportRequest` + `run_resize` … `run_export` | `python -m anime_tools.stages.cli.{resize_images,correct_captions,autotag_captions,export_workspace}` | Resize (preflight), Curate → Correct, Autotag, Export |
| `tagger.py` | The Anima Tagger on one image | `tagger.tagger.AnimaTagger.predict` / `predict_caption`, `ensure_tagger_checkpoint` | `python -m anime_tools.tagger.cli.autotag`, `python -m anime_tools.tagger.cli --mode predict` | the per-image Autotag button |
| `position_clauses.py` | Position clauses (SAM3 + tagger), `--flatten`, `--from_report` replay; the multiview audit | `stages.PositionRequest` / `AuditRequest` with a nested `DetectionRequest`, `run_position`, `run_audit` | `python -m anime_tools.stages.cli.{position_captions,audit_multiview}` | Curate → Position / Audit |
| `ocr.py` | PP-OCRv6 text recognition → `{stem}.ocr.txt` | `stages.OcrRequest`, `run_ocr`; `ocr.load_ocr().read()`; `captions.ocr_sidecar.read_ocr` | `python -m anime_tools.stages.cli.ocr_captions` | OCR |
| `grouping.py` | Near-twin grouping on PE-Spatial → `groups.json`; custom embedders | `grouping.GroupRequest`, `run_groups` | `python -m anime_tools.grouping.cli.build_groups` | Groups; sidebar *groups* ordering |
| `masking.py` | SAM3 subject masks, MIT / SAM3 text masks, merge; where a mask lives | `masking.SamMaskRequest` / `MitMaskRequest` / `MergeMasksRequest`, `run_*_masks`, `_masks.mask_path_for` | `python -m anime_tools.masking.cli.{generate_masks,generate_masks_mit,merge_masks}` | Masks → Subject / Text / Merge |

## The one pattern

Every stage is a frozen dataclass whose fields are its CLI flags:

```python
from anime_tools.stages import AutotagRequest, run_autotag

req = AutotagRequest(mode="merge", min_confidence=0.35)  # dry run: report.json only
rows, stats = run_autotag(req)
req.to_argv()  # ['--mode', 'merge', '--min_confidence', '0.35']
AutotagRequest.from_argv(
    argv=req.to_argv()
) == req  # the CLI is a shell over the same object
```

- The request modules are torch-free; `run_<stage>` imports the model when it runs.
- Caption stages spell flags with underscores (`--path_pattern`); grouping and masking with hyphens
  (`--source-dir`). Either spelling is accepted everywhere.
- Stages that propose are **dry-run by default** and write `report.json`; `apply=True` writes.
  Resize and correct always write. A `from_report=` request replays a dry run's proposals without
  loading a model and skips any caption that changed since.
- Paths are relative to the **curation home** (`ANIME_TOOLS_HOME` → `ANIMA_HOME` → the current
  directory). Stages read `workspace/resized/` and write under `workspace/`; only Export writes
  outside it. A write to a caption pushes the replaced text onto `{stem}.history.txt`.
- After any apply that changes captions, re-encode the text embeddings in the trainer
  (`make preprocess-te`).
