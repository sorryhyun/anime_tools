// Mirrors anime_tools/gui/stages.py (schema()) and jobs.py (Job.to_dict()).

export type FieldKind = "bool" | "int" | "float" | "str" | "enum" | "list";

export interface Field {
  dest: string;
  kind: FieldKind;
  flags: string[];
  default: unknown;
  choices: unknown[] | null;
  help: string;
  required: boolean;
  path: boolean;
  /** Which chooser the `…` opens — derived server-side from the flag's name,
      like everything else about it. */
  path_kind: "dir" | "file";
  group: string;
  negate: string | null;
  label: string;
  /** Bound to a dataset root — filled server-side from Settings, hidden here. */
  root: RootName | null;
  /** Bound to a Settings *stage default* (`path_pattern` / `tagger_dir` /
      `checkpoint` / `prompt_embed`) — filled server-side the same way, and
      hidden here for the same reason. */
  setting: string | null;
  /** This stage's own path under the Settings `report_root` (`captions/autotag`,
      `groups/groups.json`): bound server-side, so hidden here too. The audit's
      curated apply *reads* its report through the same binding. */
  report: string | null;
  /** This generator's own tail(s) under the Settings `mask_root` (`masks_sam`,
      `masks_mit`): bound and hidden the same way. A list on the merge, which
      names both generators' trees in one flag and so moves with them. */
  mask: string | string[] | null;
  /** Auto-detected by the stage (`--device`): never shown, never sent. */
  auto: boolean;
}

/** The argparse dest a replay-capable stage exposes. Like `--apply` it is
    GUI-managed, not typed into the form: the Apply dialog is its only route. */
export const REPLAY_FIELD = "from_report";

/** Mirrors `stages.REPORT_SETTING` / `stages.MASK_SETTING`: the two stage
    defaults with no argparse field behind them, so the Settings dialog renders
    their rows by hand. Neither can be a plain stage default: several stages
    spell `--report_dir` / `--mask-dir` identically, and one shared value would
    have them write over each other, so only the *root* is the setting and each
    stage keeps its own tail. */
export const REPORT_SETTING = "report_root";
export const MASK_SETTING = "mask_root";

export interface Stage {
  id: string;
  title: string;
  /** Which dock button this stage lives under; several stages share one. */
  panel: string;
  /** Label for the in-panel picker — the title minus what the panel says. */
  short: string;
  module: string;
  extra: string;
  notes: string;
  report: boolean;
  available: boolean;
  error?: string;
  doc: string;
  apply?: boolean;
  /** Has `--from_report`: Apply can write a dry run's proposals as-is. */
  replay?: boolean;
  /** Has `--path_pattern`: a run can be narrowed to the selected image. */
  scoped?: boolean;
  /** Kept out of the dock: not something you run by hand. `resize` is the one
      -- it runs as a preflight, and its knobs live in the Settings dialog. */
  hidden?: boolean;
  /** The stage id that runs automatically before this one, or null. */
  preprocess?: string | null;
  fields: Field[];
}

export type JobState = "running" | "done" | "failed" | "cancelled";

/** One `python -m <module>` invocation inside a job. A job is a sequence:
    the stage itself, preceded by its preflight when it has one. */
export interface JobStep {
  module: string;
  label: string;
  argv: string[];
}

export interface Job {
  id: string;
  stage: string;
  /** The *stage's* command (the last step) -- what the UI labels the job by. */
  argv: string[];
  steps?: JobStep[];
  state: JobState;
  started: number;
  finished: number | null;
  exit_code: number | null;
  lines: number;
  report_path: string | null;
  apply: boolean;
  values: Record<string, unknown>;
}

export interface Info {
  home: string;
  models_dir: string;
  hf_token: boolean;
  running: string | null;
  /** False while the background stage-schema dump is still running, during
      which `/api/stages` answers 503 — poll info and refetch stages then. */
  schemas_ready: boolean;
}

export interface Settings {
  values?: Record<string, Record<string, unknown>>;
  /** `path_pattern` / `tagger_dir` / `checkpoint` / `prompt_embed` /
      `report_root` / `mask_root`: set once here, not on every stage form. */
  stage_defaults?: Record<string, string>;
  /** The resize preflight's form values. It has no dock panel to carry a form,
      so its knobs are set here and apply to every stage it runs in front of. */
  preprocess?: Record<string, unknown>;
  [k: string]: unknown;
}

// ---- model weights (mirrors anime_tools/downloads.py Asset.to_dict()) ----

export interface ModelAsset {
  id: string;
  title: string;
  repo: string;
  files: string[];
  /** Which stages stop working without it. */
  used_by: string;
  /** The same, as stage ids — what the stage bar's missing-models hint keys on. */
  stages: string[];
  /** The directory it lands in, or "Hugging Face cache". */
  location: string;
  installed: boolean;
  missing: string[];
  /** Accept-the-terms URL when the repo is gated; "" when it is public. */
  gated: string;
  notes: string;
}

export interface ModelCatalog {
  models: ModelAsset[];
  models_dir: string;
}

export interface Listing {
  /** Home-relative inside the curation home, absolute outside it, `""` at the
      home itself. */
  path: string;
  /** Where `..` goes, or null at the top of what this browser may see. The
      server hands it over rather than the client splitting a path. */
  parent: string | null;
  entries: { name: string; dir: boolean }[];
}

/** What the host's own chooser said. `available: false` means there was none to
    open (a headless host, or this browser is not on the machine the server runs
    on) and the caller falls back to the in-page `Listing` browser; `available`
    with a null `path` is an ordinary cancel. */
export interface PickResult {
  available: boolean;
  path: string | null;
}

export type Values = Record<string, unknown>;

// ---- dataset browser (mirrors anime_tools/gui/dataset.py) ----

/** One row of the sidebar tree: a source image plus which sidecars it has. */
export interface DatasetItem {
  rel: string;
  dir: string;
  name: string;
  stem: string;
  /** One flag per `DatasetList.ladder` rung — the row's caption dot strip.
      Keyed by rung rather than spelled out as fields, so a rung the server
      adds appears here without a type change. */
  captions: Record<string, boolean>;
  /** The resized image exists under the dst root — matched on stem, since
      resize may have re-encoded. Everything downstream of resize walks that
      tree, so a row without it is invisible to every stage but autotag. */
  resized: boolean;
  mask: boolean;
}

/** One rung of the server's caption ladder (`dataset.CAPTION_LADDER`). The
    sidebar draws its dot strip from these rather than from a copy of the names,
    which is what keeps the strip and `DatasetItem.captions` in step. */
export interface Rung {
  kind: VersionKind;
  editable: boolean;
}

export interface DatasetList {
  root: string;
  missing: boolean;
  total: number;
  truncated?: boolean;
  items: DatasetItem[];
  ladder: Rung[];
}

/** One near-twin component out of the grouping stage's `groups.json`. Rels
    only: the group view draws the `DatasetItem` rows the listing already has,
    so both sidebar modes agree on the filter and the truncation. */
export interface DatasetGroup {
  id: number;
  /** The top-level folder grouping is scoped to — components never cross it. */
  artist: string;
  mean_cosine: number | null;
  members: string[];
}

export interface DatasetGroups {
  path: string;
  /** The Groups stage has not run yet — the panel says so and points at it. */
  missing: boolean;
  /** Built by an older manifest version; the components still list. */
  stale: boolean;
  /** The tree it was built from. Not the `src` root ⇒ the rels join onto
      nothing, which is the one failure worth saying out loud. */
  source_dir: string;
  groups: DatasetGroup[];
}

export interface RootInfo {
  path: string;
  exists: boolean;
}

export type RootName = "src" | "master" | "dst" | "masks" | "out";

export interface DatasetRoots {
  roots: Record<RootName, RootInfo>;
  defaults: Record<RootName, string>;
  /** What a blank `report_root` resolves to (the parent of `dst`, i.e. the
      workspace) — the Settings placeholder, derived server-side. */
  report_root: string;
  /** The same for a blank `mask_root`: the parent of the `masks` root, so each
      generator's own tree sits beside the merged one it feeds. */
  mask_root: string;
}

export interface Clause {
  header: string;
  prefix: string;
  position: string;
  tags: string[];
}

/** One tag's half-open `[start, end)` slice of the caption text the parse was
    run on — `position_clauses.TagSpan`. The editor draws a box around each one,
    which is how a tag is delimited on screen without the browser ever deciding
    where a tag ends. `clause` is -1 in the flat bag, else the clause it is in. */
export interface Span {
  start: number;
  end: number;
  kind: "tag" | "header" | "artist";
  clause: number;
}

/** The caption grammar, already parsed server-side — never split(",") here. */
export interface Parsed {
  flat_tags: string[];
  spans: Span[];
  clauses: Clause[];
}

/** A writable caption file — what `PUT /api/dataset/item` accepts and what a
    stage proposes against (`proposals.CAPTION_KIND`). */
export type CaptionKind = "master" | "derived";

/** One rung of the caption ladder, by id. The file rungs are named
    (`master`, `derived`, the `variants` placeholder); a sidecar rung expands
    server-side into one id per label it holds, so `v0`, `v1`, `r1` … are rung
    ids too and the set is not closed here. */
export type VersionKind = CaptionKind | "variants" | (string & {});

/** What a tree row can select: the image itself, or one caption under it. */
export type NodeKind = "image" | VersionKind;

/** The selection: one dataset image, and which of its files is on screen. It
    is mirrored into the URL hash (`#rel|kind`), so a GUI link points at it. */
export interface Sel {
  rel: string;
  kind: NodeKind;
}

/** The sidebar's two orderings of the same rows: the folder tree the dataset
    is stored in, and the near-twin components the Groups stage found in it. */
export type TreeMode = "tree" | "groups";

/** One caption of one image: a rung of the ladder, already parsed.
    `editable` is the rung's, not the file's — a variant is read-only whether or
    not it is on disk, and Phase 2 makes the original master read-only the same
    way. */
export interface CaptionEntry {
  kind: VersionKind;
  path: string;
  exists: boolean;
  editable: boolean;
  text: string;
  /** `null` when the file does not exist yet. */
  mtime: number | null;
  parsed: Parsed | null;
  /** Set by the PUT: the .variants.txt sidecar no longer matches v0. */
  variants_stale?: boolean;
}

// ---- proposals (mirrors anime_tools/gui/proposals.py) ----

/** One image's pending change, as a finished Run wrote it down. Both texts
    arrive already parsed — the browser never splits a caption itself. */
export interface Proposal {
  rel: string;
  image: string;
  kind: CaptionKind;
  path: string;
  before: string;
  after: string;
  status: string;
  before_parsed: Parsed | null;
  after_parsed: Parsed | null;
}

/** The index of a Run's proposals: which images it wants to change. The full
    text of one comes from `api.proposal` as the selection lands on it. */
export interface ProposalIndex {
  stage: string;
  apply: boolean;
  kind: CaptionKind;
  total: number;
  rels: string[];
}

export interface UndoResult {
  stage: string;
  report: string;
  restored: number;
  removed: number;
  skipped: Record<string, number>;
  /** Dataset rels to re-stat — the same contract as a job's `written`. */
  written: string[];
}

export interface ImageInfo {
  path: string;
  bytes: number;
  width?: number;
  height?: number;
}

export interface ItemDetail {
  rel: string;
  dir: string;
  name: string;
  stem: string;
  image: ImageInfo | null;
  resized: ImageInfo | null;
  mask: ImageInfo | null;
  /** Every caption this image has, oldest first: the ladder's file rungs, then
      the sidecar expanded into one entry per label. The panel's badge row. */
  versions: CaptionEntry[];
}

/** One Danbooru tag as the KB knows it — `/api/tags/describe`, behind a click
    on any tag chip. `installed` false means the KB CSV was never downloaded
    (Settings › Models › Danbooru tag KB); `known` false means it is downloaded
    and this simply is not a Danbooru tag (an Anima quality tag, a typo). */
export interface TagInfo {
  tag: string;
  installed: boolean;
  known: boolean;
  /** Which CSV the description came from, for the panel's footer. */
  source: string | null;
  /** The catalog row that installs it, for the download button. */
  download_id: string;
  name?: string;
  kind?: string;
  category_path?: string;
  description?: string;
  post_count?: number;
  /** False when the KB answered under another spelling than the one clicked. */
  exact?: boolean;
}

/** What the run bar (and the Settings download row) is saying right now.
    `state` is a job state — `running` / `done` / `failed` — and doubles as the
    badge's class; absent means "just this text, no badge". */
export interface JobStatus {
  text: string;
  state?: string;
}
