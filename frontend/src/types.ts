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
  group: string;
  negate: string | null;
  label: string;
  /** Bound to a dataset root — filled server-side from Settings, hidden here. */
  root: RootName | null;
}

/** The argparse dest a replay-capable stage exposes. Like `--apply` it is
    GUI-managed, not typed into the form: the Apply dialog is its only route. */
export const REPLAY_FIELD = "from_report";

export interface Stage {
  id: string;
  title: string;
  group: string;
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
  fields: Field[];
}

export type JobState = "running" | "done" | "failed" | "cancelled";

export interface Job {
  id: string;
  stage: string;
  argv: string[];
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
}

export interface Settings {
  values?: Record<string, Record<string, unknown>>;
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
  path: string;
  entries: { name: string; dir: boolean }[];
}

export type Values = Record<string, unknown>;

// ---- dataset browser (mirrors anime_tools/gui/dataset.py) ----

/** One row of the sidebar tree: a source image plus which sidecars it has. */
export interface DatasetItem {
  rel: string;
  dir: string;
  name: string;
  stem: string;
  master: boolean;
  derived: boolean;
  variants: boolean;
  mask: boolean;
}

export interface DatasetList {
  root: string;
  missing: boolean;
  total: number;
  truncated?: boolean;
  items: DatasetItem[];
}

export interface RootInfo {
  path: string;
  exists: boolean;
}

export type RootName = "src" | "dst" | "masks";

export interface DatasetRoots {
  roots: Record<RootName, RootInfo>;
  defaults: Record<RootName, string>;
}

export interface Clause {
  header: string;
  prefix: string;
  position: string;
  tags: string[];
}

/** The caption grammar, already parsed server-side — never split(",") here. */
export interface Parsed {
  flat_tags: string[];
  clauses: Clause[];
}

export type CaptionKind = "master" | "derived";
/** What a tree row can select: the image itself, or one caption under it. */
export type NodeKind = "image" | CaptionKind | "variants";

export interface CaptionEntry {
  kind: CaptionKind;
  path: string;
  exists: boolean;
  text: string;
  mtime?: number;
  parsed: Parsed | null;
  /** Set by the PUT: the .variants.txt sidecar no longer matches v0. */
  variants_stale?: boolean;
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
  captions: CaptionEntry[];
  variants: { path: string; exists: boolean; rows: { label: string; text: string }[] };
}
