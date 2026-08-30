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
}

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

export interface Listing {
  path: string;
  entries: { name: string; dir: boolean }[];
}

export type Values = Record<string, unknown>;
