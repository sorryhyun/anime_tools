import type {
  CaptionEntry,
  CaptionKind,
  DatasetGroups,
  DatasetItem,
  DatasetList,
  DatasetRoots,
  Info,
  ItemDetail,
  Job,
  JobStatus,
  Listing,
  ModelCatalog,
  Parsed,
  Proposal,
  ProposalIndex,
  Settings,
  Stage,
  TagInfo,
  UndoResult,
  Values,
} from "./types";

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    const detail = await r
      .json()
      .then((j) => j.detail)
      .catch(() => r.statusText);
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return r.json() as Promise<T>;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  info: () => req<Info>("/api/info"),
  stages: () => req<Stage[]>("/api/stages"),
  settings: () => req<Settings>("/api/settings"),
  putSettings: (body: Record<string, unknown>) => req<Settings>("/api/settings", json("PUT", body)),
  models: () => req<ModelCatalog>("/api/models"),
  /** Fetch weights as a job; `[]` means every missing model. */
  downloadModels: (ids: string[]) => req<Job>("/api/models/download", json("POST", { ids })),
  jobs: () => req<Job[]>("/api/jobs"),
  job: (id: string) => req<Job>(`/api/jobs/${id}`),
  /** `rel` narrows the run to that one dataset image (the stage's own
      `--path_pattern`); omit it to run the batch the Settings pattern names. */
  start: (stage: string, values: Values, apply: boolean, rel?: string | null) =>
    req<Job>("/api/jobs", json("POST", { stage, values, apply, rel: rel ?? "" })),
  cancel: (id: string) => req<{ cancelled: boolean }>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  report: (id: string) => req<{ path: string; report: unknown }>(`/api/jobs/${id}/report`),
  /** Which images a finished Run wants to change -- the index only. */
  proposals: (id: string) => req<ProposalIndex>(`/api/jobs/${id}/proposals`),
  /** One image's before/after, both already parsed server-side. */
  proposal: (id: string, rel: string) =>
    req<Proposal>(`/api/jobs/${id}/proposal?rel=${encodeURIComponent(rel)}`),
  /** Put back the captions an Apply wrote. */
  undo: (id: string) => req<UndoResult>(`/api/jobs/${id}/undo`, { method: "POST" }),
  ls: (path: string) => req<Listing>(`/api/ls?path=${encodeURIComponent(path)}`),
  fileUrl: (path: string) => `/api/files?path=${encodeURIComponent(path)}`,
  thumbUrl: (path: string, size = 96) => `/api/thumb?path=${encodeURIComponent(path)}&size=${size}`,

  // ---- dataset ----
  datasetRoots: () => req<DatasetRoots>("/api/dataset/roots"),
  putDatasetRoots: (body: Record<string, string>) =>
    req<DatasetRoots>("/api/dataset/roots", json("PUT", body)),
  dataset: (q: { q?: string; pattern?: string; limit?: number } = {}) => {
    const p = new URLSearchParams();
    for (const [k, v] of Object.entries(q)) if (v) p.set(k, String(v));
    return req<DatasetList>(`/api/dataset?${p}`);
  },
  /** The near-twin components the Groups stage wrote -- rels only, joined
      against the listing above by the sidebar's group view. */
  groups: () => req<DatasetGroups>("/api/dataset/groups"),
  item: (rel: string) => req<ItemDetail>(`/api/dataset/item?rel=${encodeURIComponent(rel)}`),
  /** Re-stat named sidebar rows -- what a finished job actually touched. */
  items: (rels: string[]) =>
    req<{ items: DatasetItem[] }>("/api/dataset/items", json("POST", { rels })),
  /** Parse an unsaved caption server-side; the grammar has one implementation. */
  parse: (text: string) => req<Parsed>("/api/dataset/parse", json("POST", { text })),
  /** What one tag means, out of the Danbooru KB. */
  describeTag: (tag: string) => req<TagInfo>(`/api/tags/describe?tag=${encodeURIComponent(tag)}`),
  saveCaption: (rel: string, kind: CaptionKind, text: string) =>
    req<CaptionEntry>("/api/dataset/item", json("PUT", { rel, kind, text })),
};

/** Any thrown error as a failed status line — the four call sites that start
    or undo a job all did this by hand. */
export const toStatus = (e: unknown): JobStatus => ({
  text: e instanceof Error ? e.message : String(e),
  state: "failed",
});

/** Follow a job's stdout. Resolves with the final job dict; `onLine` per line. */
export function followLog(
  id: string,
  onLine: (line: string) => void,
  onDone: (job: Job) => void,
  onError: () => void,
): EventSource {
  const es = new EventSource(`/api/jobs/${id}/log`);
  es.onmessage = (e) => onLine(JSON.parse(e.data));
  es.addEventListener("done", (e) => {
    es.close();
    onDone(JSON.parse((e as MessageEvent).data));
  });
  es.onerror = () => {
    es.close();
    onError();
  };
  return es;
}
