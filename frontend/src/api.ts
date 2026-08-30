import type {
  CaptionEntry,
  CaptionKind,
  DatasetItem,
  DatasetList,
  DatasetRoots,
  Info,
  ItemDetail,
  Job,
  Listing,
  ModelCatalog,
  Parsed,
  Settings,
  Stage,
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
  putSettings: (body: Record<string, unknown>) =>
    req<Settings>("/api/settings", json("PUT", body)),
  models: () => req<ModelCatalog>("/api/models"),
  /** Fetch weights as a job; `[]` means every missing model. */
  downloadModels: (ids: string[]) => req<Job>("/api/models/download", json("POST", { ids })),
  jobs: () => req<Job[]>("/api/jobs"),
  job: (id: string) => req<Job>(`/api/jobs/${id}`),
  start: (stage: string, values: Values, apply: boolean) =>
    req<Job>("/api/jobs", json("POST", { stage, values, apply })),
  cancel: (id: string) => req<{ cancelled: boolean }>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  report: (id: string) => req<{ path: string; report: unknown }>(`/api/jobs/${id}/report`),
  ls: (path: string) => req<Listing>(`/api/ls?path=${encodeURIComponent(path)}`),
  fileUrl: (path: string) => `/api/files?path=${encodeURIComponent(path)}`,
  thumbUrl: (path: string, size = 96) =>
    `/api/thumb?path=${encodeURIComponent(path)}&size=${size}`,

  // ---- dataset ----
  datasetRoots: () => req<DatasetRoots>("/api/dataset/roots"),
  putDatasetRoots: (body: Record<string, string>) =>
    req<DatasetRoots>("/api/dataset/roots", json("PUT", body)),
  dataset: (q: { q?: string; pattern?: string; limit?: number } = {}) => {
    const p = new URLSearchParams();
    for (const [k, v] of Object.entries(q)) if (v) p.set(k, String(v));
    return req<DatasetList>(`/api/dataset?${p}`);
  },
  item: (rel: string) => req<ItemDetail>(`/api/dataset/item?rel=${encodeURIComponent(rel)}`),
  /** Re-stat named sidebar rows -- what a finished job actually touched. */
  items: (rels: string[]) =>
    req<{ items: DatasetItem[] }>("/api/dataset/items", json("POST", { rels })),
  /** Parse an unsaved caption server-side; the grammar has one implementation. */
  parse: (text: string) => req<Parsed>("/api/dataset/parse", json("POST", { text })),
  saveCaption: (rel: string, kind: CaptionKind, text: string) =>
    req<CaptionEntry>("/api/dataset/item", json("PUT", { rel, kind, text })),
};

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
