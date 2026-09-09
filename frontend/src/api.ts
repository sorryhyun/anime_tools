import type {
  CaptionKind,
  DatasetGroups,
  DatasetItem,
  DatasetList,
  DatasetRoots,
  ExcludeResult,
  Guidebook,
  Info,
  ItemDetail,
  Job,
  JobStatus,
  Listing,
  ModelCatalog,
  Parsed,
  PickResult,
  Proposal,
  ProposalIndex,
  SavedCaption,
  Settings,
  Stage,
  TagInfo,
  UndoResult,
  UpdateInfo,
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
  /** Installed vs latest. Cached six hours server-side; `force` is "Check
      now" and is the only thing that always goes out to GitHub. */
  update: (force = false) => req<UpdateInfo>(`/api/update${force ? "?force=true" : ""}`),
  /** Install a release as a job; a blank tag lets the child resolve the latest. */
  runUpdate: (version?: string) =>
    req<Job>("/api/update/run", json("POST", { version: version ?? "" })),
  /** The manual, in one language; the browser renders the markdown. */
  guidebook: (lang: string) => req<Guidebook>(`/api/guidebook?lang=${lang}`),
  jobs: () => req<Job[]>("/api/jobs"),
  job: (id: string) => req<Job>(`/api/jobs/${id}`),
  /** `rel` narrows the run to that one dataset image (the stage's own
      `--path_pattern`); omit it to run the batch the Settings pattern names. */
  start: (stage: string, values: Values, apply: boolean, rel?: string | null) =>
    req<Job>("/api/jobs", json("POST", { stage, values, apply, rel: rel ?? "" })),
  cancel: (id: string) => req<{ cancelled: boolean }>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  report: (id: string) => req<{ path: string; report: unknown }>(`/api/jobs/${id}/report`),
  /** Which images a finished Run changed -- the index only. */
  proposals: (id: string) => req<ProposalIndex>(`/api/jobs/${id}/proposals`),
  /** One image's before/after, both already parsed server-side. */
  proposal: (id: string, rel: string) =>
    req<Proposal>(`/api/jobs/${id}/proposal?rel=${encodeURIComponent(rel)}`),
  undo: (id: string) => req<UndoResult>(`/api/jobs/${id}/undo`, { method: "POST" }),
  ls: (path: string) => req<Listing>(`/api/ls?path=${encodeURIComponent(path)}`),
  /** Ask the host to open its own chooser, starting from `path`. Localhost
      only, by the server's rule. */
  pick: (kind: "dir" | "file", path: string, title: string) =>
    req<PickResult>("/api/pick", json("POST", { kind, path, title })),
  /** Show a path in the host's file manager — a folder opened, a file selected
      inside its own. Localhost only, and only when `Info.can_reveal` said so. */
  reveal: (path: string) => req<{ revealed: boolean }>("/api/reveal", json("POST", { path })),
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
  items: (rels: string[]) =>
    req<{ items: DatasetItem[] }>("/api/dataset/items", json("POST", { rels })),
  /** Take one image out of the pipeline, or put it back. Instant, not a job:
      it moves the image's files between `workspace/` and `workspace/_excluded/`
      and answers with the row as it now is. */
  setExcluded: (rel: string, excluded: boolean, note = "") =>
    req<ExcludeResult>("/api/dataset/exclude", json("POST", { rel, excluded, note })),
  /** Parse an unsaved caption server-side; the grammar has one implementation. */
  parse: (text: string) => req<Parsed>("/api/dataset/parse", json("POST", { text })),
  describeTag: (tag: string) => req<TagInfo>(`/api/tags/describe?tag=${encodeURIComponent(tag)}`),
  saveCaption: (rel: string, kind: CaptionKind, text: string) =>
    req<SavedCaption>("/api/dataset/item", json("PUT", { rel, kind, text })),
};

/** Any thrown error as a failed status line. */
export const toStatus = (e: unknown): JobStatus => ({
  text: e instanceof Error ? e.message : String(e),
  state: "failed",
});

/** Follow a job's stdout over SSE; `onDone` gets the final job dict. */
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

/** Hold `/api/alive` open for as long as this page is: the server it was
    launched with (`make gui`, i.e. `--open`) exits a few seconds after the last
    window goes -- closing the app window stops the server rather than leaving it
    in the terminal. Nothing is read off the stream; EventSource reconnects on
    its own, so a reload gets back inside the grace. */
export function holdAlive(): EventSource {
  return new EventSource("/api/alive");
}
