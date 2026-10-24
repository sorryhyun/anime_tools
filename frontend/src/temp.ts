import { createResource, createSignal, onCleanup } from "solid-js";
import { api } from "./api";
import type { Dataset } from "./dataset";

/** What a drop may bring: the server's own list (`gui/dataset.py::_TEMP_SUFFIXES`),
    checked here too so a folder full of other files is not sent file by file
    only to be refused. */
const ACCEPT = /\.(png|jpe?g|webp|bmp|avif|txt)$/i;
/** A `{stem}.*.txt` sidecar is not a caption, and the server refuses it. */
const SIDECAR = /\.[^./]+\.txt$/i;
/** Uploads in flight at once: enough to keep the pipe full, few enough that a
    drop of a thousand files does not open a thousand requests. */
const PARALLEL = 4;

interface Dropped {
  rel: string;
  file: File;
}

/** Everything under the dropped entries, each at the path it had under what was
    dropped — a folder keeps its subfolders. `webkitGetAsEntry` has to be called
    while the drop event is still being dispatched, so the entries are taken
    synchronously by the caller and only walked here. */
async function walk(entries: FileSystemEntry[]): Promise<Dropped[]> {
  const out: Dropped[] = [];
  const visit = async (e: FileSystemEntry): Promise<void> => {
    if (e.isFile) {
      const file = await new Promise<File>((ok, fail) => (e as FileSystemFileEntry).file(ok, fail));
      out.push({ rel: e.fullPath.replace(/^\/+/, ""), file });
      return;
    }
    if (!e.isDirectory) return;
    const reader = (e as FileSystemDirectoryEntry).createReader();
    // readEntries answers in batches (100 in Chrome) until an empty one.
    for (;;) {
      const batch = await new Promise<FileSystemEntry[]>((ok, fail) =>
        reader.readEntries(ok, fail),
      );
      if (!batch.length) break;
      for (const c of batch) await visit(c);
    }
  };
  for (const e of entries) await visit(e);
  return out;
}

/** The drop area: files dragged onto the window from outside it land in
 * `workspace/_temp/`, and the sidebar opens its temp mode on them.
 *
 * The listeners are on the window, so a drop anywhere counts — the page has no
 * other use for one. Only a drag that carries files is answered; dragging text
 * or a row inside the page is left to the browser.
 */
export function createTemp(dataset: Dataset) {
  const [summary, { refetch }] = createResource(api.temp);
  /** A file drag is over the window — the overlay's cue. */
  const [dragging, setDragging] = createSignal(false);
  const [upload, setUpload] = createSignal<{ done: number; total: number } | null>(null);
  const [error, setError] = createSignal("");

  const carriesFiles = (e: DragEvent) => !!e.dataTransfer?.types.includes("Files");
  // dragenter/dragleave fire for every child crossed, so the window is "left"
  // only when as many leaves as enters have gone by.
  let depth = 0;
  const onEnter = (e: DragEvent) => {
    if (!carriesFiles(e)) return;
    e.preventDefault();
    depth++;
    setDragging(true);
  };
  const onOver = (e: DragEvent) => {
    if (!carriesFiles(e)) return;
    // Without this the drop is the browser's, which opens the file in place
    // of the page.
    e.preventDefault();
    e.dataTransfer!.dropEffect = "copy";
  };
  const onLeave = (e: DragEvent) => {
    if (!carriesFiles(e)) return;
    depth = Math.max(0, depth - 1);
    if (!depth) setDragging(false);
  };
  const onDrop = (e: DragEvent) => {
    if (!carriesFiles(e)) return;
    e.preventDefault();
    depth = 0;
    setDragging(false);
    const dt = e.dataTransfer!;
    const entries = [...dt.items]
      .map((i) => i.webkitGetAsEntry?.())
      .filter((x): x is FileSystemEntry => !!x);
    const flat = [...dt.files].map((file) => ({ rel: file.name, file }));
    void (entries.length ? walk(entries) : Promise.resolve(flat)).then(send, (err) =>
      setError(String(err)),
    );
  };
  window.addEventListener("dragenter", onEnter);
  window.addEventListener("dragover", onOver);
  window.addEventListener("dragleave", onLeave);
  window.addEventListener("drop", onDrop);
  onCleanup(() => {
    window.removeEventListener("dragenter", onEnter);
    window.removeEventListener("dragover", onOver);
    window.removeEventListener("dragleave", onLeave);
    window.removeEventListener("drop", onDrop);
  });

  /** Upload, then open the temp mode on the first image the drop brought. A
      file the server refuses does not stop the rest; the first refusal is what
      the sidebar says. */
  async function send(files: Dropped[]) {
    const keep = files.filter((f) => ACCEPT.test(f.rel) && !SIDECAR.test(f.rel));
    if (!keep.length) return;
    setError("");
    setUpload({ done: 0, total: keep.length });
    let first: string | null = null;
    let next = 0;
    const worker = async () => {
      while (next < keep.length) {
        const f = keep[next++];
        try {
          const got = await api.putTempFile(f.rel, f.file);
          if (got.image && (first === null || got.rel < first)) first = got.rel;
        } catch (err) {
          if (!error()) setError(err instanceof Error ? err.message : String(err));
        }
        setUpload((u) => u && { ...u, done: u.done + 1 });
      }
    };
    await Promise.all(Array.from({ length: PARALLEL }, worker));
    setUpload(null);
    void refetch();
    dataset.setTreeMode("temp", first ? { rel: first, kind: "image" } : undefined);
    dataset.refresh();
  }

  /** Delete the drop area — what was dropped and everything a run made of it. */
  async function clear() {
    setError("");
    try {
      await api.clearTemp();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      return;
    }
    void refetch();
    dataset.leaveTemp();
    dataset.refresh();
  }

  return {
    /** Images the drop area holds; the sidebar draws its temp mode while there
        are any. */
    count: () => summary()?.count ?? 0,
    dragging,
    upload,
    error,
    clear,
  };
}

export type Temp = ReturnType<typeof createTemp>;
