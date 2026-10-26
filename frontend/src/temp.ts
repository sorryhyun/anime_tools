import { createResource, createSignal, onCleanup } from "solid-js";
import { api } from "./api";
import type { Dataset } from "./dataset";

/** What a drop may bring: the server's own list (`gui/dataset.py::_TEMP_SUFFIXES`),
    checked here too so a folder full of other files is not sent file by file
    only to be refused. */
const ACCEPT = /\.(png|jpe?g|webp|bmp|avif|txt)$/i;
/** A `{stem}.*.txt` sidecar is not a caption, and the server refuses it. */
const SIDECAR = /\.[^./]+\.txt$/i;
/** A caption rather than an image. */
const CAPTION = /\.txt$/i;
/** Uploads in flight at once: enough to keep the pipe full, few enough that a
    drop of a thousand files does not open a thousand requests. */
const PARALLEL = 4;

const stemOf = (rel: string) => rel.replace(/\.[^./]+$/, "");

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

/** A drop's files by folder and stem. An image and the caption beside it are
    one unit: the image may land under another stem (`a.png` already there makes
    this one `a_1.png`), and its caption has to follow it there. */
function byStem(files: Dropped[]): Dropped[][] {
  const units = new Map<string, Dropped[]>();
  for (const f of files) {
    const k = stemOf(f.rel).toLowerCase();
    const unit = units.get(k);
    if (unit) unit.push(f);
    else units.set(k, [f]);
  }
  return [...units.values()];
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
      the sidebar says.

      An image never replaces one already there: it goes up `unique`, and the
      server answers with the `{stem}_N` it picked. The caption that came with
      it goes up second, under that stem. A caption dropped with no image is
      written as named, since what it is for is the image of that name already
      in the drop area — renamed, it would caption nothing. One whose image was
      refused is not sent at all, for the same reason. */
  async function send(files: Dropped[]) {
    const keep = files.filter((f) => ACCEPT.test(f.rel) && !SIDECAR.test(f.rel));
    if (!keep.length) return;
    setError("");
    setUpload({ done: 0, total: keep.length });
    let first: string | null = null;
    const put = async (f: Dropped, rel: string, unique: boolean) => {
      try {
        const got = await api.putTempFile(rel, f.file, unique);
        if (got.image && (first === null || got.rel < first)) first = got.rel;
        return got.rel;
      } catch (err) {
        if (!error()) setError(err instanceof Error ? err.message : String(err));
        return null;
      } finally {
        setUpload((u) => u && { ...u, done: u.done + 1 });
      }
    };
    const units = byStem(keep);
    let next = 0;
    const worker = async () => {
      while (next < units.length) {
        const unit = units[next++];
        const images = unit.filter((f) => !CAPTION.test(f.rel));
        let stem: string | null = null;
        for (const f of images) {
          const rel = await put(f, f.rel, true);
          stem ??= rel && stemOf(rel);
        }
        for (const f of unit.filter((f) => CAPTION.test(f.rel))) {
          if (!images.length) await put(f, f.rel, false);
          else if (stem !== null) await put(f, `${stem}.txt`, false);
          else setUpload((u) => u && { ...u, done: u.done + 1 });
        }
      }
    };
    await Promise.all(Array.from({ length: PARALLEL }, worker));
    setUpload(null);
    void refetch();
    dataset.setTreeMode("temp", first ? { rel: first, kind: "image" } : undefined);
    dataset.refresh();
  }

  /** Delete the open image — or every picked one — and land on the row after
      the last of them (before the first, at the end of the list), so holding
      Delete walks down the sidebar. No confirmation: what is here is a copy of
      something dropped from elsewhere, and the clear below is the gesture that
      asks first. A key repeat while one is in flight is dropped. */
  let removing = false;
  async function remove(rels: string[]) {
    if (removing) return;
    removing = true;
    setError("");
    const order = dataset.order();
    const gone = new Set<string>();
    let left = count();
    for (const rel of rels) {
      try {
        left = (await api.deleteTempFile(rel)).count;
        gone.add(rel);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        break;
      }
    }
    removing = false;
    if (!gone.size) return;
    void refetch();
    if (!left) {
      dataset.leaveTemp();
    } else {
      const last = Math.max(...[...gone].map((r) => order.indexOf(r)));
      const land =
        order.slice(last + 1).find((r) => !gone.has(r)) ??
        order
          .slice(0, Math.max(0, last))
          .reverse()
          .find((r) => !gone.has(r)) ??
        null;
      dataset.setTreeMode("temp", land ? { rel: land, kind: "image" } : null);
    }
    dataset.refresh();
  }

  /** Delete (and ⌘⌫, the Mac keyboard's spelling of it) in the temp mode,
      outside text fields and dialogs — the editor's Delete is a character's, and
      a Settings dialog over the sidebar is not looking at the image. */
  const onKey = (e: KeyboardEvent) => {
    if (dataset.space() !== "temp") return;
    if (!(e.key === "Delete" || (e.metaKey && e.key === "Backspace"))) return;
    const t = e.target as HTMLElement | null;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    if (t?.closest?.("dialog")) return;
    const rel = dataset.rel();
    const rels = dataset.picked().length > 1 ? dataset.picked() : rel ? [rel] : [];
    if (!rels.length) return;
    e.preventDefault();
    void remove(rels);
  };
  window.addEventListener("keydown", onKey);
  onCleanup(() => window.removeEventListener("keydown", onKey));

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

  /** Images the drop area holds; the sidebar draws its temp mode while there
      are any. */
  const count = () => summary()?.count ?? 0;

  return {
    count,
    dragging,
    upload,
    error,
    clear,
  };
}

export type Temp = ReturnType<typeof createTemp>;
