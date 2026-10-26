import {
  batch,
  createEffect,
  createMemo,
  createResource,
  createSignal,
  on,
  onCleanup,
} from "solid-js";
import { api } from "./api";
import { persisted } from "./state";
import { build, drawOrder, regroup } from "./tree";
import type { ItemDetail, NodeKind, SavedCaption, Sel, Space, TreeMode } from "./types";
import type { Config } from "./config";

/** `#rel|kind` — the dataset item is what a GUI link should point at now. */
function parseHash(): Sel | null {
  const raw = decodeURIComponent(location.hash.slice(1));
  if (!raw) return null;
  const cut = raw.lastIndexOf("|");
  if (cut < 0) return { rel: raw, kind: "image" };
  return { rel: raw.slice(0, cut), kind: raw.slice(cut + 1) as NodeKind };
}

/** How a click on a sidebar row lands: alone, as the far end of a range from
    the selected row (shift), or toggled in and out of the pick (⌘/Ctrl). */
export type PickHow = "only" | "range" | "toggle";

/** The image browser: the listing the sidebar draws, which row is selected, and
 * the detail of that one row.
 *
 * The selection is mirrored into the location hash both ways, so a link into the
 * GUI opens on an image and Back walks the images you looked at. `reloadRels`
 * and `onSaved` keep the listing truthful without re-walking the tree.
 */
export function createDataset(config: Config) {
  const [query, setQuery] = createSignal("");
  const [debouncedQuery, setDebouncedQuery] = createSignal("");
  const [reload, setReload] = createSignal(0);
  /** Which ordering the sidebar is showing. The grouping manifest is fetched
      only while group view is up -- the Groups stage may never have written
      it. */
  const [treeMode, setMode] = persisted<TreeMode>("treemode", "tree", (raw) =>
    raw === "groups" || raw === "temp" ? raw : "tree",
  );
  /** The temp mode is not a third ordering but another set of trees: every
      dataset call the page makes is answered against the drop area while it is
      up, which is why the space is bound into `api` rather than passed. */
  const space = createMemo<Space>(() => (treeMode() === "temp" ? "temp" : "dataset"));
  api.bindSpace(space);
  const [list, { refetch: refetchList, mutate: mutateList }] = createResource(
    () => [debouncedQuery(), config.roots(), reload(), space()] as const,
    ([q]) => api.dataset({ q }),
  );
  const [sel, setSel] = createSignal<Sel | null>(parseHash());
  const [groups] = createResource(
    () => (treeMode() === "groups" ? ([config.roots(), reload()] as const) : false),
    api.groups,
  );
  /** The rows a shift- or ⌘/Ctrl-click gathered, in draw order. Two or more is
      compare view; the selection stays on the row the range was drawn from, so
      the address, the hash and a per-image Run all still name one image. */
  const [picked, setPicked] = createSignal<string[]>([]);
  const pickedSet = createMemo(() => new Set(picked()));
  // A pick hangs off the row it was drawn from; the selection moving anywhere
  // else (the arrows, Back, a link) is a new place, not a wider range.
  createEffect(
    on(
      () => sel()?.rel,
      () => setPicked([]),
      { defer: true },
    ),
  );

  /** Each space keeps the image it was left on: going to the drop area and back
      is a look at something else, not a reason to lose your place. */
  const left: Record<Space, Sel | null> = { dataset: null, temp: null };
  /** Switch the sidebar's mode. `land` is where a switch into another space
      should open instead of where that space was left (a drop opens on what it
      brought). */
  function setTreeMode(mode: TreeMode, land?: Sel | null) {
    const from = space();
    const to: Space = mode === "temp" ? "temp" : "dataset";
    batch(() => {
      if (from !== to) {
        left[from] = sel();
        setPicked([]);
        setSel(land !== undefined ? land : left[to]);
      } else if (land !== undefined) setSel(land);
      setMode(mode);
    });
  }
  /** The drop area was emptied: nothing there to come back to. */
  function leaveTemp() {
    left.temp = null;
    if (space() === "temp") setTreeMode("tree");
  }
  const [item, { mutate: mutateItem, refetch: refetchItem }] = createResource<
    ItemDetail | undefined,
    string
  >(() => sel()?.rel, api.item);
  /** What the position stage last saw in the selected image — the caption
      panel's analysis badge. Keyed on the item itself rather than its rel, so
      every refetch of the item (a job that wrote it, a reload) re-reads it: a
      run that captioned the image is the run that re-analysed it. */
  const [analysis] = createResource(
    () => item(),
    (it) => api.analysis(it.rel),
  );

  /** The two shapes the sidebar draws the listing as, built here rather than in
      the component: the keyboard walks the same model the eye reads, so there
      is one answer to what order the images are in. */
  const tree = createMemo(() => build(list()?.items ?? []));
  const grouped = createMemo(() => regroup(list()?.items ?? [], groups()));
  const order = createMemo(() => drawOrder(treeMode(), tree(), grouped()));

  let queryTimer: ReturnType<typeof setTimeout> | undefined;
  createEffect(
    on(query, (q) => {
      clearTimeout(queryTimer);
      queryTimer = setTimeout(() => setDebouncedQuery(q), 200);
    }),
  );
  createEffect(
    on(sel, (s) => {
      location.hash = s ? encodeURIComponent(s.rel) + (s.kind === "image" ? "" : `|${s.kind}`) : "";
    }),
  );
  const onHash = () => {
    const h = parseHash();
    if (h?.rel !== sel()?.rel || h?.kind !== sel()?.kind) setSel(h);
  };
  window.addEventListener("hashchange", onHash);

  const refresh = () => setReload((n) => n + 1);

  /** ↑/↓ (and j/k) walk the images in the order the sidebar draws them,
      outside text fields. The flat listing is not that order -- a folder's
      subfolders are drawn before its own files, and group view is a different
      sequence entirely -- so the walk is over `order`, not `list().items`.

      ⌘/Ctrl+R and F5 re-walk the listing instead of reloading the page, and
      ⌘/Ctrl+W is swallowed. The page is a chromeless app window over a local
      server: a reload throws away the caption being typed and the running
      job's log to arrive back at the same URL, and there is no tab to close.
      Both are answered above the text-field guard, since the editor is exactly
      where the reflex costs the most. (Chrome reserves ⌘/Ctrl+W, so that half
      is best-effort.) */
  const onKey = (e: KeyboardEvent) => {
    const mod = e.metaKey || e.ctrlKey;
    if (e.key === "Escape" && picked().length > 1) {
      e.preventDefault();
      setPicked([]);
      return;
    }
    const k = e.key.toLowerCase();
    if (e.key === "F5" || (mod && k === "r")) {
      e.preventDefault();
      refresh();
      return;
    }
    if (mod && k === "w") {
      e.preventDefault();
      return;
    }
    const t = e.target as HTMLElement | null;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    const step =
      e.key === "ArrowDown" || e.key === "j" ? 1 : e.key === "ArrowUp" || e.key === "k" ? -1 : 0;
    if (!step) return;
    const rels = order();
    if (!rels.length) return;
    const i = rels.indexOf(sel()?.rel ?? "");
    const at =
      i < 0 ? (step > 0 ? 0 : rels.length - 1) : Math.min(rels.length - 1, Math.max(0, i + step));
    e.preventDefault();
    // Keep the caption kind, so arrowing down a column compares the same file.
    setSel({ rel: rels[at], kind: sel()?.kind ?? "image" });
  };
  window.addEventListener("keydown", onKey, true);
  onCleanup(() => {
    window.removeEventListener("hashchange", onHash);
    window.removeEventListener("keydown", onKey, true);
  });

  /** A click on a row. A range runs from the selected row to this one along
      `order` — the rows as drawn, so it is what the eye sees between them in
      either view. A toggle starts from the selected row too, so ⌘/Ctrl-clicking
      one other image is already a pair. A pick that shrinks back to one image
      is a plain selection of it. */
  function pick(rel: string, kind: NodeKind, how: PickHow = "only") {
    const anchor = sel()?.rel;
    if (how === "only" || !anchor || anchor === rel) {
      setPicked([]);
      setSel({ rel, kind });
      return;
    }
    if (how === "range") {
      const rels = order();
      const a = rels.indexOf(anchor);
      const b = rels.indexOf(rel);
      if (a < 0 || b < 0) return pick(rel, kind);
      setPicked(rels.slice(Math.min(a, b), Math.max(a, b) + 1));
      return;
    }
    const now = new Set(picked().length ? picked() : [anchor]);
    if (!now.delete(rel)) now.add(rel);
    const next = order().filter((r) => now.has(r));
    if (next.length > 1) return setPicked(next);
    setPicked([]);
    if (next[0]) setSel({ rel: next[0], kind: "image" });
  }

  /** Fold a just-saved caption back into the loaded item and its tree row,
      so neither has to be re-fetched.

      The whole ladder is taken from the answer rather than the one entry
      patched into place: a write *adds* a rung -- the text it replaced becomes
      a `revised@N` history badge -- so replacing by `kind` would drop the very
      version the save just created. The row's dots follow, one flag per rung
      that has a file behind it. */
  const onSaved = (saved: SavedCaption) => {
    const versions = saved.versions;
    mutateItem((prev) => (prev ? { ...prev, versions } : prev));
    const captions: Record<string, boolean> = {};
    for (const v of versions) captions[v.rung] ||= v.exists;
    const rel = sel()?.rel;
    mutateList((prev) =>
      prev
        ? {
            ...prev,
            items: prev.items.map((x) =>
              x.rel === rel ? { ...x, captions: { ...x.captions, ...captions } } : x,
            ),
          }
        : prev,
    );
  };

  /** Take the open image out of the pipeline, or put it back.

      Instant, like a caption save and unlike a Run: the server moves the files
      and hands back the row it left behind, which is folded into the listing
      here. The item itself is re-fetched rather than patched — excluding
      changes what the panel can say about the resized copy and the mask, and
      those moved. */
  async function setExcluded(excluded: boolean, note = "") {
    const rel = sel()?.rel;
    if (!rel) return;
    const result = await api.setExcluded(rel, excluded, note);
    mutateList((prev) =>
      prev ? { ...prev, items: prev.items.map((x) => (x.rel === rel ? result.row : x)) } : prev,
    );
    void refetchItem();
    return result;
  }

  /** Re-stat named sidebar rows in place, and the open item if it is one. */
  async function reloadRels(rels: string[]) {
    if (!rels.length) return;
    const { items } = await api.items(rels);
    const by = new Map(items.map((i) => [i.rel, i]));
    mutateList((prev) =>
      prev ? { ...prev, items: prev.items.map((x) => by.get(x.rel) ?? x) } : prev,
    );
    const rel = sel()?.rel;
    if (rel && by.has(rel)) void refetchItem();
  }

  /** Re-walk the whole tree, for a job whose report named nothing. */
  function reloadAll() {
    void refetchList();
    void refetchItem();
  }

  return {
    list,
    analysis,
    query,
    setQuery,
    debouncedQuery,
    reload,
    refresh,
    sel,
    setSel,
    rel: () => sel()?.rel ?? null,
    picked,
    pickedSet,
    setPicked,
    pick,
    space,
    treeMode,
    setTreeMode,
    leaveTemp,
    groups,
    tree,
    grouped,
    order,
    item,
    onSaved,
    setExcluded,
    reloadRels,
    reloadAll,
  };
}

export type Dataset = ReturnType<typeof createDataset>;
