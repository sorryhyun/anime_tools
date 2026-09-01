import { createEffect, createResource, createSignal, on, onCleanup } from "solid-js";
import { api } from "./api";
import { persisted } from "./state";
import type { CaptionEntry, ItemDetail, NodeKind, Sel, TreeMode } from "./types";
import type { Config } from "./config";

/** `#rel|kind` — the dataset item is what a GUI link should point at now. */
function parseHash(): Sel | null {
  const raw = decodeURIComponent(location.hash.slice(1));
  if (!raw) return null;
  const cut = raw.lastIndexOf("|");
  if (cut < 0) return { rel: raw, kind: "image" };
  return { rel: raw.slice(0, cut), kind: raw.slice(cut + 1) as NodeKind };
}

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
  const [list, { refetch: refetchList, mutate: mutateList }] = createResource(
    () => [debouncedQuery(), config.roots(), reload()] as const,
    ([q]) => api.dataset({ q }),
  );
  const [sel, setSel] = createSignal<Sel | null>(parseHash());
  /** Which ordering the sidebar is showing. The grouping manifest is fetched
      only while group view is up -- the Groups stage may never have written
      it. */
  const [treeMode, setTreeMode] = persisted<TreeMode>("treemode", "tree", (raw) =>
    raw === "groups" ? "groups" : "tree",
  );
  const [groups] = createResource(
    () => (treeMode() === "groups" ? ([config.roots(), reload()] as const) : false),
    api.groups,
  );
  const [item, { mutate: mutateItem, refetch: refetchItem }] = createResource<
    ItemDetail | undefined,
    string
  >(() => sel()?.rel, api.item);

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

  /** ↑/↓ (and j/k) walk the images in listing order, outside text fields. */
  const onKey = (e: KeyboardEvent) => {
    const t = e.target as HTMLElement | null;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    const step =
      e.key === "ArrowDown" || e.key === "j" ? 1 : e.key === "ArrowUp" || e.key === "k" ? -1 : 0;
    if (!step) return;
    const items = list()?.items ?? [];
    if (!items.length) return;
    const i = items.findIndex((x) => x.rel === sel()?.rel);
    const at =
      i < 0 ? (step > 0 ? 0 : items.length - 1) : Math.min(items.length - 1, Math.max(0, i + step));
    e.preventDefault();
    // Keep the caption kind, so arrowing down a column compares the same file.
    setSel({ rel: items[at].rel, kind: sel()?.kind ?? "image" });
  };
  window.addEventListener("keydown", onKey);
  onCleanup(() => {
    window.removeEventListener("hashchange", onHash);
    window.removeEventListener("keydown", onKey);
  });

  /** Fold a just-saved caption back into the loaded item and its tree row,
      so neither has to be re-fetched. */
  const onSaved = (entry: CaptionEntry) => {
    mutateItem((prev) =>
      prev
        ? { ...prev, versions: prev.versions.map((c) => (c.kind === entry.kind ? entry : c)) }
        : prev,
    );
    const rel = sel()?.rel;
    mutateList((prev) =>
      prev
        ? {
            ...prev,
            items: prev.items.map((x) =>
              x.rel === rel ? { ...x, captions: { ...x.captions, [entry.kind]: true } } : x,
            ),
          }
        : prev,
    );
  };

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
    query,
    setQuery,
    debouncedQuery,
    reload,
    refresh: () => setReload((n) => n + 1),
    sel,
    setSel,
    rel: () => sel()?.rel ?? null,
    treeMode,
    setTreeMode,
    groups,
    item,
    onSaved,
    reloadRels,
    reloadAll,
  };
}

export type Dataset = ReturnType<typeof createDataset>;
