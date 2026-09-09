import { createEffect, createMemo, createSignal, on } from "solid-js";
import type { DatasetGroups, DatasetItem, TreeMode } from "./types";

/** The shapes the sidebar draws the listing as, and the fold/paging state over
 * them. Pure model: nothing here knows a row is a `<div>`. `build` nests the
 * flat listing into folders, `regroup` joins the grouping manifest onto it,
 * and `createFolding` is the click state both views share.
 */

/** Folders render lazily, but one flat folder can still hold thousands of
    images; rows past this need an explicit click. */
export const PAGE = 200;
/** Above this many images the tree opens collapsed; expanding 5k rows on load
    is slow. */
export const AUTO_EXPAND_MAX = 400;

export interface Folder {
  path: string;
  name: string;
  folders: Folder[];
  items: DatasetItem[];
  count: number;
}

/** Nest the flat `dir` strings the API returns into a real folder tree. */
export function build(items: DatasetItem[]): Folder {
  const root: Folder = { path: "", name: "", folders: [], items: [], count: 0 };
  const byPath = new Map<string, Folder>([["", root]]);
  const folder = (path: string): Folder => {
    const hit = byPath.get(path);
    if (hit) return hit;
    const cut = path.lastIndexOf("/");
    const parent = folder(cut < 0 ? "" : path.slice(0, cut));
    const node: Folder = {
      path,
      name: path.slice(cut + 1),
      folders: [],
      items: [],
      count: 0,
    };
    parent.folders.push(node);
    byPath.set(path, node);
    return node;
  };
  for (const it of items) folder(it.dir).items.push(it);
  // Bubble the counts up so a collapsed folder can show its whole subtree size.
  const total = (f: Folder): number =>
    (f.count = f.items.length + f.folders.reduce((n, c) => n + total(c), 0));
  total(root);
  return root;
}

/** One near-twin component, resolved against the listing on screen. */
export interface Comp {
  key: string;
  id: number;
  cos: number | null;
  items: DatasetItem[];
}
export interface Artist {
  name: string;
  comps: Comp[];
  count: number;
}
export interface Grouped {
  artists: Artist[];
  ungrouped: DatasetItem[];
}

/** `<dir>/<stem>` -- the manifest's rels are keyed to the *resized* tree, whose
    files resize may have re-encoded (a `.webp` master lands as `.png`), so the
    extension is the one part of a rel the two trees need not agree on. */
const stemKey = (dir: string, stem: string) => (dir ? `${dir}/${stem}` : stem);
function relStemKey(rel: string) {
  const cut = rel.lastIndexOf("/");
  const dot = rel.lastIndexOf(".");
  return dot > cut + 1 ? rel.slice(0, dot) : rel;
}

/** The manifest's rels joined onto the rows the listing already has. A member
 * the filter dropped (or the truncation never sent) is not in the group, so a
 * group's size on screen is the number of rows you can click; whatever is left
 * over is the `ungrouped` bucket, so switching modes cannot lose an image.
 */
export function regroup(items: DatasetItem[], groups?: DatasetGroups): Grouped {
  const byRel = new Map<string, DatasetItem>();
  for (const it of items) {
    const k = stemKey(it.dir, it.stem);
    if (!byRel.has(k)) byRel.set(k, it); // same-stem collision: the listing's first
  }
  const artists: Artist[] = [];
  const seen = new Set<string>();
  const byArtist = new Map<string, Artist>();
  for (const g of groups?.groups ?? []) {
    const rows = g.members
      .map((r) => byRel.get(relStemKey(r)))
      .filter((x): x is DatasetItem => !!x);
    if (rows.length < 2) continue; // a component the listing cut down to nothing
    for (const r of rows) seen.add(r.rel);
    let a = byArtist.get(g.artist);
    if (!a) {
      a = { name: g.artist, comps: [], count: 0 };
      byArtist.set(g.artist, a);
      artists.push(a);
    }
    a.comps.push({ key: `${g.artist}#${g.id}`, id: g.id, cos: g.mean_cosine, items: rows });
    a.count += rows.length;
  }
  return { artists, ungrouped: items.filter((it) => !seen.has(it.rel)) };
}

/** The rels in the order the sidebar *draws* them, which is not the order the
 * flat listing arrives in: a folder's subfolders come before its own files, and
 * group view is a different sequence altogether. ↑/↓ walk this, so the keyboard
 * and the eye agree on what "the next image" is.
 *
 * Fold and paging state are deliberately not consulted. A collapsed folder and
 * a row past a "N more" are still rows of the listing; skipping them would make
 * an image the arrows cannot reach, and the order of what is on screen is the
 * same either way.
 */
export function drawOrder(mode: TreeMode, tree: Folder, grouped: Grouped): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  // A rel can be drawn twice (the manifest may cluster one image under two
  // components); the walk is a permutation of the listing, so first place wins.
  const push = (it: DatasetItem) => {
    if (!seen.has(it.rel)) {
      seen.add(it.rel);
      out.push(it.rel);
    }
  };
  if (mode === "groups") {
    for (const a of grouped.artists) for (const c of a.comps) c.items.forEach(push);
    grouped.ungrouped.forEach(push);
  } else {
    const walk = (f: Folder) => {
      f.folders.forEach(walk);
      f.items.forEach(push);
    };
    walk(tree);
  }
  return out;
}

/** Which nodes are unfolded and how many rows of each are shown. A node is
 * named by a key (a folder path, `a:<artist>`, `c:<component>`, `ungrouped`),
 * and what is stored is which keys differ from their *default*, not which are
 * open -- the default flips with dataset size, and an override has to survive
 * that. `resetKey` is the listing's identity: a new listing (filter change,
 * rescan) drops the paging and every override, while a row patched in place
 * after a save keeps them.
 */
export function createFolding(opts: { size: () => number; resetKey: () => string }) {
  const [flipped, setFlipped] = createSignal(new Set<string>());
  const [shown, setShown] = createSignal(new Map<string, number>());

  // Small trees open so the whole dataset is visible at a glance.
  const openByDefault = createMemo(() => opts.size() <= AUTO_EXPAND_MAX);
  /** `key` is the collapse identity; `dflt` what it does before a click. */
  const open = (key: string, dflt = openByDefault()) => (flipped().has(key) ? !dflt : dflt);
  const toggle = (key: string) =>
    setFlipped((prev) => {
      const next = new Set(prev);
      if (!next.delete(key)) next.add(key);
      return next;
    });

  createEffect(
    on(
      opts.resetKey,
      () => {
        setShown(new Map<string, number>());
        setFlipped(new Set<string>());
      },
      { defer: true },
    ),
  );

  const limitOf = (key: string) => shown().get(key) ?? PAGE;
  const more = (key: string) => setShown((m) => new Map(m).set(key, limitOf(key) + PAGE * 4));

  return { open, toggle, limitOf, more };
}

export type Folding = ReturnType<typeof createFolding>;
