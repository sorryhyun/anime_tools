import { createEffect, createMemo, createSignal, For, on, Show } from "solid-js";
import { api } from "../api";
import type { DatasetItem, DatasetList, NodeKind } from "../types";

export interface Sel {
  rel: string;
  kind: NodeKind;
}

/** Folders render lazily, but one flat folder can still hold thousands of
    images; rows past this need an explicit click. */
const PAGE = 200;
/** Above this many images the tree opens collapsed — expanding 5k rows on load
    is the one thing that makes this sidebar feel slow. */
const AUTO_EXPAND_MAX = 400;

interface Folder {
  path: string;
  name: string;
  folders: Folder[];
  items: DatasetItem[];
  count: number;
}

/** Nest the flat `dir` strings the API returns into a real folder tree. */
function build(items: DatasetItem[]): Folder {
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

export function DatasetTree(props: {
  list?: DatasetList;
  /** Changes only when the *listing* changes (filter, rescan) -- not when one
      row is patched in place after a save, which must not collapse the tree. */
  resetKey: string;
  loading: boolean;
  error?: string;
  sel: Sel | null;
  onSelect: (sel: Sel) => void;
  query: string;
  onQuery: (q: string) => void;
  onRefresh: () => void;
}) {
  const tree = createMemo(() => build(props.list?.items ?? []));
  // Folders whose state differs from the default, not folders that are open --
  // the default flips with dataset size, and an override has to survive that.
  const [flipped, setFlipped] = createSignal(new Set<string>());
  const [openImages, setOpenImages] = createSignal(new Set<string>());
  const [shown, setShown] = createSignal(new Map<string, number>());

  // Small trees open so the whole dataset is visible at a glance; big ones stay
  // collapsed, because expanding thousands of rows on load is the one thing
  // that makes this sidebar feel slow.
  const openByDefault = createMemo(() => (props.list?.items.length ?? 0) <= AUTO_EXPAND_MAX);
  const isOpen = (f: Folder) =>
    f.path === "" || (flipped().has(f.path) ? !openByDefault() : openByDefault());

  const flip = (key: string) => (prev: Set<string>) => {
    const next = new Set(prev);
    if (!next.delete(key)) next.add(key);
    return next;
  };
  const toggleDir = (f: Folder) => setFlipped(flip(f.path));
  const toggleImage = (rel: string) => setOpenImages(flip(rel));

  // A new listing (filter change, rescan) drops the per-folder paging and any
  // override that no longer refers to a folder that exists.
  createEffect(
    on(
      () => props.resetKey,
      () => {
        setShown(new Map<string, number>());
        setFlipped(new Set<string>());
      },
      { defer: true },
    ),
  );

  // Whatever selects an image -- a click, the ↑/↓ keys, a #rel link pasted into
  // a fresh tab -- opens its caption children, since those are the next thing
  // to click.
  createEffect(
    on(
      () => props.sel?.rel,
      (rel) => rel && setOpenImages((s) => (s.has(rel) ? s : new Set(s).add(rel))),
    ),
  );
  const select = (rel: string, kind: NodeKind) => props.onSelect({ rel, kind });

  const limitOf = (f: Folder) => shown().get(f.path) ?? PAGE;
  const more = (f: Folder) =>
    setShown((m) => new Map(m).set(f.path, limitOf(f) + PAGE * 4));

  const FolderNode = (p: { f: Folder; depth: number }) => (
    <>
      <Show when={p.f.path !== ""}>
        <div
          class="tn dir"
          style={{ "padding-left": `${4 + p.depth * 12}px` }}
          onClick={() => toggleDir(p.f)}
        >
          <span class="tw">{isOpen(p.f) ? "▾" : "▸"}</span>
          <span class="tl">{p.f.name}</span>
          <span class="tc">{p.f.count}</span>
        </div>
      </Show>
      <Show when={isOpen(p.f)}>
        <For each={p.f.folders}>{(c) => <FolderNode f={c} depth={p.depth + 1} />}</For>
        <For each={p.f.items.slice(0, limitOf(p.f))}>
          {(it) => <ImageNode it={it} depth={p.f.path === "" ? p.depth : p.depth + 1} />}
        </For>
        <Show when={p.f.items.length > limitOf(p.f)}>
          <div
            class="tn more"
            style={{ "padding-left": `${16 + p.depth * 12}px` }}
            onClick={() => more(p.f)}
          >
            + {p.f.items.length - limitOf(p.f)} more
          </div>
        </Show>
      </Show>
    </>
  );

  const ImageNode = (p: { it: DatasetItem; depth: number }) => {
    const open = () => openImages().has(p.it.rel);
    const on = (k: NodeKind) => props.sel?.rel === p.it.rel && props.sel.kind === k;
    const child = (kind: NodeKind, label: string, present: boolean, hint: string) => (
      <div
        classList={{ tn: true, cap: true, sel: on(kind), missing: !present }}
        style={{ "padding-left": `${20 + p.depth * 12}px` }}
        title={hint}
        onClick={(e) => {
          e.stopPropagation();
          select(p.it.rel, kind);
        }}
      >
        <span class={`dot ${kind}`} />
        <span class="tl">{label}</span>
        <Show when={!present}>
          <span class="tc">—</span>
        </Show>
      </div>
    );
    return (
      <>
        <div
          classList={{ tn: true, img: true, sel: on("image") }}
          style={{ "padding-left": `${4 + p.depth * 12}px` }}
          title={p.it.rel}
          onClick={() => select(p.it.rel, "image")}
        >
          <span
            class="tw"
            onClick={(e) => {
              e.stopPropagation();
              toggleImage(p.it.rel);
            }}
          >
            {open() ? "▾" : "▸"}
          </span>
          <img class="tt" loading="lazy" src={api.thumbUrl(`${props.list?.root}/${p.it.rel}`, 48)} alt="" />
          <span class="tl">{p.it.name}</span>
          <span class="flags">
            <Show when={p.it.mask}>
              <span class="flag mask" title="has a mask">
                ◑
              </span>
            </Show>
          </span>
        </div>
        <Show when={open()}>
          {child("master", "master", p.it.master, "image_dataset — the hand-written caption")}
          {child("derived", "derived", p.it.derived, "post_image_dataset/resized — the stage output")}
          {child("variants", "variants", p.it.variants, ".variants.txt — generated, read-only")}
        </Show>
      </>
    );
  };

  return (
    <aside>
      <div class="treebar">
        <input
          type="text"
          placeholder="filter…"
          value={props.query}
          onInput={(e) => props.onQuery(e.currentTarget.value)}
        />
        <button title="Rescan the dataset" onClick={props.onRefresh}>
          ↻
        </button>
      </div>
      <div class="tree">
        <Show when={!props.error} fallback={<div class="err pad">{props.error}</div>}>
          <Show when={props.list} fallback={<div class="dim pad">{props.loading ? "scanning…" : ""}</div>}>
            {(l) => (
              <Show
                when={!l().missing}
                fallback={
                  <div class="dim pad">
                    No <code>{l().root}</code> under the curation home. Point the roots at your
                    dataset in ⚙ Settings.
                  </div>
                }
              >
                <Show when={l().total} fallback={<div class="dim pad">No images match.</div>}>
                  <FolderNode f={tree()} depth={0} />
                  <Show when={l().truncated}>
                    <div class="dim pad">
                      showing {l().items.length} of {l().total} — narrow it with the filter
                    </div>
                  </Show>
                </Show>
              </Show>
            )}
          </Show>
        </Show>
      </div>
    </aside>
  );
}
