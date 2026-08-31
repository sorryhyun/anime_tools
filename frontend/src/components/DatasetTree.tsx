import { createEffect, createMemo, createSignal, For, on, Show } from "solid-js";
import { slots, t } from "../i18n";
import type {
  DatasetGroups,
  DatasetItem,
  DatasetList,
  NodeKind,
  Rung,
  Sel,
  TreeMode,
  VersionKind,
} from "../types";

/** Folders render lazily, but one flat folder can still hold thousands of
    images; rows past this need an explicit click. */
const PAGE = 200;
/** Above this many images the tree opens collapsed — expanding 5k rows on load
    is the one thing that makes this sidebar feel slow. */
const AUTO_EXPAND_MAX = 400;

/** The caption ladder, as the dots on an image row: a filled dot is a file on
    disk, a hollow one is not, and clicking either opens that version in the
    panel. A fixed strip rather than child rows under a chevron, because every
    image has the same rungs.

    Which rungs there are is the server's answer (`DatasetList.ladder`), not a
    list retyped here — only what to *call* one is ours, and a rung with nothing
    to call it wears its own id. */
const CAP_HINT: Record<string, () => string> = {
  master: () => t().tree.capMaster,
  derived: () => t().tree.capDerived,
  variants: () => t().tree.capVariants,
};
const capHint = (k: VersionKind) => CAP_HINT[k]?.() ?? k;

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

/** One near-twin component, resolved against the listing on screen. */
interface Comp {
  key: string;
  id: number;
  cos: number | null;
  items: DatasetItem[];
}
interface Artist {
  name: string;
  comps: Comp[];
  count: number;
}

/** `<dir>/<stem>` -- the manifest's rels are keyed to the *resized* tree, whose
    files resize may have re-encoded (a `.webp` master lands as `.png`), so the
    extension is the one part of a rel the two trees need not agree on. Same
    stem-match `_row`'s `resized`/`mask` flags and `_sibling_image` use. */
const stemKey = (dir: string, stem: string) => (dir ? `${dir}/${stem}` : stem);
function relStemKey(rel: string) {
  const cut = rel.lastIndexOf("/");
  const dot = rel.lastIndexOf(".");
  return dot > cut + 1 ? rel.slice(0, dot) : rel;
}

/** The manifest's rels joined onto the rows the listing already has.
 *
 * The join is what keeps the two modes honest: a member the filter dropped (or
 * that the truncation never sent) is simply not in the group, so a group's size
 * on screen is always the number of rows you can actually click. Whatever is
 * left over is the `ungrouped` bucket, so switching modes cannot lose an image.
 */
function regroup(items: DatasetItem[], groups?: DatasetGroups) {
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

export function DatasetTree(props: {
  list?: DatasetList;
  /** Changes only when the *listing* changes (filter, rescan) -- not when one
      row is patched in place after a save, which must not collapse the tree. */
  resetKey: string;
  loading: boolean;
  error?: string;
  mode: TreeMode;
  onMode: (mode: TreeMode) => void;
  /** The grouping manifest, fetched only while group view is up. */
  groups?: DatasetGroups;
  groupsLoading: boolean;
  groupsError?: string;
  sel: Sel | null;
  onSelect: (sel: Sel) => void;
  query: string;
  onQuery: (q: string) => void;
  onRefresh: () => void;
  /** Rels the last Run wants to change. Marked here so a batch's diff is
      something you can walk, instead of clicking around looking for it. */
  pending?: Set<string>;
  /** Collapse the tree; the rail's ⟩ on the left edge brings it back. */
  onCollapse: () => void;
}) {
  const tree = createMemo(() => build(props.list?.items ?? []));
  /** The rungs the dot strip draws — the listing carries them, so the strip
      cannot come apart from the `captions` map on each row. */
  const ladder = createMemo<Rung[]>(() => props.list?.ladder ?? []);
  const grouped = createMemo(() => regroup(props.list?.items ?? [], props.groups));
  // Folders whose state differs from the default, not folders that are open --
  // the default flips with dataset size, and an override has to survive that.
  const [flipped, setFlipped] = createSignal(new Set<string>());
  const [shown, setShown] = createSignal(new Map<string, number>());

  // Small trees open so the whole dataset is visible at a glance.
  const openByDefault = createMemo(() => (props.list?.items.length ?? 0) <= AUTO_EXPAND_MAX);
  /** `key` is the collapse identity; `dflt` what it does before a click. */
  const open = (key: string, dflt = openByDefault()) => (flipped().has(key) ? !dflt : dflt);
  const isOpen = (f: Folder) => f.path === "" || open(f.path);

  const flip = (key: string) => (prev: Set<string>) => {
    const next = new Set(prev);
    if (!next.delete(key)) next.add(key);
    return next;
  };
  const toggle = (key: string) => setFlipped(flip(key));

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

  const select = (rel: string, kind: NodeKind) => props.onSelect({ rel, kind });

  const limitOf = (key: string) => shown().get(key) ?? PAGE;
  const more = (key: string) => setShown((m) => new Map(m).set(key, limitOf(key) + PAGE * 4));

  const Twisty = (p: { open: boolean }) => <span class="tw">{p.open ? "▾" : "▸"}</span>;

  const More = (p: { key: string; total: number; depth: number }) => (
    <Show when={p.total > limitOf(p.key)}>
      <div
        class="tn more"
        style={{ "padding-left": `${16 + p.depth * 12}px` }}
        onClick={() => more(p.key)}
      >
        {t().tree.more(p.total - limitOf(p.key))}
      </div>
    </Show>
  );

  const FolderNode = (p: { f: Folder; depth: number }) => (
    <>
      <Show when={p.f.path !== ""}>
        <div
          class="tn dir"
          style={{ "padding-left": `${4 + p.depth * 12}px` }}
          onClick={() => toggle(p.f.path)}
        >
          <Twisty open={isOpen(p.f)} />
          <span class="tl">{p.f.name}</span>
          <span class="tc">{p.f.count}</span>
        </div>
      </Show>
      <Show when={isOpen(p.f)}>
        <For each={p.f.folders}>{(c) => <FolderNode f={c} depth={p.depth + 1} />}</For>
        <For each={p.f.items.slice(0, limitOf(p.f.path))}>
          {(it) => <ImageNode it={it} depth={p.f.path === "" ? p.depth : p.depth + 1} />}
        </For>
        <More key={p.f.path} total={p.f.items.length} depth={p.depth} />
      </Show>
    </>
  );

  /** One image: the thumbnail, the name, its flags — and the caption ladder's
      dots, which are both the "what exists" readout and the way to open one in
      the panel. In group view the name carries its folder, since the row is no
      longer sitting under it. */
  const ImageNode = (p: { it: DatasetItem; depth: number; withDir?: boolean }) => {
    const on = (k: NodeKind) => props.sel?.rel === p.it.rel && props.sel.kind === k;
    return (
      <div
        classList={{ tn: true, img: true, sel: props.sel?.rel === p.it.rel }}
        style={{ "padding-left": `${8 + p.depth * 12}px` }}
        title={p.it.rel}
        onClick={() => select(p.it.rel, "image")}
      >
        <span class="tl">
          <Show when={p.withDir && p.it.dir}>
            <span class="dim">{p.it.dir}/</span>
          </Show>
          {p.it.name}
        </span>
        {/* Row flags are about the *image*, not its captions -- the caption
            files are the dot strip below. Both of these say a stage has been
            here: `r` that resize has produced the copy every stage downstream
            of it walks, so a row without one is invisible to them. */}
        <span class="flags">
          <Show when={props.pending?.has(p.it.rel)}>
            <span class="flag prop" title={t().tree.flagPending}>
              ●
            </span>
          </Show>
          <Show when={p.it.resized}>
            <span class="flag rz" title={t().tree.flagResized}>
              r
            </span>
          </Show>
          <Show when={p.it.mask}>
            <span class="flag mask" title={t().tree.flagMask}>
              😷
            </span>
          </Show>
        </span>
        <span class="caps">
          <For each={ladder()}>
            {(c) => {
              const present = () => !!p.it.captions[c.kind];
              return (
                <button
                  classList={{ dot: true, [c.kind]: true, off: !present(), on: on(c.kind) }}
                  title={`${capHint(c.kind)} — ${present() ? t().tree.onDisk : t().tree.capMissing}`}
                  aria-label={c.kind}
                  onClick={(e) => {
                    e.stopPropagation();
                    select(p.it.rel, c.kind);
                  }}
                />
              );
            }}
          </For>
        </span>
      </div>
    );
  };

  /** Group view: artist ▸ component ▸ images, then everything the manifest did
      not cluster. The bucket is collapsed by default — it is the big one, and
      the point of this mode is the components above it. */
  const GroupView = () => (
    <>
      <For each={grouped().artists}>
        {(a) => (
          <>
            <div class="tn dir" onClick={() => toggle(`a:${a.name}`)}>
              <Twisty open={open(`a:${a.name}`, true)} />
              <span class="tl">{a.name || t().tree.root}</span>
              <span class="tc">
                {a.comps.length}g · {a.count}
              </span>
            </div>
            <Show when={open(`a:${a.name}`, true)}>
              <For each={a.comps}>
                {(c) => (
                  <>
                    <div
                      class="tn grp"
                      style="padding-left:16px"
                      onClick={() => toggle(`c:${c.key}`)}
                      title={t().tree.groupHint(c.id, c.cos ?? "?")}
                    >
                      <Twisty open={open(`c:${c.key}`)} />
                      <span class="tl">{t().tree.group(c.id)}</span>
                      <span class="tc">
                        <Show when={c.cos !== null}>{c.cos!.toFixed(3)} · </Show>
                        {c.items.length}
                      </span>
                    </div>
                    <Show when={open(`c:${c.key}`)}>
                      <For each={c.items}>{(it) => <ImageNode it={it} depth={2} withDir />}</For>
                    </Show>
                  </>
                )}
              </For>
            </Show>
          </>
        )}
      </For>
      <Show when={grouped().ungrouped.length}>
        <div class="tn dir" onClick={() => toggle("ungrouped")}>
          <Twisty open={open("ungrouped", false)} />
          <span class="tl">{t().tree.ungrouped}</span>
          <span class="tc">{grouped().ungrouped.length}</span>
        </div>
        <Show when={open("ungrouped", false)}>
          <For each={grouped().ungrouped.slice(0, limitOf("ungrouped"))}>
            {(it) => <ImageNode it={it} depth={1} withDir />}
          </For>
          <More key="ungrouped" total={grouped().ungrouped.length} depth={0} />
        </Show>
      </Show>
    </>
  );

  /** Everything that can stand between group view and a list of groups: the
      fetch, a manifest the Groups stage has not written, and one built from a
      different tree — whose rels join onto nothing, so it would otherwise look
      like an empty dataset. */
  const GroupNotice = () => {
    const g = () => props.groups;
    return (
      <Show when={!props.groupsError} fallback={<div class="err pad">{props.groupsError}</div>}>
        <Show
          when={g()}
          fallback={<div class="dim pad">{props.groupsLoading ? t().tree.reading : ""}</div>}
        >
          {(m) => (
            <Show
              when={!m().missing}
              fallback={
                <div class="dim pad">
                  {slots(t().tree.noManifest, (i) =>
                    i === 0 ? <code>{m().path}</code> : <b>{t().tree.buildGroups}</b>,
                  )}
                </div>
              }
            >
              <Show when={m().stale}>
                <div class="dim pad">
                  <span class="warn">{t().tree.staleLabel}</span> {t().tree.staleHint}
                </div>
              </Show>
              <Show when={!grouped().artists.length}>
                <div class="dim pad">
                  {slots(t().tree.clustersNothing, () => (
                    <code>{m().path}</code>
                  ))}
                  <Show when={m().source_dir}>
                    {slots(t().tree.builtFrom, () => (
                      <code>{m().source_dir}</code>
                    ))}
                  </Show>
                  .
                </div>
              </Show>
            </Show>
          )}
        </Show>
      </Show>
    );
  };

  return (
    <aside>
      <div class="treebar">
        <div class="modes">
          <button
            classList={{ sel: props.mode === "tree" }}
            title={t().tree.modeTreeHint}
            onClick={() => props.onMode("tree")}
          >
            {t().tree.modeTree}
          </button>
          <button
            classList={{ sel: props.mode === "groups" }}
            title={t().tree.modeGroupsHint}
            onClick={() => props.onMode("groups")}
          >
            {t().tree.modeGroups}
          </button>
        </div>
        <span class="sp" />
        <button title={t().tree.rescan} onClick={props.onRefresh}>
          ↻
        </button>
        <button title={t().tree.collapse} onClick={props.onCollapse}>
          ⟨
        </button>
      </div>
      <div class="treebar">
        <input
          type="text"
          placeholder={t().tree.filter}
          value={props.query}
          onInput={(e) => props.onQuery(e.currentTarget.value)}
        />
      </div>
      <div class="tree">
        <Show when={!props.error} fallback={<div class="err pad">{props.error}</div>}>
          <Show
            when={props.list}
            fallback={<div class="dim pad">{props.loading ? t().tree.scanning : ""}</div>}
          >
            {(l) => (
              <Show
                when={!l().missing}
                fallback={
                  <div class="dim pad">
                    {slots(t().tree.noRoot, () => (
                      <code>{l().root}</code>
                    ))}
                  </div>
                }
              >
                <Show when={l().total} fallback={<div class="dim pad">{t().tree.noImages}</div>}>
                  <Show
                    when={props.mode === "groups"}
                    fallback={<FolderNode f={tree()} depth={0} />}
                  >
                    <GroupNotice />
                    <GroupView />
                  </Show>
                  <Show when={l().truncated}>
                    <div class="dim pad">{t().tree.truncated(l().items.length, l().total)}</div>
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
