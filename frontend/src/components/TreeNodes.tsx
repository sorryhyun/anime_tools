import { For, Show } from "solid-js";
import { t } from "../i18n";
import type { Folder, Folding, Grouped } from "../tree";
import type { DatasetItem, NodeKind, Rung, Sel, VersionKind } from "../types";

/** The rows of the sidebar, in both of its views. Each takes the same `ctx`:
 * the fold state, the listing's caption ladder, what is selected and how to
 * select, and the rels the last Run touched. The tree that owns them builds
 * one `ctx` and never reaches back in.
 */
export interface TreeCtx {
  fold: Folding;
  /** The rungs the dot strip draws — the listing carries them, so the strip
      cannot come apart from the `captions` map on each row. */
  ladder: () => Rung[];
  sel: () => Sel | null;
  select: (rel: string, kind: NodeKind) => void;
  /** Rels the last Run changed, so a batch's diff can be walked down. */
  pending: () => Set<string> | undefined;
}

/** The caption ladder, as the dots on an image row: a filled dot is a file on
    disk, a hollow one is not, and clicking either opens that version in the
    panel. Which rungs there are is the server's answer (`DatasetList.ladder`);
    only what to *call* one is ours, and a rung with no name wears its own id. */
const CAP_HINT: Record<string, () => string> = {
  master: () => t().tree.capMaster,
  history: () => t().tree.capHistory,
  revised: () => t().tree.capRevised,
  variants: () => t().tree.capVariants,
};
const capHint = (k: VersionKind) => CAP_HINT[k]?.() ?? k;

const Twisty = (p: { open: boolean }) => <span class="tw">{p.open ? "▾" : "▸"}</span>;

/** The "N more" row under a paged list. */
function More(p: { ctx: TreeCtx; key: string; total: number; depth: number }) {
  const limit = () => p.ctx.fold.limitOf(p.key);
  return (
    <Show when={p.total > limit()}>
      <div
        class="tn more"
        style={{ "padding-left": `${16 + p.depth * 12}px` }}
        onClick={() => p.ctx.fold.more(p.key)}
      >
        {t().tree.more(p.total - limit())}
      </div>
    </Show>
  );
}

/** A folder and, while it is open, everything under it. The root has no row of
    its own. */
export function FolderNode(p: { ctx: TreeCtx; f: Folder; depth: number }) {
  const isOpen = () => p.f.path === "" || p.ctx.fold.open(p.f.path);
  return (
    <>
      <Show when={p.f.path !== ""}>
        <div
          class="tn dir"
          style={{ "padding-left": `${4 + p.depth * 12}px` }}
          onClick={() => p.ctx.fold.toggle(p.f.path)}
        >
          <Twisty open={isOpen()} />
          <span class="tl">{p.f.name}</span>
          <span class="tc">{p.f.count}</span>
        </div>
      </Show>
      <Show when={isOpen()}>
        <For each={p.f.folders}>{(c) => <FolderNode ctx={p.ctx} f={c} depth={p.depth + 1} />}</For>
        <For each={p.f.items.slice(0, p.ctx.fold.limitOf(p.f.path))}>
          {(it) => (
            <ImageNode ctx={p.ctx} it={it} depth={p.f.path === "" ? p.depth : p.depth + 1} />
          )}
        </For>
        <More ctx={p.ctx} key={p.f.path} total={p.f.items.length} depth={p.depth} />
      </Show>
    </>
  );
}

/** One image: the name, its flags, and the caption ladder's dots. In group
    view the name carries its folder, since the row is no longer under it. */
export function ImageNode(p: { ctx: TreeCtx; it: DatasetItem; depth: number; withDir?: boolean }) {
  const isSel = () => p.ctx.sel()?.rel === p.it.rel;
  const on = (k: NodeKind) => isSel() && p.ctx.sel()!.kind === k;
  return (
    <div
      classList={{ tn: true, img: true, sel: isSel(), excluded: p.it.excluded }}
      style={{ "padding-left": `${8 + p.depth * 12}px` }}
      title={p.it.rel}
      onClick={() => p.ctx.select(p.it.rel, "image")}
    >
      <span class="tl">
        <Show when={p.withDir && p.it.dir}>
          <span class="dim">{p.it.dir}/</span>
        </Show>
        {p.it.name}
      </span>
      {/* Row flags are about the *image*, not its captions. `r` = resize has
          produced the copy every stage downstream of it walks, so a row
          without one is invisible to them; ⊘ says curation took it out, which
          is *why* such a row has no `r`. */}
      <span class="flags">
        <Show when={p.it.excluded}>
          <span class="flag excl" title={t().tree.flagExcluded}>
            ⊘
          </span>
        </Show>
        <Show when={p.ctx.pending()?.has(p.it.rel)}>
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
        <For each={p.ctx.ladder()}>
          {(c) => {
            const present = () => !!p.it.captions[c.kind];
            return (
              <button
                classList={{ dot: true, [c.kind]: true, off: !present(), on: on(c.kind) }}
                title={`${capHint(c.kind)} — ${present() ? t().tree.onDisk : t().tree.capMissing}`}
                aria-label={c.kind}
                onClick={(e) => {
                  e.stopPropagation();
                  p.ctx.select(p.it.rel, c.kind);
                }}
              />
            );
          }}
        </For>
      </span>
    </div>
  );
}

/** Group view: artist ▸ component ▸ images, then everything the manifest did
    not cluster. That bucket is the big one, so it opens collapsed. */
export function GroupView(p: { ctx: TreeCtx; grouped: Grouped }) {
  const { open, toggle, limitOf } = p.ctx.fold;
  return (
    <>
      <For each={p.grouped.artists}>
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
                      <For each={c.items}>
                        {(it) => <ImageNode ctx={p.ctx} it={it} depth={2} withDir />}
                      </For>
                    </Show>
                  </>
                )}
              </For>
            </Show>
          </>
        )}
      </For>
      <Show when={p.grouped.ungrouped.length}>
        <div class="tn dir" onClick={() => toggle("ungrouped")}>
          <Twisty open={open("ungrouped", false)} />
          <span class="tl">{t().tree.ungrouped}</span>
          <span class="tc">{p.grouped.ungrouped.length}</span>
        </div>
        <Show when={open("ungrouped", false)}>
          <For each={p.grouped.ungrouped.slice(0, limitOf("ungrouped"))}>
            {(it) => <ImageNode ctx={p.ctx} it={it} depth={1} withDir />}
          </For>
          <More ctx={p.ctx} key="ungrouped" total={p.grouped.ungrouped.length} depth={0} />
        </Show>
      </Show>
    </>
  );
}
