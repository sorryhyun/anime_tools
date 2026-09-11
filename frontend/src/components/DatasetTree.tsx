import { createMemo, Show } from "solid-js";
import { slots, t } from "../i18n";
import { createFolding, type Folder, type Grouped } from "../tree";
import type { DatasetGroups, DatasetList, Rung, Sel, TreeMode } from "../types";
import { Icon } from "./Icon";
import { FolderNode, GroupView, type TreeCtx } from "./TreeNodes";

/** The sidebar: the mode and filter bars, and under them the listing as a
    folder tree or as the grouping manifest's clusters. The rows themselves are
    `TreeNodes`; what is built here is the one `ctx` they all read. */
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
  /** The listing as the two shapes this draws it in. Both are built in
      `dataset.ts`, since ↑/↓ walk the same order the rows are drawn in. */
  tree: Folder;
  grouped: Grouped;
  groupsLoading: boolean;
  groupsError?: string;
  sel: Sel | null;
  onSelect: (sel: Sel) => void;
  query: string;
  onQuery: (q: string) => void;
  onRefresh: () => void;
  /** Rels the last Run changed, so a batch's diff can be walked down. */
  pending?: Set<string>;
  /** Collapse the tree; the rail's chevron on the left edge brings it back. */
  onCollapse: () => void;
}) {
  const fold = createFolding({
    size: () => props.list?.items.length ?? 0,
    resetKey: () => props.resetKey,
  });
  const ctx: TreeCtx = {
    fold,
    ladder: createMemo<Rung[]>(() => props.list?.ladder ?? []),
    sel: () => props.sel,
    select: (rel, kind) => props.onSelect({ rel, kind }),
    pending: () => props.pending,
  };

  /** Everything that can stand between group view and a list of groups: the
      fetch, a manifest the Groups stage has not written, and one built from a
      different tree — whose rels join onto nothing, so it would otherwise look
      like an empty dataset. */
  const GroupNotice = () => (
    <Show when={!props.groupsError} fallback={<div class="err pad">{props.groupsError}</div>}>
      <Show
        when={props.groups}
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
            <Show when={!props.grouped.artists.length}>
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
        <button class="icon" title={t().tree.rescan} onClick={props.onRefresh}>
          <Icon name="refresh" />
        </button>
        <button class="icon" title={t().tree.collapse} onClick={props.onCollapse}>
          <Icon name="chevronLeft" />
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
                    fallback={<FolderNode ctx={ctx} f={props.tree} depth={0} />}
                  >
                    <GroupNotice />
                    <GroupView ctx={ctx} grouped={props.grouped} />
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
