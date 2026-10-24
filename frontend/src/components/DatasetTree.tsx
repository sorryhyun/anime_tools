import { createEffect, createMemo, createSignal, on, onCleanup, Show } from "solid-js";
import type { PickHow } from "../dataset";
import { slots, t } from "../i18n";
import { createFolding, revealPath, type Folder, type Grouped } from "../tree";
import type { DatasetGroups, DatasetList, NodeKind, Rung, Sel, TreeMode } from "../types";
import { Icon } from "./Icon";
import { FolderNode, GroupView, type TreeCtx } from "./TreeNodes";

/** The sidebar: the mode and filter bars, and under them the listing as a
    folder tree or as the grouping manifest's clusters — or, in temp mode, the
    folder tree of the drop area. The rows themselves are `TreeNodes`; what is
    built here is the one `ctx` they all read. */
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
  onPick: (rel: string, kind: NodeKind, how?: PickHow) => void;
  /** The rows gathered for compare view. */
  picked: Set<string>;
  /** Images in the drop area. The temp mode's button is drawn while there are
      any, or while it is the mode on screen. */
  tempCount: number;
  /** A drop being uploaded, and what the last one could not send. */
  upload: { done: number; total: number } | null;
  tempError: string;
  onClearTemp: () => void;
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
  /** The sidebar follows the selection into whatever it is folded behind.
      ↑/↓ walk the listing rather than what is unfolded, so the walk crosses
      into the next folder whether or not it is open — and a selection whose
      row is not drawn is a panel pointing at nothing. Opening is all that
      happens: a fold you made elsewhere is left as you left it. */
  createEffect(
    on(
      // The listing is a dependency, not just the rel: a link opened straight
      // onto an image arrives before the tree that holds it, and a filter or a
      // rescan drops every fold (`createFolding`'s reset, which runs first)
      // with the selection still standing.
      [() => props.sel?.rel, () => props.tree, () => props.grouped],
      ([rel, tree, grouped]) => {
        if (!rel) return;
        const path = revealPath(props.mode, tree, grouped, rel);
        if (path) fold.reveal(path);
      },
    ),
  );

  const ctx: TreeCtx = {
    fold,
    ladder: createMemo<Rung[]>(() => props.list?.ladder ?? []),
    sel: () => props.sel,
    select: (rel, kind, how) => props.onPick(rel, kind, how),
    picked: () => props.picked,
    pending: () => props.pending,
  };

  /** Clearing deletes what was dropped and every run's output over it, so it
      takes a second click on the same button — armed for a few seconds, then
      back to its resting state. No dialog: one would block the page. */
  const [armed, setArmed] = createSignal(false);
  let disarm: ReturnType<typeof setTimeout> | undefined;
  onCleanup(() => clearTimeout(disarm));
  function clearTemp() {
    clearTimeout(disarm);
    if (armed()) {
      setArmed(false);
      props.onClearTemp();
      return;
    }
    setArmed(true);
    disarm = setTimeout(() => setArmed(false), 3000);
  }
  const isTemp = () => props.mode === "temp";
  /** The drop area before anything was dropped, or after it was cleared. */
  const TempEmpty = () => <div class="dim pad">{t().tree.tempEmpty}</div>;

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
          <Show when={props.tempCount > 0 || isTemp()}>
            <button
              classList={{ sel: isTemp(), temp: true }}
              title={t().tree.modeTempHint}
              onClick={() => props.onMode("temp")}
            >
              {t().tree.modeTemp}
            </button>
          </Show>
        </div>
        <span class="sp" />
        <Show when={isTemp() && props.tempCount > 0}>
          <button
            classList={{ icon: true, armed: armed() }}
            title={armed() ? t().tree.tempClearArmed : t().tree.tempClear}
            onClick={clearTemp}
          >
            {armed() ? t().tree.tempClearConfirm : <Icon name="trash" />}
          </button>
        </Show>
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
        <Show when={props.upload}>
          {(u) => <div class="dim pad">{t().tree.uploading(u().done, u().total)}</div>}
        </Show>
        <Show when={props.tempError}>
          <div class="err pad">{props.tempError}</div>
        </Show>
        <Show when={!props.error} fallback={<div class="err pad">{props.error}</div>}>
          <Show
            when={props.list}
            fallback={<div class="dim pad">{props.loading ? t().tree.scanning : ""}</div>}
          >
            {(l) => (
              <Show
                when={!l().missing}
                fallback={
                  <Show when={!isTemp()} fallback={<TempEmpty />}>
                    <div class="dim pad">
                      {slots(t().tree.noRoot, () => (
                        <code>{l().root}</code>
                      ))}
                    </div>
                  </Show>
                }
              >
                <Show
                  when={l().total}
                  fallback={
                    <Show
                      when={isTemp() && !props.query}
                      fallback={<div class="dim pad">{t().tree.noImages}</div>}
                    >
                      <TempEmpty />
                    </Show>
                  }
                >
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
