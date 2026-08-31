import { batch, createEffect, createSignal, on } from "solid-js";
import { api } from "./api";
import { createConfig } from "./config";
import { createDataset } from "./dataset";
import { createDownloads, isDownloadJob } from "./downloads";
import { createLayout } from "./layout";
import { createRunner } from "./runner";
import { createStages } from "./stages";
import type { Stage } from "./types";
import { ApplyDialog } from "./components/ApplyDialog";
import { DatasetTree } from "./components/DatasetTree";
import { Dock } from "./components/Dock";
import { Header } from "./components/Header";
import { ItemView } from "./components/ItemView";
import { SettingsDialog } from "./components/SettingsDialog";
import { StagePanel } from "./components/StagePanel";
import { TagLens } from "./components/TagLens";

/** The whole GUI is five pieces of state and the layout that shows them:
 *
 *     config   — what the server says about itself: home, roots, weights,
 *                settings, and the dialog that edits them
 *     layout   — which panes are open, and how tall the dock is
 *     dataset  — the image listing, the selection (mirrored into the hash),
 *                and the detail of the selected row
 *     stages   — the stage registry and the form over the open one
 *     runner   — running that stage: Run → diff → Apply → Undo
 *
 * Nothing below is business logic: this file wires those five together and
 * hands their signals to the components. The two seams worth naming are the
 * dock (a stage start opens it, so `runner` is given `openDock`) and the single
 * job slot the server has — a stage run and a weights download share it, which
 * is why an adopted job has to be told apart before it is followed.
 */
export default function App() {
  const config = createConfig();
  const layout = createLayout();
  const dataset = createDataset(config);
  const stages = createStages(config);
  const runner = createRunner({
    config,
    stages,
    dataset,
    openDock: () => layout.setDockOpen(true),
  });
  const downloads = createDownloads(config);
  const [confirmOpen, setConfirmOpen] = createSignal(false);

  // A job the page did not start -- one already running when it loaded, or
  // started from another tab -- belongs to whichever follower it came from.
  createEffect(
    on(config.info, (i) => {
      const id = i?.running;
      // `running`, not the ids: an id is kept after its job ends, and a
      // second externally-started job deserves adopting too.
      if (!id || runner.busy() || downloads.busy()) return;
      // Only the id comes back on /api/info, so ask what it is before deciding
      // which follower gets it -- a download must not land in the dock.
      void api
        .job(id)
        .then((j) => (isDownloadJob(j) ? downloads.adopt(j) : runner.attach(id)))
        .catch(() => runner.attach(id));
    }),
  );

  /** A dock button: open the panel on the stage it was last left on, or close
      the dock when it is already the open one. */
  function pickPanel(panel: string, ss: Stage[]) {
    if (layout.dockOpen() && stages.curPanel() === panel) return layout.setDockOpen(false);
    const want = stages.curPanel() === panel ? stages.curId() : stages.lastInPanel[panel];
    const s = ss.find((x) => x.id === want) ?? ss.find((x) => x.available) ?? ss[0];
    if (!s) return;
    batch(() => {
      stages.setCurId(s.id);
      layout.setDockOpen(true);
    });
  }

  return (
    <>
      <Header
        info={config.info()}
        list={dataset.list()}
        sidebar={layout.sidebar()}
        onToggleSidebar={layout.toggleSidebar}
        downloading={downloads.busy()}
        onSettings={config.openSettings}
      />

      <DatasetTree
        list={dataset.list()}
        loading={dataset.list.loading}
        error={dataset.list.error ? String(dataset.list.error) : undefined}
        mode={dataset.treeMode()}
        onMode={dataset.setTreeMode}
        groups={dataset.groups()}
        groupsLoading={dataset.groups.loading}
        groupsError={dataset.groups.error ? String(dataset.groups.error) : undefined}
        resetKey={`${dataset.treeMode()}|${dataset.debouncedQuery()}|${dataset.reload()}`}
        sel={dataset.sel()}
        onSelect={dataset.setSel}
        query={dataset.query()}
        onQuery={dataset.setQuery}
        onRefresh={dataset.refresh}
        pending={runner.pendingSet()}
        onCollapse={() => layout.setSidebar(false)}
      />

      <ItemView
        item={dataset.item()}
        loading={dataset.item.loading}
        error={dataset.item.error ? String(dataset.item.error) : undefined}
        kind={dataset.sel()?.kind ?? "image"}
        proposal={runner.shownProposal()}
        proposalStage={stages.cur()?.title}
        droppedKind={runner.droppedKind()}
        onSaved={dataset.onSaved}
      />

      <Dock
        open={layout.dockOpen()}
        height={layout.dockH()}
        panels={stages.panels()}
        curPanel={stages.curPanel()}
        busy={runner.busy()}
        onPick={pickPanel}
        onToggle={layout.toggleDock}
        onResize={layout.grip}
      >
        <StagePanel
          cur={stages.cur()}
          siblings={stages.siblings()}
          onPick={(id) => {
            stages.setLastInPanel(stages.curPanel(), id);
            stages.setCurId(id);
          }}
          error={stages.all.error}
          values={stages.values()}
          setValue={stages.setValue}
          reset={stages.resetForm}
          busy={runner.busy()}
          locked={downloads.busy()}
          status={runner.status()}
          rel={dataset.rel()}
          onRun={runner.run}
          onApply={() => setConfirmOpen(true)}
          applyBlocked={runner.applyBlocked()}
          onUndo={runner.undo}
          undoBlocked={runner.undoBlocked()}
          onCancel={runner.cancel}
          missingModels={stages.missingModels().map((m) => m.title)}
          onSettings={() => config.openSettings("models")}
          help={layout.help()}
          onHelp={layout.toggleHelp}
        />
      </Dock>

      <ApplyDialog
        open={confirmOpen()}
        title={stages.cur()?.title}
        pending={runner.pending()}
        pattern={config.stageDefaults().path_pattern}
        onConfirm={runner.apply}
        onClose={() => setConfirmOpen(false)}
      />

      {/* One card for every tag chip in the app; it floats, so it is mounted
          at the root rather than inside the caption panel. */}
      <TagLens onInstall={() => config.openSettings("models")} />

      <SettingsDialog
        open={config.settingsOpen()}
        initialTab={config.settingsTab()}
        info={config.info()}
        roots={config.roots()}
        fields={stages.settingFields()}
        defaults={config.stageDefaults()}
        preprocess={stages.preprocessStage()}
        preprocessValues={config.settings.preprocess ?? {}}
        models={config.models()}
        busy={runner.busy() || downloads.busy()}
        downloading={downloads.busy()}
        downloadIds={downloads.ids()}
        progress={downloads.status()}
        help={layout.help()}
        onHelp={layout.toggleHelp}
        onDownload={downloads.start}
        onCancelDownload={downloads.cancel}
        onClose={config.closeSettings}
      />
    </>
  );
}
