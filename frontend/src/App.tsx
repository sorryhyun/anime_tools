import { batch, createEffect, createSignal, on, Show } from "solid-js";
import { api } from "./api";
import { createConfig } from "./config";
import { createDataset } from "./dataset";
import { createDownloads, isDownloadJob } from "./downloads";
import { t } from "./i18n";
import { createLayout } from "./layout";
import { createRunner } from "./runner";
import { createStages } from "./stages";
import type { Stage } from "./types";
import { DatasetTree } from "./components/DatasetTree";
import { Dock } from "./components/Dock";
import { Header } from "./components/Header";
import { ItemView } from "./components/ItemView";
import { SettingsDialog } from "./components/SettingsDialog";
import { StagePanel } from "./components/StagePanel";
import { TagLens } from "./components/TagLens";

/** Wiring only: creates the five state modules and hands their signals to the
 * components. A stage start opens the dock, so `runner` is given `openDock`;
 * the server has a single job slot shared by stage runs and weights downloads,
 * so an adopted job has to be told apart before it is followed.
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

  // Adopt a job the page did not start (already running on load, or started
  // from another tab). Keyed on `running`, not the ids: an id is kept after its
  // job ends, and a second externally-started job deserves adopting too.
  createEffect(
    on(config.info, (i) => {
      const id = i?.running;
      if (!id || runner.busy() || downloads.busy()) return;
      // Only the id comes back on /api/info, so ask what it is before deciding
      // which follower gets it -- a download must not land in the dock.
      void api
        .job(id)
        .then((j) => (isDownloadJob(j) ? downloads.adopt(j) : runner.attach(id)))
        .catch(() => runner.attach(id));
    }),
  );

  /** A dock button: open the panel on the stage it was last left on. Pressing
      the open panel's own button is a no-op; the dock folds by the ▾ in the
      strip's corner, so a mis-aimed second click cannot take the form out from
      under a run. */
  function pickPanel(panel: string, ss: Stage[]) {
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
        help={layout.help()}
        onHelp={layout.toggleHelp}
        downloading={downloads.busy()}
        missingModels={(config.models()?.models ?? []).filter((m) => !m.installed).length}
        onSettings={config.openSettings}
      />

      {/* The tree folds from its own bar and comes back on the rail it leaves
          behind. */}
      <Show when={!layout.sidebar()}>
        <button
          class="unfold"
          title={t().header.showSidebar}
          onClick={() => layout.setSidebar(true)}
        >
          ⟩
        </button>
      </Show>

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
        onSelectCaption={(kind) => dataset.setSel({ rel: dataset.sel()?.rel ?? "", kind })}
        proposal={runner.shownProposal()}
        proposalStage={stages.cur()?.title}
        help={layout.help()}
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
          onUndo={runner.undo}
          undoBlocked={runner.undoBlocked()}
          onCancel={runner.cancel}
          missingModels={stages.missingModels().map((m) => m.title)}
          onSettings={() => config.openSettings("models")}
          help={layout.help()}
          onHelp={layout.toggleHelp}
        />
      </Dock>

      {/* One card for every tag chip in the app; it floats, so it is mounted at
          the root. */}
      <TagLens onInstall={() => config.openSettings("models")} />

      <SettingsDialog
        open={config.settingsOpen()}
        pane={config.settingsPane()}
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
