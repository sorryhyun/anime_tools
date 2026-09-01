import { createResource, createSignal } from "solid-js";
import { createStore } from "solid-js/store";
import { api } from "./api";
import type { DatasetRoots, Info, ModelCatalog, Settings } from "./types";
import type { SettingsOut, SettingsPane } from "./components/SettingsDialog";

/** What the server says about *itself* — the home it is running in, the dataset
 * roots, the weights catalog and the saved settings — plus the dialog that
 * edits them.
 *
 * Every other module reads its configuration from here rather than fetching its
 * own copy: `/api/settings` is read exactly once (`loaded`, which the stage
 * forms are seeded from), and the four resources are the single refetch points,
 * so a finished download and a saved root land everywhere at once.
 */
export function createConfig() {
  const [info, { refetch: refetchInfo }] = createResource<Info>(api.info);
  const [roots, { refetch: refetchRoots }] = createResource<DatasetRoots>(api.datasetRoots);
  const [models, { refetch: refetchModels }] = createResource<ModelCatalog>(api.models);
  const [settings, setSettings] = createStore<Settings>({});
  /** The one read of `/api/settings`. It resolves with what the file held, so
      the stage forms can be seeded from the same fetch the store was. */
  const loaded = api.settings().then((s) => {
    setSettings(s);
    return s;
  });
  /** One value each, from Settings, for every stage that takes them. The
      server fills the flags; this is only the copy the run bar and the Settings
      dialog show. */
  const stageDefaults = () => settings.stage_defaults ?? {};

  const [settingsOpen, setSettingsOpen] = createSignal(false);
  /** *Which* Settings dialog is open. The three are separate: a hint about
      weights or the token opens the one that fixes it and nothing else, and
      there is no way from one to another but closing it. */
  const [settingsPane, setSettingsPane] = createSignal<SettingsPane>("general");
  const openSettings = (pane: SettingsPane = "general") => {
    setSettingsPane(pane);
    setSettingsOpen(true);
    void refetchModels();
  };
  /** Close the dialog, writing back only the blocks it says were touched —
      which is at most the open pane's, since the other two are not mounted. */
  const closeSettings = async (out: SettingsOut | null) => {
    setSettingsOpen(false);
    if (!out) return;
    if (out.token) {
      await api.putSettings({ hf_token: out.token });
      void refetchInfo();
    }
    if (out.roots) {
      await api.putDatasetRoots(out.roots);
      void refetchRoots();
    }
    if (out.defaults) setSettings(await api.putSettings({ stage_defaults: out.defaults }));
    if (out.preprocess) setSettings(await api.putSettings({ preprocess: out.preprocess }));
  };

  return {
    info,
    refetchInfo,
    roots,
    refetchRoots,
    models,
    refetchModels,
    settings,
    setSettings,
    loaded,
    stageDefaults,
    settingsOpen,
    settingsPane,
    openSettings,
    closeSettings,
  };
}

export type Config = ReturnType<typeof createConfig>;
