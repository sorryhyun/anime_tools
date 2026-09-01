import { Show } from "solid-js";
import { t } from "../i18n";
import type { DatasetList, Info } from "../types";
import { HeaderMenu } from "./HeaderMenu";
import type { SettingsPane } from "./SettingsDialog";

/** The title bar: where the dataset is, how big it is, and the two things that
    can be wrong before anything is run (no HF token, a download in flight).
    Settings live behind the ☰ menu at the left. */
export function Header(props: {
  info?: Info;
  list?: DatasetList;
  help: boolean;
  onHelp: () => void;
  /** A weights pull is never shown in the dock, so this badge is the only sign
      of one while the dialog it belongs to is closed. */
  downloading: boolean;
  /** How many catalog models are not downloaded; badged on the menu row that
      opens the Models dialog. */
  missingModels: number;
  onSettings: (pane?: SettingsPane) => void;
}) {
  return (
    <header>
      <HeaderMenu
        hasToken={!!props.info?.hf_token}
        missingModels={props.missingModels}
        help={props.help}
        onHelp={props.onHelp}
        onSettings={props.onSettings}
      />
      {/* The home every path in the panel is written against, on the name. */}
      <b title={props.info?.home}>anime_tools</b>
      <Show when={props.list}>
        {(l) => <span class="dim">{t().header.images(l().total, l().root)}</span>}
      </Show>
      <span class="sp" />
      {/* A set token is a menu row; a missing one is a blocker, so it stays in
          the bar. */}
      <Show when={!props.info?.hf_token}>
        <button
          class="link warn"
          title={t().header.noTokenHint}
          onClick={() => props.onSettings("models")}
        >
          {t().header.noToken}
        </button>
      </Show>
      <Show when={props.downloading}>
        <span class="badge running">{t().common.downloading}</span>
      </Show>
    </header>
  );
}
