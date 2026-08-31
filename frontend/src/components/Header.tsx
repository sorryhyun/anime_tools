import { Show } from "solid-js";
import { t } from "../i18n";
import type { DatasetList, Info } from "../types";
import { HeaderMenu } from "./HeaderMenu";
import type { SettingsTab } from "./SettingsDialog";

/** The title bar: where the dataset is, how big it is, and the two things that
    can be wrong before anything is run (no HF token, a download in flight).
    Settings live behind the ☰ menu at the left; what stays in the bar itself is
    only what you need to see without opening anything. */
export function Header(props: {
  info?: Info;
  list?: DatasetList;
  help: boolean;
  onHelp: () => void;
  /** A weights pull is never shown in the dock, so this badge is the only sign
      of one while the dialog it belongs to is closed. */
  downloading: boolean;
  onSettings: (tab?: SettingsTab) => void;
}) {
  return (
    <header>
      <HeaderMenu
        hasToken={!!props.info?.hf_token}
        help={props.help}
        onHelp={props.onHelp}
        onSettings={props.onSettings}
      />
      {/* One path in the bar, not two: the home is what every path shown in
          the panel is written against, so it reads back off the name here and
          the listing's root is the one that says something. */}
      <b title={props.info?.home}>anime_tools</b>
      <Show when={props.list}>
        {(l) => <span class="dim">{t().header.images(l().total, l().root)}</span>}
      </Show>
      <span class="sp" />
      {/* A set token is a menu row; a missing one is a blocker, so it stays in
          the bar where it cannot be missed. */}
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
