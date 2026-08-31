import { Show } from "solid-js";
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
      <b>anime_tools</b>
      <span class="dim mono" title={props.info?.home}>
        {props.info?.home}
      </span>
      <Show when={props.list}>
        {(l) => (
          <span class="dim">
            {l().total} image{l().total === 1 ? "" : "s"} in {l().root}
          </span>
        )}
      </Show>
      <span class="sp" />
      {/* A set token is a menu row; a missing one is a blocker, so it stays in
          the bar where it cannot be missed. */}
      <Show when={!props.info?.hf_token}>
        <button
          class="link warn"
          title="The tagger backbone and SAM3 are gated on the Hub — set a token in Settings"
          onClick={() => props.onSettings("models")}
        >
          ⚠ no HF token
        </button>
      </Show>
      <Show when={props.downloading}>
        <span class="badge running">downloading</span>
      </Show>
    </header>
  );
}
