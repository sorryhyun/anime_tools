import { Show } from "solid-js";
import type { DatasetList, Info } from "../types";
import type { SettingsTab } from "./SettingsDialog";

/** The title bar: where the dataset is, how big it is, and the two things that
    can be wrong before anything is run (no HF token, a download in flight). */
export function Header(props: {
  info?: Info;
  list?: DatasetList;
  sidebar: boolean;
  onToggleSidebar: () => void;
  /** A weights pull is never shown in the dock, so this badge is the only sign
      of one while the dialog it belongs to is closed. */
  downloading: boolean;
  onSettings: (tab?: SettingsTab) => void;
}) {
  return (
    <header>
      <button
        class="link fold"
        title={props.sidebar ? "Hide the dataset sidebar" : "Show the dataset sidebar"}
        onClick={props.onToggleSidebar}
      >
        ☰
      </button>
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
      <Show
        when={props.info?.hf_token}
        fallback={
          <button
            class="link warn"
            title="The tagger backbone and SAM3 are gated on the Hub — set a token in Settings"
            onClick={() => props.onSettings("models")}
          >
            ⚠ no HF token
          </button>
        }
      >
        <span class="dim">HF token ✓</span>
      </Show>
      <Show when={props.downloading}>
        <span class="badge running">downloading</span>
      </Show>
      <button onClick={() => props.onSettings()}>⚙ Settings</button>
    </header>
  );
}
