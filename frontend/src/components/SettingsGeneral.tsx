import { createSignal, For, Show } from "solid-js";
import { slots, t } from "../i18n";
import type { SettingsOut } from "../config";
import type { HelpArea } from "../layout";
import { ROOT_NAMES } from "../types";
import type { DatasetRoots, Info, RootName } from "../types";
import { HelpToggle } from "./HelpToggle";
import { browsePath, PathPicker } from "./PathPicker";
import { SettingRow, SettingsShell } from "./SettingsShell";

const rootHelp = (n: RootName) => t().settings.rootHelp[n];

/** The General dialog: where the server is running and the five dataset roots.
    OK writes the roots only if one of them was changed. */
export function SettingsGeneral(props: {
  open: boolean;
  info?: Info;
  roots?: DatasetRoots;
  helpOpen: (area: HelpArea) => boolean;
  onHelp: (area: HelpArea) => void;
  onClose: (out: SettingsOut | null) => void;
}) {
  const rootEls: Partial<Record<RootName, HTMLInputElement>> = {};
  /** Which root row's fallback browser is open -- only hosts with no chooser of
      their own get here. A pick is written straight onto the (uncontrolled)
      input. */
  const [picking, setPicking] = createSignal<RootName | null>(null);
  const current = (n: RootName) => props.roots?.roots[n];

  const close = (ok: boolean) => {
    if (!ok) return props.onClose(null);
    const picked = Object.fromEntries(ROOT_NAMES.map((n) => [n, rootEls[n]?.value.trim() ?? ""]));
    const changed = ROOT_NAMES.some((n) => picked[n] !== (current(n)?.path ?? ""));
    props.onClose(changed ? { roots: picked } : {});
  };

  return (
    <>
      <SettingsShell open={props.open} pane="general" onClose={close}>
        <div class="kv">
          <b>{t().settings.home}</b>
          <span class="mono">{props.info?.home}</span>
          <b>{t().settings.modelsDir}</b>
          <span class="mono">{props.info?.models_dir}</span>
        </div>

        <h4>
          {t().settings.roots}
          <HelpToggle open={props.helpOpen("roots")} onToggle={() => props.onHelp("roots")} />
        </h4>
        <Show when={props.helpOpen("roots")}>
          <p class="dim" style="margin:0 0 8px">
            {slots(t().settings.rootsHelp, () => (
              <code>out</code>
            ))}
          </p>
        </Show>
        <div class="kv">
          <For each={ROOT_NAMES}>
            {(n) => {
              const gone = () => !!current(n) && !current(n)!.exists;
              return (
                <SettingRow
                  label={n}
                  ref={(el) => (rootEls[n] = el)}
                  value={current(n)?.path ?? props.roots?.defaults[n] ?? ""}
                  placeholder={props.roots?.defaults[n]}
                  err={gone()}
                  hint={(gone() ? t().settings.rootMissing : "") + rootHelp(n)}
                  onPick={() =>
                    void browsePath(
                      "dir",
                      rootEls[n]?.value ?? "",
                      (path) => rootEls[n] && (rootEls[n]!.value = path),
                      () => setPicking(n),
                    )
                  }
                />
              );
            }}
          </For>
        </div>
      </SettingsShell>
      {/* A sibling of the dialog, not a child: <dialog> puts both in the top
          layer so the picker still lands over the settings, while nesting its
          <form method="dialog"> inside this one's would be a nested form. */}
      <PathPicker
        open={picking() !== null}
        onClose={(path) => {
          const n = picking();
          if (path !== null && n && rootEls[n]) rootEls[n]!.value = path;
          setPicking(null);
        }}
      />
    </>
  );
}
