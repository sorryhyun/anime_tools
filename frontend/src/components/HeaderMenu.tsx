import { createSignal, For, Show } from "solid-js";
import { locale, localeName, LOCALES, setLocale, t } from "../i18n";
import { onDismiss } from "../state";
import type { SettingsPane } from "../config";

/** The ☰ menu: the four Settings dialogs, the Hub token they all need, the
    guidebook, and the row that opens every explanation at once — each (?) on the
    page speaks for its own spot, so this is the only thing that reaches all of
    them. The four Settings rows are the only way to each dialog, since they are
    separate windows rather than tabs. The sidebar fold is not here — it lives on
    the edge it moves. */
export function HeaderMenu(props: {
  hasToken: boolean;
  /** Catalog rows still to download — the badge the Models tab used to carry. */
  missingModels: number;
  /** A newer release exists — the only sign of one while the dialog is shut. */
  updateReady: boolean;
  /** Every help area is open — what the row's on/off reports. */
  help: boolean;
  onHelp: () => void;
  onGuide: () => void;
  onSettings: (pane?: SettingsPane) => void;
}) {
  const [open, setOpen] = createSignal(false);

  // A press anywhere but inside the menu dismisses it -- not the ☰ itself,
  // which is inside and toggles.
  onDismiss({ open, inside: ".hmenu", close: () => setOpen(false) });

  /** Every item closes the menu, so a dialog does not open behind one. */
  const pick = (run: () => void) => () => {
    setOpen(false);
    run();
  };

  return (
    <div class="hmenu">
      <button class="link fold" title={t().menu.open} onClick={() => setOpen(!open())}>
        ☰
      </button>
      <Show when={open()}>
        <div class="hmenu-pop" role="menu">
          <button onClick={pick(() => props.onSettings("general"))}>{t().menu.settings}</button>
          <button onClick={pick(() => props.onSettings("advanced"))}>{t().menu.advanced}</button>
          <button onClick={pick(() => props.onSettings("models"))}>
            <span>{t().menu.models}</span>
            <Show when={props.missingModels}>
              <span class="badge miss">{props.missingModels}</span>
            </Show>
          </button>
          <button onClick={pick(() => props.onSettings("update"))}>
            <span>{t().menu.update}</span>
            <Show when={props.updateReady}>
              <span class="badge running">{t().menu.updateReady}</span>
            </Show>
          </button>
          <hr />
          <button onClick={pick(() => props.onSettings("models"))}>
            <span>{t().menu.hfToken}</span>
            <span class={props.hasToken ? "ok" : "warn"}>
              {props.hasToken ? "✓" : t().menu.notSet}
            </span>
          </button>
          <hr />
          {/* The manual and the (?) prose are the same shelf: one explains the
              whole workflow, the other the spot under the cursor. */}
          <button onClick={pick(props.onGuide)}>{t().menu.guidebook}</button>
          <button onClick={pick(props.onHelp)}>
            <span>{t().menu.showHelp}</span>
            <span class={props.help ? "ok" : "dim"}>
              {props.help ? t().common.on : t().common.off}
            </span>
          </button>
          <hr />
          {/* The language buttons switch in place and leave the menu open, so
              the label just picked can be read in its own language. */}
          <div class="hmenu-lang">
            <span>{t().menu.language}</span>
            <span>
              <For each={LOCALES}>
                {(l) => (
                  <button classList={{ on: locale() === l }} lang={l} onClick={() => setLocale(l)}>
                    {localeName(l)}
                  </button>
                )}
              </For>
            </span>
          </div>
        </div>
      </Show>
    </div>
  );
}
