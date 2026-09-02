import { createSignal, For, onCleanup, onMount, Show } from "solid-js";
import { locale, localeName, LOCALES, setLocale, t } from "../i18n";
import type { SettingsPane } from "./SettingsDialog";

/** The ☰ menu: the three Settings dialogs, the Hub token they all need, and the
    row that opens every explanation at once — each (?) on the page speaks for
    its own spot, so this is the only thing that reaches all of them. The three
    Settings rows are the only way to each dialog, since they are separate
    windows rather than tabs. The sidebar fold is not here — it lives on the edge
    it moves. */
export function HeaderMenu(props: {
  hasToken: boolean;
  /** Catalog rows still to download — the badge the Models tab used to carry. */
  missingModels: number;
  /** Every help area is open — what the row's on/off reports. */
  help: boolean;
  onHelp: () => void;
  onSettings: (pane?: SettingsPane) => void;
}) {
  const [open, setOpen] = createSignal(false);

  onMount(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && open()) {
        e.stopPropagation();
        setOpen(false);
      }
    };
    // Capture phase, like the tag lens: a click anywhere but inside the menu
    // dismisses it, including the ☰ itself (whose own handler then toggles).
    const onDown = (e: MouseEvent) => {
      if ((e.target as HTMLElement | null)?.closest(".hmenu")) return;
      setOpen(false);
    };
    window.addEventListener("keydown", onKey, true);
    window.addEventListener("mousedown", onDown, true);
    onCleanup(() => {
      window.removeEventListener("keydown", onKey, true);
      window.removeEventListener("mousedown", onDown, true);
    });
  });

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
          <hr />
          <button onClick={pick(() => props.onSettings("models"))}>
            <span>{t().menu.hfToken}</span>
            <span class={props.hasToken ? "ok" : "warn"}>
              {props.hasToken ? "✓" : t().menu.notSet}
            </span>
          </button>
          <hr />
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
