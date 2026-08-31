import { createSignal, For, onCleanup, onMount, Show } from "solid-js";
import { locale, localeName, LOCALES, setLocale, t } from "../i18n";
import type { SettingsTab } from "./SettingsDialog";

/** The ☰ menu: everything that is about the app rather than about the dataset
    — the three Settings panels, the Hub token they all need, and the one
    global "show the prose" preference. The sidebar is deliberately not in
    here: it folds and unfolds on the edge it moves (`⟨` in the tree's own bar,
    `⟩` on the rail it leaves behind), so the control is where the motion is. */
export function HeaderMenu(props: {
  hasToken: boolean;
  help: boolean;
  onHelp: () => void;
  onSettings: (tab?: SettingsTab) => void;
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

  /** Every item closes the menu; a menu that stays open behind a dialog is a
      menu you have to dismiss twice. */
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
          <button onClick={pick(() => props.onSettings("models"))}>{t().menu.models}</button>
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
          {/* The one row that is not a command: the language buttons switch in
              place and deliberately leave the menu open, so the label you just
              picked can be read in its own language before it closes. */}
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
