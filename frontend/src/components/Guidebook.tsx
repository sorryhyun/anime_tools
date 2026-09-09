import { createResource, Show } from "solid-js";
import { api } from "../api";
import { locale, t } from "../i18n";
import { Dialog } from "./Dialog";
import { Markdown } from "./Markdown";

/** The manual, read in the window it is about.
 *
 * The book is fetched per language and keyed on the locale signal, so switching
 * language in the ☰ menu re-reads it rather than leaving the reader on an
 * English page of a Korean panel. The source is the open flag as well, so the
 * ~30 KB of markdown is fetched by opening the window and by nothing else —
 * it is not in the bundle, and a panel that is never asked never reads it.
 */
export function Guidebook(props: { open: boolean; onClose: () => void }) {
  const [book] = createResource(
    () => (props.open ? locale() : null),
    (lang) => api.guidebook(lang),
  );

  let box: HTMLDivElement | undefined;
  /** A book's table of contents links to its own headings. The dialog is the
      scroll box, so those clicks are caught here and answered by scrolling it —
      following the href would put `#1-install` in the page's location, which is
      where `dataset.ts` keeps the selected image. */
  const onClick = (e: MouseEvent) => {
    const a = (e.target as HTMLElement | null)?.closest("a");
    const href = a?.getAttribute("href");
    if (!href?.startsWith("#")) return;
    e.preventDefault();
    box?.querySelector(`[id="${CSS.escape(href.slice(1))}"]`)?.scrollIntoView({ block: "start" });
  };

  return (
    <Dialog open={props.open} class="guide" onClose={() => props.onClose()}>
      <h3 class="dlgh">
        {t().guide.title}
        <span class="sp" />
        <button value="cancel" class="dlgx" title={t().common.close} aria-label={t().common.close}>
          ×
        </button>
      </h3>
      <div class="guidebox" ref={box} onClick={onClick}>
        <Show
          when={book()}
          fallback={<p class="dim">{book.error ? String(book.error) : t().guide.loading}</p>}
        >
          {(b) => <Markdown text={b().markdown} base={b().base} />}
        </Show>
      </div>
      <div class="dlg-actions">
        <button value="ok" class="primary">
          {t().common.close}
        </button>
      </div>
    </Dialog>
  );
}
