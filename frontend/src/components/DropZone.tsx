import { Show } from "solid-js";
import { t } from "../i18n";

/** The cue while files are dragged over the window: wherever they land, they go
    to the drop area, so the whole window says so. It takes no pointer events —
    the drop is the window's (`temp.ts`), not this element's. */
export function DropZone(props: { on: boolean }) {
  return (
    <Show when={props.on}>
      <div class="dropzone">
        <div>
          {t().drop.title}
          <span class="dim">{t().drop.hint}</span>
        </div>
      </div>
    </Show>
  );
}
