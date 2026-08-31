import { createResource, createSignal, For, Show } from "solid-js";
import { api } from "../api";
import type { PickResult } from "../types";
import { t } from "../i18n";
import { Dialog } from "./Dialog";

/** Pressing the `…`: the host's own chooser when there is one, the dialog below
 *  when there isn't.
 *
 * The panel and the dataset are normally on the same machine, so the desktop's
 * chooser is the one that already knows where things are — it opens anywhere,
 * not just under the curation home, and types nothing. `onFallback` runs only
 * when the server says there was no chooser to open (a headless host, or a
 * browser somewhere else); a cancel is a cancel and leaves the field alone.
 */
export async function browsePath(
  kind: "dir" | "file",
  current: string,
  onPicked: (path: string) => void,
  onFallback: () => void,
) {
  let res: PickResult | null = null;
  try {
    res = await api.pick(kind, current, t().picker.title);
  } catch {
    res = null; // a request that never landed is not a reason to lose the …
  }
  if (!res?.available) onFallback();
  else if (res.path) onPicked(res.path);
}

/** Home-relative folder/file browser over /api/ls. */
export function PathPicker(props: { open: boolean; onClose: (path: string | null) => void }) {
  const [path, setPath] = createSignal("");
  const [listing] = createResource(
    () => (props.open ? path() : null),
    (p) => api.ls(p),
  );
  const join = (name: string) => (path() ? `${path()}/${name}` : name);
  /** A root outside the curation home comes back as an absolute path, so the
      leading separator is already there. */
  const shown = () => (path().startsWith("/") ? path() : `/${path()}`);

  return (
    <Dialog
      open={props.open}
      class="picker"
      onClose={(v) => {
        props.onClose(v === "ok" ? path() : null);
        setPath("");
      }}
    >
      <div>
        <b>{t().picker.title}</b> <span class="dim">{shown()}</span>
      </div>
      <ul>
        {/* Up is wherever the server says up is -- past the curation home too,
            which is how a root gets pinned to a tree beside it. */}
        <Show when={listing()?.parent !== null && listing()?.parent !== undefined}>
          <li onClick={() => setPath(listing()!.parent!)}>..</li>
        </Show>
        <Show when={listing.error}>
          <li class="err">{String(listing.error)}</li>
        </Show>
        <For each={listing()?.entries ?? []}>
          {(e) => (
            <li
              onClick={() => {
                if (e.dir) setPath(join(e.name));
                else {
                  props.onClose(join(e.name));
                  setPath("");
                }
              }}
            >
              {e.dir ? "📁" : "·"} {e.name}
            </li>
          )}
        </For>
      </ul>
      <div class="dlg-actions">
        <button value="cancel">{t().common.cancel}</button>
        <button value="ok" class="primary">
          {t().picker.use}
        </button>
      </div>
    </Dialog>
  );
}
