import { createResource, createSignal, For, Show } from "solid-js";
import { api } from "../api";
import { Dialog } from "./Dialog";

/** Home-relative folder/file browser over /api/ls. */
export function PathPicker(props: { open: boolean; onClose: (path: string | null) => void }) {
  const [path, setPath] = createSignal("");
  const [listing] = createResource(() => (props.open ? path() : null), (p) => api.ls(p));
  const parent = () => path().split("/").slice(0, -1).join("/");
  const join = (name: string) => (path() ? `${path()}/${name}` : name);

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
        <b>Choose a path</b> <span class="dim">/{path()}</span>
      </div>
      <ul>
        <Show when={path()}>
          <li onClick={() => setPath(parent())}>..</li>
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
        <button value="cancel">Cancel</button>
        <button value="ok" class="primary">Use this folder</button>
      </div>
    </Dialog>
  );
}
