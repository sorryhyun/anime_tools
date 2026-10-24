import { createResource, For, Show } from "solid-js";
import { api } from "../api";
import { t } from "../i18n";
import type { CaptionEntry } from "../types";

/** Tiles past this are named, not drawn: a shift-click across a big folder is
    one gesture, and the panel would otherwise fetch every image in it. */
export const COMPARE_MAX = 12;

/** The caption a tile shows: the revised one when a stage has written it, else
    the hand-written master — the text the image says now, which is what the
    row's last filled dot is too. */
function current(versions: CaptionEntry[]): CaptionEntry | undefined {
  const on = (kind: string) => versions.find((v) => v.kind === kind && v.exists && v.text);
  return on("revised") ?? on("master");
}

/** One picked image: the picture, its name and its caption, read on its own so
    a slow one does not hold the rest of the grid back. */
function Tile(props: { rel: string; onOpen: (rel: string) => void }) {
  const [item] = createResource(() => props.rel, api.item);
  const img = () => item()?.image ?? item()?.resized ?? null;
  const cap = () => current(item()?.versions ?? []);
  return (
    <div class="ctile" title={t().compare.open} onClick={() => props.onOpen(props.rel)}>
      <div class="cimg">
        <Show
          when={img()}
          fallback={
            <span class={item.error ? "err" : "dim"}>
              {item.error ? String(item.error) : t().item.loading}
            </span>
          }
        >
          {(i) => <img src={api.fileUrl(i().path)} alt={props.rel} draggable={false} />}
        </Show>
      </div>
      <div class="cname mono">
        <Show when={item()?.dir}>
          <span class="dim">{item()!.dir}/</span>
        </Show>
        {item()?.name ?? props.rel}
      </div>
      <Show
        when={cap()}
        fallback={<div class="ccap dim">{item() ? t().compare.noCaption : ""}</div>}
      >
        {(c) => (
          <div class="ccap">
            <span
              classList={{ dot: true, [c().kind]: true }}
              title={c().kind === "revised" ? t().tree.capRevised : t().tree.capMaster}
            />
            {c().text}
          </div>
        )}
      </Show>
    </div>
  );
}

/** Compare view: the images a shift- or ⌘/Ctrl-click gathered, side by side
 * with each one's caption under it. It stands in for the item panel while the
 * pick holds two or more; a tile clicked opens that image alone, and Esc gives
 * back the one the pick was drawn from.
 */
export function CompareView(props: {
  rels: string[];
  onOpen: (rel: string) => void;
  onClose: () => void;
}) {
  const shown = () => props.rels.slice(0, COMPARE_MAX);
  return (
    <main>
      <div class="crumbs">
        <b>{t().compare.title(props.rels.length)}</b>
        <span class="dim">{t().compare.hint}</span>
        <span class="sp" />
        <button onClick={props.onClose}>{t().common.close}</button>
      </div>
      <div class="compare">
        <For each={shown()}>{(rel) => <Tile rel={rel} onOpen={props.onOpen} />}</For>
      </div>
      <Show when={props.rels.length > COMPARE_MAX}>
        <div class="dim hint">{t().compare.more(props.rels.length - COMPARE_MAX)}</div>
      </Show>
    </main>
  );
}
