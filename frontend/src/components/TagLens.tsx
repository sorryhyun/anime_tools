import { createResource, createSignal, onCleanup, onMount, Show } from "solid-js";
import { api } from "../api";
import { t } from "../i18n";

/** Click a tag chip anywhere -- the caption's bag, a clause, a diff -- and ask
    the Danbooru KB what it means. A floating card, not a tooltip, so its often
    long wiki blurb can be read.

    The inspected tag is module state rather than a prop: chips are rendered in
    three components at two depths, and only one card is ever open. */
const [target, setTarget] = createSignal<{ tag: string; x: number; y: number } | null>(null);

/** Open the card on `tag`, hanging from the box the caller clicked. The chip
    below is one caller; the boxed caption editor is the other, where the "chip"
    is a rectangle painted behind the text of a real textarea. */
export function showTag(tag: string, from: DOMRect) {
  setTarget({ tag, x: from.left, y: from.bottom });
}

/** The chip. Every tag in the UI goes through here, so every tag is clickable. */
export function Tag(props: { tag: string; class?: string; prefix?: string }) {
  return (
    <button
      type="button"
      class={`tag${props.class ? " " + props.class : ""}`}
      title={t().tag.what(props.tag)}
      onClick={(e) => showTag(props.tag, e.currentTarget.getBoundingClientRect())}
    >
      {props.prefix}
      {props.tag}
    </button>
  );
}

const CARD_W = 380;

/** The card itself. Mounted once, at the app root, so it floats over the dock
    and the dialog alike. */
export function TagLens(props: { onInstall: () => void }) {
  const [info] = createResource(target, (t) => api.describeTag(t.tag));
  const close = () => setTarget(null);

  onMount(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && target()) {
        e.stopPropagation();
        close();
      }
    };
    // Capture phase: a click on another chip re-targets the card (its own
    // handler runs after this one), anything else dismisses it.
    const onDown = (e: MouseEvent) => {
      const el = e.target as HTMLElement | null;
      if (el?.closest(".taglens") || el?.closest(".tag")) return;
      close();
    };
    window.addEventListener("keydown", onKey, true);
    window.addEventListener("mousedown", onDown, true);
    onCleanup(() => {
      window.removeEventListener("keydown", onKey, true);
      window.removeEventListener("mousedown", onDown, true);
    });
  });

  /** Below the chip, nudged in from the right edge; flipped above when the
      bottom of the window is closer than the card is tall. */
  const pos = () => {
    const t = target();
    if (!t) return {};
    const left = Math.max(8, Math.min(t.x, window.innerWidth - CARD_W - 8));
    const room = window.innerHeight - t.y;
    return room > 240
      ? { left: `${left}px`, top: `${t.y + 6}px` }
      : { left: `${left}px`, bottom: `${window.innerHeight - t.y + 22}px` };
  };

  return (
    <Show when={target()}>
      {(tgt) => (
        <div class="taglens" style={pos()}>
          <div class="card-h">
            <b class="mono">{info()?.name ?? tgt().tag}</b>
            <Show when={info()?.kind}>
              <span class="badge">{info()!.kind}</span>
            </Show>
            <Show when={info()?.post_count}>
              <span class="dim">{t().tag.posts(info()!.post_count!.toLocaleString())}</span>
            </Show>
            <span class="sp" />
            <button type="button" class="link" title={t().tag.close} onClick={close}>
              ✕
            </button>
          </div>
          <Show when={!info.loading} fallback={<div class="dim hint">{t().tag.looking}</div>}>
            <Show
              when={info()?.installed}
              fallback={
                <div class="hint">
                  {t().tag.notInstalled}{" "}
                  <button type="button" class="link" onClick={props.onInstall}>
                    {t().tag.getIt}
                  </button>
                </div>
              }
            >
              <Show when={info()?.known} fallback={<div class="dim hint">{t().tag.unknown}</div>}>
                <Show when={info()?.category_path}>
                  <div class="tlcat">[{info()!.category_path}]</div>
                </Show>
                <Show
                  when={info()?.description}
                  fallback={<div class="dim hint">{t().tag.noDescription}</div>}
                >
                  <div class="tlbody">{info()!.description}</div>
                </Show>
                <Show when={info()?.exact === false}>
                  <div class="dim hint">{t().tag.matchedAs(info()!.name ?? "")}</div>
                </Show>
              </Show>
            </Show>
          </Show>
          <Show when={info.error}>
            <div class="hint err">{String(info.error)}</div>
          </Show>
          <Show when={info()?.source}>
            <div class="dim tlsrc mono">{info()!.source}</div>
          </Show>
        </div>
      )}
    </Show>
  );
}
