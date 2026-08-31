import { createResource, createSignal, onCleanup, onMount, Show } from "solid-js";
import { api } from "../api";

/** Click a tag chip anywhere -- the caption's bag, a clause, a proposed diff --
    and ask the Danbooru KB what it means. The trainer's Qt GUI answers this
    with a tooltip at the cursor; the same idea, as a small floating card that
    stays put until dismissed, so its (often long) wiki blurb can be read.

    The inspected tag is module state rather than a prop: chips are rendered in
    three components at two depths, and there is only ever one card open. */
const [target, setTarget] = createSignal<{ tag: string; x: number; y: number } | null>(null);

/** The chip. Every tag in the UI goes through here, so every tag is clickable. */
export function Tag(props: { tag: string; class?: string; prefix?: string }) {
  return (
    <button
      type="button"
      class={`tag${props.class ? " " + props.class : ""}`}
      title={`What is "${props.tag}"?`}
      onClick={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        setTarget({ tag: props.tag, x: r.left, y: r.bottom });
      }}
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
      {(t) => (
        <div class="taglens" style={pos()}>
          <div class="card-h">
            <b class="mono">{info()?.name ?? t().tag}</b>
            <Show when={info()?.kind}>
              <span class="badge">{info()!.kind}</span>
            </Show>
            <Show when={info()?.post_count}>
              <span class="dim">{info()!.post_count!.toLocaleString()} posts</span>
            </Show>
            <span class="sp" />
            <button type="button" class="link" title="Close (Esc)" onClick={close}>
              ✕
            </button>
          </div>
          <Show when={!info.loading} fallback={<div class="dim hint">looking it up…</div>}>
            <Show
              when={info()?.installed}
              fallback={
                <div class="hint">
                  The Danbooru tag KB is not downloaded. It is what caption correction types tags
                  against — and what this panel reads.{" "}
                  <button type="button" class="link" onClick={props.onInstall}>
                    Get it in Settings › Models
                  </button>
                </div>
              }
            >
              <Show
                when={info()?.known}
                fallback={
                  <div class="dim hint">
                    not a Danbooru tag — an Anima quality tag, a position phrase, or a typo.
                  </div>
                }
              >
                <Show when={info()?.category_path}>
                  <div class="tlcat">[{info()!.category_path}]</div>
                </Show>
                <Show
                  when={info()?.description}
                  fallback={<div class="dim hint">no wiki description for this tag.</div>}
                >
                  <div class="tlbody">{info()!.description}</div>
                </Show>
                <Show when={info()?.exact === false}>
                  <div class="dim hint">matched as “{info()!.name}”.</div>
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
