import { createEffect, createMemo, createSignal, For, Index, on, Show } from "solid-js";
import { api } from "../api";
import { CaptionCard } from "./CaptionCard";
import type {
  CaptionEntry,
  CaptionKind,
  ImageInfo,
  ItemDetail,
  NodeKind,
  Proposal,
} from "../types";

/** The three files a preview can show, plus `overlay`: the mask drawn over the
    image at 40%, which is how a mask is actually audited — flipping tabs and
    comparing from memory is not. */
const FILE_VIEWS = ["image", "resized", "mask"] as const;
type FileView = (typeof FILE_VIEWS)[number];
type View = FileView | "overlay";
const VIEW_LABEL: Record<View, string> = {
  image: "source",
  resized: "resized",
  mask: "mask",
  overlay: "overlay",
};

const dims = (i: ImageInfo | null) =>
  i ? `${i.width ?? "?"}×${i.height ?? "?"} · ${(i.bytes / 1024).toFixed(0)} KB` : "";

/** The caption column's width, in px. Kept out of `persisted` for the reason
    the dock height is: it moves on every pointermove of a drag, so it is saved
    once on pointerup instead of at frame rate. */
const CAP_W = "capw";
const CAP_MIN = 300;

export function ItemView(props: {
  item?: ItemDetail;
  loading: boolean;
  error?: string;
  kind: NodeKind;
  /** What the last Run proposed for this image, or undefined. It lands on the
      caption card of the kind the stage writes -- master or derived. */
  proposal?: Proposal;
  /** The stage that proposed it, for the diff's header. */
  proposalStage?: string;
  /** The caption kind whose diff was dropped (the form changed since the run)
      — its card says so where the diff stood, instead of vanishing silently. */
  droppedKind?: CaptionKind;
  onSaved: (entry: CaptionEntry) => void;
}) {
  const [view, setView] = createSignal<View>("image");
  const [capW, setCapW] = createSignal(Number(localStorage.getItem(CAP_W)) || 420);
  let split!: HTMLDivElement;

  /** Drag the preview|caption boundary. The image and the caption both want the
      width, and which one wins changes with what you are auditing -- a mask
      overlay wants the picture, a clause rewrite wants the text. */
  function grip(e: PointerEvent) {
    e.preventDefault();
    const x0 = e.clientX;
    const w0 = capW();
    const move = (ev: PointerEvent) =>
      setCapW(Math.max(CAP_MIN, Math.min(split.clientWidth - 240, w0 + (x0 - ev.clientX))));
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      localStorage.setItem(CAP_W, String(capW()));
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }
  /** Overlay needs a mask and something to draw it over. */
  const base = () => props.item?.image ?? props.item?.resized ?? null;
  const overlayOk = createMemo(() => !!props.item?.mask && !!base());
  const available = createMemo(() => {
    const vs: View[] = FILE_VIEWS.filter((v) => props.item?.[v]);
    if (overlayOk()) vs.push("overlay");
    return vs;
  });
  // Keep the toggle on something that exists as the selection moves.
  createEffect(
    on(
      () => props.item,
      () => {
        if (!available().includes(view())) setView(available()[0] ?? "image");
      },
    ),
  );
  const shown = () => (view() === "overlay" ? null : (props.item?.[view() as FileView] ?? null));

  // Picking `variants` in the tree has to bring its card into view, the same
  // way CaptionCard does for the two editable ones.
  let variantsCard!: HTMLDivElement;
  createEffect(() => {
    if (props.kind === "variants" && props.item)
      variantsCard?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  });

  return (
    <main>
      <Show
        when={props.item}
        fallback={
          <div class="empty">
            <div>
              <Show
                when={props.error}
                fallback={props.loading ? "loading…" : "Pick an image on the left."}
              >
                <span class="err">{props.error}</span>
              </Show>
              <Show when={!props.error && !props.loading}>
                <div class="dim" style="margin-top:8px">
                  ↑/↓ or j/k walk the images · ⌘/Ctrl+Enter saves a caption
                </div>
              </Show>
            </div>
          </div>
        }
      >
        {(it) => (
          <>
            <div class="crumbs">
              <Show when={it().dir}>
                <span class="dim">{it().dir}/</span>
              </Show>
              <b>{it().name}</b>
              <span class="sp" />
              <span class="dim">{dims(it().image)}</span>
            </div>

            <div class="split" ref={split} style={{ "--cap-w": `${capW()}px` }}>
              <div class="splitgrip" onPointerDown={grip} title="Drag to resize" />
              <div class="preview">
                <div class="tabs sub">
                  <For each={[...FILE_VIEWS, "overlay"] as View[]}>
                    {(v) => {
                      const ok = () => (v === "overlay" ? overlayOk() : !!it()[v]);
                      const hint = () =>
                        v === "overlay"
                          ? overlayOk()
                            ? "the mask at 40% over the image"
                            : "needs an image and a mask"
                          : (it()[v]?.path ?? "not generated");
                      return (
                        <a
                          classList={{ sel: view() === v, na: !ok() }}
                          onClick={() => ok() && setView(v)}
                          title={hint()}
                        >
                          {VIEW_LABEL[v]}
                        </a>
                      );
                    }}
                  </For>
                </div>
                <Show
                  when={view() !== "overlay"}
                  fallback={
                    <Show
                      when={overlayOk()}
                      fallback={<div class="empty dim">no overlay for this image</div>}
                    >
                      <div class="frame">
                        <div class="overlay">
                          <img src={api.fileUrl(base()!.path)} alt={it().rel} />
                          <img class="ov" src={api.fileUrl(it().mask!.path)} alt="" />
                        </div>
                      </div>
                      <div class="dim hint mono">
                        {it().mask!.path} · at 40% over {base()!.path}
                      </div>
                    </Show>
                  }
                >
                  <Show
                    when={shown()}
                    fallback={<div class="empty dim">no {VIEW_LABEL[view()]} for this image</div>}
                  >
                    {(img) => (
                      <>
                        <a class="frame" href={api.fileUrl(img().path)} target="_blank">
                          <img src={api.fileUrl(img().path)} alt={it().rel} />
                        </a>
                        <div class="dim hint mono">
                          {img().path} · {dims(img())}
                        </div>
                      </>
                    )}
                  </Show>
                </Show>
              </div>

              <div class="captions">
                {/* Index, not For: the pair is fixed and in fixed order, so a
                    saved entry should update its card, not replace it. */}
                <Index each={it().captions}>
                  {(c) => (
                    <CaptionCard
                      rel={it().rel}
                      entry={c()}
                      selected={props.kind === c().kind}
                      proposal={props.proposal?.kind === c().kind ? props.proposal : undefined}
                      proposalStage={props.proposalStage}
                      dropped={props.droppedKind === c().kind}
                      onSaved={props.onSaved}
                    />
                  )}
                </Index>

                <div classList={{ card: true, sel: props.kind === "variants" }} ref={variantsCard}>
                  <div class="card-h">
                    <span class="dot variants" />
                    <b>variants</b>
                    <span class="dim path" title={it().variants.path}>
                      {it().variants.path}
                    </span>
                    <span class="sp" />
                    <span class="badge">read-only</span>
                  </div>
                  <div class="dim hint">
                    Generated by the correct stage — v0 is the pristine derived caption. Hand edits
                    here are overwritten.
                  </div>
                  <Show
                    when={it().variants.rows.length}
                    fallback={<div class="dim hint">no sidecar</div>}
                  >
                    <table class="variants">
                      <tbody>
                        <For each={it().variants.rows}>
                          {(r) => (
                            <tr>
                              <td class="vlabel">{r.label}</td>
                              <td>{r.text}</td>
                            </tr>
                          )}
                        </For>
                      </tbody>
                    </table>
                  </Show>
                </div>
              </div>
            </div>
          </>
        )}
      </Show>
    </main>
  );
}
