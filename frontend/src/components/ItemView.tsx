import { createEffect, createMemo, createSignal, For, Index, on, Show } from "solid-js";
import { api } from "../api";
import { t } from "../i18n";
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
const viewLabel = (v: View) => t().item.views[v];

const dims = (i: ImageInfo | null) =>
  i ? `${i.width ?? "?"}×${i.height ?? "?"} · ${(i.bytes / 1024).toFixed(0)} KB` : "";

/** The caption column's width, in px. Kept out of `persisted` for the reason
    the dock height is: it moves on every pointermove of a drag, so it is saved
    once on pointerup instead of at frame rate. */
const CAP_W = "capw";
const CAP_MIN = 300;

/** The preview is a magnifier, not a link: a click used to hand the file to the
    browser, which is the one place the mask, the resized pixels and the caption
    are *not* side by side. Ctrl/⌘+scroll scales in place instead, anchored on
    the pointer so the detail you aimed at stays under it. */
const ZOOM_MAX = 12;

export function ItemView(props: {
  item?: ItemDetail;
  loading: boolean;
  error?: string;
  kind: NodeKind;
  /** What the last Run proposed for this image, or undefined. It lands on the
      caption card of the kind the stage writes -- master or derived. */
  proposal?: Proposal;
  proposalStage?: string;
  /** The caption kind whose diff was dropped, because the form changed since
      the run. */
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
  /** Magnification of whatever the preview is showing, and the pan that keeps
      the zoomed-in point where it was. `pan` is in screen px and is only ever
      non-zero while zoomed, so leaving the zoom leaves the picture centred. */
  const [zoom, setZoom] = createSignal(1);
  const [pan, setPan] = createSignal({ x: 0, y: 0 });
  const reset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };
  const shownStyle = () => ({
    transform: `translate(${pan().x}px, ${pan().y}px) scale(${zoom()})`,
  });

  /** Zoom about the pointer. The frame centres the picture, so the untransformed
      centre is the frame's own centre: a point sits at `centre + pan + zoom*v`,
      and holding it still across a zoom change is one solve for the new pan. */
  function wheel(e: WheelEvent & { currentTarget: HTMLDivElement }) {
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    const z0 = zoom();
    const z = Math.min(ZOOM_MAX, Math.max(1, z0 * Math.exp(-e.deltaY / 400)));
    if (z === z0) return;
    if (z === 1) return reset();
    const r = e.currentTarget.getBoundingClientRect();
    const p = pan();
    const vx = (e.clientX - (r.left + r.width / 2) - p.x) / z0;
    const vy = (e.clientY - (r.top + r.height / 2) - p.y) / z0;
    setZoom(z);
    setPan({
      x: e.clientX - (r.left + r.width / 2) - z * vx,
      y: e.clientY - (r.top + r.height / 2) - z * vy,
    });
  }

  /** Drag to pan, but only once there is something off-frame to reach. */
  function drag(e: PointerEvent) {
    if (zoom() === 1 || e.button !== 0) return;
    e.preventDefault();
    const p0 = pan();
    const x0 = e.clientX;
    const y0 = e.clientY;
    const move = (ev: PointerEvent) =>
      setPan({ x: p0.x + (ev.clientX - x0), y: p0.y + (ev.clientY - y0) });
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
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
  // A zoom belongs to the picture it was aimed at, not to the pane: moving to
  // another image or another view starts fitted again.
  createEffect(on([() => props.item, view], reset, { defer: true }));

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
              <Show when={props.error} fallback={props.loading ? t().item.loading : t().item.pick}>
                <span class="err">{props.error}</span>
              </Show>
              <Show when={!props.error && !props.loading}>
                <div class="dim" style="margin-top:8px">
                  {t().item.keys}
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
              <div class="splitgrip" onPointerDown={grip} title={t().common.dragToResize} />
              <div class="preview">
                <div class="tabs sub">
                  <For each={[...FILE_VIEWS, "overlay"] as View[]}>
                    {(v) => {
                      const ok = () => (v === "overlay" ? overlayOk() : !!it()[v]);
                      const hint = () =>
                        v === "overlay"
                          ? overlayOk()
                            ? t().item.overlayHint
                            : t().item.overlayNeeds
                          : (it()[v]?.path ?? t().item.notGenerated);
                      return (
                        <a
                          classList={{ sel: view() === v, na: !ok() }}
                          onClick={() => ok() && setView(v)}
                          title={hint()}
                        >
                          {viewLabel(v)}
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
                      fallback={<div class="empty dim">{t().item.noOverlay}</div>}
                    >
                      <div
                        classList={{ frame: true, zoomed: zoom() > 1 }}
                        onWheel={wheel}
                        onPointerDown={drag}
                        onDblClick={reset}
                        title={t().item.zoomHint}
                      >
                        <div class="overlay" style={shownStyle()}>
                          <img src={api.fileUrl(base()!.path)} alt={it().rel} draggable={false} />
                          <img class="ov" src={api.fileUrl(it().mask!.path)} alt="" />
                        </div>
                      </div>
                      <div class="dim hint mono">
                        {it().mask!.path} · {t().item.overlayOver(base()!.path)}
                        <Show when={zoom() > 1}> · {zoom().toFixed(1)}×</Show>
                      </div>
                    </Show>
                  }
                >
                  <Show
                    when={shown()}
                    fallback={<div class="empty dim">{t().item.noView(viewLabel(view()))}</div>}
                  >
                    {(img) => (
                      <>
                        <div
                          classList={{ frame: true, zoomed: zoom() > 1 }}
                          onWheel={wheel}
                          onPointerDown={drag}
                          onDblClick={reset}
                          title={t().item.zoomHint}
                        >
                          <img
                            src={api.fileUrl(img().path)}
                            alt={it().rel}
                            draggable={false}
                            style={shownStyle()}
                          />
                        </div>
                        <div class="dim hint mono">
                          {img().path} · {dims(img())}
                          <Show when={zoom() > 1}> · {zoom().toFixed(1)}×</Show>
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
                    <b>{t().item.variants}</b>
                    <span class="dim path" title={it().variants.path}>
                      {it().variants.path}
                    </span>
                    <span class="sp" />
                    <span class="badge">{t().item.readOnly}</span>
                  </div>
                  <div class="dim hint">{t().item.variantsHint}</div>
                  <Show
                    when={it().variants.rows.length}
                    fallback={<div class="dim hint">{t().item.noSidecar}</div>}
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
