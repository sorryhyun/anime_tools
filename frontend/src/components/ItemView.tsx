import { createEffect, createMemo, createSignal, For, on, Show } from "solid-js";
import { api } from "../api";
import { t } from "../i18n";
import { CaptionCard } from "./CaptionCard";
import { OcrPanel } from "./OcrPanel";
import type {
  CaptionEntry,
  ImageInfo,
  ItemDetail,
  NodeKind,
  Proposal,
  VersionKind,
} from "../types";

/** The two files a preview can show, plus `overlay`: the mask drawn over the
    image at 40%, which is how a mask is audited. The resized pixels get no tab
    of their own (the source re-encoded onto the bucket geometry is the same
    picture), but are still the overlay's base when there is no source. */
const FILE_VIEWS = ["image", "mask"] as const;
type FileView = (typeof FILE_VIEWS)[number];
type View = FileView | "overlay";
const viewLabel = (v: View) => t().item.views[v];

const mp = (px: number) => `${(px / 1e6).toFixed(2)} MP`;

/** The size read-out under the preview. An image below the resize floor is
    skipped by the preflight, so it never lands in `workspace/resized` — the tree
    every stage walks — and a stage over it writes nothing at all; the chip is
    where that is said. `too_small` null means the floor was never applied to
    this file (the mask, the resized copy, or a floor turned off), and an
    unmeasured image gets no chip. */
function Dims(props: { info: ImageInfo | null; floor: number }) {
  return (
    <Show when={props.info}>
      {(i) => (
        <>
          {i().width ?? "?"}×{i().height ?? "?"} · {(i().bytes / 1024).toFixed(0)} KB
          <Show when={i().pixels != null && i().too_small != null}>
            {" "}
            <span
              classList={{ badge: true, px: true, below: !!i().too_small }}
              title={
                i().too_small
                  ? t().item.belowFloor(mp(props.floor))
                  : t().item.aboveFloor(mp(props.floor))
              }
            >
              {mp(i().pixels!)}
            </span>
          </Show>
        </>
      )}
    </Show>
  );
}

/** The caption column's width, in px. Kept out of `persisted` like the dock
    height: it moves on every pointermove, so it saves once on pointerup. */
const CAP_W = "capw";
const CAP_MIN = 300;

/** Ctrl/⌘+scroll scales the preview in place, anchored on the pointer so the
    detail aimed at stays under it. */
const ZOOM_MAX = 12;

export function ItemView(props: {
  item?: ItemDetail;
  loading: boolean;
  error?: string;
  kind: NodeKind;
  /** Which caption version the panel shows. The panel picks for itself when
      this is `image` — see `CaptionCard`. */
  onSelectCaption: (kind: VersionKind) => void;
  /** What the last Run changed about this image, or undefined. It rides under
      the editor when the version it rewrote is the one on screen. */
  proposal?: Proposal;
  proposalStage?: string;
  /** The one global "show explanations" preference. */
  help: boolean;
  onSaved: (entry: CaptionEntry) => void;
}) {
  const [view, setView] = createSignal<View>("image");
  const [capW, setCapW] = createSignal(Number(localStorage.getItem(CAP_W)) || 420);
  let split!: HTMLDivElement;

  /** Drag the preview|caption boundary. */
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

  /** Zoom about the pointer. The frame centres the picture, so a point sits at
      `centre + pan + zoom*v`; holding it still across a zoom change is one solve
      for the new pan. */
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
            {/* The address only: the geometry belongs to whichever file is on
                screen, and the line under the picture reports it. */}
            <div class="crumbs">
              <Show when={it().dir}>
                <span class="dim">{it().dir}/</span>
              </Show>
              <b>{it().name}</b>
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
                      <div class="dim hint mono" title={`${it().mask!.path} · ${base()!.path}`}>
                        {t().item.overlayHint} · <Dims info={base()} floor={it().min_pixels} />
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
                        <div class="dim hint mono" title={img().path}>
                          <Dims info={img()} floor={it().min_pixels} />
                          <Show when={zoom() > 1}> · {zoom().toFixed(1)}×</Show>
                        </div>
                      </>
                    )}
                  </Show>
                </Show>
              </div>

              {/* One caption panel, not one card per tree: an image has a
                  *ladder* of captions and the panel's badges are that
                  ladder. */}
              <div class="captions">
                <CaptionCard
                  help={props.help}
                  rel={it().rel}
                  versions={it().versions}
                  kind={props.kind}
                  onSelect={props.onSelectCaption}
                  proposal={props.proposal}
                  proposalStage={props.proposalStage}
                  onSaved={props.onSaved}
                />
                <OcrPanel lines={it().ocr} />
              </div>
            </div>
          </>
        )}
      </Show>
    </main>
  );
}
