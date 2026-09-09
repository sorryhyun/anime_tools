import { createEffect, createMemo, createSignal, For, on, Show } from "solid-js";
import { api } from "../api";
import { t } from "../i18n";
import { trackPointer } from "../state";
import { createZoomPan } from "../zoomPan";
import { CaptionCard } from "./CaptionCard";
import { OcrPanel } from "./OcrPanel";
import { RevealButton } from "./RevealButton";
import type {
  ExcludeResult,
  ImageInfo,
  ItemDetail,
  NodeKind,
  Proposal,
  SavedCaption,
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
  /** The host can show a file in its own file manager (`Info.can_reveal`);
      false hides the ↗ on the address line. */
  canReveal: boolean;
  /** The caption panel's own explanations, and the (?) in its header. */
  help: boolean;
  onHelp: () => void;
  onSaved: (saved: SavedCaption) => void;
  /** Take this image out of the pipeline, or put it back. Resolves once the
      files have moved, which is what the button waits on. */
  onSetExcluded: (excluded: boolean) => Promise<ExcludeResult | undefined>;
}) {
  const [view, setView] = createSignal<View>("image");
  /** The exclusion round-trip: one at a time, and its failure said on the line
      it was clicked from rather than in the run bar — nothing was run. */
  const [moving, setMoving] = createSignal(false);
  const [moveErr, setMoveErr] = createSignal("");
  const [kept, setKept] = createSignal(0);
  async function toggleExcluded(excluded: boolean) {
    setMoving(true);
    setMoveErr("");
    try {
      setKept((await props.onSetExcluded(excluded))?.skipped.length ?? 0);
    } catch (e) {
      setMoveErr(e instanceof Error ? e.message : String(e));
    } finally {
      setMoving(false);
    }
  }
  const [capW, setCapW] = createSignal(Number(localStorage.getItem(CAP_W)) || 420);
  let split!: HTMLDivElement;

  /** Drag the preview|caption boundary. */
  function grip(e: PointerEvent) {
    const x0 = e.clientX;
    const w0 = capW();
    trackPointer(
      e,
      (ev) => setCapW(Math.max(CAP_MIN, Math.min(split.clientWidth - 240, w0 + (x0 - ev.clientX)))),
      () => localStorage.setItem(CAP_W, String(capW())),
    );
  }
  const zp = createZoomPan();

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
  createEffect(on([() => props.item, view], zp.reset, { defer: true }));

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
              {/* The source file, or the resized copy for an image that only
                  exists in the workspace -- the address line names one image
                  and this opens the file it stands for, whichever tab is up. */}
              <RevealButton path={(it().image ?? it().resized)?.path} can={props.canReveal} />
            </div>

            <Show when={it().excluded}>
              {(e) => (
                <div class="excluded-bar">
                  <b>⊘ {t().item.excluded}</b>
                  <span class="dim">
                    {t().item.excludedAt(new Date(e().at * 1000).toLocaleString())} ·{" "}
                    {t().item.excludedFiles(e().moved.length)}
                  </span>
                  <Show when={e().note}>
                    <span class="note">{e().note}</span>
                  </Show>
                </div>
              )}
            </Show>
            <Show when={moveErr() || kept()}>
              <div classList={{ "excluded-bar": true, err: !!moveErr() }}>
                {moveErr() || t().item.excludeKept(kept())}
              </div>
            </Show>

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
                  {/* After the view tabs and set apart from them: it is an
                      action on the picture you have just been looking at, not a
                      fourth way of looking at it. One click moves the image out
                      of the trees the stages walk, and the same button moves it
                      back. */}
                  <button
                    classList={{ excl: true, on: !!it().excluded }}
                    disabled={moving()}
                    title={it().excluded ? t().item.restoreHint : t().item.excludeHint}
                    onClick={() => void toggleExcluded(!it().excluded)}
                  >
                    {it().excluded ? `↩ ${t().item.restore}` : `⊘ ${t().item.exclude}`}
                  </button>
                </div>
                <Show
                  when={view() !== "overlay"}
                  fallback={
                    <Show
                      when={overlayOk()}
                      fallback={<div class="empty dim">{t().item.noOverlay}</div>}
                    >
                      <div
                        classList={{ frame: true, zoomed: zp.zoom() > 1 }}
                        onWheel={zp.wheel}
                        onPointerDown={zp.drag}
                        onDblClick={zp.reset}
                        title={t().item.zoomHint}
                      >
                        <div class="overlay" style={zp.style()}>
                          <img src={api.fileUrl(base()!.path)} alt={it().rel} draggable={false} />
                          <img class="ov" src={api.fileUrl(it().mask!.path)} alt="" />
                        </div>
                      </div>
                      <div class="dim hint mono" title={`${it().mask!.path} · ${base()!.path}`}>
                        {t().item.overlayHint} · <Dims info={base()} floor={it().min_pixels} />
                        <Show when={zp.zoom() > 1}> · {zp.zoom().toFixed(1)}×</Show>
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
                          classList={{ frame: true, zoomed: zp.zoom() > 1 }}
                          onWheel={zp.wheel}
                          onPointerDown={zp.drag}
                          onDblClick={zp.reset}
                          title={t().item.zoomHint}
                        >
                          <img
                            src={api.fileUrl(img().path)}
                            alt={it().rel}
                            draggable={false}
                            style={zp.style()}
                          />
                        </div>
                        <div class="dim hint mono" title={img().path}>
                          <Dims info={img()} floor={it().min_pixels} />
                          <Show when={zp.zoom() > 1}> · {zp.zoom().toFixed(1)}×</Show>
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
                  onHelp={props.onHelp}
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
