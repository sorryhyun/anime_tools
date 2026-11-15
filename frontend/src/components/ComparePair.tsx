import {
  createEffect,
  createMemo,
  createResource,
  createSignal,
  on,
  onCleanup,
  onMount,
  Show,
} from "solid-js";
import { api } from "../api";
import { t } from "../i18n";
import { DIFF_CHANGED, drawDiff, loadPair, pixelDiff } from "../imageDiff";
import { trackPointer } from "../state";
import type { ImageInfo, ItemDetail } from "../types";
import { createZoomPan } from "../zoomPan";
import { Icon } from "./Icon";
import { Seg } from "./Seg";
import { ZoomFrame } from "./ZoomFrame";

const GAINS = ["1", "4", "16"] as const;
type Gain = (typeof GAINS)[number];

/** The picture a pair view lays over the other: the source, as the grid shows. */
const source = (it: ItemDetail | undefined): ImageInfo | null => it?.image ?? it?.resized ?? null;
const sizeOf = (i: HTMLImageElement) => `${i.naturalWidth}×${i.naturalHeight}`;
const pct = (f: number) => (f * 100).toFixed(f < 0.1 ? 2 : 1);

/** Two picked images in one frame: `slider` stacks the right one over the left
 * and clips it at a line you drag, `diff` paints their per-pixel difference.
 * Both draw from one decoded, aligned pair (`imageDiff.loadPair`): a pair of
 * different shapes is center-cropped or stretched onto one frame, whichever
 * matches better, and the bar says which — so a wipe and a diff line the pair
 * up the same way. Ctrl/⌘+scroll zooms either, as in the preview.
 */
export function Pair(props: {
  a: string;
  b: string;
  mode: "slider" | "diff";
  onOpen: (rel: string) => void;
  onSwap: () => void;
}) {
  const [ia] = createResource(() => props.a, api.item);
  const [ib] = createResource(() => props.b, api.item);
  const srcs = createMemo(() => {
    const a = source(ia());
    const b = source(ib());
    return a && b ? { a, b } : null;
  });
  const [pair] = createResource(
    () => {
      const s = srcs();
      return s ? `${api.fileUrl(s.a.path)}\n${api.fileUrl(s.b.path)}` : null;
    },
    (key) => {
      const [a, b] = key.split("\n");
      return loadPair(a, b);
    },
  );
  const mismatch = () => {
    const p = pair.error ? undefined : pair();
    return p && (p.a.naturalWidth !== p.b.naturalWidth || p.a.naturalHeight !== p.b.naturalHeight)
      ? p
      : null;
  };

  const zp = createZoomPan();
  createEffect(on([() => props.a, () => props.b, () => props.mode], zp.reset, { defer: true }));

  // The stage is sized in script, not CSS: it has to be the alignment's shape
  // contain-fitted into whatever the frame is, so both images and the diff
  // canvas fill it exactly and the split line is a fraction of it.
  let frame!: HTMLDivElement;
  const [box, setBox] = createSignal({ w: 0, h: 0 });
  onMount(() => {
    const ro = new ResizeObserver(([e]) =>
      setBox({ w: e.contentRect.width, h: e.contentRect.height }),
    );
    ro.observe(frame);
    onCleanup(() => ro.disconnect());
  });
  const fit = (ar: number) => {
    const w = Math.min(box().w, box().h * ar);
    return { width: `${w}px`, height: `${w / ar}px`, ...zp.style() };
  };

  const [split, setSplit] = createSignal(0.5);
  let stage: HTMLDivElement | undefined;
  const moveTo = (ev: PointerEvent) => {
    const r = stage!.getBoundingClientRect();
    setSplit(Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)));
  };
  const grab = (e: PointerEvent) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    moveTo(e);
    trackPointer(e, moveTo);
  };

  const [gain, setGain] = createSignal<Gain>("4");
  const diff = createMemo(() => {
    const p = pair.error ? undefined : pair();
    return props.mode === "diff" && p ? pixelDiff(p) : undefined;
  });
  const [canvas, setCanvas] = createSignal<HTMLCanvasElement>();
  createEffect(() => {
    const d = diff();
    const c = canvas();
    if (d && c) drawDiff(c, d, Number(gain()));
  });

  const error = () => ia.error ?? ib.error ?? pair.error;
  const name = (it: ItemDetail | undefined, rel: string) => it?.name ?? rel;

  return (
    <>
      <div class="pairbar">
        <button class="pname mono" title={t().compare.open} onClick={() => props.onOpen(props.a)}>
          {name(ia(), props.a)}
        </button>
        <button class="icon" title={t().compare.swap} onClick={props.onSwap}>
          <Icon name="swap" />
        </button>
        <button class="pname mono" title={t().compare.open} onClick={() => props.onOpen(props.b)}>
          {name(ib(), props.b)}
        </button>
        <span class="sp" />
        <Show when={props.mode === "diff"}>
          <Seg
            options={GAINS}
            value={gain()}
            label={(g) => `×${g}`}
            title={t().compare.gainHint}
            onPick={setGain}
          />
        </Show>
      </div>
      <div class="pairnote dim">
        <Show
          when={diff()}
          fallback={props.mode === "slider" ? t().compare.sliderHint : t().compare.diffHint}
        >
          {(d) =>
            t().compare.diffStats(
              pct(d().mean),
              pct(d().changed),
              DIFF_CHANGED,
              `${d().width}×${d().height}`,
            )
          }
        </Show>
        <Show when={mismatch()}>
          {(m) => (
            <span classList={{ warn: m().align.fit === "fill" }}>
              {" · "}
              {m().align.fit === "fill"
                ? t().compare.alignStretch(sizeOf(m().a), sizeOf(m().b))
                : t().compare.alignCrop(sizeOf(m().a), sizeOf(m().b), m().align.cropped === "a")}
            </span>
          )}
        </Show>
      </div>
      <ZoomFrame zp={zp} class="pairframe" ref={frame}>
        <Show
          when={!error() && pair()}
          fallback={
            <span class={error() ? "err" : "dim"}>
              {error() ? String(error()) : t().item.loading}
            </span>
          }
        >
          {(p) => (
            <Show
              when={props.mode === "slider"}
              fallback={
                <div class="pstage" style={fit(p().align.aspect)}>
                  <canvas class="diff" ref={setCanvas} />
                </div>
              }
            >
              <div
                class="pstage slider"
                ref={stage}
                style={fit(p().align.aspect)}
                onPointerDown={(e) => zp.zoom() === 1 && grab(e)}
              >
                <img
                  src={p().a.src}
                  alt={props.a}
                  draggable={false}
                  style={{ "object-fit": p().align.fit }}
                />
                <img
                  src={p().b.src}
                  alt={props.b}
                  draggable={false}
                  style={{
                    "object-fit": p().align.fit,
                    "clip-path": `inset(0 0 0 ${split() * 100}%)`,
                  }}
                />
                <div class="split" style={{ left: `${split() * 100}%` }} onPointerDown={grab}>
                  <span
                    class="knob"
                    style={{ transform: `translate(-50%, -50%) scale(${1 / zp.zoom()})` }}
                  />
                </div>
              </div>
            </Show>
          )}
        </Show>
      </ZoomFrame>
    </>
  );
}
