import {
  createEffect,
  createMemo,
  createResource,
  createSignal,
  For,
  on,
  onCleanup,
  onMount,
  Show,
} from "solid-js";
import { api } from "../api";
import { t } from "../i18n";
import { DIFF_CHANGED, drawDiff, pixelDiff } from "../imageDiff";
import { COMPARE_MODES, type CompareMode } from "../layout";
import { trackPointer } from "../state";
import type { CaptionEntry, ImageInfo, ItemDetail } from "../types";
import { createZoomPan } from "../zoomPan";
import { Icon } from "./Icon";
import { Seg } from "./Seg";

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

const GAINS = ["1", "4", "16"] as const;
type Gain = (typeof GAINS)[number];

/** The picture a pair view lays over the other: the source, as the grid shows. */
const source = (it: ItemDetail | undefined): ImageInfo | null => it?.image ?? it?.resized ?? null;
const sizeOf = (i: ImageInfo) => (i.width && i.height ? `${i.width}×${i.height}` : "?");
const pct = (f: number) => (f * 100).toFixed(f < 0.1 ? 2 : 1);

/** Two picked images in one frame: `slider` stacks the right one over the left
 * and clips it at a line you drag, `diff` paints their per-pixel difference.
 * Both put the right image on the left one's frame — stretched if the shapes
 * differ, which the bar says — so a wipe and a diff line the pair up the same
 * way. Ctrl/⌘+scroll zooms either, as in the preview.
 */
function Pair(props: {
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
  const mismatch = () => {
    const s = srcs();
    return s && (s.a.width !== s.b.width || s.a.height !== s.b.height) ? s : null;
  };

  const zp = createZoomPan();
  createEffect(on([() => props.a, () => props.b, () => props.mode], zp.reset, { defer: true }));

  // The stage is sized in script, not CSS: it has to be the left image's shape
  // contain-fitted into whatever the frame is, so the right image and the
  // diff canvas can fill it exactly and the split line is a fraction of it.
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

  /** The header's width/height first, then the decoded image's own, since a
      header that would not read leaves them out. */
  const [natural, setNatural] = createSignal(0);
  const aspect = () => {
    const a = srcs()?.a;
    return natural() || (a?.width && a.height ? a.width / a.height : 1);
  };
  createEffect(
    on(
      () => props.a,
      () => setNatural(0),
    ),
  );

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
  const [diff] = createResource(
    () => {
      const s = srcs();
      return props.mode === "diff" && s
        ? `${api.fileUrl(s.a.path)}\n${api.fileUrl(s.b.path)}`
        : null;
    },
    (key) => {
      const [a, b] = key.split("\n");
      return pixelDiff(a, b);
    },
  );
  const [canvas, setCanvas] = createSignal<HTMLCanvasElement>();
  createEffect(() => {
    const d = diff.error ? undefined : diff();
    const c = canvas();
    if (d && c) drawDiff(c, d, Number(gain()));
  });

  const error = () => ia.error ?? ib.error ?? (props.mode === "diff" ? diff.error : undefined);
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
          when={props.mode === "diff" && diff() && !diff.error ? diff() : null}
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
            <span class="warn"> · {t().compare.sizeMismatch(sizeOf(m().a), sizeOf(m().b))}</span>
          )}
        </Show>
      </div>
      <div
        class="pairframe"
        classList={{ zoomed: zp.zoom() > 1 }}
        ref={frame}
        onWheel={zp.wheel}
        onPointerDown={zp.drag}
      >
        <Show
          when={!error() && srcs()}
          fallback={
            <span class={error() ? "err" : "dim"}>
              {error() ? String(error()) : t().item.loading}
            </span>
          }
        >
          {(s) => (
            <Show
              when={props.mode === "slider"}
              fallback={
                <Show when={diff()} fallback={<span class="dim">{t().compare.diffing}</span>}>
                  {(d) => (
                    <div class="pstage" style={fit(d().width / d().height)}>
                      <canvas class="diff" ref={setCanvas} />
                    </div>
                  )}
                </Show>
              }
            >
              <div
                class="pstage slider"
                ref={stage}
                style={fit(aspect())}
                onPointerDown={(e) => zp.zoom() === 1 && grab(e)}
              >
                <img
                  src={api.fileUrl(s().a.path)}
                  alt={props.a}
                  draggable={false}
                  onLoad={(e) =>
                    setNatural(e.currentTarget.naturalWidth / e.currentTarget.naturalHeight)
                  }
                />
                <img
                  src={api.fileUrl(s().b.path)}
                  alt={props.b}
                  draggable={false}
                  style={{ "clip-path": `inset(0 0 0 ${split() * 100}%)` }}
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
      </div>
    </>
  );
}

/** Compare view: the images a shift- or ⌘/Ctrl-click gathered, side by side
 * with each one's caption under it. It stands in for the item panel while the
 * pick holds two or more; a tile clicked opens that image alone, and Esc gives
 * back the one the pick was drawn from. A pick of exactly two can also be laid
 * one over the other (`Pair`); the mode is a saved preference, and a pick of
 * any other size is the grid whatever it says.
 */
export function CompareView(props: {
  rels: string[];
  mode: CompareMode;
  onMode: (m: CompareMode) => void;
  onOpen: (rel: string) => void;
  onClose: () => void;
}) {
  const shown = () => props.rels.slice(0, COMPARE_MAX);
  const pair = () => props.rels.length === 2;
  const mode = () => (pair() ? props.mode : "grid");
  /** Which of the two is on the left. View-local: it is a look at this pair. */
  const [flip, setFlip] = createSignal(false);
  createEffect(
    on(
      () => props.rels,
      () => setFlip(false),
      { defer: true },
    ),
  );
  return (
    <main>
      <div class="crumbs">
        <b>{t().compare.title(props.rels.length)}</b>
        <span class="dim">{t().compare.hint}</span>
        <span class="sp" />
        <Show when={pair()}>
          <Seg
            options={COMPARE_MODES}
            value={mode()}
            label={(m) => t().compare.modes[m]}
            title={t().compare.modeHint}
            onPick={props.onMode}
          />
        </Show>
        <button onClick={props.onClose}>{t().common.close}</button>
      </div>
      <Show
        when={mode() !== "grid"}
        fallback={
          <>
            <div class="compare">
              <For each={shown()}>{(rel) => <Tile rel={rel} onOpen={props.onOpen} />}</For>
            </div>
            <Show when={props.rels.length > COMPARE_MAX}>
              <div class="dim hint">{t().compare.more(props.rels.length - COMPARE_MAX)}</div>
            </Show>
          </>
        }
      >
        <Pair
          a={props.rels[flip() ? 1 : 0]}
          b={props.rels[flip() ? 0 : 1]}
          mode={mode() as "slider" | "diff"}
          onOpen={props.onOpen}
          onSwap={() => setFlip(!flip())}
        />
      </Show>
    </main>
  );
}
