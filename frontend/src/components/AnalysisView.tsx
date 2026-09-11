import { createEffect, createSignal, For, on, onCleanup, Show } from "solid-js";
import { api } from "../api";
import { t } from "../i18n";
import type { Analysis, AuditAnalysis, PositionAnalysis } from "../types";

/** One hue per instance, in label order: the swatch in a legend row is the
    colour its region is painted in. Past the eighth the hues repeat, which a
    position proposal (`max_instances`) does not reach. */
const HUES: [number, number, number][] = [
  [122, 162, 247],
  [158, 206, 106],
  [224, 175, 104],
  [247, 118, 142],
  [187, 154, 247],
  [125, 207, 255],
  [255, 158, 100],
  [115, 218, 202],
];
const hue = (i: number) => HUES[i % HUES.length];
const rgb = (i: number) => `rgb(${hue(i).join(",")})`;
const pct = (v: number | null | undefined) => (v == null ? "—" : `${Math.round(v * 100)}%`);

/** The instance label map drawn over the image it was cut from. The map is the
    stage's own 8-bit PNG (`0` background, `i + 1` instance `i`), coloured here
    per pixel; `focus` is the legend row under the pointer, drawn alone so an
    instance hidden behind another can still be found. */
function MaskFigure(props: { base: string; mask: string; focus: number | null }) {
  let canvas!: HTMLCanvasElement;
  const [labels, setLabels] = createSignal<ImageData | null>(null);

  createEffect(
    on(
      () => props.mask,
      (src) => {
        let live = true;
        onCleanup(() => (live = false));
        const img = new Image();
        img.onload = () => {
          if (!live) return;
          const c = document.createElement("canvas");
          c.width = img.naturalWidth;
          c.height = img.naturalHeight;
          const g = c.getContext("2d")!;
          g.drawImage(img, 0, 0);
          setLabels(g.getImageData(0, 0, c.width, c.height));
        };
        img.src = src;
      },
    ),
  );

  createEffect(() => {
    const src = labels();
    if (!src) return;
    canvas.width = src.width;
    canvas.height = src.height;
    const out = new ImageData(src.width, src.height);
    const f = props.focus;
    for (let p = 0; p < src.data.length; p += 4) {
      const v = src.data[p];
      if (!v || (f !== null && v - 1 !== f)) continue;
      const [r, g, b] = hue(v - 1);
      out.data[p] = r;
      out.data[p + 1] = g;
      out.data[p + 2] = b;
      out.data[p + 3] = f === null ? 120 : 170;
    }
    canvas.getContext("2d")!.putImageData(out, 0, 0);
  });

  return (
    <div class="anfig">
      <img src={api.fileUrl(props.base)} alt="" draggable={false} />
      <canvas ref={canvas} />
    </div>
  );
}

/** A legend row: the swatch, then what the stage said about that region. */
function Legend(props: {
  rows: { head: string; body: string; dim?: string }[];
  onFocus: (i: number | null) => void;
}) {
  return (
    <div class="anlegend" onPointerLeave={() => props.onFocus(null)}>
      <For each={props.rows}>
        {(r, i) => (
          <div class="anrow" onPointerEnter={() => props.onFocus(i())}>
            <span class="swatch" style={{ background: rgb(i()) }} />
            <b>{r.head}</b>
            <span class="mono">{r.body}</span>
            <Show when={r.dim}>
              <span class="dim">{r.dim}</span>
            </Show>
          </div>
        )}
      </For>
    </div>
  );
}

function PositionSection(props: {
  rec: PositionAnalysis;
  base: string | null;
  mask: string | null;
}) {
  const [focus, setFocus] = createSignal<number | null>(null);
  const ok = () => props.rec.status === "proposed";
  const rows = () =>
    props.rec.labels === "instances"
      ? props.rec.instances.map((inst) => ({
          head: inst.position || "—",
          body: inst.tags.join(", ") || t().analysis.noTags,
          dim: `${pct(inst.score)}${inst.source !== "subject" ? ` · ${inst.source}` : ""}`,
        }))
      : props.rec.detections.map((d, i) => ({
          head: `#${i + 1}`,
          body: "",
          dim: `${pct(d.score)}${d.source !== "subject" ? ` · ${d.source}` : ""}`,
        }));
  return (
    <section class="ansec">
      <div class="card-h">
        <b>{t().analysis.position}</b>
        <span classList={{ badge: true, ok: ok(), warn: !ok() }}>
          {ok() ? t().analysis.proposed : props.rec.status.replace(/^skip:/, "")}
        </span>
        <span class="sp" />
        <span class="dim">{t().analysis.detected(props.rec.detected, props.rec.expected)}</span>
      </div>
      <Show when={props.base && props.mask}>
        <MaskFigure base={props.base!} mask={props.mask!} focus={focus()} />
      </Show>
      <Legend rows={rows()} onFocus={setFocus} />
      <Show when={props.rec.moved.length}>
        <div class="dim hint">
          {t().analysis.moved}{" "}
          <span class="mono">
            {props.rec.moved.map((m) => `${m.tag} → ${m.position}`).join(" · ")}
          </span>
        </div>
      </Show>
    </section>
  );
}

function AuditSection(props: { rec: AuditAnalysis; base: string | null; mask: string | null }) {
  const [focus, setFocus] = createSignal<number | null>(null);
  const rows = () =>
    props.rec.crops.map((c) => ({
      head: c.position || "—",
      body: [c.name, ...Object.values(c.groups)].filter(Boolean).join(", ") || t().analysis.noTags,
      dim: `${pct(c.score)}${c.source !== "subject" ? ` · ${c.source}` : ""}`,
    }));
  return (
    <section class="ansec">
      <div class="card-h">
        <b>{t().analysis.audit}</b>
        <span classList={{ badge: true, ok: props.rec.confidence === "strong" }}>
          {props.rec.verdict} · {props.rec.confidence}
        </span>
      </div>
      <Show when={props.base && props.mask}>
        <MaskFigure base={props.base!} mask={props.mask!} focus={focus()} />
      </Show>
      <Legend rows={rows()} onFocus={setFocus} />
      <div class="dim hint">
        {t().analysis.auditFacts(
          pct(props.rec.tagger_multiple_views),
          props.rec.people_count ?? "—",
          pct(props.rec.identity_agreement),
        )}
      </div>
      <Show when={props.rec.witnesses.length}>
        <div class="dim hint">
          {t().analysis.witnesses} <span class="mono">{props.rec.witnesses.join(", ")}</span>
        </div>
      </Show>
      <Show when={props.rec.suggested_tag}>
        <div class="dim hint">
          {t().analysis.suggested} <span class="mono">{props.rec.suggested_tag}</span>
        </div>
      </Show>
    </section>
  );
}

/** What the caption panel shows under its analysis badge: the position sweep's
 * proposal for this image and, when its audit phase looked at it, the audit's
 * finding — each with the instance masks the stage cut, over the resized image
 * those masks were cut from. It draws the stage's record and decides nothing:
 * which image had what said about it is the stage's, kept per image beside its
 * report (`stages/_analysis.py`).
 */
export function AnalysisView(props: { analysis: Analysis; base: string | null }) {
  return (
    <div class="analysis">
      <Show when={props.analysis.position}>
        {(p) => <PositionSection rec={p().record} base={props.base} mask={p().mask} />}
      </Show>
      <Show when={props.analysis.audit}>
        {(a) => <AuditSection rec={a().record} base={props.base} mask={a().mask} />}
      </Show>
      <div class="dim hint">{t().analysis.where}</div>
    </div>
  );
}
