import { createEffect, createResource, createSignal, on, Show } from "solid-js";
import { api } from "../api";
import { t } from "../i18n";
import { BoxedCaption } from "./BoxedCaption";
import { CaptionDiff } from "./CaptionDiff";
import type { CaptionEntry, CaptionKind, Parsed, Proposal } from "../types";

/** A caption and the parse of it, kept together: the parse arrives ~a debounce
    behind the buffer, and its spans are offsets into the text it was run on, so
    which text that was has to travel with them. */
interface Snap {
  text: string;
  parsed: Parsed | null;
}

const label = (k: CaptionKind) => t().caption[k];
const where = (k: CaptionKind) =>
  k === "master" ? t().caption.whereMaster : t().caption.whereDerived;

/** Debounce the live parse so a keystroke isn't a request. */
function debounced<T>(source: () => T, ms: number) {
  const [out, setOut] = createSignal(source());
  let timer: ReturnType<typeof setTimeout> | undefined;
  createEffect(
    on(source, (v) => {
      clearTimeout(timer);
      timer = setTimeout(() => setOut(() => v), ms);
    }),
  );
  return out;
}

export function CaptionCard(props: {
  rel: string;
  entry: CaptionEntry;
  selected: boolean;
  /** What the last Run proposed for *this* caption, or undefined. Shown below
      the editor until an Apply writes it (or another Run replaces it). */
  proposal?: Proposal;
  proposalStage?: string;
  /** The run's diff for this caption was dropped (the form changed since); say
      so where it stood instead of letting it vanish silently. */
  dropped?: boolean;
  onSaved: (entry: CaptionEntry) => void;
}) {
  const [text, setText] = createSignal(props.entry.text);
  const [busy, setBusy] = createSignal(false);
  const [msg, setMsg] = createSignal<{ text: string; bad?: boolean } | null>(null);
  let card!: HTMLDivElement;

  // Re-sync the buffer whenever the tree moves to another image or the file
  // itself changed (our own save, or a stage that rewrote it)...
  createEffect(
    on(
      () => [props.rel, props.entry.text] as const,
      ([, t]) => setText(t),
    ),
  );
  // ...but only drop the status line when the selection actually moves, or the
  // "saved" note would be wiped by the entry it just produced.
  createEffect(
    on(
      () => props.rel,
      () => {
        setMsg(null);
        setLast(null);
      },
      { defer: true },
    ),
  );
  createEffect(() => {
    if (props.selected) card.scrollIntoView({ block: "nearest", behavior: "smooth" });
  });

  const dirty = () => text().trim() !== props.entry.text.trim();
  /** Which *tree* the file is in. The tail is identical for master and derived
      (same relative path by contract), so the root is the only telling part —
      the full path stays in the tooltip. */
  const root = () => {
    const tail = props.rel.replace(/\.[^./]+$/, ".txt");
    return props.entry.path.endsWith(tail)
      ? props.entry.path.slice(0, -tail.length).replace(/\/$/, "")
      : props.entry.path;
  };
  const live = debounced(text, 180);
  // Saved captions arrive parsed; only the edited buffer needs a round trip.
  const [preview] = createResource(
    () => (dirty() ? live() : null),
    async (t): Promise<Snap> => ({ text: t, parsed: await api.parse(t) }),
  );
  // Hold the last parse that came back rather than reading the resource: while
  // the next one is in flight the editor still has boxes to draw, and
  // `BoxedCaption` re-anchors them onto whatever has been typed since.
  const [last, setLast] = createSignal<Snap | null>(null);
  createEffect(() => {
    const p = preview();
    if (p) setLast(p);
  });
  const saved = (): Snap => ({ text: props.entry.text, parsed: props.entry.parsed });
  const snap = (): Snap => (dirty() && last() ? last()! : saved());
  const parsed = (): Parsed | null => snap().parsed;

  async function save() {
    setBusy(true);
    setMsg(null);
    try {
      const saved = await api.saveCaption(props.rel, props.entry.kind, text());
      props.onSaved(saved);
      setMsg({ text: saved.variants_stale ? t().caption.savedStale : t().caption.saved });
    } catch (e) {
      setMsg({ text: (e as Error).message, bad: true });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div classList={{ card: true, sel: props.selected }} ref={card}>
      <div class="card-h">
        <span class={`dot ${props.entry.kind}`} />
        <b>{label(props.entry.kind)}</b>
        <span class="dim path" title={props.entry.path}>
          {root()}
        </span>
        <Show when={!props.entry.exists}>
          <span class="badge">{t().caption.new}</span>
        </Show>
        <span class="sp" />
        <button disabled={!dirty() || busy()} onClick={() => setText(props.entry.text)}>
          {t().caption.revert}
        </button>
        <button
          class="primary"
          disabled={!dirty() || busy()}
          title={t().caption.saveHint}
          onClick={save}
        >
          {t().caption.save}
        </button>
      </div>
      <div class="dim hint">{where(props.entry.kind)}</div>
      <BoxedCaption
        text={text()}
        spans={parsed()?.spans ?? []}
        parsedText={snap().text}
        dirty={dirty()}
        placeholder={props.entry.exists ? "" : t().caption.empty}
        onInput={setText}
        onKeyDown={(e) => {
          // The core loop is typing in here; saving must not need the mouse.
          // Cmd/Ctrl+S is caught too, or the browser offers to save the page.
          if ((e.metaKey || e.ctrlKey) && (e.key === "Enter" || e.key.toLowerCase() === "s")) {
            e.preventDefault();
            if (dirty() && !busy()) void save();
          }
        }}
      />
      <Show when={parsed()} fallback={<div class="dim hint">{t().caption.noCaption}</div>}>
        {(pp) => (
          <div class="dim hint">
            {t().caption.tags(pp().flat_tags.length)} · {t().caption.clauses(pp().clauses.length)} ·{" "}
            {t().caption.lookUpHint}
            <Show when={dirty()}> · {t().caption.unsaved}</Show>
          </div>
        )}
      </Show>
      <Show when={msg()}>
        {(m) => <div classList={{ hint: true, err: !!m().bad, ok: !m().bad }}>{m().text}</div>}
      </Show>
      <Show when={props.proposal}>
        {(p) => (
          <CaptionDiff
            proposal={p()}
            stage={props.proposalStage ?? t().diff.lastRun}
            stale={props.entry.text.trim() !== p().before.trim()}
          />
        )}
      </Show>
      <Show when={!props.proposal && props.dropped}>
        <div class="dim hint">{t().caption.dropped}</div>
      </Show>
    </div>
  );
}
