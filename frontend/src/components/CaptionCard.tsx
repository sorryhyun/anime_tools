import { createEffect, createResource, createSignal, For, on, Show } from "solid-js";
import { api } from "../api";
import { CaptionDiff } from "./CaptionDiff";
import type { CaptionEntry, CaptionKind, Parsed, Proposal } from "../types";

const LABEL: Record<CaptionKind, string> = { master: "master", derived: "derived" };
const WHERE: Record<CaptionKind, string> = {
  master: "hand-written; the stages only read it",
  derived: "stage output; overwritten by the next correct/position run",
};

const count = (n: number, noun: string) => `${n} ${noun}${n === 1 ? "" : "s"}`;

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
  /** The stage that proposed it, for the diff's header. */
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
  createEffect(on(() => [props.rel, props.entry.text] as const, ([, t]) => setText(t)));
  // ...but only drop the status line when the selection actually moves, or the
  // "saved" note would be wiped by the entry it just produced.
  createEffect(on(() => props.rel, () => setMsg(null), { defer: true }));
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
  const live = debounced(text, 250);
  // Saved captions arrive parsed; only the edited buffer needs a round trip.
  const [preview] = createResource(
    () => (dirty() ? live() : null),
    (t) => api.parse(t),
  );
  const parsed = (): Parsed | null | undefined => (dirty() ? preview() : props.entry.parsed);

  async function save() {
    setBusy(true);
    setMsg(null);
    try {
      const saved = await api.saveCaption(props.rel, props.entry.kind, text());
      props.onSaved(saved);
      setMsg({
        text: saved.variants_stale
          ? "saved — .variants.txt is now stale; re-run correct + the trainer's TE re-encode"
          : "saved — follow with the trainer's TE re-encode",
      });
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
        <b>{LABEL[props.entry.kind]}</b>
        <span class="dim path" title={props.entry.path}>
          {root()}
        </span>
        <Show when={!props.entry.exists}>
          <span class="badge">new</span>
        </Show>
        <span class="sp" />
        <button disabled={!dirty() || busy()} onClick={() => setText(props.entry.text)}>
          Revert
        </button>
        <button class="primary" disabled={!dirty() || busy()} title="⌘/Ctrl+Enter" onClick={save}>
          Save
        </button>
      </div>
      <div class="dim hint">{WHERE[props.entry.kind]}</div>
      <textarea
        classList={{ cap: true, dirty: dirty() }}
        spellcheck={false}
        value={text()}
        placeholder={props.entry.exists ? "" : "no caption file yet — type one and save"}
        onInput={(e) => setText(e.currentTarget.value)}
        onKeyDown={(e) => {
          // The core loop is typing in here; saving must not need the mouse.
          // Cmd/Ctrl+S is caught too, or the browser offers to save the page.
          if ((e.metaKey || e.ctrlKey) && (e.key === "Enter" || e.key.toLowerCase() === "s")) {
            e.preventDefault();
            if (dirty() && !busy()) void save();
          }
        }}
      />
      <Show when={msg()}>
        {(m) => <div classList={{ hint: true, err: !!m().bad, ok: !m().bad }}>{m().text}</div>}
      </Show>
      <Show when={props.proposal}>
        {(p) => (
          <CaptionDiff
            proposal={p()}
            stage={props.proposalStage ?? "the last run"}
            stale={props.entry.text.trim() !== p().before.trim()}
          />
        )}
      </Show>
      <Show when={!props.proposal && props.dropped}>
        <div class="dim hint">
          diff dropped — the form changed since the run; Run again to recompute
        </div>
      </Show>
      <Show when={parsed()} fallback={<div class="dim hint">no caption</div>}>
        {(pp) => (
          <div class="parsed">
            <div class="dim hint">
              {count(pp().flat_tags.length, "tag")} · {count(pp().clauses.length, "clause")}
              <Show when={dirty()}> · unsaved preview</Show>
            </div>
            <div class="clause">
              <span class="ck">bag</span>
              <span class="tags">
                <For each={pp().flat_tags}>{(t) => <span class="tag">{t}</span>}</For>
              </span>
            </div>
            <For each={pp().clauses}>
              {(c) => (
                <div class="clause">
                  <span class="ck pos">{c.header}</span>
                  <span class="tags">
                    <For each={c.tags}>{(t) => <span class="tag">{t}</span>}</For>
                  </span>
                </div>
              )}
            </For>
          </div>
        )}
      </Show>
    </div>
  );
}
