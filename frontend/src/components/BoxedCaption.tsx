import { createEffect, createMemo, createSignal, For, on, onCleanup, onMount } from "solid-js";
import { t } from "../i18n";
import type { Span } from "../types";
import { showTag } from "./TagLens";

/** The caption editor: a plain `<textarea>` with a box drawn around every tag.
 *
 * A caption is one string and is edited as one -- select across two tags, retype
 * a comma, paste a line -- so the boxes are painted on a backdrop *behind* a
 * real textarea instead of replacing it with chips. The backdrop mirrors the
 * same text in the same font at the same width with the glyphs transparent, and
 * each tag's slice gets a `box-shadow` outline: a shadow costs no layout, so the
 * mirrored line breaks exactly where the textarea's do.
 *
 * Where a tag *ends* is the server's answer: the spans come from
 * `parse_caption`, and the only thing done to them here is `alignSpans`.
 */

/** Re-anchor spans parsed from `was` onto the live `now`, since a keystroke
 * lands ~200ms before the parse of it comes back. The two strings differ by one
 * edit: the unchanged head keeps its spans, the unchanged tail keeps them
 * shifted by the length change, and the tag straddling the edit loses its box
 * until the parse lands.
 *
 * Not a caption split -- it compares two strings and invents no boundary. */
export function alignSpans(was: string, spans: Span[], now: string): Span[] {
  if (was === now) return spans;
  const max = Math.min(was.length, now.length);
  let head = 0;
  while (head < max && was[head] === now[head]) head++;
  let tail = 0;
  while (tail < max - head && was[was.length - 1 - tail] === now[now.length - 1 - tail]) tail++;
  const delta = now.length - was.length;
  const from = was.length - tail;
  const out: Span[] = [];
  for (const s of spans) {
    if (s.end <= head) out.push(s);
    else if (s.start >= from) out.push({ ...s, start: s.start + delta, end: s.end + delta });
  }
  return out;
}

/** The mirror's content: the caption cut at the span boundaries, so a run is
    either one tag (boxed) or the punctuation between two (bare). */
function pieces(text: string, spans: Span[]) {
  const out: { text: string; span?: Span; i: number }[] = [];
  let at = 0;
  spans.forEach((s, i) => {
    if (s.start > at) out.push({ text: text.slice(at, s.start), i: -1 });
    out.push({ text: text.slice(s.start, s.end), span: s, i });
    at = s.end;
  });
  out.push({ text: text.slice(at), i: -1 });
  return out;
}

const boxClass = (s: Span) =>
  `tb${s.kind === "header" ? " hdr" : s.kind === "artist" ? " art" : ""}${
    s.clause >= 0 && s.kind !== "header" ? " cl" : ""
  }`;

export function BoxedCaption(props: {
  /** The live buffer — what the textarea holds. */
  text: string;
  /** Spans, and the text they were parsed from (the two travel together). */
  spans: Span[];
  parsedText: string;
  dirty: boolean;
  /** A generated version (a `v0…vN` sidecar row) is shown in the same editor,
      dimmed: a read-only textarea still selects, copies and looks a tag up. */
  readOnly?: boolean;
  placeholder: string;
  onInput: (v: string) => void;
  onKeyDown: (e: KeyboardEvent & { currentTarget: HTMLTextAreaElement }) => void;
}) {
  const [caret, setCaret] = createSignal(-1);
  let ta!: HTMLTextAreaElement;
  let hl!: HTMLDivElement;

  const spans = createMemo(() => alignSpans(props.parsedText, props.spans, props.text));

  /** Grow to fit instead of scrolling: a scrollbar would narrow the textarea's
      text column but not the backdrop's, and the two must wrap identically. */
  const fit = () => {
    ta.style.height = "auto";
    ta.style.height = `${ta.scrollHeight + (ta.offsetHeight - ta.clientHeight)}px`;
  };
  createEffect(on(() => props.text, fit));
  onMount(() => {
    fit();
    // The caption pane is drag-resizable, so a rewrap can happen with the text
    // untouched. Width only: refitting on our own height change would loop.
    let width = ta.clientWidth;
    const ro = new ResizeObserver(() => {
      if (ta.clientWidth === width) return;
      width = ta.clientWidth;
      fit();
    });
    ro.observe(ta);
    onCleanup(() => ro.disconnect());
  });

  /** The tag the caret is in — what a double-click looks up. */
  const track = () => setCaret(ta.selectionStart);
  const lookUp = () => {
    const i = spans().findIndex((s) => caret() >= s.start && caret() <= s.end);
    if (i < 0) return;
    const box = hl.querySelector(`[data-tb="${i}"]`);
    if (box)
      showTag(props.text.slice(spans()[i].start, spans()[i].end), box.getBoundingClientRect());
  };

  return (
    <div class="capwrap">
      <div class="caphl" aria-hidden="true" ref={hl}>
        <For each={pieces(props.text, spans())}>
          {(p) =>
            p.span ? (
              <span
                class={`${boxClass(p.span)}${caret() >= p.span.start && caret() <= p.span.end ? " on" : ""}`}
                data-tb={p.i}
              >
                {p.text}
              </span>
            ) : (
              p.text
            )
          }
        </For>
      </div>
      <textarea
        classList={{ cap: true, dirty: props.dirty, ro: !!props.readOnly }}
        spellcheck={false}
        readOnly={!!props.readOnly}
        ref={ta}
        value={props.text}
        placeholder={props.placeholder}
        title={t().caption.lookUpHint}
        onInput={(e) => {
          props.onInput(e.currentTarget.value);
          track();
        }}
        onKeyUp={track}
        onClick={track}
        onSelect={track}
        onBlur={() => setCaret(-1)}
        onDblClick={lookUp}
        onKeyDown={props.onKeyDown}
      />
    </div>
  );
}
