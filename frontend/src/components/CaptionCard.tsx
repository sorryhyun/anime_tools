import { createEffect, createMemo, createResource, createSignal, For, on, Show } from "solid-js";
import { api } from "../api";
import { t } from "../i18n";
import { BoxedCaption } from "./BoxedCaption";
import { CaptionDiff } from "./CaptionDiff";
import type { CaptionEntry, CaptionKind, Parsed, Proposal, VersionKind } from "../types";

/** The caption panel: one editor, and a badge per version of this image's
 * caption.
 *
 * An image's captions are a *ladder* (`dataset.CAPTION_LADDER`) -- the
 * hand-written master, what the stages derived from it, then the generated
 * `v0…vN` -- not a set of unrelated files, and the workspace/Export split is
 * what makes that true: the trees are where the tools keep their working
 * copies, and Export is the moment one of them becomes training data. So the
 * panel does not ask which *tree* you meant before you can type. It opens on
 * the caption you are almost always here for -- what the last run wrote, or
 * failing that the newest writable one on disk -- and the older versions are
 * badges beside it, one click away, filled when the file exists.
 */

/** A caption and the parse of it, kept together: the parse arrives ~a debounce
    behind the buffer, and its spans are offsets into the text it was run on, so
    which text that was has to travel with them. */
interface Snap {
  text: string;
  parsed: Parsed | null;
}

/** A rung's name. The three file rungs are translated; a sidecar label
    (`v0`, `r1`, …) is its own name in every language and is passed through. */
const label = (k: VersionKind) => (t().caption as Record<string, unknown>)[k] as string | undefined;
const vlabel = (k: VersionKind) => label(k) ?? k;
/** Which of the dot hues a badge wears — the same colours the sidebar's dot
    strip uses, so a badge and its dot are recognisably the same file. Anything
    the palette does not name is a sidecar label, which wears the sidecar's. */
const hue = (k: VersionKind) => (label(k) ? k : "variants");
const where = (e: CaptionEntry) =>
  e.editable
    ? e.kind === "master"
      ? t().caption.whereMaster
      : t().caption.whereDerived
    : t().caption.whereVariants;

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
  /** The ladder, oldest first, as `/api/dataset/item` sent it. */
  versions: CaptionEntry[];
  /** The selected node. A caption rung puts that version in the editor;
      anything else (an image row was clicked) leaves the choice to us. */
  kind: VersionKind;
  onSelect: (kind: VersionKind) => void;
  /** What the last Run proposed for *this* image, or undefined. Shown below the
      editor until an Apply writes it (or another Run replaces it). */
  proposal?: Proposal;
  proposalStage?: string;
  /** The run's diff for this caption was dropped (the form changed since); say
      so where it stood instead of letting it vanish silently. */
  dropped?: boolean;
  /** The global "show explanations" preference — the same ? that folds the
      prose in the dock and in Settings folds this card's. */
  help: boolean;
  onSaved: (entry: CaptionEntry) => void;
}) {
  let card!: HTMLDivElement;

  /** The version on screen. An explicit badge (or caption dot) wins; otherwise
      the panel opens on what the last run touched, then on the newest writable
      caption that exists — the working copy, never a `v3` picked for being
      last on the ladder. */
  const entry = createMemo(() => {
    const vs = props.versions;
    const pick = vs.find((v) => v.kind === props.kind);
    if (pick) return pick;
    const run = props.proposal && vs.find((v) => v.kind === props.proposal!.kind);
    if (run) return run;
    return (
      [...vs].reverse().find((v) => v.editable && v.exists) ?? vs.find((v) => v.exists) ?? vs[0]
    );
  });

  const [busy, setBusy] = createSignal(false);
  const [msg, setMsg] = createSignal<{ text: string; bad?: boolean } | null>(null);

  /** What has been typed against each rung and not yet saved, for as long as
      the panel stays on this image. The two-card panel kept a buffer per card,
      and one editor has to keep that promise with one field: comparing a draft
      against `v1` is a click, and a click must not be able to throw away what
      you typed. A rung with no draft simply shows the file. */
  const [drafts, setDrafts] = createSignal<Record<string, string>>({});
  const key = () => entry()?.kind ?? "";
  const text = () => drafts()[key()] ?? entry()?.text ?? "";
  const setText = (v: string) => setDrafts((d) => ({ ...d, [key()]: v }));
  const clearDraft = (k: string) =>
    setDrafts((d) => {
      const { [k]: _gone, ...rest } = d;
      return rest;
    });

  // Another image: nothing typed here belongs to it, and the status line under
  // the field was about the last one.
  createEffect(
    on(
      () => props.rel,
      () => {
        setDrafts({});
        setMsg(null);
        setLast(null);
      },
      { defer: true },
    ),
  );
  // The file under a draft changed — our own save, or a stage that rewrote it.
  // The draft was an edit of text that is gone, so it goes with it. Keyed on
  // the rung *staying the same*, or switching badges would clear the one being
  // switched to.
  createEffect(
    on(
      () => [entry()?.kind, entry()?.text] as const,
      ([k, txt], prev) => {
        if (k && prev && prev[0] === k && prev[1] !== txt) clearDraft(k);
      },
      { defer: true },
    ),
  );
  createEffect(() => {
    if (props.kind !== "image") card.scrollIntoView({ block: "nearest", behavior: "smooth" });
  });

  const editable = () => !!entry()?.editable;
  const dirty = () => editable() && text().trim() !== (entry()?.text ?? "").trim();
  /** Does *that* rung hold something unsaved? Its badge says so, so a draft
      left on another version is visible from the one you are on. */
  const dirtyIn = (k: VersionKind) => {
    const d = drafts()[k];
    const v = props.versions.find((x) => x.kind === k);
    return !!v?.editable && d !== undefined && d.trim() !== v.text.trim();
  };
  /** Which *tree* the file is in. The tail is identical for every writable rung
      (same relative path by contract) and the crumbs above already name it, so
      the root is the only telling part — and it rides with the prose the ?
      folds, since it is the same answer to the same question. The full path
      stays in the tooltip either way. */
  const root = () => {
    const p = entry()?.path ?? "";
    const tail = props.rel.replace(/\.[^./]+$/, ".txt");
    return p.endsWith(tail) ? p.slice(0, -tail.length).replace(/\/$/, "") : p;
  };
  const live = debounced(text, 180);
  // Saved captions arrive parsed — variants included, so a badge costs no round
  // trip. Only the edited buffer needs one.
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
  const saved = (): Snap => ({ text: entry()?.text ?? "", parsed: entry()?.parsed ?? null });
  const snap = (): Snap => (dirty() && last() ? last()! : saved());
  const parsed = (): Parsed | null => snap().parsed;

  /** The run's diff belongs to one rung. It rides under the editor when that is
      the rung on screen; from any other the badge above is the way to it. */
  const diffHere = () => !!props.proposal && props.proposal.kind === entry()?.kind;

  async function save() {
    const e = entry();
    if (!e?.editable) return;
    setBusy(true);
    setMsg(null);
    try {
      const out = await api.saveCaption(props.rel, e.kind as CaptionKind, text());
      clearDraft(e.kind);
      props.onSaved(out);
      setMsg({ text: out.variants_stale ? t().caption.savedStale : t().caption.saved });
    } catch (err) {
      setMsg({ text: (err as Error).message, bad: true });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div classList={{ card: true, sel: props.kind !== "image" }} ref={card}>
      {/* The ladder. A filled dot is a file on disk and a hollow one is not --
          the sidebar row's readout, at the size a name fits beside it -- and
          the badge is also the switch that puts that version in the editor. */}
      <div class="vbs">
        <For each={props.versions}>
          {(v) => (
            <button
              classList={{
                vb: true,
                on: v.kind === entry()?.kind,
                off: !v.exists,
                mod: dirtyIn(v.kind),
              }}
              title={`${v.path} — ${v.exists ? t().tree.onDisk : t().tree.capMissing}${
                dirtyIn(v.kind) ? ` · ${t().caption.unsaved}` : ""
              }`}
              onClick={() => props.onSelect(v.kind)}
            >
              <span class={`dot ${hue(v.kind)}`} />
              {vlabel(v.kind)}
              <Show when={props.proposal?.kind === v.kind}>
                <span class="dot proposal" title={t().caption.diffHere} />
              </Show>
            </button>
          )}
        </For>
      </div>

      <Show when={entry()}>
        {(e) => (
          <>
            <div class="card-h" style="margin-top:8px">
              <b title={e().path}>{vlabel(e().kind)}</b>
              <Show when={!e().exists}>
                <span class="badge">{t().caption.new}</span>
              </Show>
              <Show when={!e().editable}>
                <span class="badge">{t().item.readOnly}</span>
              </Show>
              <span class="sp" />
              <Show when={e().editable}>
                <button disabled={!dirty() || busy()} onClick={() => clearDraft(e().kind)}>
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
              </Show>
            </div>
            <Show when={props.help}>
              <div class="dim hint" title={e().path}>
                {where(e())} · {root()}
              </div>
            </Show>
            <BoxedCaption
              text={text()}
              spans={parsed()?.spans ?? []}
              parsedText={snap().text}
              dirty={dirty()}
              readOnly={!e().editable}
              placeholder={e().exists ? "" : t().caption.empty}
              onInput={setText}
              onKeyDown={(ev) => {
                // The core loop is typing in here; saving must not need the
                // mouse. Cmd/Ctrl+S is caught too, or the browser offers to
                // save the page.
                if (
                  (ev.metaKey || ev.ctrlKey) &&
                  (ev.key === "Enter" || ev.key.toLowerCase() === "s")
                ) {
                  ev.preventDefault();
                  if (dirty() && !busy()) void save();
                }
              }}
            />
          </>
        )}
      </Show>
      <Show when={parsed()} fallback={<div class="dim hint">{t().caption.noCaption}</div>}>
        {(pp) => (
          <div class="dim hint">
            {t().caption.tags(pp().flat_tags.length)} · {t().caption.clauses(pp().clauses.length)}
            <Show when={props.help}> · {t().caption.lookUpHint}</Show>
            <Show when={dirty()}> · {t().caption.unsaved}</Show>
          </div>
        )}
      </Show>
      <Show when={msg()}>
        {(m) => <div classList={{ hint: true, err: !!m().bad, ok: !m().bad }}>{m().text}</div>}
      </Show>
      <Show when={props.proposal}>
        {(p) => (
          <Show
            when={diffHere()}
            fallback={
              <button class="link hint" onClick={() => props.onSelect(p().kind)}>
                {t().caption.diffElsewhere(vlabel(p().kind))}
              </button>
            }
          >
            <CaptionDiff
              proposal={p()}
              stage={props.proposalStage ?? t().diff.lastRun}
              stale={(entry()?.text ?? "").trim() !== p().before.trim()}
            />
          </Show>
        )}
      </Show>
      <Show when={!props.proposal && props.dropped}>
        <div class="dim hint">{t().caption.dropped}</div>
      </Show>
    </div>
  );
}
