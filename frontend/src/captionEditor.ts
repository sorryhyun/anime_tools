import { createEffect, createMemo, createResource, createSignal, on } from "solid-js";
import { api } from "./api";
import { t } from "./i18n";
import { debounced } from "./state";
import type {
  CaptionEntry,
  CaptionKind,
  Parsed,
  Proposal,
  SavedCaption,
  VersionKind,
} from "./types";

/** A caption and the parse of it, kept together: the parse arrives ~a debounce
    behind the buffer, and its spans are offsets into the text it was run on, so
    which text that was has to travel with them. */
export interface Snap {
  text: string;
  parsed: Parsed | null;
}

/** The state behind the caption panel: which version is on screen, the unsaved
 * draft against each rung, the live parse of the buffer, and Save. One is
 * created per `CaptionCard`, over the props it reads, so it is a view-scoped
 * primitive rather than one of the page's composables -- but the rules of what
 * a draft is and when it goes are here and not in the markup.
 *
 * An image's captions are a *ladder* (`dataset.CAPTION_LADDER`) -- the
 * hand-written master, every version the revised caption used to be, that
 * caption itself, then the generated `v0…vN`. The editor opens on what the
 * last run wrote, or failing that the newest writable caption on disk.
 */
export function createCaptionEditor(src: {
  rel: () => string;
  /** The ladder, oldest first, as `/api/dataset/item` sent it. */
  versions: () => CaptionEntry[];
  /** The selected node. A caption rung puts that version in the editor;
      anything else (an image row was clicked) leaves the choice to us. */
  kind: () => VersionKind;
  /** What the last Run changed about *this* image, or undefined. */
  proposal: () => Proposal | undefined;
  onSaved: (saved: SavedCaption) => void;
}) {
  /** The version on screen. An explicit badge (or caption dot) wins; otherwise
      what the last run touched, then the newest writable caption that exists. */
  const entry = createMemo(() => {
    const vs = src.versions();
    const pick = vs.find((v) => v.kind === src.kind());
    if (pick) return pick;
    const run = src.proposal() && vs.find((v) => v.kind === src.proposal()!.kind);
    if (run) return run;
    return (
      [...vs].reverse().find((v) => v.editable && v.exists) ?? vs.find((v) => v.exists) ?? vs[0]
    );
  });

  const [busy, setBusy] = createSignal(false);
  const [msg, setMsg] = createSignal<{ text: string; bad?: boolean } | null>(null);

  /** What has been typed against each rung and not yet saved, for as long as
      the panel stays on this image: comparing a draft against `v1` is one click,
      and a click must not throw away what you typed. A rung with no draft shows
      the file. */
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
      src.rel,
      () => {
        setDrafts({});
        setMsg(null);
        setLast(null);
      },
      { defer: true },
    ),
  );
  // The file under a draft changed — our own save, or a stage that rewrote it —
  // so the draft goes with it. Keyed on the rung *staying the same*, or
  // switching badges would clear the one being switched to.
  createEffect(
    on(
      () => [entry()?.kind, entry()?.text] as const,
      ([k, txt], prev) => {
        if (k && prev && prev[0] === k && prev[1] !== txt) clearDraft(k);
      },
      { defer: true },
    ),
  );

  const editable = () => !!entry()?.editable;
  const dirty = () => editable() && text().trim() !== (entry()?.text ?? "").trim();
  /** Does *that* rung hold something unsaved? Its badge says so. */
  const dirtyIn = (k: VersionKind) => {
    const d = drafts()[k];
    const v = src.versions().find((x) => x.kind === k);
    return !!v?.editable && d !== undefined && d.trim() !== v.text.trim();
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
  /** The text the boxes are drawn from and its parse: the last live parse while
      the buffer is dirty, the file's own otherwise. */
  const snap = (): Snap => (dirty() && last() ? last()! : saved());
  const parsed = (): Parsed | null => snap().parsed;

  async function save() {
    const e = entry();
    if (!e?.editable || !dirty() || busy()) return;
    setBusy(true);
    setMsg(null);
    try {
      const out = await api.saveCaption(src.rel(), e.kind as CaptionKind, text());
      clearDraft(e.kind);
      src.onSaved(out);
      setMsg({ text: out.variants_stale ? t().caption.savedStale : t().caption.saved });
    } catch (err) {
      setMsg({ text: (err as Error).message, bad: true });
    } finally {
      setBusy(false);
    }
  }

  return { entry, text, setText, clearDraft, dirty, dirtyIn, busy, msg, snap, parsed, save };
}

export type CaptionEditor = ReturnType<typeof createCaptionEditor>;
