import { createEffect, For, Show } from "solid-js";
import { createCaptionEditor } from "../captionEditor";
import { t } from "../i18n";
import { BoxedCaption } from "./BoxedCaption";
import { CaptionDiff } from "./CaptionDiff";
import { HelpToggle } from "./HelpToggle";
import type { CaptionEntry, Proposal, SavedCaption, VersionKind } from "../types";

/** The caption panel: one editor, and a badge per version of this image's
 * caption. What a draft is, which version is on screen and what Save does are
 * `createCaptionEditor`'s; this draws the ladder, the field and the diff.
 */

/** A rung's name. The file rungs are translated; an expanded label (`v0`, `r1`,
    `revised@2`) is passed through. A version knows which rung it came out of, so
    neither the label nor the colour is inferred from the shape of an id. */
const label = (k: VersionKind) => (t().caption as Record<string, unknown>)[k] as string | undefined;
const vlabel = (e: CaptionEntry) => label(e.kind) ?? e.kind;
/** Which of the dot hues a badge wears — the same colours the sidebar's dot
    strip uses. */
const hue = (e: CaptionEntry) => e.rung;
/** The one line under the header saying what this version *is*, by rung. */
const where = (e: CaptionEntry) =>
  (t().caption as Record<string, unknown>)[`where_${e.rung}`] as string | undefined;

export function CaptionCard(props: {
  rel: string;
  /** The ladder, oldest first, as `/api/dataset/item` sent it. */
  versions: CaptionEntry[];
  /** The selected node. A caption rung puts that version in the editor;
      anything else (an image row was clicked) leaves the choice to the editor. */
  kind: VersionKind;
  onSelect: (kind: VersionKind) => void;
  /** What the last Run changed about *this* image, or undefined. Shown below
      the editor until another Run replaces it. */
  proposal?: Proposal;
  proposalStage?: string;
  /** This panel's own explanations — where the version on screen lives and what
      a double-click on a tag does. The (?) in the header is the only thing that
      opens them; the stage form's speaks for the dock alone. */
  help: boolean;
  onHelp: () => void;
  onSaved: (saved: SavedCaption) => void;
}) {
  let card!: HTMLDivElement;
  const ed = createCaptionEditor({
    rel: () => props.rel,
    versions: () => props.versions,
    kind: () => props.kind,
    proposal: () => props.proposal,
    onSaved: (s) => props.onSaved(s),
  });

  createEffect(() => {
    if (props.kind !== "image") card.scrollIntoView({ block: "nearest", behavior: "smooth" });
  });

  /** Which *tree* the file is in. The tail is identical for every writable rung
      (same relative path by contract), so only the root says anything; the full
      path stays in the tooltip. */
  const root = () => {
    const p = ed.entry()?.path ?? "";
    const tail = props.rel.replace(/\.[^./]+$/, ".txt");
    return p.endsWith(tail) ? p.slice(0, -tail.length).replace(/\/$/, "") : p;
  };

  /** The run's diff belongs to one rung. It rides under the editor when that is
      the rung on screen; from any other the badge above is the way to it. */
  const diffHere = () => !!props.proposal && props.proposal.kind === ed.entry()?.kind;

  return (
    <div classList={{ card: true, sel: props.kind !== "image" }} ref={card}>
      {/* The ladder. A filled dot is a file on disk and a hollow one is not,
          and the badge is also the switch that puts that version in the
          editor. */}
      <div class="vbs">
        <For each={props.versions}>
          {(v) => (
            <button
              classList={{
                vb: true,
                on: v.kind === ed.entry()?.kind,
                off: !v.exists,
                mod: ed.dirtyIn(v.kind),
              }}
              title={`${v.path} — ${v.exists ? t().tree.onDisk : t().tree.capMissing}${
                v.note ? ` · ${v.note}` : ""
              }${ed.dirtyIn(v.kind) ? ` · ${t().caption.unsaved}` : ""}`}
              onClick={() => props.onSelect(v.kind)}
            >
              <span class={`dot ${hue(v)}`} />
              {vlabel(v)}
              <Show when={props.proposal?.kind === v.kind}>
                <span class="dot proposal" title={t().caption.diffHere} />
              </Show>
            </button>
          )}
        </For>
      </div>

      <Show when={ed.entry()}>
        {(e) => (
          <>
            <div class="card-h" style="margin-top:8px">
              <b title={e().path}>{vlabel(e())}</b>
              <Show when={!e().exists}>
                <span class="badge">{t().caption.new}</span>
              </Show>
              <Show when={!e().editable}>
                <span class="badge">{t().item.readOnly}</span>
              </Show>
              <HelpToggle open={props.help} onToggle={props.onHelp} />
              <span class="sp" />
              <Show when={e().editable}>
                <button disabled={!ed.dirty() || ed.busy()} onClick={() => ed.clearDraft(e().kind)}>
                  {t().caption.revert}
                </button>
                <button
                  class="primary"
                  disabled={!ed.dirty() || ed.busy()}
                  title={t().caption.saveHint}
                  onClick={() => void ed.save()}
                >
                  {t().caption.save}
                </button>
              </Show>
            </div>
            {/* A history badge says *when* it stopped being the caption. */}
            <Show when={e().note}>
              <div class="dim hint">{e().note}</div>
            </Show>
            <Show when={props.help}>
              <div class="dim hint" title={e().path}>
                {where(e())} · {root()}
              </div>
            </Show>
            <BoxedCaption
              text={ed.text()}
              spans={ed.parsed()?.spans ?? []}
              parsedText={ed.snap().text}
              dirty={ed.dirty()}
              readOnly={!e().editable}
              placeholder={e().exists ? "" : t().caption.empty}
              onInput={ed.setText}
              onKeyDown={(ev) => {
                // Cmd/Ctrl+Enter saves; Cmd/Ctrl+S is caught too, or the
                // browser offers to save the page.
                if (
                  (ev.metaKey || ev.ctrlKey) &&
                  (ev.key === "Enter" || ev.key.toLowerCase() === "s")
                ) {
                  ev.preventDefault();
                  void ed.save();
                }
              }}
            />
          </>
        )}
      </Show>
      <Show when={ed.parsed()} fallback={<div class="dim hint">{t().caption.noCaption}</div>}>
        {(pp) => (
          <div class="dim hint">
            {t().caption.tags(pp().flat_tags.length)} · {t().caption.clauses(pp().clauses.length)}
            <Show when={props.help}> · {t().caption.lookUpHint}</Show>
            <Show when={ed.dirty()}> · {t().caption.unsaved}</Show>
          </div>
        )}
      </Show>
      <Show when={ed.msg()}>
        {(m) => <div classList={{ hint: true, err: !!m().bad, ok: !m().bad }}>{m().text}</div>}
      </Show>
      <Show when={props.proposal}>
        {(p) => (
          <Show
            when={diffHere()}
            fallback={
              <button class="link hint" onClick={() => props.onSelect(p().kind)}>
                {t().caption.diffElsewhere(label(p().kind) ?? p().kind)}
              </button>
            }
          >
            <CaptionDiff
              proposal={p()}
              stage={props.proposalStage ?? t().diff.lastRun}
              stale={(ed.entry()?.text ?? "").trim() !== p().after.trim()}
            />
          </Show>
        )}
      </Show>
    </div>
  );
}
