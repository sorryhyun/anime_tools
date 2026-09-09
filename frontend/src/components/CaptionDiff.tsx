import { createMemo, For, Show } from "solid-js";
import { t } from "../i18n";
import type { Clause, Parsed } from "../types";
import { ClauseRow } from "./ClauseRow";
import { Tag } from "./TagLens";

/** One change to this caption, as tags and clauses rather than characters: what
    a finished **Run** wrote, or the change a history badge stands for (that
    version against the one that replaced it). Both sides arrive already
    parsed — by `/api/jobs/{id}/proposal` for a run and by `/api/dataset/item`
    for a version — since the browser never splits a caption. */

/** A clause keyed the way the grammar keys it: its header names the position,
    so two clauses with the same header are the same clause, changed. */
const key = (c: Clause) => c.header;
const lower = (ts: string[]) => new Set(ts.map((t) => t.trim().toLowerCase()));

interface TagDelta {
  added: string[];
  removed: string[];
}

function delta(before: string[], after: string[]): TagDelta {
  const was = lower(before);
  const now = lower(after);
  return {
    added: after.filter((t) => !was.has(t.trim().toLowerCase())),
    removed: before.filter((t) => !now.has(t.trim().toLowerCase())),
  };
}

const empty = (d: TagDelta) => !d.added.length && !d.removed.length;

/** Bag + every clause, as one flat list of labelled deltas. A clause on only
    one side shows entirely as added or removed, which is what the position
    rewrite does when it moves a bound tag out of the bag. */
function rows(before: Parsed | null, after: Parsed | null) {
  const b = before ?? { flat_tags: [], clauses: [] };
  const a = after ?? { flat_tags: [], clauses: [] };
  const out: { label: string; pos?: boolean; d: TagDelta }[] = [
    { label: t().caption.bag, d: delta(b.flat_tags, a.flat_tags) },
  ];
  const bc = new Map(b.clauses.map((c) => [key(c), c]));
  const ac = new Map(a.clauses.map((c) => [key(c), c]));
  for (const k of new Set([...bc.keys(), ...ac.keys()]))
    out.push({
      label: k,
      pos: true,
      d: delta(bc.get(k)?.tags ?? [], ac.get(k)?.tags ?? []),
    });
  return out.filter((r) => !empty(r.d));
}

export function CaptionDiff(props: {
  /** The two sides, as the server parsed them. */
  before: Parsed | null;
  after: Parsed | null;
  /** The text the after side holds, printed whole under the deltas. */
  text: string;
  /** The header: what this change is, and the dim line saying whose it was. */
  title: string;
  note: string;
  /** The dot's hue — the rung this change belongs to. */
  hue: string;
  /** Whether the after side is still what the file holds: `false` draws the
      "on disk" badge, `true` the "superseded" one, and `undefined` no badge —
      a change read off the ladder is on disk by construction. */
  stale?: boolean;
}) {
  const d = createMemo(() => rows(props.before, props.after));
  return (
    <div classList={{ diff: true, stale: !!props.stale, past: props.stale === undefined }}>
      <div class="card-h">
        <span class={`dot ${props.hue}`} />
        <b>{props.title}</b>
        <span class="dim">{props.note}</span>
        <span class="sp" />
        <Show when={props.stale !== undefined}>
          <Show when={props.stale} fallback={<span class="badge">{t().diff.onDisk}</span>}>
            <span class="badge miss" title={t().diff.staleHint}>
              {t().diff.stale}
            </span>
          </Show>
        </Show>
      </div>
      <Show when={d().length} fallback={<div class="dim hint">{t().diff.reordered}</div>}>
        <div class="parsed">
          <For each={d()}>
            {(r) => (
              <ClauseRow label={r.label} pos={r.pos}>
                <For each={r.d.removed}>{(t) => <Tag tag={t} class="del" prefix="− " />}</For>
                <For each={r.d.added}>{(t) => <Tag tag={t} class="add" prefix="+ " />}</For>
              </ClauseRow>
            )}
          </For>
        </div>
      </Show>
      <div class="proposed mono">{props.text}</div>
    </div>
  );
}
