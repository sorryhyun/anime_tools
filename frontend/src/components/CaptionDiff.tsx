import { createMemo, For, Show } from "solid-js";
import { t } from "../i18n";
import type { Clause, Parsed, Proposal } from "../types";
import { ClauseRow } from "./ClauseRow";
import { Tag } from "./TagLens";

/** What a finished **Run** changed about this caption — it writes, so this is a
    record of the edit, not an offer. The version it replaced is a badge above.

    Both sides arrive already parsed (`/api/jobs/{id}/proposal` runs them
    through `parse_caption`), so the diff compares *tags and clauses* — the
    units the grammar actually has — instead of the characters of a line the
    browser is not allowed to split. */

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

/** Bag + every clause, as one flat list of labelled deltas. A clause that only
    exists on one side shows entirely as added or removed, which is exactly what
    the position rewrite does when it *moves* a bound tag out of the bag. */
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
  proposal: Proposal;
  /** The stage that proposed it, for the header ("proposed by Autotag captions"). */
  stage: string;
  /** True once the caption on disk no longer holds what the Run wrote — an
      edit, or a later run, has moved past it, so this diff is history. */
  stale?: boolean;
}) {
  const d = createMemo(() => rows(props.proposal.before_parsed, props.proposal.after_parsed));
  return (
    <div classList={{ diff: true, stale: !!props.stale }}>
      <div class="card-h">
        <span class="dot proposal" />
        <b>{t().diff.written}</b>
        <span class="dim">{t().diff.by(props.stage)}</span>
        <span class="sp" />
        <Show when={props.stale} fallback={<span class="badge">{t().diff.onDisk}</span>}>
          <span class="badge miss" title={t().diff.staleHint}>
            {t().diff.stale}
          </span>
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
      <div class="proposed mono">{props.proposal.after}</div>
    </div>
  );
}
