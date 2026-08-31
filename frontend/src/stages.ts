import { createEffect, createMemo, createResource, on } from "solid-js";
import { createStore, reconcile } from "solid-js/store";
import { api } from "./api";
import { persisted } from "./state";
import { REPLAY_FIELD, type Field, type Stage, type Values } from "./types";
import type { Config } from "./config";

/** The stage registry and the forms over it: which stage is open, what its form
 * says, and how the dock's buttons bucket the stages.
 *
 * Every field here comes from the stage CLI's own argparse (`gui/stages.py`
 * dumps it in a child interpreter), so nothing about a flag is re-typed in the
 * browser — including the ones bound to Settings, which the server fills and
 * this never shows. Running one of these is `runner.ts`; this module only knows
 * what *would* be run.
 */
export function createStages(config: Config) {
  const [all, { refetch }] = createResource<Stage[]>(api.stages);
  const [curId, setCurId] = persisted("stage", "", String);
  const cur = createMemo(() => all()?.find((s) => s.id === curId()));

  const [forms, setForms] = createStore<Record<string, Values>>({});
  const values = () => forms[curId()] ?? {};
  const setValue = (dest: string, v: unknown) =>
    setForms(curId(), (prev) => ({ ...(prev ?? {}), [dest]: v }));
  const resetForm = () => setForms(curId(), reconcile({}));
  // The saved forms ride in on the same settings read the config store took.
  void config.loaded.then((s) => setForms(reconcile(structuredClone(s.values ?? {}))));

  /** The form as it matters to a replay: everything the stage actually reads,
      minus the managed field itself. */
  const formKey = (v: Values) => {
    const { [REPLAY_FIELD]: _managed, ...rest } = v ?? {};
    return JSON.stringify(rest);
  };

  /** The stages the dock offers: everything but the hidden ones, which are not
      run by hand (`resize` runs itself, in front of the stages that need it). */
  const shown = createMemo(() => (all() ?? []).filter((s) => !s.hidden));
  /** Stages, in registry order, bucketed into the dock's panels. One button
      per panel; which of its stages runs is picked inside the panel. */
  const panels = createMemo(() => {
    const m = new Map<string, Stage[]>();
    for (const s of shown()) m.set(s.panel, [...(m.get(s.panel) ?? []), s]);
    return [...m];
  });
  const curPanel = createMemo(() => cur()?.panel ?? "");
  /** Per panel, the stage last picked in it, so re-opening a panel comes back
      to where you left it instead of always its first stage. */
  const [lastInPanel, setLastInPanel] = createStore<Record<string, string>>({});

  /** One Field descriptor per Settings-bound dest, first stage that has it
      wins: the input's label, help and placeholder all come from the real
      argparse action, so Settings never re-describes a flag. */
  const settingFields = createMemo(() => {
    const m = new Map<string, Field>();
    for (const st of shown())
      for (const f of st.fields) if (f.setting && !m.has(f.setting)) m.set(f.setting, f);
    return [...m.values()];
  });
  /** The hidden preflight stage, so Settings can render its form: it has no
      dock panel of its own, and its knobs apply to every stage it precedes. */
  const preprocessStage = createMemo(() =>
    (all() ?? []).find((s) => s.hidden && s.id === "resize"),
  );
  /** Catalog rows the open stage needs that are not downloaded yet — surfaced
      in the stage bar, so a Run does not stall on a surprise multi-GB fetch. */
  const missingModels = createMemo(() =>
    (config.models()?.models ?? []).filter((m) => !m.installed && m.stages.includes(curId())),
  );

  createEffect(
    on(all, (ss) => {
      // A hidden stage is never the landing stage: it has no dock button to
      // show it, so selecting one would open the dock on nothing.
      if (ss && !shown().some((s) => s.id === curId()))
        setCurId(shown().find((s) => s.available)?.id ?? "");
    }),
  );
  // Cold start: /api/stages answers 503 while the server's background schema
  // dump is still running, and a resource error is otherwise permanent. Poll
  // info until `schemas_ready` flips, then refetch stages out of its error.
  createEffect(
    on(config.info, (i) => {
      if (!i) return;
      if (!i.schemas_ready) setTimeout(() => void config.refetchInfo(), 1000);
      else if (all.error) void refetch();
    }),
  );

  return {
    all,
    curId,
    setCurId,
    cur,
    byId: (id: string) => all()?.find((s) => s.id === id),
    values,
    setValue,
    resetForm,
    formKey,
    shown,
    panels,
    curPanel,
    /** Every stage under the open dock button, the current one included. */
    siblings: createMemo(() => panels().find(([p]) => p === curPanel())?.[1] ?? []),
    lastInPanel,
    setLastInPanel,
    settingFields,
    preprocessStage,
    missingModels,
  };
}

export type Stages = ReturnType<typeof createStages>;
