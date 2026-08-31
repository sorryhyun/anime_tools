import { createEffect, createMemo, createResource, on, onCleanup } from "solid-js";
import { createStore } from "solid-js/store";
import { api, toStatus } from "./api";
import { t } from "./i18n";
import { createJobFollower } from "./state";
import { REPLAY_FIELD, type CaptionKind, type Job, type Values } from "./types";
import type { Config } from "./config";
import type { Dataset } from "./dataset";
import type { Stages } from "./stages";

/** A finished **Run**: the report it wrote, the form and scope it ran at, and
    the images it wants to change. */
export interface RunResult {
  jobId: string;
  report: string;
  /** The form as it was when the run started; the result is dropped when the
      form moves on, or the proposals would not be this form's. */
  values: string;
  /** The image it was narrowed to, or null for the batch. Apply inherits it. */
  rel: string | null;
  /** Which caption the stage writes, so the diff lands on the right card. */
  kind: CaptionKind;
  rels: string[];
}

/** Apply jobs already undone, so a reload does not re-offer an Undo that would
    only skip every row. Local, like the undo decision itself. */
const undone = (): string[] => {
  try {
    return JSON.parse(localStorage.getItem("undone") ?? "[]") as string[];
  } catch {
    return [];
  }
};
const markUndone = (id: string) =>
  localStorage.setItem("undone", JSON.stringify([...undone(), id].slice(-100)));

/** Running the open stage: the **Run → diff → Apply → Undo** loop, and the one
 * job the dock is following.
 *
 * Run computes the proposals and writes only the report. Apply replays *that
 * report* (`--from_report`), so it loads no model and can write nothing the
 * caption panel's diff did not show — which is why it stays blocked until a Run
 * has produced one, and why the run is dropped the moment the form moves on.
 * Undo reads the apply report's before-text back. A stage with no `--apply`
 * (correct, the mask generators, groups) has no dry pass: its Run *is* the
 * write, and there is nothing to replay or undo.
 */
export function createRunner(deps: {
  config: Config;
  stages: Stages;
  dataset: Dataset;
  /** A start (and a failure to start) has to be visible: both open the dock. */
  openDock: () => void;
}) {
  const { config, stages, dataset } = deps;

  /** No log panel yet, so the newest line *is* the dock's status. */
  const run$ = createJobFollower({ done: (job) => onStageDone(job) });
  const { id: jobId, running: busy, status, setStatus } = run$;
  onCleanup(() => run$.close());

  /** Per stage, the last Run that finished cleanly -- and only for as long as
      the form still says what it said. */
  const [dry, setDry] = createStore<Record<string, RunResult | undefined>>({});
  /** Per stage, the last Apply, so Undo has a report to read the before-text
      out of. Seeded from the server's job list on load (the jobs outlive this
      page), so a reload does not silently take Undo away. */
  const [applied, setApplied] = createStore<Record<string, string | undefined>>({});

  // Re-adopt the newest finished Apply per stage: the server still holds the
  // job and its report, so Undo survives a reload instead of evaporating.
  // Session state is never overwritten -- this only fills gaps on load.
  createEffect(
    on(stages.all, (ss) => {
      if (!ss) return;
      void api
        .jobs()
        .then((jobs) => {
          const done = new Set(undone());
          const latest = new Map<string, Job>();
          for (const j of jobs) {
            const st = ss.find((s) => s.id === j.stage);
            if (!j.apply || j.state !== "done" || !j.report_path) continue;
            if (!st?.replay || done.has(j.id)) continue;
            const prev = latest.get(j.stage);
            if (!prev || j.started > prev.started) latest.set(j.stage, j);
          }
          for (const [stage, j] of latest) if (!applied[stage]) setApplied(stage, j.id);
        })
        .catch(() => {});
    }),
  );

  /** The open stage's last Run, and whether the form still says what it said.
      Both halves matter: Apply wants the fresh one, while its refusal message
      and the caption card both need to know a stale run existed at all. */
  const lastRun = createMemo(() => {
    const s = stages.cur();
    const run = s?.replay ? dry[s.id] : undefined;
    return { run, fresh: !!run && run.values === stages.formKey(stages.values()) };
  });
  /** The Run whose proposals are still on the table for the open stage, or
      null: no such run, the stage cannot replay, or the form moved on since. */
  const pending = createMemo(() => (lastRun().fresh ? lastRun().run! : null));
  const pendingSet = createMemo(() => new Set(pending()?.rels ?? []));
  /** The selected image's before/after, fetched one at a time: a batch's index
      is thousands of rels, and only the one on screen needs its text. */
  const [proposal] = createResource(
    () => {
      const p = pending();
      const rel = dataset.rel();
      return p && rel && pendingSet().has(rel) ? ([p.jobId, rel] as const) : false;
    },
    ([id, rel]) => api.proposal(id, rel),
  );
  /** The proposal to render, or undefined. `createResource` keeps its last
      value when its source goes falsy, so both guards are repeated here: an
      image with no proposal must not show the previous image's, and an Apply
      that consumed the run must not leave its diff on screen. */
  const shownProposal = createMemo(() => {
    const p = proposal();
    return p && pending() && p.rel === dataset.rel() && pendingSet().has(p.rel) ? p : undefined;
  });
  /** The caption kind whose card should explain a vanished diff: the last Run
      still exists, but the form moved on, so its proposals were dropped. */
  const droppedKind = createMemo(() => {
    const { run, fresh } = lastRun();
    const rel = dataset.rel();
    if (!run || fresh || !rel) return undefined;
    return run.rels.includes(rel) ? run.kind : undefined;
  });

  /** Why Apply is off, or "" when it is on. */
  const applyBlocked = createMemo(() => {
    const s = stages.cur();
    if (!s?.apply) return t().runner.noDryPass;
    if (!s.replay) return "";
    const { run, fresh } = lastRun();
    if (!run) return t().runner.runFirst;
    if (!fresh) return t().runner.formChanged;
    if (!run.rels.length) return t().runner.noChanges;
    return "";
  });
  const undoBlocked = createMemo(() => {
    const s = stages.cur();
    if (!s) return t().runner.nothingToUndo;
    return applied[s.id] ? "" : t().runner.nothingApplied;
  });

  /** Follow a job in the dock, opening it so the status line is visible. */
  function attach(id: string) {
    run$.follow(id, { text: t().runner.following(id), state: "running" });
    deps.openDock();
  }

  function onStageDone(job: Job) {
    setStatus({ text: t().runner.exit(job.exit_code), state: job.state });
    void config.refetchInfo();
    // A finished download job changed what the Settings rows should say.
    void config.refetchModels();
    if (job.state === "done") void finished(job);
    // A finished stage rewrote captions/masks under our feet -- but a Run of a
    // stage that has an --apply wrote nothing, so nothing to do.
    const dryRun = stages.byId(job.stage)?.apply && !job.apply;
    if (!dryRun) void reloadTouched(job);
  }

  /** What a job was started with, so a clean finish can be filed against the
      form as *this* page sent it -- a snapshot that fails to compare equal
      would silently stop offering the replay. */
  const sent = new Map<string, { form: string; rel: string | null }>();

  /** File a finished job: a Run becomes the proposals Apply will write and the
      diff the caption panel shows; an Apply becomes what Undo can put back. */
  async function finished(job: Job) {
    const s = stages.byId(job.stage);
    const meta = sent.get(job.id);
    sent.delete(job.id);
    if (!s) return;
    if (job.apply) {
      // The proposals are on disk now, so they stop being pending -- and the
      // report they were written from is exactly what Undo reads back.
      setDry(job.stage, undefined);
      if (job.report_path) setApplied(job.stage, job.id);
      return;
    }
    if (!s.replay || !s.apply || !job.report_path || !meta) return;
    try {
      const idx = await api.proposals(job.id);
      setDry(job.stage, {
        jobId: job.id,
        report: job.report_path,
        values: meta.form,
        rel: meta.rel,
        kind: idx.kind,
        rels: idx.rels,
      });
      setStatus((st) => ({ ...st, text: `${st.text} — ${t().runner.proposals(idx.total)}` }));
    } catch {
      // No readable report (a failed or cancelled run): nothing to propose.
      setDry(job.stage, undefined);
    }
  }

  /** Reload the dataset after a job that wrote. A replay's report names the
      images it touched (`written`), so only those rows are re-stat'd; anything
      else falls back to re-walking the tree. */
  async function reloadTouched(job: Job) {
    let touched: string[] | null = null;
    if (job.report_path) {
      try {
        const { report } = await api.report(job.id);
        const w = (report as { written?: unknown } | null)?.written;
        if (Array.isArray(w)) touched = w.map(String);
      } catch {
        // No report on disk (a failed or cancelled run): re-walk instead.
      }
    }
    if (!touched) return dataset.reloadAll();
    await dataset.reloadRels(touched);
    setStatus((st) => ({ ...st, text: `${st.text} — ${t().runner.changed(touched.length)}` }));
  }

  /** Start the open stage and follow it. The one place a start can fail — a
      rejected form, or the single job slot already taken — and it says so on
      the run bar, with the dock open so the message is visible. */
  async function startStage(v: Values, apply: boolean, rel: string | null) {
    const s = stages.cur();
    if (!s) return null;
    try {
      const job = await api.start(s.id, v, apply, rel);
      config.setSettings("values", (prev) => ({ ...(prev ?? {}), [s.id]: job.values }));
      attach(job.id);
      return job;
    } catch (e) {
      setStatus(toStatus(e));
      deps.openDock();
      return null;
    }
  }

  /** **Run** the current stage: compute the proposals and write the report,
      not the captions. `rel` narrows it to that one image; null runs the batch
      the Settings `path_pattern` names. */
  async function run(rel: string | null) {
    // `--from_report` is rebuilt per start and never carried over from the
    // saved form: a leftover path would quietly turn a Run into a replay.
    const { [REPLAY_FIELD]: _stale, ...v } = stages.values() ?? {};
    const job = await startStage(v, false, rel);
    if (job) sent.set(job.id, { form: stages.formKey(stages.values()), rel });
  }

  /** **Apply**: write what the Run proposed, at the scope the Run ran at. A
      stage that cannot replay (`audit_apply`) has no report to stand on, so
      Apply re-runs it for real. */
  async function apply() {
    const d = pending();
    const { [REPLAY_FIELD]: _stale, ...rest } = stages.values() ?? {};
    const v = d ? { ...rest, [REPLAY_FIELD]: d.report } : rest;
    await startStage(v, true, d?.rel ?? null);
  }

  /** **Undo**: put back the captions the last Apply wrote, from the very report
      it wrote them out of. A caption edited since is left alone and counted --
      the run bar says so rather than quietly restoring less than it claims. */
  async function undo() {
    const s = stages.cur();
    const id = s && applied[s.id];
    if (!s || !id) return;
    setStatus({ text: t().runner.undoing, state: "running" });
    try {
      const out = await api.undo(id);
      markUndone(id);
      setApplied(s.id, undefined);
      const skipped = Object.entries(out.skipped)
        .map(([k, n]) => `${k} ${n}`)
        .join(", ");
      setStatus({
        text:
          t().runner.undone(out.restored, out.removed) +
          (skipped ? ` ${t().runner.skipped(skipped)}` : ""),
        state: "done",
      });
      await dataset.reloadRels(out.written);
    } catch (e) {
      setStatus(toStatus(e));
    }
  }

  return {
    jobId,
    busy,
    status,
    attach,
    run,
    apply,
    undo,
    cancel: () => jobId() && api.cancel(jobId()!),
    pending,
    pendingSet,
    shownProposal,
    droppedKind,
    applyBlocked,
    undoBlocked,
  };
}

export type Runner = ReturnType<typeof createRunner>;
