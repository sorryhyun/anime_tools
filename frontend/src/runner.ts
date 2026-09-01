import { createEffect, createMemo, createResource, on, onCleanup } from "solid-js";
import { createStore } from "solid-js/store";
import { api, toStatus } from "./api";
import { t } from "./i18n";
import { createJobFollower } from "./state";
import { REPLAY_FIELD, type CaptionKind, type Job, type Values } from "./types";
import type { Config } from "./config";
import type { Dataset } from "./dataset";
import type { Stages } from "./stages";

/** A finished **Run**: the report it wrote and the images it changed. */
export interface RunResult {
  jobId: string;
  report: string;
  /** The image it was narrowed to, or null for the batch. */
  rel: string | null;
  /** Which caption the stage writes, so the diff lands on the right card. */
  kind: CaptionKind;
  rels: string[];
}

/** Runs already undone, so a reload does not re-offer an Undo that would only
    skip every row. Local, like the undo decision itself. */
const undone = (): string[] => {
  try {
    return JSON.parse(localStorage.getItem("undone") ?? "[]") as string[];
  } catch {
    return [];
  }
};
const markUndone = (id: string) =>
  localStorage.setItem("undone", JSON.stringify([...undone(), id].slice(-100)));

/** Running the open stage: the **Run → versions → Undo** loop, and the one job
 * the dock is following.
 *
 * A Run writes. There is no Apply gate in front of it, because the caption
 * ladder is what a gate was standing in for: the text a run replaces is pushed
 * onto its rung's history and shows up as a badge (`revised@2`) beside the
 * caption, so "what did that just do to my caption?" is answered *after* the
 * write and by the panel, not by a diff you had to read before agreeing to one.
 *
 * The report a run writes is still read back — as the diff of what it *did*,
 * keyed by image, plus the dots on the sidebar rows it touched — and Undo is
 * that report replayed with the two texts swapped (`gui/proposals.undo`), which
 * is itself a write and so leaves its own version behind.
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

  /** Per stage, the last Run that finished cleanly: the diff it wrote and the
      rows the sidebar should dot. */
  const [dry, setDry] = createStore<Record<string, RunResult | undefined>>({});
  /** Per stage, the last run that wrote, so Undo has a report to read the
      before-text out of. Seeded from the server's job list on load (the jobs
      outlive this page), so a reload does not silently take Undo away. */
  const [applied, setApplied] = createStore<Record<string, string | undefined>>({});

  // Re-adopt the newest finished run per stage: the server still holds the
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

  /** The open stage's last Run — the diff on screen is what it wrote, so it
      stands until another run replaces it. */
  const pending = createMemo(() => {
    const s = stages.cur();
    return (s?.replay ? dry[s.id] : undefined) ?? null;
  });
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
  /** The diff to render, or undefined. `createResource` keeps its last value
      when its source goes falsy, so both guards are repeated here: an image the
      run did not touch must not show the previous image's diff. */
  const shownProposal = createMemo(() => {
    const p = proposal();
    return p && pending() && p.rel === dataset.rel() && pendingSet().has(p.rel) ? p : undefined;
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
    // Every stage run writes, so every one of them rewrote captions, masks or
    // pixels under our feet.
    void reloadTouched(job);
  }

  /** What a job was started with, so a clean finish can be filed against the
      scope *this* page sent it at. */
  const sent = new Map<string, { rel: string | null }>();

  /** File a finished job: its report is both the diff the caption panel shows
      and what Undo puts back, so a run that wrote one is filed under both. */
  async function finished(job: Job) {
    const s = stages.byId(job.stage);
    const meta = sent.get(job.id);
    sent.delete(job.id);
    if (!s || !job.report_path) return;
    if (job.apply) setApplied(job.stage, job.id);
    if (!s.replay || !meta) return;
    try {
      const idx = await api.proposals(job.id);
      setDry(job.stage, {
        jobId: job.id,
        report: job.report_path,
        rel: meta.rel,
        kind: idx.kind,
        rels: idx.rels,
      });
      setStatus((st) => ({ ...st, text: `${st.text} — ${t().runner.changed(idx.total)}` }));
    } catch {
      // No readable report (a failed or cancelled run): no diff to show.
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
  async function startStage(v: Values, rel: string | null) {
    const s = stages.cur();
    if (!s) return null;
    try {
      // `--apply` whenever the stage has the flag: a Run writes, and there is
      // no dry pass to opt into any more. A stage without one (correct, the
      // mask generators, groups) never had the distinction, and is sent
      // `false` rather than a flag it does not take -- so `job.apply` stays
      // exactly "this run was told to write", which is what the undo route and
      // the Undo button both read it as.
      const job = await api.start(s.id, v, !!s.apply, rel);
      config.setSettings("values", (prev) => ({ ...(prev ?? {}), [s.id]: job.values }));
      attach(job.id);
      return job;
    } catch (e) {
      setStatus(toStatus(e));
      deps.openDock();
      return null;
    }
  }

  /** **Run** the current stage, for real: `--apply`, so the captions are
      written and the report it leaves is a record of what changed rather than
      an offer. `rel` narrows it to that one image; null runs the batch the
      Settings `path_pattern` names.

      What stands between a mistaken run and a lost caption is the ladder, not a
      confirmation: the replaced text is a version now, and Undo replays this
      very report backwards. */
  async function run(rel: string | null) {
    // `--from_report` is rebuilt per start and never carried over from the
    // saved form: a leftover path would quietly turn a Run into a replay of
    // some older report.
    const { [REPLAY_FIELD]: _stale, ...v } = stages.values() ?? {};
    const job = await startStage(v, rel);
    if (job) sent.set(job.id, { rel });
  }

  /** **Undo**: put back the captions the last Run wrote, from the very report
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
    undo,
    cancel: () => jobId() && api.cancel(jobId()!),
    pending,
    pendingSet,
    shownProposal,
    undoBlocked,
  };
}

export type Runner = ReturnType<typeof createRunner>;
