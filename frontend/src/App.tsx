import {
  batch,
  createEffect,
  createMemo,
  createResource,
  createSignal,
  For,
  on,
  onCleanup,
  Show,
} from "solid-js";
import { createStore, reconcile } from "solid-js/store";
import { api, toStatus } from "./api";
import type {
  CaptionEntry,
  CaptionKind,
  DatasetRoots,
  Field,
  Info,
  ItemDetail,
  ModelCatalog,
  NodeKind,
  Settings,
  Job,
  Stage,
  Values,
} from "./types";
import { REPLAY_FIELD } from "./types";
import { DatasetTree, type Sel } from "./components/DatasetTree";
import { ItemView } from "./components/ItemView";
import { StagePanel } from "./components/StagePanel";
import { Dialog } from "./components/Dialog";
import { HelpToggle } from "./components/HelpToggle";
import { TagLens } from "./components/TagLens";
import { asFlag, createJobFollower, fromFlag, persisted } from "./state";
import { SettingsDialog, type SettingsTab } from "./components/SettingsDialog";

/** `/api/models/download` names its job `download:<ids>`; that prefix is the
    only thing that tells an *adopted* job (one already running when the page
    loaded) apart from a stage run, and the two go to different places. */
const DOWNLOAD_STAGE = "download:";

/** A finished **Run**: the report it wrote, the form and scope it ran at, and
    the images it wants to change. Apply replays exactly this, so what Apply
    writes can only be what the caption panel's diff already showed. */
interface RunResult {
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

/** `#rel|kind` — the dataset item is what a GUI link should point at now. */
function parseHash(): Sel | null {
  const raw = decodeURIComponent(location.hash.slice(1));
  if (!raw) return null;
  const cut = raw.lastIndexOf("|");
  if (cut < 0) return { rel: raw, kind: "image" };
  return { rel: raw.slice(0, cut), kind: raw.slice(cut + 1) as NodeKind };
}

export default function App() {
  const [info, { refetch: refetchInfo }] = createResource<Info>(api.info);
  const [stages, { refetch: refetchStages }] = createResource<Stage[]>(api.stages);
  const [settings, setSettings] = createStore<Settings>({});
  const [roots, { refetch: refetchRoots }] = createResource<DatasetRoots>(api.datasetRoots);
  const [models, { refetch: refetchModels }] = createResource<ModelCatalog>(api.models);

  // ---- dataset ----
  const [query, setQuery] = createSignal("");
  const [debouncedQuery, setDebouncedQuery] = createSignal("");
  const [reload, setReload] = createSignal(0);
  const [list, { refetch: refetchList, mutate: mutateList }] = createResource(
    () => [debouncedQuery(), roots(), reload()] as const,
    ([q]) => api.dataset({ q }),
  );
  const [sel, setSel] = createSignal<Sel | null>(parseHash());
  const [item, { mutate: mutateItem, refetch: refetchItem }] = createResource<
    ItemDetail | undefined,
    string
  >(() => sel()?.rel, api.item);

  let queryTimer: ReturnType<typeof setTimeout> | undefined;
  createEffect(
    on(query, (q) => {
      clearTimeout(queryTimer);
      queryTimer = setTimeout(() => setDebouncedQuery(q), 200);
    }),
  );
  createEffect(
    on(sel, (s) => {
      location.hash = s ? encodeURIComponent(s.rel) + (s.kind === "image" ? "" : `|${s.kind}`) : "";
    }),
  );
  const onHash = () => {
    const h = parseHash();
    if (h?.rel !== sel()?.rel || h?.kind !== sel()?.kind) setSel(h);
  };
  window.addEventListener("hashchange", onHash);

  /** ↑/↓ (and j/k) walk the images in listing order, outside text fields. */
  const onKey = (e: KeyboardEvent) => {
    const t = e.target as HTMLElement | null;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    const step =
      e.key === "ArrowDown" || e.key === "j" ? 1 : e.key === "ArrowUp" || e.key === "k" ? -1 : 0;
    if (!step) return;
    const items = list()?.items ?? [];
    if (!items.length) return;
    const i = items.findIndex((x) => x.rel === sel()?.rel);
    const at =
      i < 0 ? (step > 0 ? 0 : items.length - 1) : Math.min(items.length - 1, Math.max(0, i + step));
    e.preventDefault();
    // Keep the caption kind, so arrowing down a column compares the same file.
    setSel({ rel: items[at].rel, kind: sel()?.kind ?? "image" });
  };
  window.addEventListener("keydown", onKey);

  /** Fold a just-saved caption back into the loaded item and its tree row,
      so neither has to be re-fetched. */
  const onSaved = (entry: CaptionEntry) => {
    mutateItem((prev) =>
      prev
        ? { ...prev, captions: prev.captions.map((c) => (c.kind === entry.kind ? entry : c)) }
        : prev,
    );
    const rel = sel()?.rel;
    mutateList((prev) =>
      prev
        ? {
            ...prev,
            items: prev.items.map((x) => (x.rel === rel ? { ...x, [entry.kind]: true } : x)),
          }
        : prev,
    );
  };

  // ---- stages ----
  const [curId, setCurId] = persisted("stage", "", String);
  const cur = createMemo(() => stages()?.find((s) => s.id === curId()));
  const [forms, setForms] = createStore<Record<string, Values>>({});
  const values = () => forms[curId()] ?? {};
  const setValue = (dest: string, v: unknown) =>
    setForms(curId(), (prev) => ({ ...(prev ?? {}), [dest]: v }));
  const resetForm = () => setForms(curId(), reconcile({}));

  /** The stages the dock offers: everything but the hidden ones, which are not
      run by hand (`resize` runs itself, in front of the stages that need it). */
  const shown = createMemo(() => (stages() ?? []).filter((s) => !s.hidden));
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

  /** The dataset sidebar. The stage forms and the caption panel both want the
      width, and which one wins changes by the minute -- so it folds away and
      the header's toggle brings it back. */
  const [sidebar, setSidebar] = persisted("sidebar", true, asFlag, fromFlag);
  const [dockOpen, setDockOpen] = persisted("dock", true, asFlag, fromFlag);
  // Not `persisted`: this moves on every pointermove of a drag, so it is saved
  // once on pointerup (see `grip`) rather than at frame rate.
  const [dockH, setDockH] = createSignal(Number(localStorage.getItem("dockh")) || 320);
  /** One global "show the prose" preference, off by default: the stage doc and
      the Settings blurbs are behind the (?) buttons until it is on. */
  const [help, setHelp] = persisted("help", false, asFlag, fromFlag);
  /** The stage runner: the dock's job, its "running" badge and its status line.
      No log panel yet, so the newest line *is* the status. */
  const run$ = createJobFollower({
    done: (job) => onStageDone(job),
  });
  const jobId = run$.id;
  const busy = run$.running;
  const status = run$.status;
  const setStatus = run$.setStatus;
  /** A weights download is a job like any other to the server -- it takes the
      same single slot -- but it belongs to the Settings dialog, not the dock:
      its own id/status keep the modal open over it and keep the run bar from
      saying "running" for a fetch no stage form started. */
  const dl$ = createJobFollower({
    // The downloader prints a blank line between models (and hub progress bars
    // arrive as \r-terminated lines): the newest *non-empty* one is the status.
    line: (line) => (line.trim() ? { text: line.trim(), state: "running" } : null),
    done: (job) => {
      dl$.setStatus({
        text: job.state === "done" ? "download finished" : `exit ${job.exit_code}`,
        state: job.state,
      });
      refetchInfo();
      // The rows say installed/missing; a finished pull just changed that.
      refetchModels();
    },
  });
  const dlBusy = dl$.running;
  /** The ids this download asked for; `[]` = every missing model. Only used to
      mark the rows in flight. */
  const [dlIds, setDlIds] = createSignal<string[]>([]);
  const [confirmOpen, setConfirmOpen] = createSignal(false);
  const [settingsOpen, setSettingsOpen] = createSignal(false);
  /** Which Settings panel to open on: a hint about weights or the token
      opens the one that fixes it, the ⚙ button opens the first. */
  const [settingsTab, setSettingsTab] = createSignal<SettingsTab>("general");
  const openSettings = (t: SettingsTab = "general") => {
    setSettingsTab(t);
    setSettingsOpen(true);
    refetchModels();
  };
  /** Per stage, the last Run that finished cleanly. Apply replays its report
      (`--from_report`) instead of re-running the tagger/SAM3 pass that produced
      it, so it writes exactly the text the diff showed -- and only while the
      form still says what it said. */
  const [dry, setDry] = createStore<Record<string, RunResult | undefined>>({});
  /** Per stage, the last Apply, so Undo has a report to read the before-text
      out of. Seeded from the server's job list on load (the jobs outlive this
      page), so a reload does not silently take Undo away. */
  const [applied, setApplied] = createStore<Record<string, string | undefined>>({});
  /** Apply jobs already undone, so a reload does not re-offer an Undo that
      would only skip every row. Local, like the undo decision itself. */
  const undone = (): string[] => {
    try {
      return JSON.parse(localStorage.getItem("undone") ?? "[]") as string[];
    } catch {
      return [];
    }
  };
  const markUndone = (id: string) =>
    localStorage.setItem("undone", JSON.stringify([...undone(), id].slice(-100)));
  /** `path_pattern` / `tagger_dir` / `checkpoint` / `report_root`: one value
      each, from Settings, for every stage that takes them. The server fills the
      flags; this is only the copy the run bar and the Settings dialog show. */
  const stageDefaults = () => settings.stage_defaults ?? {};
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
    (stages() ?? []).find((s) => s.hidden && s.id === "resize"),
  );
  /** The form as it matters to a replay: everything the stage actually reads,
      minus the managed field itself. */
  const formKey = (v: Values) => {
    const { [REPLAY_FIELD]: _managed, ...rest } = v ?? {};
    return JSON.stringify(rest);
  };
  /** The open stage's last Run, and whether the form still says what it said.
      Both halves matter and they were derived three times with the polarity
      flipped: Apply wants the fresh one, its refusal message wants "there is a
      run but it is stale", and the caption card wants to explain the diff that
      staleness took away. */
  const lastRun = createMemo(() => {
    const s = cur();
    const run = s?.replay ? dry[s.id] : undefined;
    return { run, fresh: !!run && run.values === formKey(values()) };
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
      const rel = sel()?.rel;
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
    return p && pending() && p.rel === sel()?.rel && pendingSet().has(p.rel) ? p : undefined;
  });

  /** Why Apply is off, or "" when it is on. A replay-capable stage must have a
      Run to apply: that is what makes Apply a plain write of the shown diff. */
  const applyBlocked = createMemo(() => {
    const s = cur();
    if (!s?.apply) return "this stage has no dry pass — Run writes";
    if (!s.replay) return "";
    const { run, fresh } = lastRun();
    if (!run) return "Run first — Apply writes what the run proposed";
    if (!fresh) return "the form changed since the run — Run again";
    if (!run.rels.length) return "that run proposed no changes — nothing to write";
    return "";
  });
  const undoBlocked = createMemo(() => {
    const s = cur();
    if (!s) return "nothing to undo";
    return applied[s.id] ? "" : "nothing applied yet — Undo puts back an Apply";
  });
  /** Catalog rows the open stage needs that are not downloaded yet — surfaced
      in the stage bar, so a Run does not stall on a surprise multi-GB fetch. */
  const missingModels = createMemo(() =>
    (models()?.models ?? []).filter((m) => !m.installed && m.stages.includes(curId())),
  );
  /** The caption kind whose card should explain a vanished diff: the last Run
      still exists, but the form moved on, so its proposals were dropped. */
  const droppedKind = createMemo(() => {
    const { run, fresh } = lastRun();
    const rel = sel()?.rel;
    if (!run || fresh || !rel) return undefined;
    return run.rels.includes(rel) ? run.kind : undefined;
  });
  api.settings().then((s) => {
    setSettings(s);
    setForms(reconcile(structuredClone(s.values ?? {})));
  });

  createEffect(
    on(stages, (ss) => {
      // A hidden stage is never the landing stage: it has no dock button to
      // show it, so selecting one would open the dock on nothing.
      if (ss && !shown().some((s) => s.id === curId()))
        setCurId(shown().find((s) => s.available)?.id ?? "");
    }),
  );
  // Re-adopt the newest finished Apply per stage: the server still holds the
  // job and its report, so Undo survives a reload instead of evaporating.
  // Session state is never overwritten -- this only fills gaps on load.
  createEffect(
    on(stages, (ss) => {
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

  // A class on <body>, not a <Show>: the tree keeps its expanded folders and
  // its scroll position while it is folded away.
  createEffect(on(sidebar, (v) => document.body.classList.toggle("nosidebar", !v)));
  createEffect(
    on(info, (i) => {
      const id = i?.running;
      // `running`, not the ids: an id is kept after its job ends, and a
      // second externally-started job deserves adopting too.
      if (!id || busy() || dlBusy()) return;
      // Only the id comes back on /api/info, so ask what it is before deciding
      // which follower gets it -- a download must not land in the dock.
      void api
        .job(id)
        .then((j) => (j.stage.startsWith(DOWNLOAD_STAGE) ? adoptDownload(j) : attach(id)))
        .catch(() => attach(id));
    }),
  );
  // Cold start: /api/stages answers 503 while the server's background schema
  // dump is still running, and a resource error is otherwise permanent. Poll
  // info until `schemas_ready` flips, then refetch stages out of its error.
  createEffect(
    on(info, (i) => {
      if (!i) return;
      if (!i.schemas_ready) setTimeout(() => void refetchInfo(), 1000);
      else if (stages.error) void refetchStages();
    }),
  );
  onCleanup(() => {
    run$.close();
    dl$.close();
    window.removeEventListener("hashchange", onHash);
    window.removeEventListener("keydown", onKey);
  });

  /** Drag the dock's top edge. The dataset view and the stage form both want
      the vertical space, and which one wins changes by the minute. */
  function grip(e: PointerEvent) {
    e.preventDefault();
    const y0 = e.clientY;
    const h0 = dockH();
    const move = (ev: PointerEvent) =>
      setDockH(Math.max(120, Math.min(window.innerHeight - 220, h0 + (y0 - ev.clientY))));
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      localStorage.setItem("dockh", String(dockH()));
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  /** The dock's button strip is the stage picker; the open stage's button is
      also its close button. */
  function pickStage(id: string) {
    if (dockOpen() && curId() === id) return setDockOpen(false);
    batch(() => {
      setCurId(id);
      setDockOpen(true);
    });
  }

  /** A dock button: open the panel on the stage it was last left on, or close
      the dock when it is already the open one. */
  function pickPanel(panel: string, ss: Stage[]) {
    if (dockOpen() && curPanel() === panel) return setDockOpen(false);
    const want = curPanel() === panel ? curId() : lastInPanel[panel];
    const s = ss.find((x) => x.id === want) ?? ss.find((x) => x.available) ?? ss[0];
    if (s) pickStage(s.id);
  }

  function attach(id: string) {
    run$.follow(id, { text: `running ${id}`, state: "running" });
    setDockOpen(true);
  }

  /** A stage job that reached the end of its stream. */
  function onStageDone(job: Job) {
    setStatus({ text: `exit ${job.exit_code}`, state: job.state });
    refetchInfo();
    // A finished download job changed what the Settings rows should say.
    refetchModels();
    if (job.state === "done") void finished(job);
    // A finished stage rewrote captions/masks under our feet -- but a Run of a
    // stage that has an --apply wrote nothing, so nothing to do.
    const dryRun = stages()?.find((s) => s.id === job.stage)?.apply && !job.apply;
    if (!dryRun) void reloadTouched(job);
  }

  /** What a job was started with, so a clean finish can be filed against the
      form as *this* page sent it -- a snapshot that fails to compare equal
      would silently stop offering the replay. */
  const sent = new Map<string, { form: string; rel: string | null }>();

  /** File a finished job: a Run becomes the proposals Apply will write and the
      diff the caption panel shows; an Apply becomes what Undo can put back. */
  async function finished(job: Job) {
    const s = stages()?.find((x) => x.id === job.stage);
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
      setStatus((st) => ({
        ...st,
        text: `${st.text} — ${idx.total} proposal${idx.total === 1 ? "" : "s"}`,
      }));
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
    if (!touched) {
      refetchList();
      refetchItem();
      return;
    }
    await reloadRels(touched);
    setStatus((st) => ({ ...st, text: `${st.text} — ${touched.length} file(s) changed` }));
  }

  /** Re-stat named sidebar rows in place, and the open item if it is one. */
  async function reloadRels(rels: string[]) {
    if (!rels.length) return;
    const { items } = await api.items(rels);
    const by = new Map(items.map((i) => [i.rel, i]));
    mutateList((prev) =>
      prev ? { ...prev, items: prev.items.map((x) => by.get(x.rel) ?? x) } : prev,
    );
    const rel = sel()?.rel;
    if (rel && by.has(rel)) refetchItem();
  }

  /** **Run** the current stage: compute the proposals and write the report,
      not the captions. `rel` narrows it to that one image; null runs the batch
      the Settings `path_pattern` names. A stage with no `--apply` has no dry
      pass, so for those this *is* the write. */
  async function run(rel: string | null) {
    // `--from_report` is rebuilt per start and never carried over from the
    // saved form: a leftover path would quietly turn a Run into a replay.
    const { [REPLAY_FIELD]: _stale, ...v } = values() ?? {};
    const job = await startStage(v, false, rel);
    if (job) sent.set(job.id, { form: formKey(values()), rel });
  }

  /** Start the open stage and follow it. The one place a start can fail — a
      rejected form, or the single job slot already taken — and it says so on
      the run bar, with the dock open so the message is visible. */
  async function startStage(v: Values, apply: boolean, rel: string | null) {
    const s = cur();
    if (!s) return null;
    try {
      const job = await api.start(s.id, v, apply, rel);
      setSettings("values", (prev) => ({ ...(prev ?? {}), [s.id]: job.values }));
      attach(job.id);
      return job;
    } catch (e) {
      setStatus(toStatus(e));
      setDockOpen(true);
      return null;
    }
  }

  /** **Apply**: write what the Run proposed, at the scope the Run ran at.
      Replaying its report is the whole point -- no model loads, and nothing can
      be written that the diff did not show. A stage that cannot replay
      (`audit_apply`) has no report to stand on, so Apply re-runs it for real. */
  async function apply() {
    const d = pending();
    const { [REPLAY_FIELD]: _stale, ...rest } = values() ?? {};
    const v = d ? { ...rest, [REPLAY_FIELD]: d.report } : rest;
    await startStage(v, true, d?.rel ?? null);
  }

  /** **Undo**: put back the captions the last Apply wrote, from the very report
      it wrote them out of. A caption edited since is left alone and counted --
      the run bar says so rather than quietly restoring less than it claims. */
  async function undo() {
    const s = cur();
    const jobId = s && applied[s.id];
    if (!s || !jobId) return;
    setStatus({ text: "undoing…", state: "running" });
    try {
      const out = await api.undo(jobId);
      markUndone(jobId);
      setApplied(s.id, undefined);
      const skipped = Object.entries(out.skipped)
        .map(([k, n]) => `${k} ${n}`)
        .join(", ");
      setStatus({
        text:
          `undone: ${out.restored} restored, ${out.removed} removed` +
          (skipped ? ` — skipped ${skipped}` : ""),
        state: "done",
      });
      await reloadRels(out.written);
    } catch (e) {
      setStatus(toStatus(e));
    }
  }

  /** Follow a weights job inside the Settings dialog. Deliberately none of what
      `attach` does: no dock, no dock status line -- the modal stays open and
      reports the pull itself. */
  function attachDownload(id: string, ids: string[]) {
    batch(() => {
      setDlIds(ids);
      dl$.follow(id, { text: "starting…", state: "running" });
    });
  }

  /** Re-attach to a download that was already running when the page loaded.
      `download:<ids>` is the only record of what it asked for. */
  function adoptDownload(job: Job) {
    const rest = job.stage.slice(DOWNLOAD_STAGE.length);
    attachDownload(job.id, rest === "missing" ? [] : rest.split(",").filter(Boolean));
    setSettingsTab("models");
    setSettingsOpen(true);
  }

  /** Weights download: a job like any other on the server, but it reports into
      the Settings dialog that started it, which stays open over it. */
  async function download(ids: string[]) {
    try {
      const job = await api.downloadModels(ids);
      attachDownload(job.id, ids);
    } catch (e) {
      // A stage holding the one job slot lands here (409) -- say so in the
      // dialog, where the button that failed is.
      dl$.setStatus(toStatus(e));
    }
  }

  return (
    <>
      <header>
        <button
          class="link fold"
          title={sidebar() ? "Hide the dataset sidebar" : "Show the dataset sidebar"}
          onClick={() => setSidebar(!sidebar())}
        >
          ☰
        </button>
        <b>anime_tools</b>
        <span class="dim mono" title={info()?.home}>
          {info()?.home}
        </span>
        <Show when={list()}>
          {(l) => (
            <span class="dim">
              {l().total} image{l().total === 1 ? "" : "s"} in {l().root}
            </span>
          )}
        </Show>
        <span class="sp" />
        <Show
          when={info()?.hf_token}
          fallback={
            <button
              class="link warn"
              title="The tagger backbone and SAM3 are gated on the Hub — set a token in Settings"
              onClick={() => openSettings("models")}
            >
              ⚠ no HF token
            </button>
          }
        >
          <span class="dim">HF token ✓</span>
        </Show>
        <Show when={dlBusy()}>
          {/* The dock never shows a download, so this is the only sign of one
              while the dialog it belongs to is closed. */}
          <span class="badge running">downloading</span>
        </Show>
        <button onClick={() => openSettings()}>⚙ Settings</button>
      </header>

      <DatasetTree
        list={list()}
        loading={list.loading}
        error={list.error ? String(list.error) : undefined}
        resetKey={`${debouncedQuery()}|${reload()}`}
        sel={sel()}
        onSelect={setSel}
        query={query()}
        onQuery={setQuery}
        onRefresh={() => setReload((n) => n + 1)}
        pending={pendingSet()}
        onCollapse={() => setSidebar(false)}
      />

      <ItemView
        item={item()}
        loading={item.loading}
        error={item.error ? String(item.error) : undefined}
        kind={sel()?.kind ?? "image"}
        proposal={shownProposal()}
        proposalStage={cur()?.title}
        droppedKind={droppedKind()}
        onSaved={onSaved}
      />

      <div classList={{ dock: true, closed: !dockOpen() }} style={{ "--dock-h": `${dockH()}px` }}>
        <Show when={dockOpen()}>
          <div class="dockgrip" onPointerDown={grip} title="Drag to resize" />
        </Show>
        <div class="tabs stagetabs">
          <For each={panels()}>
            {([p, ss]) => (
              <a
                classList={{
                  sel: dockOpen() && curPanel() === p,
                  na: !ss.some((s) => s.available),
                }}
                title={ss.map((s) => s.title).join(" · ")}
                onClick={() => pickPanel(p, ss)}
              >
                {p}
              </a>
            )}
          </For>
          <span class="sp" />
          <Show when={busy()}>
            <span class="badge running">running</span>
          </Show>
          <button class="link" onClick={() => setDockOpen(!dockOpen())}>
            {dockOpen() ? "▾" : "▴"}
          </button>
        </div>

        <Show when={dockOpen()}>
          <div class="dockbody">
            <StagePanel
              cur={cur()}
              siblings={panels().find(([p]) => p === curPanel())?.[1] ?? []}
              onPick={(id) => {
                setLastInPanel(curPanel(), id);
                setCurId(id);
              }}
              error={stages.error}
              values={values()}
              setValue={setValue}
              reset={resetForm}
              busy={busy()}
              locked={dlBusy()}
              status={status()}
              rel={sel()?.rel ?? null}
              onRun={run}
              onApply={() => setConfirmOpen(true)}
              applyBlocked={applyBlocked()}
              onUndo={undo}
              undoBlocked={undoBlocked()}
              onCancel={() => jobId() && api.cancel(jobId()!)}
              missingModels={missingModels().map((m) => m.title)}
              onSettings={() => openSettings("models")}
              help={help()}
              onHelp={() => setHelp(!help())}
            />
          </div>
        </Show>
      </div>

      <Dialog
        open={confirmOpen()}
        onClose={(v) => {
          setConfirmOpen(false);
          if (v === "ok") apply();
        }}
      >
        <h3>Apply for real?</h3>
        <p style="max-width:520px">
          {cur()?.title} will write to{" "}
          <Show
            when={pending()?.rel}
            fallback={
              <>
                every image <code>{stageDefaults().path_pattern || "*"}</code> names
              </>
            }
          >
            {(rel) => <code>{rel()}</code>}
          </Show>
          <Show when={pending()}>
            {(d) => (
              <>
                {" "}
                — <b>{d().rels.length}</b> caption{d().rels.length === 1 ? "" : "s"} change
              </>
            )}
          </Show>
          .
        </p>
        {/* Apply is a replay by construction: the run already computed and wrote
            down every proposal, so this pass loads no model and can only write
            text the diff showed. A stage with no report to stand on says so. */}
        <Show
          when={pending()}
          fallback={
            <p class="dim" style="max-width:520px">
              This stage keeps no replayable report — Apply runs it again with <code>--apply</code>,
              so it writes what <em>this</em> pass computes.
            </p>
          }
        >
          {(d) => (
            <p class="dim" style="max-width:520px">
              Writing the run's proposals (<code>{d().report}</code>) — no model loads. A caption
              edited since that run is skipped, not overwritten.
            </p>
          )}
        </Show>
        <p class="dim">
          Caption stages write under <code>post_image_dataset/resized/</code> (autotag{" "}
          <code>missing</code> creates masters under <code>image_dataset/</code>). Any caption
          change must be followed by the trainer's TE re-encode (<code>make preprocess-te</code>).{" "}
          <b>Undo</b> puts these captions back, from the same report.
        </p>
        <div class="dlg-actions">
          <button value="cancel">Cancel</button>
          <button value="ok" class="danger">
            Apply
          </button>
        </div>
      </Dialog>

      {/* One card for every tag chip in the app; it floats, so it is mounted
          at the root rather than inside the caption panel. */}
      <TagLens onInstall={() => openSettings("models")} />

      <SettingsDialog
        open={settingsOpen()}
        initialTab={settingsTab()}
        info={info()}
        roots={roots()}
        fields={settingFields()}
        defaults={stageDefaults()}
        preprocess={preprocessStage()}
        preprocessValues={settings.preprocess ?? {}}
        models={models()}
        busy={busy() || dlBusy()}
        downloading={dlBusy()}
        downloadIds={dlIds()}
        progress={dl$.status()}
        help={help()}
        onHelp={() => setHelp(!help())}
        onDownload={download}
        onCancelDownload={() => dl$.id() && api.cancel(dl$.id()!)}
        onClose={async (out) => {
          setSettingsOpen(false);
          if (!out) return;
          if (out.token) {
            await api.putSettings({ hf_token: out.token });
            refetchInfo();
          }
          if (out.roots) {
            await api.putDatasetRoots(out.roots);
            refetchRoots();
          }
          if (out.defaults) setSettings(await api.putSettings({ stage_defaults: out.defaults }));
          if (out.preprocess) setSettings(await api.putSettings({ preprocess: out.preprocess }));
        }}
      />
    </>
  );
}
