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
import { api, followLog } from "./api";
import type {
  CaptionEntry,
  DatasetRoots,
  Field,
  Info,
  ItemDetail,
  ModelAsset,
  ModelCatalog,
  NodeKind,
  RootName,
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

const ROOT_NAMES: RootName[] = ["src", "dst", "masks"];
const ROOT_HELP: Record<RootName, string> = {
  src: "source images + hand-written master captions",
  dst: "resized images + derived captions + .variants.txt",
  masks: "{stem}_mask.png, mirroring the source subdirs",
};

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
  const [stages] = createResource<Stage[]>(api.stages);
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
    const step = e.key === "ArrowDown" || e.key === "j" ? 1 : e.key === "ArrowUp" || e.key === "k" ? -1 : 0;
    if (!step) return;
    const items = list()?.items ?? [];
    if (!items.length) return;
    const i = items.findIndex((x) => x.rel === sel()?.rel);
    const at =
      i < 0
        ? step > 0
          ? 0
          : items.length - 1
        : Math.min(items.length - 1, Math.max(0, i + step));
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
            items: prev.items.map((x) =>
              x.rel === rel ? { ...x, [entry.kind]: true } : x,
            ),
          }
        : prev,
    );
  };

  // ---- stages ----
  const [curId, setCurId] = createSignal(localStorage.getItem("stage") ?? "");
  const cur = createMemo(() => stages()?.find((s) => s.id === curId()));
  const [forms, setForms] = createStore<Record<string, Values>>({});
  const values = () => forms[curId()] ?? {};
  const setValue = (dest: string, v: unknown) =>
    setForms(curId(), (prev) => ({ ...(prev ?? {}), [dest]: v }));
  const resetForm = () => setForms(curId(), reconcile({}));

  /** Stages, in registry order, bucketed by their argparse group. */
  const groups = createMemo(() => {
    const m = new Map<string, Stage[]>();
    for (const s of stages() ?? []) m.set(s.group, [...(m.get(s.group) ?? []), s]);
    return [...m];
  });

  const [dockOpen, setDockOpen] = createSignal(localStorage.getItem("dock") !== "0");
  const [dockH, setDockH] = createSignal(Number(localStorage.getItem("dockh")) || 320);
  /** One global "show the prose" preference, off by default: the stage doc and
      the Settings blurbs are behind the (?) buttons until it is on. */
  const [help, setHelp] = createSignal(localStorage.getItem("help") === "1");
  const [jobId, setJobId] = createSignal<string | null>(null);
  const [busy, setBusy] = createSignal(false);
  const [status, setStatus] = createSignal<{ text: string; state?: string }>({ text: "" });
  const [confirmOpen, setConfirmOpen] = createSignal(false);
  /** What the pending Apply is aimed at: one image's `rel`, or null = the
      batch the Settings `path_pattern` names. Rebuilt per click so the dialog
      and the run it starts can never disagree about the scope. */
  const [applyRel, setApplyRel] = createSignal<string | null>(null);
  const [settingsOpen, setSettingsOpen] = createSignal(false);
  /** Per stage, the last dry run that finished cleanly: its report, and the
      form it ran on. Apply replays that report (`--from_report`) instead of
      re-running the tagger/SAM3 pass that produced it -- but only while the
      form still says what it said, or the proposals would not be this form's. */
  const [dry, setDry] = createStore<Record<string, { report: string; values: string }>>({});
  /** `path_pattern` / `tagger_dir`: one value each, from Settings, for every
      stage that takes them. The server fills the flags; this is only the copy
      the run bar and the Settings dialog show. */
  const stageDefaults = () => settings.stage_defaults ?? {};
  /** One Field descriptor per Settings-bound dest, first stage that has it
      wins: the input's label, help and placeholder all come from the real
      argparse action, so Settings never re-describes a flag. */
  const settingFields = createMemo(() => {
    const m = new Map<string, Field>();
    for (const st of stages() ?? [])
      for (const f of st.fields) if (f.setting && !m.has(f.setting)) m.set(f.setting, f);
    return [...m.values()];
  });
  const [reuse, setReuse] = createSignal(true);
  /** The form as it matters to a replay: everything the stage actually reads,
      minus the managed field itself. */
  const formKey = (v: Values) => {
    const { [REPLAY_FIELD]: _managed, ...rest } = v ?? {};
    return JSON.stringify(rest);
  };
  /** The dry-run report Apply would replay, or null: no such run, the stage
      cannot replay, or the form moved on since. */
  const replayable = createMemo(() => {
    const s = cur();
    const d = s && dry[s.id];
    if (!s?.replay || !d || d.values !== formKey(values())) return null;
    return d.report;
  });
  let es: EventSource | null = null;

  api.settings().then((s) => {
    setSettings(s);
    setForms(reconcile(structuredClone(s.values ?? {})));
  });

  createEffect(
    on(stages, (ss) => {
      if (ss && !ss.some((s) => s.id === curId())) setCurId(ss.find((s) => s.available)?.id ?? "");
    }),
  );
  createEffect(on(curId, (id) => id && localStorage.setItem("stage", id)));
  createEffect(on(dockOpen, (o) => localStorage.setItem("dock", o ? "1" : "0")));
  createEffect(on(help, (h) => localStorage.setItem("help", h ? "1" : "0")));
  createEffect(on(info, (i) => { if (i?.running && !jobId()) attach(i.running); }));
  onCleanup(() => {
    es?.close();
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

  function attach(id: string) {
    es?.close();
    batch(() => {
      setJobId(id);
      setBusy(true);
      setStatus({ text: `running ${id}`, state: "running" });
    });
    setDockOpen(true);
    es = followLog(
      id,
      // No log panel yet: the newest line is the status line.
      (line) => setStatus({ text: line, state: "running" }),
      (job) => {
        es = null;
        setBusy(false);
        setStatus({ text: `exit ${job.exit_code}`, state: job.state });
        refetchInfo();
        // A finished download job changed what the Settings rows should say.
        refetchModels();
        if (job.state === "done") remember(job);
        // A finished stage rewrote captions/masks under our feet -- but a dry
        // run of a stage that has an --apply wrote nothing, so nothing to do.
        const dryRun = stages()?.find((s) => s.id === job.stage)?.apply && !job.apply;
        if (!dryRun) void reloadTouched(job);
      },
      () => {
        es = null;
        setBusy(false);
        setStatus({ text: "log stream closed" });
      },
    );
  }

  /** File a clean dry run's report so Apply can replay it. Keyed on the form
      as *this* page sent it, not as the server echoed it back: a snapshot that
      fails to compare equal would silently stop offering the replay. */
  const sent = new Map<string, string>();
  function remember(job: Job) {
    const s = stages()?.find((x) => x.id === job.stage);
    const form = sent.get(job.id);
    sent.delete(job.id);
    if (!s?.replay || !s.apply || job.apply || !job.report_path || form === undefined) return;
    setDry(job.stage, { report: job.report_path, values: form });
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
    if (touched.length) {
      const { items } = await api.items(touched);
      const by = new Map(items.map((i) => [i.rel, i]));
      mutateList((prev) =>
        prev ? { ...prev, items: prev.items.map((x) => by.get(x.rel) ?? x) } : prev,
      );
      const rel = sel()?.rel;
      if (rel && by.has(rel)) refetchItem();
    }
    setStatus((st) => ({ ...st, text: `${st.text} — ${touched.length} file(s) changed` }));
  }

  /** Start the current stage. `rel` narrows it to that one image; null (the
      default) runs the batch the Settings `path_pattern` names. */
  async function run(apply: boolean, rel?: string | null) {
    const s = cur();
    if (!s) return;
    // Apply replays the dry run when the form has not moved since it ran. The
    // field is always rebuilt from that decision, never carried over from the
    // saved form -- a leftover path would replay an old run behind a Dry run.
    const report = apply && reuse() ? replayable() : null;
    const { [REPLAY_FIELD]: _stale, ...rest } = values() ?? {};
    const v = report ? { ...rest, [REPLAY_FIELD]: report } : rest;
    try {
      const job = await api.start(s.id, v, apply, rel);
      if (!apply) sent.set(job.id, formKey(values()));
      setSettings("values", (prev) => ({ ...(prev ?? {}), [s.id]: job.values }));
      attach(job.id);
    } catch (e) {
      setStatus({ text: (e as Error).message, state: "failed" });
      setDockOpen(true);
    }
  }

  /** Weights download: a job like any other, so it streams into the stage bar.
      The dialog gets out of the way -- the progress is in the dock. */
  async function download(ids: string[]) {
    try {
      const job = await api.downloadModels(ids);
      setSettingsOpen(false);
      attach(job.id);
    } catch (e) {
      setStatus({ text: (e as Error).message, state: "failed" });
      setDockOpen(true);
    }
  }

  return (
    <>
      <header>
        <b>anime_tools</b>
        <span class="dim mono" title={info()?.home}>{info()?.home}</span>
        <Show when={list()}>
          {(l) => (
            <span class="dim">
              {l().total} image{l().total === 1 ? "" : "s"} in {l().root}
            </span>
          )}
        </Show>
        <span class="sp" />
        <span class="dim">{info()?.hf_token ? "HF token ✓" : "no HF token"}</span>
        <button onClick={() => { setSettingsOpen(true); refetchModels(); }}>⚙ Settings</button>
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
      />

      <ItemView
        item={item()}
        loading={item.loading}
        error={item.error ? String(item.error) : undefined}
        kind={sel()?.kind ?? "image"}
        onSaved={onSaved}
      />

      <div classList={{ dock: true, closed: !dockOpen() }} style={{ "--dock-h": `${dockH()}px` }}>
        <Show when={dockOpen()}>
          <div class="dockgrip" onPointerDown={grip} title="Drag to resize" />
        </Show>
        <div class="tabs stagetabs">
          <For each={groups()}>
            {([g, ss]) => (
              <>
                <span class="tabgroup">{g}</span>
                <For each={ss}>
                  {(s) => (
                    <a
                      classList={{ sel: dockOpen() && curId() === s.id, na: !s.available }}
                      title={s.available ? s.module : s.error}
                      onClick={() => pickStage(s.id)}
                    >
                      {s.title}
                    </a>
                  )}
                </For>
              </>
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
              stages={stages()}
              error={stages.error}
              curId={curId()}
              values={values()}
              setValue={setValue}
              reset={resetForm}
              busy={busy()}
              status={status()}
              rel={sel()?.rel ?? null}
              onRun={run}
              onApply={(rel) => { setReuse(true); setApplyRel(rel); setConfirmOpen(true); }}
              onCancel={() => jobId() && api.cancel(jobId()!)}
              roots={roots()}
              defaults={stageDefaults()}
              onSettings={() => setSettingsOpen(true)}
              help={help()}
              onHelp={() => setHelp(!help())}
            />
          </div>
        </Show>
      </div>

      <Dialog open={confirmOpen()} onClose={(v) => { setConfirmOpen(false); if (v === "ok") run(true, applyRel()); }}>
        <h3>Apply for real?</h3>
        <p style="max-width:520px">
          {cur()?.title} will write its changes to{" "}
          <Show when={applyRel()} fallback={<>every image <code>{stageDefaults().path_pattern || "*"}</code> names</>}>
            {(rel) => <code>{rel()}</code>}
          </Show>
          .
        </p>
        {/* Replaying the dry run is the difference between writing text that is
            already computed and a second full tagger/SAM3 pass, so it is the
            default whenever one is on file for this exact form. */}
        <Show
          when={replayable()}
          fallback={
            <Show when={cur()?.replay}>
              <p class="dim" style="max-width:520px">
                No dry run to reuse for this form — Apply runs the models again. Dry run first to
                see the proposals and make Apply a plain write.
              </p>
            </Show>
          }
        >
          {(report) => (
            <label class="reuse">
              <input type="checkbox" checked={reuse()} onChange={(e) => setReuse(e.currentTarget.checked)} />
              <span>
                Write the dry run's proposals ({<code>{report()}</code>}) instead of running the
                models again. A caption edited since that run is skipped, not overwritten.
              </span>
            </label>
          )}
        </Show>
        <p class="dim">
          Caption stages write under <code>post_image_dataset/resized/</code> (autotag <code>missing</code> creates
          masters under <code>image_dataset/</code>). Any caption change must be followed by the trainer's TE
          re-encode (<code>make preprocess-te</code>).
        </p>
        <div class="dlg-actions">
          <button value="cancel">Cancel</button>
          <button value="ok" class="danger">Apply</button>
        </div>
      </Dialog>

      <SettingsDialog
        open={settingsOpen()}
        info={info()}
        roots={roots()}
        fields={settingFields()}
        defaults={stageDefaults()}
        models={models()}
        busy={busy()}
        help={help()}
        onHelp={() => setHelp(!help())}
        onDownload={download}
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
        }}
      />
    </>
  );
}

interface SettingsOut {
  token: string | null;
  roots: Record<string, string> | null;
  /** The stage defaults (`path_pattern` / `tagger_dir`), or null if untouched. */
  defaults: Record<string, string> | null;
}

function SettingsDialog(props: {
  open: boolean;
  info?: Info;
  roots?: DatasetRoots;
  /** One argparse Field per Settings-bound stage flag. */
  fields: Field[];
  defaults: Record<string, string>;
  models?: ModelCatalog;
  busy: boolean;
  help: boolean;
  onHelp: () => void;
  onDownload: (ids: string[]) => void;
  onClose: (out: SettingsOut | null) => void;
}) {
  let tokenEl!: HTMLInputElement;
  const rootEls: Partial<Record<RootName, HTMLInputElement>> = {};
  const defEls: Record<string, HTMLInputElement> = {};
  const current = (n: RootName) => props.roots?.roots[n];
  const missing = () => (props.models?.models ?? []).filter((m) => !m.installed);

  return (
    <Dialog
      open={props.open}
      onClose={(v) => {
        const t = tokenEl.value.trim();
        tokenEl.value = "";
        if (v !== "ok") return props.onClose(null);
        const picked = Object.fromEntries(
          ROOT_NAMES.map((n) => [n, rootEls[n]?.value.trim() ?? ""]),
        );
        const changed = ROOT_NAMES.some((n) => picked[n] !== (current(n)?.path ?? ""));
        const defaults = Object.fromEntries(
          props.fields.map((f) => [f.setting!, defEls[f.setting!]?.value.trim() ?? ""]),
        );
        const defChanged = props.fields.some(
          (f) => defaults[f.setting!] !== (props.defaults[f.setting!] ?? ""),
        );
        props.onClose({
          token: t || null,
          roots: changed ? picked : null,
          defaults: defChanged ? defaults : null,
        });
      }}
    >
      {/* `type=button`, like the model rows: every other button in here submits
          the <form method="dialog"> and closes it. */}
      <h3 class="dlgh">
        Settings
        <span class="sp" />
        <HelpToggle open={props.help} onToggle={props.onHelp} />
      </h3>
      <div class="kv">
        <b>Home</b><span class="mono">{props.info?.home}</span>
        <b>Models dir</b><span class="mono">{props.info?.models_dir}</span>
      </div>

      <h4>Dataset roots</h4>
      <Show when={props.help}>
        <p class="dim" style="margin:0 0 8px">
          Relative to the curation home; the three trees are joined by the same relative path.
        </p>
      </Show>
      <div class="kv">
        <For each={ROOT_NAMES}>
          {(n) => (
            <>
              <b>{n}</b>
              <span>
                <input
                  type="text"
                  ref={(el) => (rootEls[n] = el)}
                  value={current(n)?.path ?? props.roots?.defaults[n] ?? ""}
                  placeholder={props.roots?.defaults[n]}
                />
                <span classList={{ dim: true, err: current(n) ? !current(n)!.exists : false }}>
                  {current(n) && !current(n)!.exists ? "missing — " : ""}
                  {ROOT_HELP[n]}
                </span>
              </span>
            </>
          )}
        </For>
      </div>

      <h4>Stage defaults</h4>
      <Show when={props.help}>
        <p class="dim" style="margin:0 0 8px">
          Filled into every stage that takes them, so no stage form re-asks. Leave one blank for the
          CLI's own default. <code>--device</code> is not here on purpose: each stage auto-detects it.
        </p>
      </Show>
      <div class="kv">
        <For each={props.fields}>
          {(f) => (
            <>
              <b>{f.setting}</b>
              <span>
                <input
                  type="text"
                  ref={(el) => (defEls[f.setting!] = el)}
                  value={props.defaults[f.setting!] ?? ""}
                  placeholder={f.default == null ? "(none)" : String(f.default)}
                />
                <span class="dim">{f.help}</span>
              </span>
            </>
          )}
        </For>
      </div>

      <h4>Hugging Face</h4>
      <div class="kv">
        <b>Token</b>
        <span>
          {props.info?.hf_token ? "present" : "missing"}
          <input
            ref={tokenEl}
            type="password"
            placeholder="hf_… (stored by huggingface_hub, never shown again)"
            style="margin-top:4px"
          />
          <Show when={props.help}>
            <span class="dim">
              The tagger backbone and SAM3 weights are gated on the Hub — a token with read access is
              needed on first run.
            </span>
          </Show>
        </span>
      </div>

      <h4>Models</h4>
      <Show when={props.help}>
        <p class="dim" style="margin:0 0 8px">
          Every stage fetches what it needs on first use — these buttons only move the wait, and any
          gated-repo refusal, to a moment you picked. A download runs as a job: one at a time, streaming
          into the stage bar below.
        </p>
      </Show>
      <div class="models">
        <For each={props.models?.models}>
          {(m) => <ModelRow m={m} busy={props.busy} onDownload={props.onDownload} />}
        </For>
      </div>
      <button
        type="button"
        style="margin-top:8px"
        disabled={props.busy || !missing().length}
        onClick={() => props.onDownload([])}
      >
        {missing().length ? `Download all ${missing().length} missing` : "Every model is installed"}
      </button>

      <div class="dlg-actions">
        <button value="cancel">Close</button>
        <button value="ok" class="primary">Save</button>
      </div>
    </Dialog>
  );
}

/** One catalog row. The button is `type=button` on purpose: every other button
    in the dialog submits the <form method="dialog"> and closes it. */
function ModelRow(props: { m: ModelAsset; busy: boolean; onDownload: (ids: string[]) => void }) {
  return (
    <div class="modelrow">
      <div class="mi">
        <span class="mh">
          <b>{props.m.title}</b>
          <span classList={{ badge: true, done: props.m.installed, miss: !props.m.installed }}>
            {props.m.installed ? "installed" : "missing"}
          </span>
          <span class="dim">{props.m.used_by}</span>
        </span>
        <span class="dim mono" title={props.m.location}>
          {props.m.repo} → {props.m.location}
        </span>
        <Show when={props.m.notes}>
          <span class="dim">{props.m.notes}</span>
        </Show>
        <Show when={props.m.gated}>
          <span class="dim">
            Gated —{" "}
            <a href={props.m.gated} target="_blank" rel="noreferrer">
              accept the terms
            </a>{" "}
            with the same account as the token above.
          </span>
        </Show>
      </div>
      <button type="button" disabled={props.busy} onClick={() => props.onDownload([props.m.id])}>
        {props.m.installed ? "Re-download" : "Download"}
      </button>
    </div>
  );
}
