import { createEffect, createMemo, createResource, createSignal, For, on, onCleanup, Show } from "solid-js";
import { createStore, reconcile } from "solid-js/store";
import { api, followLog } from "./api";
import type { Info, Job, Settings, Stage, Values } from "./types";
import { StageForm } from "./components/StageForm";
import { Report } from "./components/Report";
import { Dialog } from "./components/Dialog";

type Tab = "log" | "report" | "jobs";

const fmtTime = (t: number) => new Date(t * 1000).toLocaleTimeString();

export default function App() {
  const [info, { refetch: refetchInfo }] = createResource<Info>(api.info);
  const [stages] = createResource<Stage[]>(api.stages);
  const [settings, setSettings] = createStore<Settings>({});
  const [jobs, { refetch: refetchJobs }] = createResource<Job[]>(api.jobs);

  const [curId, setCurId] = createSignal(location.hash.slice(1));
  const cur = createMemo(() => stages()?.find((s) => s.id === curId()));
  // Per-stage form state; seeded from the server's last-used values.
  const [forms, setForms] = createStore<Record<string, Values>>({});
  const values = () => forms[curId()] ?? {};
  const setValue = (dest: string, v: unknown) =>
    setForms(curId(), (prev) => ({ ...(prev ?? {}), [dest]: v }));
  const resetForm = () => setForms(curId(), reconcile({}));

  const [tab, setTab] = createSignal<Tab>("log");
  const [log, setLog] = createSignal("");
  const [jobId, setJobId] = createSignal<string | null>(null);
  const [busy, setBusy] = createSignal(false);
  const [status, setStatus] = createSignal<{ text: string; state?: string }>({ text: "" });
  const [report, setReport] = createSignal<{ path: string; report: unknown } | null>(null);
  const [reportErr, setReportErr] = createSignal("");
  const [confirmOpen, setConfirmOpen] = createSignal(false);
  const [settingsOpen, setSettingsOpen] = createSignal(false);
  let es: EventSource | null = null;
  let logPane!: HTMLDivElement;

  api.settings().then((s) => {
    setSettings(s);
    setForms(reconcile(structuredClone(s.values ?? {})));
  });

  // First available stage if the hash is empty/unknown; reattach to a running job.
  createEffect(
    on(stages, (ss) => {
      if (!ss) return;
      if (!ss.some((s) => s.id === curId())) setCurId(ss.find((s) => s.available)?.id ?? "");
    }),
  );
  createEffect(on(curId, (id) => { if (id) location.hash = id; }));
  createEffect(on(info, (i) => { if (i?.running && !jobId()) attach(i.running); }));
  const onHash = () => setCurId(location.hash.slice(1));
  window.addEventListener("hashchange", onHash);
  onCleanup(() => { es?.close(); window.removeEventListener("hashchange", onHash); });

  function attach(id: string) {
    es?.close();
    setJobId(id);
    setLog("");
    setReport(null);
    setReportErr("");
    setTab("log");
    setBusy(true);
    setStatus({ text: `running ${id}`, state: "running" });
    es = followLog(
      id,
      (line) => {
        setLog((l) => l + line + "\n");
        queueMicrotask(() => { logPane.scrollTop = logPane.scrollHeight; });
      },
      (job) => {
        es = null;
        setBusy(false);
        setStatus({ text: `exit ${job.exit_code}`, state: job.state });
        refetchJobs();
        refetchInfo();
        if (job.report_path) loadReport(job.id);
      },
      () => {
        es = null;
        setBusy(false);
        setStatus({ text: "log stream closed" });
        refetchJobs();
      },
    );
  }

  async function loadReport(id: string) {
    try {
      setReport(await api.report(id));
      setTab("report");
    } catch (e) {
      setReportErr((e as Error).message);
    }
  }

  async function run(apply: boolean) {
    const s = cur();
    if (!s) return;
    try {
      const job = await api.start(s.id, values(), apply);
      setSettings("values", (prev) => ({ ...(prev ?? {}), [s.id]: job.values }));
      attach(job.id);
    } catch (e) {
      setStatus({ text: (e as Error).message, state: "failed" });
    }
  }

  const groups = createMemo(() => {
    const m = new Map<string, Stage[]>();
    for (const s of stages() ?? []) m.set(s.group, [...(m.get(s.group) ?? []), s]);
    return [...m];
  });

  return (
    <>
      <header>
        <b>anime_tools</b>
        <span class="dim">{info()?.home}</span>
        <span class="sp" />
        <span class="dim">{info()?.hf_token ? "HF token ✓" : "no HF token"}</span>
        <button onClick={() => setSettingsOpen(true)}>⚙ Settings</button>
      </header>

      <nav>
        <Show when={stages.error}><div class="err" style="padding:12px">{String(stages.error)}</div></Show>
        <For each={groups()}>
          {([g, ss]) => (
            <>
              <h4>{g}</h4>
              <For each={ss}>
                {(s) => (
                  <a
                    classList={{ sel: curId() === s.id, na: !s.available }}
                    title={s.available ? s.module : `not installed: ${s.error}`}
                    onClick={() => setCurId(s.id)}
                  >
                    {s.title}
                  </a>
                )}
              </For>
            </>
          )}
        </For>
      </nav>

      <main>
        <div class="form">
          <Show when={cur()} fallback={<div class="doc">Pick a stage on the left.</div>}>
            {(s) => (
              <Show
                when={s().available}
                fallback={
                  <div class="doc">
                    {s().title} is unavailable: {s().error}
                    {"\n\nReinstall:  uv tool install --force \"anime-tools @ git+https://github.com/sorryhyun/anime_tools\""}
                  </div>
                }
              >
                <StageForm stage={s()} values={values()} setValue={setValue} reset={resetForm} />
                <div class="actions">
                  <button classList={{ primary: !s().apply }} disabled={busy()} onClick={() => run(false)}>
                    {s().apply ? "Dry run" : "Run"}
                  </button>
                  <Show when={s().apply}>
                    <button class="primary" disabled={busy()} onClick={() => setConfirmOpen(true)}>
                      Apply…
                    </button>
                  </Show>
                  <button disabled={!busy()} onClick={() => jobId() && api.cancel(jobId()!)}>
                    Cancel
                  </button>
                  <span class="status">
                    <Show when={status().state}>
                      <span class={`badge ${status().state}`}>{status().state}</span>{" "}
                    </Show>
                    {status().text}
                  </span>
                </div>
              </Show>
            )}
          </Show>
        </div>

        <div class="right">
          <div class="tabs">
            <For each={["log", "report", "jobs"] as Tab[]}>
              {(t) => (
                <a classList={{ sel: tab() === t }} onClick={() => { setTab(t); if (t === "jobs") refetchJobs(); }}>
                  {t[0].toUpperCase() + t.slice(1)}
                </a>
              )}
            </For>
          </div>
          <div class="pane" ref={logPane} style={{ display: tab() === "log" ? "block" : "none" }}>
            <pre class="log">{log()}</pre>
          </div>
          <Show when={tab() === "report"}>
            <div class="pane">
              <Show
                when={report()}
                fallback={<div class="dim">{reportErr() || "Run a stage; its report.json shows here."}</div>}
              >
                {(r) => <Report path={r().path} report={r().report} />}
              </Show>
            </div>
          </Show>
          <Show when={tab() === "jobs"}>
            <div class="pane">
              <Show when={jobs()?.length} fallback={<span class="dim">No jobs yet.</span>}>
                <table>
                  <thead>
                    <tr><th>started</th><th>job</th><th>stage</th><th>state</th><th>apply</th><th>argv</th></tr>
                  </thead>
                  <tbody>
                    <For each={[...(jobs() ?? [])].reverse()}>
                      {(j) => (
                        <tr>
                          <td>{fmtTime(j.started)}</td>
                          <td><button class="link" onClick={() => attach(j.id)}>{j.id}</button></td>
                          <td>{j.stage}</td>
                          <td><span class={`badge ${j.state}`}>{j.state}</span></td>
                          <td>{j.apply ? "yes" : ""}</td>
                          <td class="mono">{j.argv.slice(2).join(" ")}</td>
                        </tr>
                      )}
                    </For>
                  </tbody>
                </table>
              </Show>
            </div>
          </Show>
        </div>
      </main>

      <Dialog open={confirmOpen()} onClose={(v) => { setConfirmOpen(false); if (v === "ok") run(true); }}>
        <h3>Apply for real?</h3>
        <p style="max-width:520px">{cur()?.title} will write its changes.</p>
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
        onClose={async (token) => {
          setSettingsOpen(false);
          if (token) {
            await api.putSettings({ hf_token: token });
            refetchInfo();
          }
        }}
      />
    </>
  );
}

function SettingsDialog(props: { open: boolean; info?: Info; onClose: (token: string | null) => void }) {
  let tokenEl!: HTMLInputElement;
  return (
    <Dialog
      open={props.open}
      onClose={(v) => {
        const t = tokenEl.value.trim();
        tokenEl.value = "";
        props.onClose(v === "ok" && t ? t : null);
      }}
    >
      <h3>Settings</h3>
      <div class="kv">
        <b>Home</b><span>{props.info?.home}</span>
        <b>Models dir</b><span>{props.info?.models_dir}</span>
        <b>HF token</b>
        <span>
          {props.info?.hf_token ? "present" : "missing"}
          <br />
          <input
            ref={tokenEl}
            type="password"
            placeholder="hf_… (stored by huggingface_hub, never shown again)"
            style="margin-top:4px"
          />
        </span>
      </div>
      <p class="dim">
        The tagger backbone and SAM3 weights are gated on the Hub — a token with read access is needed on first run.
      </p>
      <div class="dlg-actions">
        <button value="cancel">Close</button>
        <button value="ok" class="primary">Save</button>
      </div>
    </Dialog>
  );
}
