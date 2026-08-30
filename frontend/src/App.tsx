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
  Info,
  ItemDetail,
  NodeKind,
  RootName,
  Settings,
  Stage,
  Values,
} from "./types";
import { DatasetTree, type Sel } from "./components/DatasetTree";
import { ItemView } from "./components/ItemView";
import { StagePanel } from "./components/StagePanel";
import { Dialog } from "./components/Dialog";

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
  const [jobId, setJobId] = createSignal<string | null>(null);
  const [busy, setBusy] = createSignal(false);
  const [status, setStatus] = createSignal<{ text: string; state?: string }>({ text: "" });
  const [confirmOpen, setConfirmOpen] = createSignal(false);
  const [settingsOpen, setSettingsOpen] = createSignal(false);
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
        // A finished stage rewrote captions/masks under our feet.
        refetchList();
        refetchItem();
      },
      () => {
        es = null;
        setBusy(false);
        setStatus({ text: "log stream closed" });
      },
    );
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
        <button onClick={() => setSettingsOpen(true)}>⚙ Settings</button>
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
              onRun={run}
              onApply={() => setConfirmOpen(true)}
              onCancel={() => jobId() && api.cancel(jobId()!)}
              roots={roots()}
              onSettings={() => setSettingsOpen(true)}
            />
          </div>
        </Show>
      </div>

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
        roots={roots()}
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
        }}
      />
    </>
  );
}

interface SettingsOut {
  token: string | null;
  roots: Record<string, string> | null;
}

function SettingsDialog(props: {
  open: boolean;
  info?: Info;
  roots?: DatasetRoots;
  onClose: (out: SettingsOut | null) => void;
}) {
  let tokenEl!: HTMLInputElement;
  const rootEls: Partial<Record<RootName, HTMLInputElement>> = {};
  const current = (n: RootName) => props.roots?.roots[n];

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
        props.onClose({ token: t || null, roots: changed ? picked : null });
      }}
    >
      <h3>Settings</h3>
      <div class="kv">
        <b>Home</b><span class="mono">{props.info?.home}</span>
        <b>Models dir</b><span class="mono">{props.info?.models_dir}</span>
      </div>

      <h4>Dataset roots</h4>
      <p class="dim" style="margin:0 0 8px">
        Relative to the curation home; the three trees are joined by the same relative path.
      </p>
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
          <span class="dim">
            The tagger backbone and SAM3 weights are gated on the Hub — a token with read access is
            needed on first run.
          </span>
        </span>
      </div>

      <div class="dlg-actions">
        <button value="cancel">Close</button>
        <button value="ok" class="primary">Save</button>
      </div>
    </Dialog>
  );
}
