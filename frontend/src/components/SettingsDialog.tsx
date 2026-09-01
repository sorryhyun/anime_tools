import { createEffect, createMemo, createSignal, For, on, Show } from "solid-js";
import { createStore, reconcile, unwrap } from "solid-js/store";
import { slots, t } from "../i18n";
import { MASK_SETTING, REPORT_SETTING } from "../types";
import type {
  DatasetRoots,
  Field,
  Info,
  JobStatus,
  ModelAsset,
  ModelCatalog,
  RootName,
  Stage,
} from "../types";
import { Dialog } from "./Dialog";
import { FieldRow, grouped } from "./StageForm";
import { browsePath, PathPicker } from "./PathPicker";
import { HelpToggle } from "./HelpToggle";
import { StatusLine } from "./StatusLine";

/** The dataset trees, in the order the dialog lists them: input, then the three
    the workspace owns, then the tree Export publishes to. */
export const ROOT_NAMES: RootName[] = ["src", "master", "dst", "masks", "out"];
const rootHelp = (n: RootName) => t().settings.rootHelp[n];

/** One `label / input / hint` row of a `.kv` grid — the roots block and the
 * stage-defaults block are the same row twice. The input is deliberately
 * uncontrolled and read off its `ref` on Save, so nothing is stored per
 * keystroke and Cancel is free.
 */
function SettingRow(props: {
  label: string;
  ref: (el: HTMLInputElement) => void;
  value: string;
  placeholder?: string;
  hint: string;
  /** Marks the hint as a problem (a root that is not there yet). */
  err?: boolean;
  /** Give a path row the same `…` browse button the stage forms have; omit it
      and the row is the bare input it always was. */
  onPick?: () => void;
}) {
  return (
    <>
      <b>{props.label}</b>
      <span>
        {/* .pathrow is the stage form's wrapper, reused: a browsed root and a
            browsed stage flag are the same widget in both places. */}
        <div classList={{ pathrow: !!props.onPick }}>
          <input type="text" ref={props.ref} value={props.value} placeholder={props.placeholder} />
          <Show when={props.onPick}>
            <button type="button" title={t().common.browse} onClick={() => props.onPick!()}>
              …
            </button>
          </Show>
        </div>
        <span classList={{ dim: true, err: props.err }}>{props.hint}</span>
      </span>
    </>
  );
}

/** The three Settings dialogs. Each entry point opens *one* of them and only
    that one: the ☰ menu lists them separately, a missing-weights hint opens
    `models`, and there is no strip of tabs to wander off along. They were one
    tabbed dialog, which meant every route into Settings put the dataset roots
    you edit once a year one click from the download buttons you press weekly —
    and made every Save a save of all three blocks at once.

    So a pane is also the unit of what OK writes: only the open one's inputs are
    even mounted, and :type:`SettingsOut` carries `null` for the other two. */
export const SETTINGS_PANES = ["general", "advanced", "models"] as const;
export type SettingsPane = (typeof SETTINGS_PANES)[number];

export interface SettingsOut {
  token: string | null;
  roots: Record<string, string> | null;
  /** The stage defaults, or null if untouched. */
  defaults: Record<string, string> | null;
  /** The preflight stage's form values, or null if untouched. */
  preprocess: Record<string, unknown> | null;
}

export function SettingsDialog(props: {
  open: boolean;
  /** Which of the three dialogs this is — the entry point picks it (the
      header's token warning and a missing-weights hint both mean Models), and
      nothing in here can change it. */
  pane: SettingsPane;
  info?: Info;
  roots?: DatasetRoots;
  /** One argparse Field per Settings-bound stage flag. */
  fields: Field[];
  defaults: Record<string, string>;
  /** The hidden preflight stage, rendered as its own Settings block. */
  preprocess?: Stage;
  preprocessValues: Record<string, unknown>;
  models?: ModelCatalog;
  /** Any job holds the one slot -- a stage run disables the buttons too. */
  busy: boolean;
  /** …but only a weights job is *ours*, and only it is reported in here. */
  downloading: boolean;
  /** What that job asked for; `[]` = every missing model. */
  downloadIds: string[];
  progress: JobStatus;
  help: boolean;
  onHelp: () => void;
  onDownload: (ids: string[]) => void;
  onCancelDownload: () => void;
  onClose: (out: SettingsOut | null) => void;
}) {
  /** Only mounted on the `models` pane, so it is genuinely optional now. */
  let tokenEl: HTMLInputElement | undefined;
  const rootEls: Partial<Record<RootName, HTMLInputElement>> = {};
  const defEls: Record<string, HTMLInputElement> = {};
  /** The preflight form's edits, keyed by dest. Seeded from the saved values on
      open and diffed on OK, so an untouched block sends nothing. */
  const [pre, setPre] = createStore<Record<string, unknown>>({});
  /** Which root row's fallback browser is open -- only the hosts with no
      chooser of their own get here. A pick is written straight onto the
      (uncontrolled) input, the way typing would be. */
  const [picking, setPicking] = createSignal<RootName | null>(null);
  createEffect(
    on(
      () => props.open,
      (open) => {
        if (!open) return;
        // `unwrap` first: this arrives as a slice of `config.settings`, which is
        // a store, and a store is a Proxy -- `structuredClone` refuses one with
        // a DataCloneError and takes the whole dialog down with it. Invisible
        // while the preprocess block was absent (the `?? {}` handed over a plain
        // literal), and a thrown open the moment anyone saved a knob in it.
        setPre(reconcile(structuredClone(unwrap(props.preprocessValues))));
      },
    ),
  );
  /** Fields the preflight block shows: the same filter the stage forms use, so
      the roots and `path_pattern` stay bound server-side and out of here. */
  const preFields = createMemo(() =>
    props.preprocess ? grouped(props.preprocess.fields).flatMap(([, fs]) => fs) : [],
  );
  const current = (n: RootName) => props.roots?.roots[n];
  const missing = () => (props.models?.models ?? []).filter((m) => !m.installed);
  /** Is this row part of the running pull? An id-less job is "every missing". */
  const inFlight = (m: ModelAsset) =>
    props.downloading &&
    (props.downloadIds.length ? props.downloadIds.includes(m.id) : !m.installed);

  return (
    <>
      <Dialog
        open={props.open}
        class="settings"
        onClose={(v) => {
          // Not `t`: that is the message table, which this callback is inside.
          // Each block is read only when its own pane is the one open: the
          // other two are not mounted, so their refs are `undefined` and a
          // blanket read would post every root as "" the first time anyone
          // saved a token.
          const token = tokenEl?.value.trim() ?? "";
          if (tokenEl) tokenEl.value = "";
          if (v !== "ok") return props.onClose(null);
          const out: SettingsOut = {
            token: null,
            roots: null,
            defaults: null,
            preprocess: null,
          };
          if (props.pane === "models") out.token = token || null;
          if (props.pane === "general") {
            const picked = Object.fromEntries(
              ROOT_NAMES.map((n) => [n, rootEls[n]?.value.trim() ?? ""]),
            );
            if (ROOT_NAMES.some((n) => picked[n] !== (current(n)?.path ?? ""))) out.roots = picked;
          }
          if (props.pane === "advanced") {
            const defKeys = [...props.fields.map((f) => f.setting!), REPORT_SETTING, MASK_SETTING];
            const defaults = Object.fromEntries(
              defKeys.map((k) => [k, defEls[k]?.value.trim() ?? ""]),
            );
            if (defKeys.some((k) => defaults[k] !== (props.defaults[k] ?? "")))
              out.defaults = defaults;
            if (JSON.stringify(unwrap(pre)) !== JSON.stringify(props.preprocessValues))
              out.preprocess = { ...unwrap(pre) };
          }
          props.onClose(out);
        }}
      >
        {/* The (?) sits against the title, because it explains *this* dialog; the
          x is pushed to the far corner, because it leaves it. HelpToggle is
          `type=button`, like the model rows -- every other button in here
          submits the <form method="dialog"> and closes it, which is exactly
          what the x wants, so it is a plain `value="cancel"` submitter. */}
        <h3 class="dlgh">
          {t().settings.paneTitle[props.pane]}
          <HelpToggle open={props.help} onToggle={props.onHelp} />
          <span class="sp" />
          <button
            value="cancel"
            class="dlgx"
            title={t().common.close}
            aria-label={t().common.close}
          >
            ×
          </button>
        </h3>

        <Show when={props.pane === "general"}>
          <div class="spane">
            <div class="kv">
              <b>{t().settings.home}</b>
              <span class="mono">{props.info?.home}</span>
              <b>{t().settings.modelsDir}</b>
              <span class="mono">{props.info?.models_dir}</span>
            </div>

            <h4>{t().settings.roots}</h4>
            <Show when={props.help}>
              <p class="dim" style="margin:0 0 8px">
                {slots(t().settings.rootsHelp, () => (
                  <code>out</code>
                ))}
              </p>
            </Show>
            <div class="kv">
              <For each={ROOT_NAMES}>
                {(n) => {
                  const gone = () => !!current(n) && !current(n)!.exists;
                  return (
                    <SettingRow
                      label={n}
                      ref={(el) => (rootEls[n] = el)}
                      value={current(n)?.path ?? props.roots?.defaults[n] ?? ""}
                      placeholder={props.roots?.defaults[n]}
                      err={gone()}
                      hint={(gone() ? t().settings.rootMissing : "") + rootHelp(n)}
                      onPick={() =>
                        void browsePath(
                          "dir",
                          rootEls[n]?.value ?? "",
                          (path) => rootEls[n] && (rootEls[n]!.value = path),
                          () => setPicking(n),
                        )
                      }
                    />
                  );
                }}
              </For>
            </div>
          </div>
        </Show>

        <Show when={props.pane === "advanced"}>
          <div class="spane">
            <h4>{t().settings.stageDefaults}</h4>
            <Show when={props.help}>
              <p class="dim" style="margin:0 0 8px">
                {slots(t().settings.stageDefaultsHelp, (i) => (
                  <code>
                    {["--device", "report_root", "captions/autotag", "groups/groups.json"][i]}
                  </code>
                ))}
              </p>
            </Show>
            <div class="kv">
              <For each={props.fields}>
                {(f) => (
                  <SettingRow
                    label={f.setting!}
                    ref={(el) => (defEls[f.setting!] = el)}
                    value={props.defaults[f.setting!] ?? ""}
                    placeholder={f.default == null ? "(none)" : String(f.default)}
                    hint={f.help}
                  />
                )}
              </For>
              <SettingRow
                label={REPORT_SETTING}
                ref={(el) => (defEls[REPORT_SETTING] = el)}
                value={props.defaults[REPORT_SETTING] ?? ""}
                placeholder={props.roots?.report_root}
                hint={t().settings.reportRootHint}
              />
              <SettingRow
                label={MASK_SETTING}
                ref={(el) => (defEls[MASK_SETTING] = el)}
                value={props.defaults[MASK_SETTING] ?? ""}
                placeholder={props.roots?.mask_root}
                hint={t().settings.maskRootHint}
              />
            </div>

            <Show when={props.preprocess}>
              {(pre_) => (
                <>
                  <h4>{pre_().title}</h4>
                  <Show when={props.help}>
                    <p class="dim" style="margin:0 0 8px">
                      {pre_().notes}{" "}
                      {slots(t().settings.preprocessHelp, () => (
                        <code>target_res</code>
                      ))}
                    </p>
                  </Show>
                  <div class="twoup">
                    <For each={preFields()}>
                      {(f) => (
                        <FieldRow
                          field={f}
                          value={pre[f.dest] ?? f.default}
                          dirty={
                            pre[f.dest] !== undefined && String(pre[f.dest]) !== String(f.default)
                          }
                          setValue={(v) => setPre(f.dest, v)}
                        />
                      )}
                    </For>
                  </div>
                </>
              )}
            </Show>
          </div>
        </Show>

        <Show when={props.pane === "models"}>
          <div class="spane">
            <h4>{t().settings.hf}</h4>
            <div class="kv">
              <b>{t().settings.token}</b>
              <span>
                {props.info?.hf_token ? t().common.present : t().common.missing}
                <input
                  ref={tokenEl}
                  type="password"
                  placeholder={t().settings.tokenPlaceholder}
                  style="margin-top:4px"
                />
                <Show when={props.help}>
                  <span class="dim">{t().settings.tokenHelp}</span>
                </Show>
              </span>
            </div>

            <h4>{t().settings.models}</h4>
            <Show when={props.help}>
              <p class="dim" style="margin:0 0 8px">
                {t().settings.modelsHelp}
              </p>
            </Show>
            <div class="models">
              <For each={props.models?.models}>
                {(m) => (
                  <ModelRow
                    m={m}
                    busy={props.busy}
                    active={inFlight(m)}
                    onDownload={props.onDownload}
                  />
                )}
              </For>
            </div>
            <div class="dlbar">
              <button
                type="button"
                disabled={props.busy || !missing().length}
                onClick={() => props.onDownload([])}
              >
                {missing().length
                  ? t().settings.downloadAll(missing().length)
                  : t().settings.allInstalled}
              </button>
              <Show when={props.downloading}>
                <button type="button" onClick={props.onCancelDownload}>
                  {t().common.cancel}
                </button>
              </Show>
              <Show when={props.progress.text}>
                <StatusLine status={props.progress} />
              </Show>
            </div>
          </div>
        </Show>

        <div class="dlg-actions">
          <button value="ok" class="primary">
            {t().common.save}
          </button>
        </div>
      </Dialog>
      {/* A sibling of the dialog, not a child of it: <dialog> puts both in the
        top layer so the picker still lands over the settings, while nesting its
        <form method="dialog"> inside this one's would be a nested form. */}
      <PathPicker
        open={picking() !== null}
        onClose={(path) => {
          const n = picking();
          if (path !== null && n && rootEls[n]) rootEls[n]!.value = path;
          setPicking(null);
        }}
      />
    </>
  );
}

function ModelRow(props: {
  m: ModelAsset;
  busy: boolean;
  active: boolean;
  onDownload: (ids: string[]) => void;
}) {
  return (
    <div class="modelrow">
      <div class="mi">
        <span class="mh">
          <b>{props.m.title}</b>
          <span
            classList={{
              badge: true,
              running: props.active,
              done: !props.active && props.m.installed,
              miss: !props.active && !props.m.installed,
            }}
          >
            {props.active
              ? t().common.downloading
              : props.m.installed
                ? t().common.installed
                : t().common.missing}
          </span>
        </span>
        <span class="dim mono" title={props.m.location}>
          {props.m.repo} → {props.m.location}
        </span>
        {/* Two-up rows are half the old width: what a model is *for* gets its
            own wrapping line rather than a clipped tail of the title's. */}
        <span class="dim wrap">{props.m.used_by}</span>
        <Show when={props.m.notes}>
          <span class="dim wrap">{props.m.notes}</span>
        </Show>
        <Show when={props.m.gated}>
          <span class="dim wrap">
            {slots(t().settings.gated, () => (
              <a href={props.m.gated} target="_blank" rel="noreferrer">
                {t().settings.gatedLink}
              </a>
            ))}
          </span>
        </Show>
      </div>
      <button type="button" disabled={props.busy} onClick={() => props.onDownload([props.m.id])}>
        {props.active
          ? t().settings.downloadingRow
          : props.m.installed
            ? t().settings.redownload
            : t().settings.download}
      </button>
    </div>
  );
}
