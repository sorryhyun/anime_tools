import { For, Show } from "solid-js";
import { slots, t } from "../i18n";
import type { SettingsOut } from "../config";
import type { HelpArea } from "../layout";
import type { Info, JobStatus, ModelAsset, ModelCatalog, ModelPack } from "../types";
import { HelpToggle } from "./HelpToggle";
import { SettingsShell } from "./SettingsShell";
import { StatusLine } from "./StatusLine";

/** The Models dialog: the Hub token and the weights catalog, with the download
    it started reporting under the rows. The token is the one thing OK writes,
    and only if something was typed. */
export function SettingsModels(props: {
  open: boolean;
  info?: Info;
  models?: ModelCatalog;
  /** Any job holds the one slot -- a stage run disables the buttons too. */
  busy: boolean;
  /** …but only a weights job is *ours*, and only it is reported in here. */
  downloading: boolean;
  /** What that job asked for; `[]` = every missing model. */
  downloadIds: string[];
  progress: JobStatus;
  helpOpen: (area: HelpArea) => boolean;
  onHelp: (area: HelpArea) => void;
  onDownload: (ids: string[]) => void;
  onCancelDownload: () => void;
  onClose: (out: SettingsOut | null) => void;
}) {
  let tokenEl!: HTMLInputElement;
  const missing = () => (props.models?.models ?? []).filter((m) => !m.installed);
  /** Is this row part of the running pull? An id-less job is "every missing". */
  const inFlight = (m: ModelAsset) =>
    props.downloading &&
    (props.downloadIds.length ? props.downloadIds.includes(m.id) : !m.installed);
  /** The rows under one pack header, in the order the catalog sent them. The
      server already left out packs with no rows, so every header has some. */
  const rowsOf = (p: ModelPack) => (props.models?.models ?? []).filter((m) => m.pack === p.id);

  /** The token field is emptied on every close, OK or not: a secret does not
      stay in a hidden dialog. */
  const close = (ok: boolean) => {
    const token = tokenEl.value.trim();
    tokenEl.value = "";
    if (!ok) return props.onClose(null);
    props.onClose(token ? { token } : {});
  };

  return (
    <SettingsShell open={props.open} pane="models" onClose={close}>
      <h4>
        {t().settings.hf}
        <HelpToggle open={props.helpOpen("token")} onToggle={() => props.onHelp("token")} />
      </h4>
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
          <Show when={props.helpOpen("token")}>
            <span class="dim">{t().settings.tokenHelp}</span>
          </Show>
        </span>
      </div>

      <h4>
        {t().settings.models}
        <HelpToggle open={props.helpOpen("models")} onToggle={() => props.onHelp("models")} />
      </h4>
      <Show when={props.helpOpen("models")}>
        <p class="dim" style="margin:0 0 8px">
          {t().settings.modelsHelp}
        </p>
      </Show>
      <For each={props.models?.packs}>
        {(p) => (
          <div class="pack">
            <PackHead
              p={p}
              rows={rowsOf(p)}
              busy={props.busy}
              active={rowsOf(p).some(inFlight)}
              onDownload={props.onDownload}
            />
            <div class="models">
              <For each={rowsOf(p)}>
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
          </div>
        )}
      </For>
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
    </SettingsShell>
  );
}

/** A pack's header: its title and description as the server wrote them, and
    the one button that pulls every row under it. The button says "Download"
    while any row is missing and "Re-download" once none is, and it hands
    `onDownload` the row ids — never the pack id — so what the dialog then
    reports (`downloadIds`) lights the same rows a per-row click would. */
function PackHead(props: {
  p: ModelPack;
  rows: ModelAsset[];
  busy: boolean;
  active: boolean;
  onDownload: (ids: string[]) => void;
}) {
  const complete = () => props.rows.every((m) => m.installed);
  return (
    <div class="packhead">
      <div class="mi">
        <b>{props.p.title}</b>
        <Show when={props.p.description}>
          <span class="dim wrap">{props.p.description}</span>
        </Show>
      </div>
      <button
        type="button"
        disabled={props.busy || !props.rows.length}
        onClick={() => props.onDownload(props.rows.map((m) => m.id))}
      >
        {props.active
          ? t().settings.downloadingRow
          : complete()
            ? t().settings.redownloadPack
            : t().settings.downloadPack}
      </button>
    </div>
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
        {/* What a model is *for* gets its own wrapping line. */}
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
