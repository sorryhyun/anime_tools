import { Show } from "solid-js";
import { t } from "../i18n";
import type { SettingsOut } from "../config";
import type { HelpArea } from "../layout";
import type { JobStatus, UpdateInfo } from "../types";
import { HelpToggle } from "./HelpToggle";
import { SettingsShell } from "./SettingsShell";
import { StatusLine } from "./StatusLine";

/** The Update dialog: which version is installed, which one GitHub has, and the
    one button between them. Nothing here is written on OK — the checkbox saves
    when it is clicked and the upgrade is a job, so its closing button says Close
    and hands `closeSettings` nothing at all. */
export function SettingsUpdate(props: {
  open: boolean;
  info?: UpdateInfo;
  /** A check is out to GitHub (the first one, or "Check now"). */
  checking: boolean;
  /** Any job holds the one slot — a stage run disables the button too. */
  busy: boolean;
  /** …but only the upgrade is *ours*, and only it is reported in here. */
  updating: boolean;
  /** An upgrade finished in this session: the process is still the old one. */
  restart: boolean;
  progress: JobStatus;
  /** The tail of the job's output, so a uv resolution failure is readable
      without opening the dock's log window (which never shows this job). */
  log: string[];
  helpOpen: (area: HelpArea) => boolean;
  onHelp: (area: HelpArea) => void;
  onCheck: () => void;
  onAuto: (on: boolean) => void;
  onUpdate: () => void;
  onCancelUpdate: () => void;
  onClose: (out: SettingsOut | null) => void;
}) {
  const state = () => props.info?.status ?? "unknown";
  const latest = () => props.info?.latest ?? "";
  /** Only an ordered pair with a newer tag on the far side is an update, and
      only an install uv owns can be moved by this button. */
  const canUpdate = () => !!props.info?.can_update && state() === "available";
  /** The button says what it would do — and when it would do nothing, which of
      the two reasons that is: already current, or nothing to compare against. */
  const label = () =>
    props.updating
      ? t().update.updating
      : canUpdate()
        ? t().update.updateTo(latest())
        : state() === "current"
          ? t().update.upToDate
          : t().update.nothingToDo;

  return (
    <SettingsShell
      open={props.open}
      pane="update"
      ok={t().common.close}
      onClose={() => props.onClose(null)}
    >
      <h4>
        {t().update.version}
        <HelpToggle open={props.helpOpen("update")} onToggle={() => props.onHelp("update")} />
      </h4>
      <Show when={props.helpOpen("update")}>
        <p class="dim" style="margin:0 0 8px">
          {t().update.help}
        </p>
      </Show>
      <div class="kv">
        <b>{t().update.installed}</b>
        <span class="mono">{props.info?.current ?? "…"}</span>
        <b>{t().update.latest}</b>
        <span class="mono">
          {props.checking ? "…" : latest() || t().update.never}{" "}
          <Show when={props.info?.url}>
            <a href={props.info!.url} target="_blank" rel="noreferrer">
              {t().update.viewRelease}
            </a>
          </Show>
        </span>
      </div>

      <div class="dlbar">
        <span
          classList={{
            badge: true,
            running: state() === "available",
            done: state() === "current",
            miss: state() === "unknown" || state() === "ahead",
          }}
        >
          {props.checking ? t().update.checking : t().update.state[state()]}
        </span>
        <span class="status dim">
          <Show when={props.info?.error}>
            <span class="err">{t().update.checkFailed(props.info!.error)}</span>
          </Show>
          <Show when={!props.info?.error && props.info?.checked_at}>
            {t().update.checkedAt(new Date(props.info!.checked_at! * 1000).toLocaleString())}
          </Show>
        </span>
        <button type="button" disabled={props.checking} onClick={props.onCheck}>
          {t().update.checkNow}
        </button>
      </div>

      <label class="urow">
        <input
          type="checkbox"
          checked={!!props.info?.auto_check}
          onChange={(e) => props.onAuto(e.currentTarget.checked)}
        />
        <span>{t().update.auto}</span>
        <span class="dim">{t().update.autoHint}</span>
      </label>

      <Show when={props.info?.notes}>
        <h4>{t().update.releaseNotes}</h4>
        {/* The body as GitHub holds it: this is release prose, not markup the
            page renders — a release that writes a heading shows the #. */}
        <div class="logbox">
          <pre class="log">{props.info!.notes}</pre>
        </div>
      </Show>

      <h4>{t().update.install}</h4>
      <p class="dim" style="margin:0 0 8px">
        {t().update.installKind[props.info?.install ?? "other"]}
        <Show when={props.info && !props.info.can_update}>
          {" — "}
          {props.info!.hint}
        </Show>
      </p>
      <Show when={props.info?.can_update}>
        <p class="notes">{t().update.warning}</p>
      </Show>
      <Show when={props.restart}>
        <p class="notes">{t().update.restart}</p>
      </Show>

      <div class="dlbar">
        <button
          type="button"
          class="primary"
          disabled={props.busy || props.checking || !canUpdate()}
          onClick={props.onUpdate}
        >
          {label()}
        </button>
        <Show when={props.updating}>
          <button type="button" onClick={props.onCancelUpdate}>
            {t().common.cancel}
          </button>
        </Show>
        <Show when={props.progress.text}>
          <StatusLine status={props.progress} />
        </Show>
      </div>

      <Show when={props.log.length}>
        <div class="logbox" style="margin-top:8px; max-height:22vh">
          <pre class="log">{props.log.join("\n")}</pre>
        </div>
      </Show>
    </SettingsShell>
  );
}
