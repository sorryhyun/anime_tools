import { Show, type JSX } from "solid-js";
import { t } from "../i18n";
import type { JobStatus } from "../types";

/** The one status line, shared by the run bar and the Settings download bar:
    an optional state badge, then the text. `title` carries the full text, since
    the line clips. */
export function StatusLine(props: { status: JobStatus }) {
  return (
    <span class="status" title={props.status.text}>
      <Show when={props.status.state}>
        <span class={`badge ${props.status.state}`}>{props.status.state}</span>{" "}
      </Show>
      {props.status.text}
    </span>
  );
}

/** A Settings dialog's job row: the one button that starts its job, Cancel
    while that job runs, and the job's status line once it has said something.
    The Models dialog's download and the Update dialog's upgrade are both this. */
export function JobActions(props: {
  label: JSX.Element;
  primary?: boolean;
  disabled: boolean;
  onRun: () => void;
  running: boolean;
  onCancel: () => void;
  status: JobStatus;
}) {
  return (
    <div class="dlbar">
      <button
        type="button"
        classList={{ primary: !!props.primary }}
        disabled={props.disabled}
        onClick={() => props.onRun()}
      >
        {props.label}
      </button>
      <Show when={props.running}>
        <button type="button" onClick={() => props.onCancel()}>
          {t().common.cancel}
        </button>
      </Show>
      <Show when={props.status.text}>
        <StatusLine status={props.status} />
      </Show>
    </div>
  );
}
