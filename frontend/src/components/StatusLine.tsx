import { Show } from "solid-js";
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
