import { Show } from "solid-js";
import { Dialog } from "./Dialog";
import type { RunResult } from "../runner";

/** The one confirmation in the app: **Apply** writes to disk.

    It spells out the scope (one image, or every image the Settings pattern
    names), the count, and — the part worth confirming — *where the text comes
    from*: a replayed report can only write what the diff showed, while a stage
    with no report to stand on is being run again for real. */
export function ApplyDialog(props: {
  open: boolean;
  /** The stage about to write. */
  title?: string;
  /** The run whose report Apply will replay, or null when there is none. */
  pending: RunResult | null;
  /** The Settings batch pattern, named when the run was not scoped to one image. */
  pattern: string;
  onConfirm: () => void;
  onClose: () => void;
}) {
  return (
    <Dialog
      open={props.open}
      onClose={(v) => {
        props.onClose();
        if (v === "ok") props.onConfirm();
      }}
    >
      <h3>Apply for real?</h3>
      <p style="max-width:520px">
        {props.title} will write to{" "}
        <Show
          when={props.pending?.rel}
          fallback={
            <>
              every image <code>{props.pattern || "*"}</code> names
            </>
          }
        >
          {(rel) => <code>{rel()}</code>}
        </Show>
        <Show when={props.pending}>
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
        when={props.pending}
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
        <code>missing</code> creates masters under <code>image_dataset/</code>). Any caption change
        must be followed by the trainer's TE re-encode (<code>make preprocess-te</code>).{" "}
        <b>Undo</b> puts these captions back, from the same report.
      </p>
      <div class="dlg-actions">
        <button value="cancel">Cancel</button>
        <button value="ok" class="danger">
          Apply
        </button>
      </div>
    </Dialog>
  );
}
