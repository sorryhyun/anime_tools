import { Show } from "solid-js";
import { slots, t } from "../i18n";
import { Dialog } from "./Dialog";
import type { RunResult } from "../runner";

/** The one confirmation in the app: **Apply** writes to disk.

    It spells out the scope (one image, or every image the Settings pattern
    names), the count, and — the part worth confirming — *where the text comes
    from*: a replayed report can only write what the diff showed, while a stage
    with no report to stand on is being run again for real. */
export function ApplyDialog(props: {
  open: boolean;
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
      <h3>{t().applyDlg.title}</h3>
      <p style="max-width:520px">
        {slots(t().applyDlg.writesTo, (i) =>
          i === 0 ? (
            props.title
          ) : (
            <Show
              when={props.pending?.rel}
              fallback={slots(t().applyDlg.everyImage, () => (
                <code>{props.pattern || "*"}</code>
              ))}
            >
              {(rel) => <code>{rel()}</code>}
            </Show>
          ),
        )}
        <Show when={props.pending}>{(d) => <> {t().applyDlg.changes(d().rels.length)}</>}</Show>.
      </p>
      {/* Apply is a replay by construction: the run already computed and wrote
          down every proposal, so this pass loads no model and can only write
          text the diff showed. A stage with no report to stand on says so. */}
      <Show
        when={props.pending}
        fallback={
          <p class="dim" style="max-width:520px">
            {slots(t().applyDlg.noReport, () => (
              <code>--apply</code>
            ))}
          </p>
        }
      >
        {(d) => (
          <p class="dim" style="max-width:520px">
            {slots(t().applyDlg.replay, () => (
              <code>{d().report}</code>
            ))}
          </p>
        )}
      </Show>
      <p class="dim">
        {slots(t().applyDlg.where, (i) =>
          i === 0 ? (
            <code>post_image_dataset/resized/</code>
          ) : i === 1 ? (
            <code>missing</code>
          ) : i === 2 ? (
            <code>image_dataset/</code>
          ) : i === 3 ? (
            <code>make preprocess-te</code>
          ) : (
            <b>{t().stage.undo}</b>
          ),
        )}
      </p>
      <div class="dlg-actions">
        <button value="cancel">{t().common.cancel}</button>
        <button value="ok" class="danger">
          {t().applyDlg.apply}
        </button>
      </div>
    </Dialog>
  );
}
