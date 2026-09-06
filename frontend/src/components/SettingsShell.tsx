import { Show, type JSX } from "solid-js";
import { t } from "../i18n";
import type { SettingsPane } from "../config";
import { Dialog } from "./Dialog";

/** The frame the three Settings dialogs share: the native dialog, its titled
 * header with the × in the corner, the blocks the pane puts in it, and the
 * Save button under them. Each dialog is its own window with its own OK, so
 * `onClose` says only whether OK was pressed and the pane decides what that
 * writes; the shell holds no inputs of its own.
 */
export function SettingsShell(props: {
  open: boolean;
  pane: SettingsPane;
  onClose: (ok: boolean) => void;
  children: JSX.Element;
}) {
  return (
    <Dialog open={props.open} class="settings" onClose={(v) => props.onClose(v === "ok")}>
      {/* The x is a plain `value="cancel"` submitter of the
          <form method="dialog">, which is exactly what it wants; the (?) beside
          each block is `type=button` so it is not. */}
      <h3 class="dlgh">
        {t().settings.paneTitle[props.pane]}
        <span class="sp" />
        <button value="cancel" class="dlgx" title={t().common.close} aria-label={t().common.close}>
          ×
        </button>
      </h3>
      <div class="spane">{props.children}</div>
      <div class="dlg-actions">
        <button value="ok" class="primary">
          {t().common.save}
        </button>
      </div>
    </Dialog>
  );
}

/** One `label / input / hint` row of a `.kv` grid — the roots block and the
 * stage-defaults block are the same row twice. The input is uncontrolled and
 * read off its `ref` on Save, so nothing is stored per keystroke and Cancel is
 * free.
 */
export function SettingRow(props: {
  label: string;
  ref: (el: HTMLInputElement) => void;
  value: string;
  placeholder?: string;
  hint: string;
  /** Marks the hint as a problem (a root that is not there yet). */
  err?: boolean;
  /** Give a path row the same `…` browse button the stage forms have; omit it
      for a bare input. */
  onPick?: () => void;
}) {
  return (
    <>
      <b>{props.label}</b>
      <span>
        {/* .pathrow is the stage form's wrapper, reused. */}
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
