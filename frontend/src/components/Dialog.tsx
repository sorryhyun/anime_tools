import { createEffect, type JSX, onCleanup } from "solid-js";

/** Native <dialog> driven by an `open` signal; `onClose(returnValue)` on close.
    A click on the backdrop closes it the way the Cancel/Close button does. */
export function Dialog(props: {
  open: boolean;
  onClose: (value: string) => void;
  class?: string;
  children: JSX.Element;
}) {
  let el!: HTMLDialogElement;
  createEffect(() => {
    if (props.open && !el.open) el.showModal();
    else if (!props.open && el.open) el.close("cancel");
  });
  const handler = () => props.onClose(el.returnValue);
  onCleanup(() => el?.removeEventListener("close", handler));
  /** The backdrop is not an element of its own: a click on it lands on the
      <dialog>, so a backdrop click is one whose target is the dialog *and* whose
      point is outside its box -- the padding around the form also hits the
      dialog and must not close it. */
  const onPointer = (e: MouseEvent) => {
    if (e.target !== el) return;
    const r = el.getBoundingClientRect();
    const inside =
      e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
    if (!inside) el.close("cancel");
  };
  /** Every close is routed through `close()`, never the browser's own
      `method="dialog"` submission: Chrome closes the dialog on that submit
      *without firing `close`*, which leaves `open` stuck at true and the dialog
      un-reopenable. `close()` always fires it, so the listener above is the only
      exit -- Esc included. */
  const onSubmit = (e: SubmitEvent) => {
    e.preventDefault();
    el.close((e.submitter as HTMLButtonElement | null)?.value ?? "");
  };
  return (
    <dialog
      ref={(d) => {
        el = d;
        d.addEventListener("close", handler);
      }}
      class={props.class}
      onMouseDown={onPointer}
    >
      <form method="dialog" onSubmit={onSubmit}>
        {props.children}
      </form>
    </dialog>
  );
}
