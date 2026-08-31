import { createEffect, type JSX, onCleanup } from "solid-js";

/** Native <dialog> driven by an `open` signal; `onClose(returnValue)` on close. */
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
  return (
    <dialog
      ref={(d) => {
        el = d;
        d.addEventListener("close", handler);
      }}
      class={props.class}
    >
      <form method="dialog">{props.children}</form>
    </dialog>
  );
}
