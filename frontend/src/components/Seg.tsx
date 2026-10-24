import { For } from "solid-js";

/** A segmented control over a closed set of choices: every choice is on screen
    at once, so the control says what it is set to without opening a select. */
export function Seg<T extends string>(props: {
  options: readonly T[];
  value: T;
  label: (v: T) => string;
  title: string;
  onPick: (v: T) => void;
}) {
  return (
    <span class="seg" title={props.title}>
      <For each={props.options}>
        {(o) => (
          <button
            type="button"
            classList={{ sel: o === props.value }}
            onClick={() => props.onPick(o)}
          >
            {props.label(o)}
          </button>
        )}
      </For>
    </span>
  );
}
