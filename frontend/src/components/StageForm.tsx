import { createMemo, createSignal, For, Show } from "solid-js";
import { t } from "../i18n";
import type { Field, Stage, Values } from "../types";
import { FieldRow, grouped, str } from "./FieldRow";
import { browsePath, PathPicker } from "./PathPicker";

/** One argparse group as a `<fieldset>`, and the fold over its own advanced
    fields. The fold is per group and sits on that group's bottom edge, and which
    fields it covers is per *field* — the server marks them, so a group can hold
    both retuned knobs and folded sweep parameters. Local, unsaved state. */
function FieldGroup(props: {
  title: string;
  fields: Field[];
  value: (f: Field) => unknown;
  dirty: (f: Field) => boolean;
  setValue: (dest: string, v: unknown) => void;
  onPick: (f: Field) => void;
}) {
  const [adv, setAdv] = createSignal(false);
  /* A gated group is a drawer: its checkbox moves up into the legend and the
     knobs it governs are folded away while it is off. The gate is the field
     naming itself, which is how the server says "this one is the switch"
     without the browser knowing any flag by name. */
  const gate = createMemo(() => props.fields.find((f) => f.gate === f.dest));
  const body = createMemo(() => props.fields.filter((f) => f !== gate()));
  const open = () => !gate() || !!props.value(gate()!);
  const shown = createMemo(() => body().filter((f) => adv() || !f.advanced));
  const folded = createMemo(() => body().filter((f) => f.advanced));
  /** A folded field that is no longer at its default still reaches the argv, so
      the fold badges it. */
  const foldedDirty = createMemo(() => (adv() ? 0 : folded().filter(props.dirty).length));

  return (
    <fieldset>
      <legend>
        <Show when={gate()} fallback={props.title || t().form.options}>
          {(gf) => (
            <label class="gate" title={gf().help}>
              <input
                type="checkbox"
                checked={!!props.value(gf())}
                onChange={(e) => props.setValue(gf().dest, e.currentTarget.checked)}
              />
              {props.title || gf().label}
            </label>
          )}
        </Show>
      </legend>
      {/* Two-up: the dock is wide and short. Same wrapper the Settings
          preflight block uses. */}
      <Show when={open()}>
        <div class="twoup">
          <For each={shown()}>
            {(f) => (
              <FieldRow
                field={f}
                value={props.value(f)}
                dirty={props.dirty(f)}
                setValue={(v) => props.setValue(f.dest, v)}
                onPick={() => props.onPick(f)}
              />
            )}
          </For>
        </div>
        {/* Only while the drawer is open: a fold hanging off a shut group would
            be a second switch for the same knobs. */}
        <Show when={folded().length}>
          <button
            class="advfold"
            classList={{ warn: !!foldedDirty() }}
            type="button"
            title={foldedDirty() ? t().form.advancedDirty(foldedDirty()) : t().form.advancedHint}
            onClick={() => setAdv(!adv())}
          >
            {adv() ? t().form.advancedHide : t().form.advanced(folded().length)}
          </button>
        </Show>
      </Show>
    </fieldset>
  );
}

/** Schema-driven form; `values` is the controlled state ({dest: value}). */
export function StageForm(props: {
  stage: Stage;
  values: Values;
  setValue: (dest: string, v: unknown) => void;
  reset: () => void;
  help: boolean;
}) {
  /** Which field's fallback browser is open. Only a host with no chooser of its
      own gets here -- see `browsePath`. */
  const [picking, setPicking] = createSignal<string | null>(null);
  const groups = createMemo(() => grouped(props.stage.fields));
  const value = (f: Field) => props.values[f.dest] ?? f.default;
  const dirty = (f: Field) => {
    const v = props.values[f.dest];
    return (
      v !== undefined && str(v) !== str(f.default) && !(f.kind === "bool" && !!v === !!f.default)
    );
  };

  return (
    <>
      {/* The prose lives behind the stage bar's (?), and the notes go with it;
          the (?) is warn-tinted while it hides a stage's notes. */}
      <Show when={props.help}>
        <div class="doc">{props.stage.doc.trim()}</div>
        <Show when={props.stage.notes}>
          <div class="notes">⚠ {props.stage.notes}</div>
        </Show>
      </Show>
      <For each={groups()}>
        {([g, fs]) => (
          <FieldGroup
            title={g}
            fields={fs}
            value={value}
            dirty={dirty}
            setValue={props.setValue}
            onPick={(f) =>
              void browsePath(
                f.path_kind,
                str(value(f)),
                (path) => props.setValue(f.dest, path),
                () => setPicking(f.dest),
              )
            }
          />
        )}
      </For>
      <div class="formfoot">
        <button class="link" type="button" onClick={props.reset}>
          {t().form.reset}
        </button>
      </div>
      <PathPicker
        open={picking() !== null}
        onClose={(p) => {
          const dest = picking();
          if (p !== null && dest) props.setValue(dest, p);
          setPicking(null);
        }}
      />
    </>
  );
}
