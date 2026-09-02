import { createMemo, createSignal, For, Show, Switch, Match } from "solid-js";
import { t } from "../i18n";
import { REPLAY_FIELD } from "../types";
import type { Field, Stage, Values } from "../types";
import { browsePath, PathPicker } from "./PathPicker";

/** Group fields by argparse group, preserving order. Left out: fields bound to
    a dataset root, a Settings stage default or the Settings `report_root` /
    `mask_root` (the server fills them); `--device`, which the stage
    auto-detects; and the run bar's own `--apply` / `--from_report`, since a
    stale path in a form field would turn the next Run into a replay.

    An `overridable` field is the one bound kind that stays: its `default`
    arrives already resolved from Settings and typing over it wins for that
    run. */
export function grouped(fields: Field[]): [string, Field[]][] {
  const m = new Map<string, Field[]>();
  for (const f of fields) {
    if (f.dest === "apply" || f.dest === REPLAY_FIELD) continue;
    if ((f.root || f.setting || f.report || f.mask || f.auto) && !f.overridable) continue;
    const g = m.get(f.group) ?? [];
    g.push(f);
    m.set(f.group, g);
  }
  return [...m];
}

const str = (v: unknown) => (v == null ? "" : Array.isArray(v) ? v.join("\n") : String(v));

/** One labelled input for one argparse action. Shared by the stage forms and
    the Settings dialog's Preprocess block. */
export function FieldRow(props: {
  field: Field;
  value: unknown;
  dirty?: boolean;
  setValue: (v: unknown) => void;
  /** Show the browse button on a path field; omit to render it as plain text. */
  onPick?: () => void;
}) {
  const f = () => props.field;
  const cls = () => ({ dirty: !!props.dirty });
  return (
    <div class="row">
      <label classList={{ req: f().required }} title={f().help}>
        {f().label || f().flags[0] || f().dest}
      </label>
      <Switch>
        <Match when={f().kind === "bool"}>
          <input
            type="checkbox"
            checked={!!props.value}
            onChange={(e) => props.setValue(e.currentTarget.checked)}
          />
        </Match>
        <Match when={f().kind === "enum"}>
          <select
            classList={cls()}
            value={str(props.value)}
            onChange={(e) => props.setValue(e.currentTarget.value)}
          >
            <For each={f().choices ?? []}>
              {(c) => <option value={String(c)}>{String(c)}</option>}
            </For>
          </select>
        </Match>
        <Match when={f().kind === "list"}>
          <textarea
            classList={cls()}
            placeholder={t().form.onePerLine}
            value={str(props.value)}
            onInput={(e) =>
              props.setValue(
                e.currentTarget.value
                  .split("\n")
                  .map((s) => s.trim())
                  .filter(Boolean),
              )
            }
          />
        </Match>
        <Match when={f().kind === "int" || f().kind === "float"}>
          <input
            type="number"
            classList={cls()}
            step={f().kind === "int" ? 1 : "any"}
            value={str(props.value)}
            onInput={(e) => props.setValue(e.currentTarget.value)}
          />
        </Match>
        <Match when={f().path && !!props.onPick}>
          <div class="pathrow">
            <input
              type="text"
              classList={cls()}
              value={str(props.value)}
              onInput={(e) => props.setValue(e.currentTarget.value)}
            />
            <button type="button" title={t().common.browse} onClick={props.onPick}>
              …
            </button>
          </div>
        </Match>
        <Match when={true}>
          <input
            type="text"
            classList={cls()}
            value={str(props.value)}
            onInput={(e) => props.setValue(e.currentTarget.value)}
          />
        </Match>
      </Switch>
    </div>
  );
}

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
