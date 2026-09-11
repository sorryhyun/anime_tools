import { For, Match, Switch } from "solid-js";
import { t } from "../i18n";
import { REPLAY_FIELD } from "../types";
import type { Field } from "../types";
import { MaskList } from "./MaskList";

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

/** A field value as the text its input shows: a list is one item per line. */
export const str = (v: unknown) => (v == null ? "" : Array.isArray(v) ? v.join("\n") : String(v));

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
    <div class="row" classList={{ wide: f().kind === "masks" }}>
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
        <Match when={f().kind === "masks"}>
          <MaskList value={props.value} dirty={props.dirty} setValue={props.setValue} />
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
