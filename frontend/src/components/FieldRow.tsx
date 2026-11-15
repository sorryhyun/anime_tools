import { For, Match, Switch } from "solid-js";
import { t } from "../i18n";
import type { Field } from "../types";
import { MaskList } from "./MaskList";

/** Group the fields the form draws by argparse group, preserving order.
    Which ones it draws is the server's answer (`Field.shown`): a field bound to
    a root or a Settings value is filled server-side, `--device` is
    auto-detected, and the run bar owns `--apply` / `--from_report`. */
export function grouped(fields: Field[]): [string, Field[]][] {
  const m = new Map<string, Field[]>();
  for (const f of fields) {
    if (!f.shown) continue;
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
    <div class="row" classList={{ wide: f().kind === "masks" || f().kind === "multi" }}>
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
        {/* Any subset of the choices, one chip per choice. The value stays in
            the choices' order whatever order they were clicked in, so the same
            pick is the same argv. A chip wears `g-<choice>`, which is only a
            colour where the choices are tag groups. */}
        <Match when={f().kind === "multi"}>
          <div class="chips" classList={cls()}>
            <For each={(f().choices ?? []).map(String)}>
              {(c) => {
                const picked = () => (Array.isArray(props.value) ? props.value.map(String) : []);
                // A button, not a label around a checkbox: `.row label` is the
                // field's name and styles every label in the row as one.
                return (
                  <button
                    type="button"
                    classList={{ chip: true, [`g-${c}`]: true, on: picked().includes(c) }}
                    aria-pressed={picked().includes(c)}
                    onClick={() => {
                      const next = new Set(picked());
                      if (next.has(c)) next.delete(c);
                      else next.add(c);
                      props.setValue((f().choices ?? []).map(String).filter((x) => next.has(x)));
                    }}
                  >
                    <span class="dot" />
                    {c}
                  </button>
                );
              }}
            </For>
          </div>
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
