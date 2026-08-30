import { createMemo, createSignal, For, Show, Switch, Match } from "solid-js";
import type { Field, Stage, Values } from "../types";
import { PathPicker } from "./PathPicker";

/** Group fields by argparse group, preserving order. */
function grouped(fields: Field[]): [string, Field[]][] {
  const m = new Map<string, Field[]>();
  for (const f of fields) {
    if (f.dest === "apply") continue;
    const g = m.get(f.group) ?? [];
    g.push(f);
    m.set(f.group, g);
  }
  return [...m];
}

const str = (v: unknown) => (v == null ? "" : Array.isArray(v) ? v.join("\n") : String(v));

/** Schema-driven form; `values` is the controlled state ({dest: value}). */
export function StageForm(props: {
  stage: Stage;
  values: Values;
  setValue: (dest: string, v: unknown) => void;
  reset: () => void;
}) {
  const [picking, setPicking] = createSignal<string | null>(null);
  const groups = createMemo(() => grouped(props.stage.fields));
  const value = (f: Field) => props.values[f.dest] ?? f.default;
  const dirty = (f: Field) => {
    const v = props.values[f.dest];
    return v !== undefined && str(v) !== str(f.default) && !(f.kind === "bool" && !!v === !!f.default);
  };

  return (
    <>
      <div class="doc">{props.stage.doc.trim()}</div>
      <Show when={props.stage.notes}>
        <div class="notes">⚠ {props.stage.notes}</div>
      </Show>
      <For each={groups()}>
        {([g, fs]) => (
          <fieldset>
            <legend>{g || "options"}</legend>
            <For each={fs}>
              {(f) => (
                <div class="row">
                  <label classList={{ req: f.required }} title={f.help}>
                    {f.label || f.flags[0] || f.dest}
                  </label>
                  <Switch>
                    <Match when={f.kind === "bool"}>
                      <input
                        type="checkbox"
                        checked={!!value(f)}
                        onChange={(e) => props.setValue(f.dest, e.currentTarget.checked)}
                      />
                    </Match>
                    <Match when={f.kind === "enum"}>
                      <select
                        classList={{ dirty: dirty(f) }}
                        value={str(value(f))}
                        onChange={(e) => props.setValue(f.dest, e.currentTarget.value)}
                      >
                        <For each={f.choices ?? []}>{(c) => <option value={String(c)}>{String(c)}</option>}</For>
                      </select>
                    </Match>
                    <Match when={f.kind === "list"}>
                      <textarea
                        classList={{ dirty: dirty(f) }}
                        placeholder="one per line"
                        value={str(value(f))}
                        onInput={(e) =>
                          props.setValue(
                            f.dest,
                            e.currentTarget.value.split("\n").map((s) => s.trim()).filter(Boolean),
                          )
                        }
                      />
                    </Match>
                    <Match when={f.kind === "int" || f.kind === "float"}>
                      <input
                        type="number"
                        classList={{ dirty: dirty(f) }}
                        step={f.kind === "int" ? 1 : "any"}
                        value={str(value(f))}
                        onInput={(e) => props.setValue(f.dest, e.currentTarget.value)}
                      />
                    </Match>
                    <Match when={f.path}>
                      <div class="pathrow">
                        <input
                          type="text"
                          classList={{ dirty: dirty(f) }}
                          value={str(value(f))}
                          onInput={(e) => props.setValue(f.dest, e.currentTarget.value)}
                        />
                        <button type="button" title="Browse (home-relative)" onClick={() => setPicking(f.dest)}>
                          …
                        </button>
                      </div>
                    </Match>
                    <Match when={true}>
                      <input
                        type="text"
                        classList={{ dirty: dirty(f) }}
                        value={str(value(f))}
                        onInput={(e) => props.setValue(f.dest, e.currentTarget.value)}
                      />
                    </Match>
                  </Switch>
                </div>
              )}
            </For>
          </fieldset>
        )}
      </For>
      <div style="margin-bottom:8px">
        <button class="link" type="button" onClick={props.reset}>
          reset to defaults
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
