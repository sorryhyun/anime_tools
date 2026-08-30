import { createMemo, createSignal, For, Show, Switch, Match } from "solid-js";
import { REPLAY_FIELD } from "../types";
import type { DatasetRoots, Field, Stage, Values } from "../types";
import { PathPicker } from "./PathPicker";

/** Group fields by argparse group, preserving order. Fields bound to a dataset
    root or to a Settings stage default (`path_pattern`, `tagger_dir`) are left
    out: the server fills them from Settings, so no two forms can disagree about
    them. So is `--device`, which the stage auto-detects. And so are the two the
    buttons own -- `--apply` (Dry run / Apply) and `--from_report` (the Apply
    dialog's reuse box): a stale path left in a form field would quietly turn
    the next Dry run into a replay of an old one. */
function grouped(fields: Field[]): [string, Field[]][] {
  const m = new Map<string, Field[]>();
  for (const f of fields) {
    if (f.dest === "apply" || f.dest === REPLAY_FIELD || f.root || f.setting || f.auto) continue;
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
  roots?: DatasetRoots;
  /** The Settings stage defaults (`path_pattern` / `tagger_dir`), for the strip. */
  defaults?: Record<string, string>;
  onSettings: () => void;
  help: boolean;
  onHelp: () => void;
}) {
  const [picking, setPicking] = createSignal<string | null>(null);
  const groups = createMemo(() => grouped(props.stage.fields));
  const bound = createMemo(() => props.stage.fields.filter((f) => f.root || f.setting));
  /** What the server will actually send for a Settings-bound field: the saved
      value, or -- when it is blank -- the CLI's own default. */
  const settingValue = (f: Field) =>
    (props.defaults?.[f.setting!] ?? "").trim() || (f.default == null ? "—" : String(f.default));
  const value = (f: Field) => props.values[f.dest] ?? f.default;
  const dirty = (f: Field) => {
    const v = props.values[f.dest];
    return v !== undefined && str(v) !== str(f.default) && !(f.kind === "bool" && !!v === !!f.default);
  };

  return (
    <>
      {/* The prose lives behind the stage bar's (?). Collapsed, the notes stay
          as a one-line chip -- they carry the destructive-mode warnings, so
          they never disappear entirely; the full text is the tooltip, and
          clicking the chip opens the doc. */}
      <Show
        when={props.help}
        fallback={
          <Show when={props.stage.notes}>
            <button
              type="button"
              classList={{ notes: true, chip: true }}
              title={props.stage.notes}
              onClick={props.onHelp}
            >
              ⚠ {props.stage.notes}
            </button>
          </Show>
        }
      >
        <div class="doc">{props.stage.doc.trim()}</div>
        <Show when={props.stage.notes}>
          <div class="notes">⚠ {props.stage.notes}</div>
        </Show>
      </Show>
      <Show when={bound().length}>
        <div class="rootstrip">
          <For each={bound()}>
            {(f) => {
              const info = () => (f.root ? props.roots?.roots[f.root] : undefined);
              return (
                <span title={f.help}>
                  <span class="mono">{f.label || f.flags[0] || f.dest}</span>{" "}
                  <span classList={{ mono: true, err: info() ? !info()!.exists : false }}>
                    {f.root ? (info()?.path ?? "…") : settingValue(f)}
                  </span>
                </span>
              );
            }}
          </For>
          <button class="link" type="button" onClick={props.onSettings}>
            set in Settings ⚙
          </button>
        </div>
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
