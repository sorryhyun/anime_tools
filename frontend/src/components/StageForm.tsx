import { createMemo, createSignal, For, Show, Switch, Match } from "solid-js";
import { REPLAY_FIELD } from "../types";
import type { Field, Stage, Values } from "../types";
import { PathPicker } from "./PathPicker";

/** Group fields by argparse group, preserving order. Fields bound to a dataset
    root, to a Settings stage default (`path_pattern`, `tagger_dir`,
    `checkpoint`, `prompt_embed`) or to the Settings `report_root`
    (`--report_dir`, groups' `--out`, and the audit report the curated apply
    reads) are left out: the server fills them from Settings, so no two forms
    can disagree about them. So is `--device`, which the stage auto-detects. And
    so are the two the run bar owns -- `--apply` (Run / Apply) and
    `--from_report` (how Apply replays the run it is applying): a stale path
    left in a form field would quietly turn the next Run into a replay of an
    old one. */
export function grouped(fields: Field[]): [string, Field[]][] {
  const m = new Map<string, Field[]>();
  for (const f of fields) {
    if (f.dest === "apply" || f.dest === REPLAY_FIELD) continue;
    if (f.root || f.setting || f.report || f.auto) continue;
    const g = m.get(f.group) ?? [];
    g.push(f);
    m.set(f.group, g);
  }
  return [...m];
}

const str = (v: unknown) => (v == null ? "" : Array.isArray(v) ? v.join("\n") : String(v));

/** One labelled input for one argparse action. Shared by the stage forms and
    the Settings dialog's Preprocess block, so a hidden stage's knobs are typed
    into exactly the same widgets as a visible one's. */
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
            placeholder="one per line"
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
            <button type="button" title="Browse (home-relative)" onClick={props.onPick}>
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

/** Schema-driven form; `values` is the controlled state ({dest: value}). */
export function StageForm(props: {
  stage: Stage;
  values: Values;
  setValue: (dest: string, v: unknown) => void;
  reset: () => void;
  help: boolean;
  onHelp: () => void;
}) {
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
      {/* The prose lives behind the stage bar's (?), and the notes go with it.
          Nothing is lost by collapsing them: the (?) is warn-tinted while it
          hides a stage's notes, which is the signal. Echoing them on the form
          as well spent a row above every field of every stage. */}
      <Show when={props.help}>
        <div class="doc">{props.stage.doc.trim()}</div>
        <Show when={props.stage.notes}>
          <div class="notes">⚠ {props.stage.notes}</div>
        </Show>
      </Show>
      <For each={groups()}>
        {([g, fs]) => (
          <fieldset>
            <legend>{g || "options"}</legend>
            {/* Two-up: the dock is wide and short, so a single column of knobs
                scrolled far more than it had to. Same wrapper the Settings
                preflight block uses. */}
            <div class="twoup">
              <For each={fs}>
                {(f) => (
                  <FieldRow
                    field={f}
                    value={value(f)}
                    dirty={dirty(f)}
                    setValue={(v) => props.setValue(f.dest, v)}
                    onPick={() => setPicking(f.dest)}
                  />
                )}
              </For>
            </div>
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
