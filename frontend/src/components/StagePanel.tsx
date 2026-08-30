import { createMemo, For, Show } from "solid-js";
import { StageForm } from "./StageForm";
import type { DatasetRoots, Stage, Values } from "../types";

/** The stage runner, now a dock panel: the sidebar belongs to the dataset, so
    the stage picker is a grouped <select> rather than a nav list. */
export function StagePanel(props: {
  stages?: Stage[];
  error?: unknown;
  curId: string;
  setCurId: (id: string) => void;
  values: Values;
  setValue: (dest: string, v: unknown) => void;
  reset: () => void;
  busy: boolean;
  status: { text: string; state?: string };
  onRun: (apply: boolean) => void;
  onApply: () => void;
  onCancel: () => void;
  roots?: DatasetRoots;
  onSettings: () => void;
}) {
  const groups = createMemo(() => {
    const m = new Map<string, Stage[]>();
    for (const s of props.stages ?? []) m.set(s.group, [...(m.get(s.group) ?? []), s]);
    return [...m];
  });
  const cur = createMemo(() => props.stages?.find((s) => s.id === props.curId));

  return (
    <div class="stagepanel">
      <Show when={!props.error} fallback={<div class="err pad">{String(props.error)}</div>}>
        <div class="stagebar">
          <select value={props.curId} onChange={(e) => props.setCurId(e.currentTarget.value)}>
            <For each={groups()}>
              {([g, ss]) => (
                <optgroup label={g}>
                  <For each={ss}>
                    {(s) => (
                      <option value={s.id} disabled={!s.available}>
                        {s.title}
                        {s.available ? "" : " (unavailable)"}
                      </option>
                    )}
                  </For>
                </optgroup>
              )}
            </For>
          </select>
          <span class="dim mono">{cur()?.module}</span>
          <span class="sp" />
          <button classList={{ primary: !cur()?.apply }} disabled={props.busy || !cur()?.available} onClick={() => props.onRun(false)}>
            {cur()?.apply ? "Dry run" : "Run"}
          </button>
          <Show when={cur()?.apply}>
            <button class="primary" disabled={props.busy || !cur()?.available} onClick={props.onApply}>
              Apply…
            </button>
          </Show>
          <button disabled={!props.busy} onClick={props.onCancel}>
            Cancel
          </button>
          <span class="status">
            <Show when={props.status.state}>
              <span class={`badge ${props.status.state}`}>{props.status.state}</span>{" "}
            </Show>
            {props.status.text}
          </span>
        </div>

        <div class="stageform">
          <Show when={cur()} fallback={<div class="dim pad">No stages.</div>}>
            {(s) => (
              <Show
                when={s().available}
                fallback={
                  <div class="doc">
                    {s().title} is unavailable: {s().error}
                    {"\n\nReinstall:  uv tool install --force \"anime-tools @ git+https://github.com/sorryhyun/anime_tools\""}
                  </div>
                }
              >
                <StageForm
                  stage={s()}
                  values={props.values}
                  setValue={props.setValue}
                  reset={props.reset}
                  roots={props.roots}
                  onSettings={props.onSettings}
                />
              </Show>
            )}
          </Show>
        </div>
      </Show>
    </div>
  );
}
