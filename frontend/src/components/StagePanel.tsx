import { createMemo, Show } from "solid-js";
import { StageForm } from "./StageForm";
import type { DatasetRoots, Stage, Values } from "../types";

/** The stage runner's body: the run bar + the schema-driven form for the stage
    picked in the dock's button strip (the picker itself lives in App). */
export function StagePanel(props: {
  stages?: Stage[];
  error?: unknown;
  curId: string;
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
  const cur = createMemo(() => props.stages?.find((s) => s.id === props.curId));

  return (
    <div class="stagepanel">
      <Show when={!props.error} fallback={<div class="err pad">{String(props.error)}</div>}>
        <div class="stagebar">
          <b>{cur()?.title}</b>
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
          <span class="status" title={props.status.text}>
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
