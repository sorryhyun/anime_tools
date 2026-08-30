import { createMemo, Show } from "solid-js";
import { StageForm } from "./StageForm";
import { HelpToggle } from "./HelpToggle";
import type { DatasetRoots, Stage, Values } from "../types";

/** The stage runner's body: the run bar + the schema-driven form for the stage
    picked in the dock's button strip (the picker itself lives in App).

    A stage that takes a `--path_pattern` runs at one of two scopes, and the run
    bar spells both out rather than hiding a mode behind a toggle:

    * the plain button acts on the **selected image** -- the server narrows the
      pattern to that one file, so the iteration loop is pick an image, run,
      look at the caption;
    * the `batch` button acts on the **Settings `path_pattern`** -- everything
      the pattern names.

    Dry run stays batch on purpose: it is the pass that writes the report both
    Applies replay, so scoping it to one image would leave the batch Apply with
    nothing to reuse. */
export function StagePanel(props: {
  stages?: Stage[];
  error?: unknown;
  curId: string;
  values: Values;
  setValue: (dest: string, v: unknown) => void;
  reset: () => void;
  busy: boolean;
  status: { text: string; state?: string };
  /** The selected image, or null -- what the per-image buttons act on. */
  rel?: string | null;
  onRun: (apply: boolean, rel?: string | null) => void;
  onApply: (rel: string | null) => void;
  onCancel: () => void;
  roots?: DatasetRoots;
  defaults?: Record<string, string>;
  onSettings: () => void;
  help: boolean;
  onHelp: () => void;
}) {
  const cur = createMemo(() => props.stages?.find((s) => s.id === props.curId));
  /** Per-image buttons need a stage that takes a pattern and an image to aim at. */
  const scoped = createMemo(() => !!cur()?.scoped);
  const noImage = createMemo(() => !props.rel);
  const off = createMemo(() => props.busy || !cur()?.available);
  const aim = () =>
    props.rel ? `just ${props.rel}` : "select an image in the sidebar first";
  const batchTitle = () => `every image the Settings path_pattern names`;

  return (
    <div class="stagepanel">
      <Show when={!props.error} fallback={<div class="err pad">{String(props.error)}</div>}>
        <div class="stagebar">
          <b>{cur()?.title}</b>
          <span class="dim mono">{cur()?.module}</span>
          <HelpToggle open={props.help} warn={!!cur()?.notes} onToggle={props.onHelp} />
          <span class="sp" />
          <Show
            when={cur()?.apply}
            fallback={
              <>
                <Show when={scoped()}>
                  <button class="primary" disabled={off() || noImage()} title={aim()} onClick={() => props.onRun(false, props.rel)}>
                    Run
                  </button>
                </Show>
                <button classList={{ primary: !scoped() }} disabled={off()} title={scoped() ? batchTitle() : undefined} onClick={() => props.onRun(false)}>
                  {scoped() ? "Run batch" : "Run"}
                </button>
              </>
            }
          >
            <button disabled={off()} title={batchTitle()} onClick={() => props.onRun(false)}>
              Dry run
            </button>
            <Show when={scoped()}>
              <button class="primary" disabled={off() || noImage()} title={aim()} onClick={() => props.onApply(props.rel ?? null)}>
                Apply
              </button>
            </Show>
            <button classList={{ primary: !scoped() }} disabled={off()} title={scoped() ? batchTitle() : undefined} onClick={() => props.onApply(null)}>
              {scoped() ? "Apply batch…" : "Apply…"}
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
                  defaults={props.defaults}
                  onSettings={props.onSettings}
                  help={props.help}
                  onHelp={props.onHelp}
                />
              </Show>
            )}
          </Show>
        </div>
      </Show>
    </div>
  );
}
