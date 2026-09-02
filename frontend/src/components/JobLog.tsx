import { createEffect, on, Show } from "solid-js";
import { t } from "../i18n";
import { LOG_TAIL } from "../state";
import type { JobProgress, JobStatus } from "../types";
import { Dialog } from "./Dialog";
import { StatusLine } from "./StatusLine";

/** How far the count has got, as a percentage of its step. */
const pct = (p: JobProgress) => Math.min(100, (p.done / p.total) * 100);
/** A step between its counter lines has no total yet, and neither has a stage
    still loading its model: the bar sweeps rather than sitting at zero, which
    would read as stuck. */
const known = (p: JobProgress | null) => !!p && p.total > 0;

/** The bar across the bottom of the window: the one thing the page says about
    the job it is following, and the button that opens the rest of it.

    It sits under everything rather than in a panel, so it reads the same
    whichever stage is open and whether the dock is folded or not — the run bar
    above it is three buttons and nothing else, and the dock's strip stays a
    picker. The state is the *colour of the line*, not a chip: a failed or
    cancelled job turns it red, and everything else is the same dim as the text
    beside it.

    All of it lives off the log, because that is all a stage subprocess tells
    anyone: `state.ts` reads the count out of it. A run that prints no counter
    still gets its text and its window; the fill is simply not drawn. */
export function JobBar(props: {
  /** The newest line, an error that stopped a start, or what a finished run
      changed — whatever the runner last had to say. */
  status: JobStatus;
  progress: JobProgress | null;
  running: boolean;
  /** A job has been followed this session, so there are lines to show. It
      outlives the run on purpose: reading why one failed is the other half of
      what the window is for. */
  hasLog: boolean;
  onLog: () => void;
}) {
  /** Only while it runs: the last count of a job that ended is not news, and a
      bar left sitting at full reads as one that stalled there. */
  const cur = () => (props.running ? props.progress : null);
  /** What the line has no room for: the step the count belongs to, the image it
      is on, and the whole of a status the bar has had to clip. */
  const hint = () => {
    const p = cur();
    const s = p?.step;
    return [s ? t().job.step(s.index, s.total, s.label) : "", p?.detail, props.status.text]
      .filter(Boolean)
      .join(" · ");
  };

  // Nothing to say and nothing to open: no bar at all, rather than an empty
  // strip along the bottom of a page where no job has ever run.
  return (
    <Show when={props.status.text || props.running || props.hasLog}>
      {/* The title carries what the line has no room for: which step is
          counting, and the image the count is on. */}
      <div class="jobbar" title={hint()}>
        <span
          classList={{
            status: true,
            err: props.status.state === "failed" || props.status.state === "cancelled",
          }}
        >
          {props.status.text}
        </span>
        <Show when={known(cur()) ? cur() : null}>
          {(p) => <span class="num">{t().job.of(p().done, p().total)}</span>}
        </Show>
        <Show when={props.hasLog}>
          <button class="link" title={t().job.logHint} onClick={props.onLog}>
            {t().job.log}
          </button>
        </Show>
        {/* The fill is the bar: the window's own bottom edge, so a long batch
            reports from under whatever is on screen. */}
        <Show when={cur()}>
          {(p) => (
            <i
              classList={{ fill: true, wait: !known(p()) }}
              style={{ width: known(p()) ? `${pct(p())}%` : undefined }}
            />
          )}
        </Show>
      </div>
    </Show>
  );
}

/** The followed job's output, as the modal the bottom bar's `log` button opens.
    The stream is already open and its lines already kept, so this window only
    draws them — there is no fetch here and none needed: a job that ended still
    has its tail, which is where a failure says what went wrong. */
export function JobLog(props: {
  open: boolean;
  jobId: string | null;
  lines: string[];
  status: JobStatus;
  progress: JobProgress | null;
  running: boolean;
  onCancel: () => void;
  onClose: () => void;
}) {
  let box: HTMLDivElement | undefined;
  /** Follow the tail, unless the reader has scrolled up off it — a window that
      yanks itself back down mid-read is worse than one that stops following. */
  let stick = true;
  const onScroll = () => {
    if (box) stick = box.scrollTop + box.clientHeight >= box.scrollHeight - 24;
  };
  createEffect(
    on(
      () => [props.open, props.lines.length] as const,
      () => {
        if (!props.open || !stick) return;
        // Deferred past this batch: the line that triggered the effect is in
        // the DOM by now, but a re-open runs before its <pre> has been laid
        // out, and scrolling to a height of zero is not scrolling to the end.
        queueMicrotask(() => box?.scrollTo({ top: box.scrollHeight }));
      },
    ),
  );

  return (
    <Dialog open={props.open} class="joblog" onClose={() => props.onClose()}>
      <h3 class="dlgh">
        {t().job.title}
        <span class="mono dim">{props.jobId}</span>
        <span class="sp" />
        <button value="cancel" class="dlgx" title={t().common.close} aria-label={t().common.close}>
          ×
        </button>
      </h3>
      {/* Mounted only while open: the tail is re-joined on every line that
          arrives, and a shut window has no reason to pay for it. */}
      <Show when={props.open}>
        <div class="jobhead">
          <StatusLine status={props.status} />
          <Show when={props.running && known(props.progress) ? props.progress : null}>
            {(p) => <span class="num">{t().job.of(p().done, p().total)}</span>}
          </Show>
        </div>
        <Show when={props.lines.length >= LOG_TAIL}>
          <p class="dim" style="margin:0 0 4px">
            {t().job.tail(LOG_TAIL)}
          </p>
        </Show>
        <div class="logbox" ref={box} onScroll={onScroll}>
          <pre class="log">{props.lines.join("\n") || t().job.empty}</pre>
        </div>
        <div class="dlg-actions">
          <Show when={props.running}>
            {/* `type=button`: every other button in a dialog submits its form
                and closes it, and a cancel leaves the window open on the lines
                the kill prints. */}
            <button type="button" onClick={props.onCancel}>
              {t().stage.cancel}
            </button>
          </Show>
          <button value="ok" class="primary">
            {t().common.close}
          </button>
        </div>
      </Show>
    </Dialog>
  );
}
