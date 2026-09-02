import { batch, createEffect, createSignal, on, type Accessor, type Setter } from "solid-js";
import { followLog } from "./api";
import { t } from "./i18n";
import type { Job, JobProgress, JobStatus } from "./types";

/** A signal mirrored into `localStorage` under `key`: read once at creation,
 * written on every change. Values that move on every pointermove of a drag (the
 * dock height, the caption column) save once on pointerup instead.
 */
export function persisted<T>(
  key: string,
  fallback: T,
  parse: (raw: string) => T,
  format: (v: T) => string = String,
): [Accessor<T>, Setter<T>] {
  const raw = localStorage.getItem(key);
  const [get, set] = createSignal<T>(raw === null ? fallback : parse(raw));
  createEffect(on(get, (v) => localStorage.setItem(key, format(v))));
  return [get, set];
}

/** `"0"` is false and anything else (including a missing key's fallback) is true. */
export const asFlag = (raw: string) => raw !== "0";
export const fromFlag = (v: boolean) => (v ? "1" : "0");

/** How many of a job's lines the browser keeps. The server holds 20 000 and
 * replays every one of them when the stream is opened, so this is only how far
 * back the log window scrolls; the job's own file under `workspace/gui_logs/`
 * is the whole of it.
 */
export const LOG_TAIL = 500;

/** `  [done/total] detail` — the one progress line every stage prints
 * (`stages/cli/_args.py::make_progress`). */
const PROGRESS_LINE = /^\s*\[(\d+)\/(\d+)\]\s*(.*)$/;
/** `── step i/n: label ──` — the header `gui/jobs.py` prints in front of each
 * step of a job that has more than one. */
const STEP_LINE = /^──\s*step (\d+)\/(\d+):\s*(.*?)\s*──\s*$/;

/** A job being followed over SSE: its id, whether it is still going, the
 * newest line as a status, the tail of its output and how far it has got. Two
 * exist — the dock's stage runner and the Settings dialog's weights download —
 * sharing the three transitions (close any previous stream, take the new job,
 * clear it on done or error); the callbacks are what differs.
 */
export function createJobFollower(opts: {
  /** The newest line as a status, or null to leave the current one standing. */
  line?: (line: string) => JobStatus | null;
  /** A clean end of stream — the follower has already cleared `running`. */
  done: (job: Job) => void;
}) {
  const [id, setId] = createSignal<string | null>(null);
  const [running, setRunning] = createSignal(false);
  const [status, setStatus] = createSignal<JobStatus>({ text: "" });
  /** The log as the window shows it. Kept whatever the status line does with a
      line: a run that fails says why in the lines the status scrolled past. */
  const [lines, setLines] = createSignal<string[]>([]);
  const [progress, setProgress] = createSignal<JobProgress | null>(null);
  let es: EventSource | null = null;

  /** Read the two counted formats off a line. A stage that prints neither just
      never sets a progress, which the bar reads as "no bar to draw". */
  const count = (line: string): void => {
    const step = STEP_LINE.exec(line);
    if (step) {
      // A step counts its own images from one, so the bar starts over with it
      // rather than jumping backwards out of the previous step's total.
      setProgress({
        done: 0,
        total: 0,
        detail: "",
        step: { index: Number(step[1]), total: Number(step[2]), label: step[3] },
      });
      return;
    }
    const m = PROGRESS_LINE.exec(line);
    // The step is carried over: it is the header this count is running under.
    if (m)
      setProgress((prev) => ({
        done: Number(m[1]),
        total: Number(m[2]),
        detail: m[3],
        step: prev?.step ?? null,
      }));
  };

  const follow = (jobId: string, initial: JobStatus) => {
    es?.close();
    batch(() => {
      setId(jobId);
      setRunning(true);
      setStatus(initial);
      setLines([]);
      setProgress(null);
    });
    es = followLog(
      jobId,
      (line) => {
        const next = opts.line ? opts.line(line) : { text: line, state: "running" };
        if (next) setStatus(next);
        setLines((prev) => [...prev, line].slice(-LOG_TAIL));
        count(line);
      },
      (job) => {
        es = null;
        setRunning(false);
        opts.done(job);
      },
      () => {
        es = null;
        setRunning(false);
        setStatus({ text: t().runner.logClosed });
      },
    );
  };

  return { id, running, status, setStatus, lines, progress, follow, close: () => es?.close() };
}
