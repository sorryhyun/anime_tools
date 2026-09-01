import { batch, createEffect, createSignal, on, type Accessor, type Setter } from "solid-js";
import { followLog } from "./api";
import { t } from "./i18n";
import type { Job, JobStatus } from "./types";

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

/** A job being followed over SSE: its id, whether it is still going, and the
 * newest line as a status. Two exist — the dock's stage runner and the Settings
 * dialog's weights download — sharing the three transitions (close any previous
 * stream, take the new job, clear it on done or error); the callbacks are what
 * differs.
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
  let es: EventSource | null = null;

  const follow = (jobId: string, initial: JobStatus) => {
    es?.close();
    batch(() => {
      setId(jobId);
      setRunning(true);
      setStatus(initial);
    });
    es = followLog(
      jobId,
      (line) => {
        const next = opts.line ? opts.line(line) : { text: line, state: "running" };
        if (next) setStatus(next);
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

  return { id, running, status, setStatus, follow, close: () => es?.close() };
}
