import { createResource, createSignal, onCleanup } from "solid-js";
import { api, toStatus } from "./api";
import { t } from "./i18n";
import { createJobFollower } from "./state";
import type { Job, UpdateInfo } from "./types";
import type { Config } from "./config";

/** `/api/update/run` names its job `update:<tag>`, the way a weights fetch is
    `download:<ids>` — the prefix is the only thing that tells an adopted job
    apart from a stage run. */
const UPDATE_STAGE = "update:";

export const isUpdateJob = (job: Job) => job.stage.startsWith(UPDATE_STAGE);

/** The Update pane's state: what the server says about installed-vs-latest, and
 * the upgrade job when one is running.
 *
 * The check is the server's to rate-limit — it caches GitHub's answer for six
 * hours and only goes out when the checkbox allows it — so the page asks once
 * on load and again on "Check now" (`force`), and never on a timer of its own.
 */
export function createUpdates(config: Config) {
  const [info, { mutate, refetch }] = createResource<UpdateInfo>(() => api.update());
  /** A forced check in flight; the resource's own `loading` covers the first. */
  const [checking, setChecking] = createSignal(false);
  /** An upgrade finished in this session: everything the page does from here on
      is still the old environment, so the pane says so until it is reloaded. */
  const [restart, setRestart] = createSignal(false);

  const up$ = createJobFollower({
    // uv writes its resolution over several lines and its own progress with
    // \r; the newest non-empty one is the status, as in the download pane.
    line: (line) => (line.trim() ? { text: line.trim(), state: "running" } : null),
    done: (job) => {
      const ok = job.state === "done";
      if (ok) setRestart(true);
      up$.setStatus({
        text: ok ? t().update.installedOk : t().runner.exit(job.exit_code),
        state: job.state,
      });
      // The version row is read from the running process, so it does not move
      // until the restart -- but the token/job fields on /api/info do.
      void config.refetchInfo();
    },
  });
  onCleanup(() => up$.close());

  /** Ask GitHub now, past the server's six-hour cache. */
  async function check() {
    setChecking(true);
    try {
      mutate(await api.update(true));
    } catch (e) {
      up$.setStatus(toStatus(e));
    } finally {
      setChecking(false);
    }
  }

  /** The checkbox. Turning it on with nothing cached checks straight away, so
      the row is not left blank until the next reload. */
  async function setAuto(on: boolean) {
    await api.putSettings({ auto_update: on });
    const had = info()?.latest;
    mutate((prev) => (prev ? { ...prev, auto_check: on } : prev));
    if (on && !had) void check();
    else void refetch();
  }

  function follow(id: string) {
    setRestart(false);
    up$.follow(id, { text: t().update.starting, state: "running" });
  }

  /** Install a release. No tag means the latest, resolved in the child rather
      than taken from a cache that may be six hours old. */
  async function start(tag?: string) {
    try {
      follow((await api.runUpdate(tag)).id);
    } catch (e) {
      // A stage or a download holding the one job slot lands here (409), as
      // does an install shape that is not ours to rewrite.
      up$.setStatus(toStatus(e));
    }
  }

  /** Re-attach to an upgrade that was already running when the page loaded. */
  function adopt(job: Job) {
    follow(job.id);
    config.openSettings("update");
  }

  return {
    info,
    checking: () => checking() || info.loading,
    busy: up$.running,
    status: up$.status,
    lines: up$.lines,
    restart,
    check,
    setAuto,
    start,
    adopt,
    cancel: () => up$.id() && api.cancel(up$.id()!),
  };
}

export type Updates = ReturnType<typeof createUpdates>;
