import { batch, createSignal, onCleanup } from "solid-js";
import { api, toStatus } from "./api";
import { t } from "./i18n";
import { createJobFollower, newestNonEmpty } from "./state";
import type { Job } from "./types";
import type { Config } from "./config";

/** A weights fetch: to the server a job like any other, taking the same single
 * slot, but it belongs to the Settings dialog rather than the dock, so it gets
 * its own follower.
 */
export function createDownloads(config: Config) {
  const dl$ = createJobFollower({
    // The downloader prints a blank line between models.
    line: newestNonEmpty,
    done: (job) => {
      dl$.setStatus({
        text: job.state === "done" ? t().downloads.finished : t().runner.exit(job.exit_code),
        state: job.state,
      });
      void config.refetchInfo();
      // The rows say installed/missing; a finished pull just changed that.
      void config.refetchModels();
    },
  });
  onCleanup(() => dl$.close());
  /** The ids this download asked for; `[]` = every missing model. Only used to
      mark the rows in flight. */
  const [ids, setIds] = createSignal<string[]>([]);

  /** Follow a weights job inside the Settings dialog: no dock, no dock status
      line -- the modal stays open and reports the pull itself. */
  function follow(id: string, want: string[]) {
    batch(() => {
      setIds(want);
      dl$.follow(id, { text: t().downloads.starting, state: "running" });
    });
  }

  /** Re-attach to a download that was already running when the page loaded;
      the job's own `ids` say what it asked for. */
  function adopt(job: Job) {
    follow(job.id, job.ids);
    config.openSettings("models");
  }

  async function start(want: string[]) {
    try {
      const job = await api.downloadModels(want);
      follow(job.id, want);
    } catch (e) {
      // A stage holding the one job slot lands here (409) -- say so in the
      // dialog, where the button that failed is.
      dl$.setStatus(toStatus(e));
    }
  }

  return {
    busy: dl$.running,
    status: dl$.status,
    ids,
    adopt,
    start,
    cancel: () => dl$.id() && api.cancel(dl$.id()!),
  };
}
