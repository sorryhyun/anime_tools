import { batch, createSignal, onCleanup } from "solid-js";
import { api, toStatus } from "./api";
import { t } from "./i18n";
import { createJobFollower } from "./state";
import type { Job } from "./types";
import type { Config } from "./config";

/** `/api/models/download` names its job `download:<ids>`; that prefix is the
    only thing that tells an *adopted* job (one already running when the page
    loaded) apart from a stage run, and the two go to different places. */
const DOWNLOAD_STAGE = "download:";

export const isDownloadJob = (job: Job) => job.stage.startsWith(DOWNLOAD_STAGE);

/** A weights fetch: a job like any other to the server — it takes the same
 * single slot — but it belongs to the Settings dialog, not the dock.
 *
 * Its own follower is what keeps the modal open over the pull and keeps the run
 * bar from saying "running" for a fetch no stage form started.
 */
export function createDownloads(config: Config) {
  const dl$ = createJobFollower({
    // The downloader prints a blank line between models (and hub progress bars
    // arrive as \r-terminated lines): the newest *non-empty* one is the status.
    line: (line) => (line.trim() ? { text: line.trim(), state: "running" } : null),
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

  /** Follow a weights job inside the Settings dialog. Deliberately none of what
      the dock's `attach` does: no dock, no dock status line -- the modal stays
      open and reports the pull itself. */
  function follow(id: string, want: string[]) {
    batch(() => {
      setIds(want);
      dl$.follow(id, { text: t().downloads.starting, state: "running" });
    });
  }

  /** Re-attach to a download that was already running when the page loaded.
      `download:<ids>` is the only record of what it asked for. */
  function adopt(job: Job) {
    const rest = job.stage.slice(DOWNLOAD_STAGE.length);
    follow(job.id, rest === "missing" ? [] : rest.split(",").filter(Boolean));
    config.openSettings("models");
  }

  /** Start a weights download; it reports into the Settings dialog that started
      it, which stays open over it. */
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

export type Downloads = ReturnType<typeof createDownloads>;
