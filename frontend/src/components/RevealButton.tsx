import { createEffect, createSignal, on, Show } from "solid-js";
import { api } from "../api";
import { t } from "../i18n";

/** The ↗ beside a path in the panel: hands that path to the host's file
    manager, a folder opened and a file selected inside its own.

    It draws nothing unless `can` — `Info.can_reveal`, which is the server
    saying it has a desktop *and* that this browser is on the machine holding
    it. On a headless box or from another machine the window would open where
    nobody is sitting, so there is no button rather than a dead one.

    The failure it can report is narrow on purpose: the server answers as soon
    as the desktop takes the request, so `revealed: false` means the command
    would not launch at all. What the file manager then did with it is out of
    reach of this page, and the button never claims otherwise. */
export function RevealButton(props: { path?: string; can: boolean; dir?: boolean }) {
  const [failed, setFailed] = createSignal(false);
  // A new path is a new question: the last one's failure says nothing about it.
  createEffect(
    on(
      () => props.path,
      () => setFailed(false),
      { defer: true },
    ),
  );

  async function go() {
    setFailed(false);
    try {
      if (!(await api.reveal(props.path!)).revealed) setFailed(true);
    } catch {
      setFailed(true);
    }
  }

  return (
    <Show when={props.can && props.path}>
      <button
        classList={{ link: true, reveal: true, warn: failed() }}
        title={
          failed()
            ? t().common.revealFailed
            : props.dir
              ? t().common.revealDir
              : t().common.revealFile
        }
        onClick={go}
      >
        ↗
      </button>
    </Show>
  );
}
