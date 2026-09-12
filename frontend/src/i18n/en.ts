/** The GUI's own chrome in English, and the schema every other locale is
 * checked against.
 *
 * The dock's panels and the stage names under them are translated here, keyed
 * by the id the registry sends (`stage.panels` / `stage.titles` / `stage.shorts`)
 * — they are a closed, rarely-changing list and they are the app's own
 * navigation. The prose behind the stage bar's (?) joins them as an *overlay*
 * (`stage.docs` / `stage.notes`): the docstring the server sent is the English,
 * and a locale with no entry for a stage falls back to it. Everything else the
 * server writes is untranslated — every argparse label and help string and the
 * model catalog's rows arrive as they were written, and captions, paths and
 * tags are data.
 *
 * `en` is the schema: every other locale is declared `: Dict` (`ko.ts`, `ja.ts`,
 * `zh.ts`), so a missing or misspelled key is a type error rather than a blank
 * label at runtime.
 */
/** The registry's stage ids (`anime_tools/stages/registry.py`), named so the
    two overlays below are checked against ids that exist rather than accepting
    any string. */
export type StageId =
  | "resize"
  | "autotag"
  | "position"
  | "correct"
  | "audit"
  | "ocr"
  | "groups"
  | "masks_sam"
  | "masks_merge"
  | "export";

const en = {
  langName: "English",
  common: {
    cancel: "Cancel",
    save: "Save",
    close: "Close",
    browse: "Browse (home-relative)",
    on: "on",
    off: "off",
    running: "running",
    downloading: "downloading",
    installed: "installed",
    missing: "missing",
    present: "present",
    dragToResize: "Drag to resize",
    revealFile: "Show this file in the file manager",
    revealDir: "Open this folder in the file manager",
    revealFailed: "The host has no file manager to open",
  },
  header: {
    images: (n: number, root: string) => `${n} image${n === 1 ? "" : "s"} in ${root}`,
    noToken: "⚠ no HF token",
    noTokenHint: "The tagger backbone and SAM3 are gated on the Hub — set a token in Settings",
    showSidebar: "Show the dataset sidebar",
  },
  menu: {
    open: "Settings and status",
    settings: "⚙ Settings…",
    advanced: "Advanced",
    models: "Models & weights",
    update: "Update",
    updateReady: "new",
    hfToken: "HF token",
    notSet: "⚠ not set",
    guidebook: "📖 Guidebook",
    showHelp: "Show every explanation",
    language: "Language",
  },
  help: {
    hide: "Hide the description",
    show: "Show the description",
    showWarn: "Show the description (this stage has a warning)",
  },
  tree: {
    modeTree: "tree",
    modeTreeHint: "The folders the dataset is stored in",
    modeGroups: "groups",
    modeGroupsHint: "The near-twin components the Groups stage found",
    rescan: "Rescan the dataset",
    collapse: "Collapse the sidebar",
    filter: "filter…",
    scanning: "scanning…",
    reading: "reading…",
    noImages: "No images match.",
    /** {0} is the root, as a <code>. */
    noRoot: "No {0} under the curation home. Point the roots at your dataset in ⚙ Settings.",
    truncated: (shown: number, total: number) =>
      `showing ${shown} of ${total} — narrow it with the filter`,
    more: (n: number) => `+ ${n} more`,
    capMaster: "master — the hand-written caption; an edit lands in workspace/master",
    capRevised: "revised — workspace/resized, the stage output",
    capVariants: "variants — .variants.txt, generated and read-only",
    onDisk: "on disk",
    capMissing: "missing",
    flagPending: "the last run changed this image",
    flagResized: "resized — workspace/resized has this image",
    flagMask: "has a mask",
    flagExcluded:
      "excluded — its files moved to workspace/_excluded, resize skips it, and Export publishes it under _excluded/",
    root: "(root)",
    group: (id: number) => `group ${id}`,
    groupHint: (id: number, cos: number | string) =>
      `component ${id} — mean pairwise CLS cosine ${cos}`,
    ungrouped: "ungrouped",
    /** {0} is the manifest path, as a <code>; {1} the Groups stage's button path. */
    noManifest: "No {0} yet — run {1} in the dock, then come back.",
    buildGroups: "Groups › Build groups",
    staleLabel: "older manifest",
    staleHint: "— rebuild it to pick up the current grouping gate.",
    /** {0} is the manifest path, as a <code>. */
    clustersNothing: "{0} clusters nothing in this listing",
    /** {0} is the directory it was built from, as a <code>. */
    builtFrom: " — it was built from {0}",
  },
  item: {
    loading: "loading…",
    pick: "Pick an image on the left.",
    keys: "↑/↓ or j/k walk the images · ⌘/Ctrl+Enter saves a caption",
    views: {
      image: "source",
      mask: "mask",
      overlay: "overlay",
      ocr: "text boxes",
    },
    overlayHint: "the mask at 40% over the image",
    overlayNeeds: "needs an image and a mask",
    notGenerated: "not generated",
    noOverlay: "no overlay for this image",
    /** e.g. "no mask for this image" — {0} is one of `views`. */
    noView: (view: string) => `no ${view} for this image`,
    readOnly: "read-only",
    /** {0} is the floor, e.g. "0.50 MP". Shown on the pixel-count chip. */
    aboveFloor: (floor: string) => `above the ${floor} resize floor`,
    /** {0} is the floor. Why a stage over this image does nothing at all. */
    belowFloor: (floor: string) =>
      `below the ${floor} resize floor — this image is never resized, so the stages, which walk workspace/resized, see nothing for it. Lower it in ⚙ Settings › Preprocess.`,
    zoomHint: "⌘/Ctrl+scroll magnifies · drag to pan · double-click fits again",
    exclude: "exclude",
    excludeHint:
      "Take this image out of the pipeline: its resized copy, mask and OCR sidecar move to workspace/_excluded, resize skips it from now on, and Export publishes it under <out>/_excluded instead of into the tree the trainer reads. The source image and its master caption are not touched.",
    restore: "put back",
    restoreHint:
      "Move every file back to the tree it came from and take this image out of the ledger, so the stages see it again.",
    excluded: "excluded",
    /** {0} is when it was excluded, already formatted. */
    excludedAt: (when: string) => `taken out of the pipeline on ${when}`,
    /** {0} is how many files moved into workspace/_excluded. */
    excludedFiles: (n: number) => `${n} file${n === 1 ? "" : "s"} in workspace/_excluded`,
    /** Files left behind because the live path is occupied again. */
    excludeKept: (n: number) =>
      `${n} file${n === 1 ? "" : "s"} stayed in _excluded — the live path is taken`,
  },
  ocr: {
    title: "text in the image",
    /** {0} is how many lines were recognized. */
    count: (n: number) => `${n} line${n === 1 ? "" : "s"}`,
    readOnly: "generated by the OCR stage",
    /** The seq badge, the header count and the preview tab: the boxes the
        detector drew, over the resized copy they were read from. */
    boxesHint: "draw every line's box over the image the stage read",
    boxesNeeds: "needs the resized copy this text was read from",
    /** {0} is the line's seq number. The hint on one line's badge. */
    showBox: (seq: number) => `show line ${seq} on the image — click again for all of them`,
    /** Under the picture while every box is drawn. {0} is how many. */
    boxesAll: (n: number) => `${n} text box${n === 1 ? "" : "es"} over the image the stage read`,
    /** Under the picture while one line is drawn alone. {0} is its seq. */
    boxesOne: (seq: number) => `line ${seq} alone — click its badge again for all of them`,
    weak: "not in captions",
    weakHint:
      "under the detector-confidence or glyph-size floor: the sidecar keeps this line, but Export's --combine_ocr leaves it out of the caption.",
    /** The row's hint: {0} is the detector's confidence in the box, {1} the reader's in the text
        (either 0 when nothing stood behind it). */
    scores: (det: number, read: number) =>
      `box ${det.toFixed(2)} · read ${read.toFixed(2)} · top-left corner in the image, in pixels`,
  },
  caption: {
    master: "master",
    history: "history",
    revised: "revised",
    variants: "variants",
    /* One line per ladder rung, looked up as `where_<rung>`: what this version
     *is*. */
    where_master: "hand-written; the stages only read it",
    where_history: "what this caption used to say, before the run that replaced it",
    where_revised: "stage output; the next run rewrites it and keeps this text as a version",
    where_variants:
      "generated — v0 is the pristine revised caption; a hand edit here is overwritten",
    diffHere: "the last run rewrote this version",
    /** {0} is the version the run wrote, e.g. "revised". */
    diffElsewhere: (kind: string) => `the last run rewrote ${kind} — open it`,
    new: "new",
    revert: "Revert",
    save: "Save",
    saveHint: "⌘/Ctrl+Enter",
    empty: "no caption file yet — type one and save",
    saved: "saved — the previous text is a version above; follow with the trainer's TE re-encode",
    savedStale: "saved — .variants.txt is now stale; re-run correct + the trainer's TE re-encode",
    noCaption: "no caption",
    noHistory: "no change recorded — this caption has only ever been written once",
    tags: (n: number) => `${n} tag${n === 1 ? "" : "s"}`,
    clauses: (n: number) => `${n} clause${n === 1 ? "" : "s"}`,
    unsaved: "unsaved preview",
    lookUpHint: "double-click a tag to look it up",
    bag: "bag",
  },
  analysis: {
    badge: "analysis",
    badgeHint:
      "what the character-position stage last saw in this image: the instance masks it cut and what it said about each",
    position: "position tagging",
    audit: "multiview audit",
    proposed: "proposed",
    detected: (n: number, expected: number | null) =>
      expected == null ? `${n} detected` : `${n} detected / ${expected} in caption`,
    noTags: "no tags",
    moved: "moved out of the bag:",
    witnesses: "witnesses:",
    suggested: "suggested tag:",
    auditFacts: (mv: string, people: string, agree: string) =>
      `multiple views ${mv} · people count ${people} · identity agreement ${agree}`,
    where: "kept per image beside the position report; the next run over this image replaces it",
  },
  diff: {
    written: "written",
    changed: "the change",
    /** The header of the change a history badge stands for: which two
        versions it is between. Who replaced the first one and when is the
        badge's own note, already under the title. */
    from: (before: string, after: string) => `${before} → ${after}`,
    by: (stage: string) => `by ${stage}`,
    lastRun: "the last run",
    onDisk: "on disk",
    stale: "superseded",
    staleHint: "the caption changed after this run — what it says now is in the editor above",
    reordered: "same tags, reordered — see the text below.",
  },
  tag: {
    what: (tag: string) => `What is "${tag}"?`,
    posts: (n: string) => `${n} posts`,
    close: "Close (Esc)",
    looking: "looking it up…",
    notInstalled:
      "The Danbooru tag KB is not downloaded. It is what caption correction types tags against — and what this panel reads.",
    getIt: "Get it in Settings › Models",
    unknown: "not a Danbooru tag — an Anima quality tag, a position phrase, or a typo.",
    noDescription: "no wiki description for this tag.",
    matchedAs: (name: string) => `matched as “${name}”.`,
  },
  picker: {
    title: "Choose a path",
    use: "Use this folder",
  },
  settings: {
    /** One per dialog: the three are separate windows, not tabs of one. */
    paneTitle: {
      general: "Settings",
      advanced: "Advanced settings",
      models: "Models & weights",
      update: "Update",
    },
    home: "Home",
    modelsDir: "Models dir",
    roots: "Dataset roots",
    /** {0} is <code>out</code>. */
    rootsHelp:
      "Relative to the curation home; the trees are joined by the same relative path. The tools only ever write the workspace — {0} is Export's alone.",
    rootHelp: {
      src: "input — source images + hand-written master captions; never written",
      master: "workspace — revised master captions",
      dst: "workspace — resized images + revised captions + their versions + .variants.txt",
      masks: "workspace — {stem}_mask.png, mirroring the source subdirs",
      out: "output — where Export publishes; the tree the trainer reads",
    },
    rootMissing: "missing — ",
    stageDefaults: "Stage defaults",
    /** {0} `--device`, {1} `report_root`, {2} `captions/autotag`, {3} `groups/groups.json`. */
    stageDefaultsHelp:
      "Filled into every stage that takes them, so no stage form re-asks. Leave one blank for the CLI's own default. {0} is not here on purpose: each stage auto-detects it. {1} is the one knob with no flag of its own: each stage keeps its own directory under it ({2}, {3}), so moving the root moves them all without ever pointing two stages at one report — and the curated audit apply reads the audit's report back out of it.",
    reportRootHint: "where every stage's report.json lands — blank = beside the dst root",
    maskRootHint:
      "where each generator's own mask tree lands — blank = beside the masks root the merge fills",
    /** {0} is `target_res`. */
    preprocessHelp:
      "Runs over the same images the stage does, so a per-image Apply resizes just that image. Already-current images are skipped, so a re-run is near-free. Tiers must match the trainer's {0}.",
    hf: "Hugging Face",
    token: "Token",
    tokenPlaceholder: "hf_… (stored by huggingface_hub, never shown again)",
    tokenHelp:
      "The tagger backbone and SAM3 weights are gated on the Hub — a token with read access is needed on first run.",
    models: "Models",
    modelsHelp:
      "Every stage fetches what it needs on first use — these buttons only move the wait, and any gated-repo refusal, to a moment you picked. A download runs as a job: one at a time, and it reports here, not in the stage bar, so this dialog can stay open over it.",
    downloadAll: (n: number) => `Download all ${n} missing`,
    allInstalled: "Every model is installed",
    download: "Download",
    redownload: "Re-download",
    downloadingRow: "downloading…",
    downloadPack: "Download pack",
    redownloadPack: "Re-download pack",
    /** {0} is the "accept the terms" link. */
    gated: "Gated — {0} with the same account as the token above.",
    gatedLink: "accept the terms",
  },
  stage: {
    /** The dock's buttons, keyed by the panel name the registry sends
        (`gui/stages.py`). A panel with no entry here shows the name as it
        arrived — a stage added server-side is never a blank button. */
    panels: {
      Resize: "Resize",
      Autotag: "Autotag",
      Curate: "Curate",
      OCR: "OCR tagging",
      Groups: "Grouping",
      Masks: "Masks",
      Export: "Export",
    },
    /** Stage titles, keyed by stage id; same fallback as `panels`. Every field
        label under them is argparse text and stays as the server wrote it; the
        doc and the notes are overlaid below. */
    titles: {
      resize: "Resize to buckets",
      autotag: "Autotag captions",
      position: "Character position tagging",
      correct: "Tag correction",
      audit: "Multiview audit",
      ocr: "OCR text",
      groups: "Group similar images",
      masks_sam: "SAM3 masks",
      masks_merge: "Merge masks",
      export: "Export workspace",
    },
    /** The in-panel picker's label, for the two panels that hold more than one
        stage. Falls back to the title. */
    shorts: {
      position: "Position tagging",
      correct: "Tag correction",
      audit: "Audit",
      masks_sam: "Setup",
      masks_merge: "Merge",
    },
    /** The prose the stage bar's (?) reveals, keyed by stage id — an *overlay*
        on the request class's docstring, not a copy of it. `en` spells nothing:
        that docstring is the English, and re-typing it here would only let the
        two drift. A stage a locale has no line for reads in English, which is
        why these are `Partial` where `titles` is not. */
    docs: {} as Partial<Record<StageId, string>>,
    /** The same overlay over the registry's `notes` — the ⚠ line under the doc.
        Read only when the server sent notes at all, so a locale can translate a
        warning but never invent one. */
    notes: {} as Partial<Record<StageId, string>>,
    run: "Run",
    runBatch: "Run batch",
    undo: "Undo",
    cancel: "Cancel",
    aim: (rel: string) => `just ${rel}`,
    noImage: "select an image in the sidebar first",
    batchHint: "every image the Settings path_pattern names",
    undoHint: "put back the captions the last run wrote",
    missingModels: (n: number) => `↓ ${n} model${n === 1 ? "" : "s"} missing`,
    missingModelsHint: (names: string) =>
      `Not downloaded yet: ${names}. The first Run fetches them itself — this only moves the wait to a moment you pick.`,
    noStages: "No stages.",
    unavailable: (title: string, error: string) => `${title} is unavailable: ${error}`,
    reinstall: "Reinstall:",
  },
  form: {
    options: "options",
    onePerLine: "one per line",
    reset: "reset to defaults",
    advanced: (n: number) => `▸ advanced (${n})`,
    advancedHide: "▾ advanced",
    advancedHint: "The knobs a run rarely changes its mind about",
    advancedDirty: (n: number) => `${n} hidden field${n === 1 ? " is" : "s are"} off its default`,
    maskRoles: { keep: "keep", ignore: "ignore" },
    maskKinds: { text: "text", soft: "soft" },
    maskRoleHint:
      "keep: only the regions you keep train; ignore: this region is left out of the loss",
    maskKindHint:
      "text: a SAM3 prompt such as speech bubble; soft: a learned prompt file (.safetensors)",
    maskTextPlaceholder: "SAM3 prompt, e.g. speech bubble",
    maskSoftPlaceholder: "soft prompt .safetensors path",
    maskRemove: "remove this mask",
    maskAdd: "+ add mask",
    maskEmpty: "no masks — a run falls back to keeping the subject",
  },
  runner: {
    nothingToUndo: "nothing to undo",
    nothingApplied: "nothing run yet — Undo puts back what a run wrote",
    following: (id: string) => `running ${id}`,
    exit: (code: number | null) => `exit ${code}`,
    changed: (n: number) => `${n} file(s) changed`,
    undoing: "undoing…",
    undone: (restored: number, removed: number) =>
      `undone: ${restored} restored, ${removed} removed`,
    skipped: (what: string) => `— skipped ${what}`,
    logClosed: "log stream closed",
  },
  guide: {
    title: "Guidebook",
    loading: "reading the guidebook…",
  },
  job: {
    log: "log",
    logHint: "The job's own output, in full",
    title: "Job log",
    empty: "no output yet",
    /** The progress counter beside the bar: images done of images in this step. */
    of: (done: number, total: number) => `${done} / ${total}`,
    step: (index: number, total: number, label: string) => `step ${index}/${total} · ${label}`,
    /** Only the tail is kept in the page; the job's log file has all of it. */
    tail: (n: number) => `showing the last ${n} lines`,
  },
  downloads: {
    starting: "starting…",
    finished: "download finished",
  },
  update: {
    version: "Version",
    help: "The panel asks GitHub for the latest release and caches the answer for six hours, so a reload is never a request. An update replaces the environment this server is running from — nothing is written to your dataset, and the GUI has to be restarted afterwards.",
    installed: "Installed",
    latest: "Latest release",
    never: "not checked yet",
    viewRelease: "release page ↗",
    checking: "checking…",
    checkNow: "Check now",
    checkedAt: (when: string) => `checked ${when}`,
    checkFailed: (err: string) => `check failed: ${err}`,
    state: {
      current: "up to date",
      available: "update available",
      ahead: "ahead of the latest release",
      unknown: "version unknown",
    },
    auto: "Check for updates on startup",
    autoHint: "Off means GitHub is asked only when you press Check now.",
    releaseNotes: "Release notes",
    install: "Install",
    installKind: {
      "uv-tool": "Installed by install.sh as its own uv tool — updated from here.",
      checkout: "Running from a git checkout of the repository.",
      other: "Installed into this environment by something other than uv tool.",
    },
    warning:
      "The update replaces the environment the GUI is running from. Stages started after it may not find their files until anime-tools-gui is restarted.",
    restart: "Update installed — restart anime-tools-gui to run it.",
    updateTo: (tag: string) => `Update to ${tag}`,
    upToDate: "Already on the latest release",
    nothingToDo: "Nothing to install",
    updating: "updating…",
    starting: "starting…",
    installedOk: "update installed",
  },
};

export default en;
export type Dict = typeof en;
