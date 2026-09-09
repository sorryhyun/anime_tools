import type { Dict } from "./en";

const ja: Dict = {
  langName: "日本語",
  common: {
    cancel: "キャンセル",
    save: "保存",
    close: "閉じる",
    browse: "参照（ホームからの相対パス）",
    on: "オン",
    off: "オフ",
    running: "実行中",
    downloading: "ダウンロード中",
    installed: "インストール済み",
    missing: "未取得",
    present: "あり",
    dragToResize: "ドラッグでサイズ変更",
  },
  header: {
    images: (n, root) => `${root} に画像 ${n} 枚`,
    noToken: "⚠ HF トークンなし",
    noTokenHint:
      "タガーのバックボーンと SAM3 は Hub で承認が必要です — 設定でトークンを入れてください",
    showSidebar: "データセットのサイドバーを開く",
  },
  menu: {
    open: "設定とステータス",
    settings: "⚙ 設定…",
    advanced: "詳細",
    models: "モデルと重み",
    update: "アップデート",
    updateReady: "新着",
    hfToken: "HF トークン",
    notSet: "⚠ 未設定",
    guidebook: "📖 ガイドブック",
    showHelp: "説明をすべて表示",
    language: "言語",
  },
  help: {
    hide: "説明を隠す",
    show: "説明を表示",
    showWarn: "説明を表示（このステージには警告があります）",
  },
  tree: {
    modeTree: "フォルダ",
    modeTreeHint: "データセットが置かれているフォルダ構成",
    modeGroups: "グループ",
    modeGroupsHint: "Groups ステージが見つけた近似画像のまとまり",
    rescan: "データセットを再スキャン",
    collapse: "サイドバーを畳む",
    filter: "絞り込み…",
    scanning: "スキャン中…",
    reading: "読み込み中…",
    noImages: "一致する画像がありません。",
    noRoot:
      "キュレーションホームに {0} がありません。⚙ 設定でルートをデータセットに向けてください。",
    truncated: (shown, total) => `${total} 件中 ${shown} 件を表示 — 絞り込みで狭めてください`,
    more: (n) => `+ 他 ${n} 件`,
    capMaster: "master — 手書きのキャプション; 編集は workspace/master に保存されます",
    capHistory: "history — .history.txt、revised キャプションが以前言っていたこと",
    capRevised: "revised — workspace/resized、ステージの出力",
    capVariants: "variants — .variants.txt、自動生成で読み取り専用",
    onDisk: "ディスク上にあり",
    capMissing: "なし",
    flagPending: "直前の実行がこの画像を変更しました",
    flagResized: "resized — workspace/resized にこの画像があります",
    flagMask: "マスクあり",
    root: "（ルート）",
    group: (id) => `グループ ${id}`,
    groupHint: (id, cos) => `まとまり ${id} — ペアごとの CLS コサイン平均 ${cos}`,
    ungrouped: "未グループ",
    noManifest: "{0} がまだありません — ドックで {1} を実行してから戻ってください。",
    buildGroups: "Groups › Build groups",
    staleLabel: "古いマニフェスト",
    staleHint: "— 今のグループ判定に合わせるには作り直してください。",
    clustersNothing: "{0} はこの一覧では何もまとめていません",
    builtFrom: " — {0} を対象に作られています",
  },
  item: {
    loading: "読み込み中…",
    pick: "左から画像を選んでください。",
    keys: "↑/↓ または j/k で画像を移動 · ⌘/Ctrl+Enter でキャプションを保存",
    views: {
      image: "元画像",
      mask: "マスク",
      overlay: "重ね表示",
    },
    overlayHint: "画像の上にマスクを 40% で重ねます",
    overlayNeeds: "画像とマスクの両方が必要です",
    notGenerated: "未生成",
    noOverlay: "この画像に重ねるものがありません",
    noView: (view) => `この画像には${view}がありません`,
    readOnly: "読み取り専用",
    /** {0} は下限値、例: "0.50 MP"。ピクセル数チップに付きます。 */
    aboveFloor: (floor: string) => `リサイズ下限 ${floor} 以上`,
    /** {0} は下限値。この画像でステージを回しても何も起きない理由です。 */
    belowFloor: (floor: string) =>
      `リサイズ下限 ${floor} 未満 — この画像はリサイズされないため、workspace/resized を歩く各ステージからは見えません。⚙ 設定 › Preprocess で下限を下げてください。`,
    zoomHint: "⌘/Ctrl+スクロールで拡大 · ドラッグで移動 · ダブルクリックで元に戻す",
  },
  ocr: {
    title: "画像内のテキスト",
    /** {0} is how many lines were recognized. */
    count: (n: number) => `${n}行`,
    readOnly: "OCR ステージが生成",
    boxHint: "画像内の左上座標（ピクセル）",
  },
  caption: {
    master: "master",
    history: "history",
    revised: "revised",
    variants: "variants",
    where_master: "手書き; ステージは読むだけです",
    where_history: "このキャプションが以前言っていたこと — 上書きした実行の直前の内容です",
    where_revised: "ステージの出力; 次の実行が書き換え、いまの本文はバージョンとして残ります",
    where_variants:
      "自動生成 — v0 が手を入れていない revised キャプションです。ここでの手編集は上書きされます",
    diffHere: "直前の実行がこのバージョンを書き換えました",
    diffElsewhere: (kind: string) => `直前の実行は ${kind} を書き換えました — 開く`,
    new: "新規",
    revert: "元に戻す",
    save: "保存",
    saveHint: "⌘/Ctrl+Enter",
    empty: "キャプションファイルがまだありません — 入力して保存してください",
    saved:
      "保存しました — 前の本文は上のバージョンに残ります; 続けてトレーナーの TE 再エンコードを実行してください",
    savedStale:
      "保存しました — .variants.txt が古くなりました; correct とトレーナーの TE 再エンコードをやり直してください",
    noCaption: "キャプションなし",
    tags: (n) => `タグ ${n} 個`,
    clauses: (n) => `節 ${n} 個`,
    unsaved: "未保存のプレビュー",
    lookUpHint: "タグをダブルクリックすると意味を引きます",
    bag: "タグ袋",
  },
  diff: {
    written: "書き込み済み",
    by: (stage) => `${stage} による`,
    lastRun: "直前の実行",
    onDisk: "ディスク上",
    stale: "過去のもの",
    staleHint: "実行後にキャプションが変わりました — Apply はこの行を飛ばします",
    reordered: "タグは同じで並びだけが変わりました — 下の本文を見てください。",
  },
  tag: {
    what: (tag) => `"${tag}" とは？`,
    posts: (n) => `投稿 ${n} 件`,
    close: "閉じる (Esc)",
    looking: "調べています…",
    notInstalled:
      "Danbooru タグ KB が未取得です。キャプション補正がタグを照合する基準であり、このパネルが読むデータです。",
    getIt: "設定 › モデル から取得",
    unknown: "Danbooru のタグではありません — Anima の品質タグ、位置句、または誤字です。",
    noDescription: "このタグには wiki の説明がありません。",
    matchedAs: (name) => `「${name}」として一致しました。`,
  },
  picker: {
    title: "パスを選ぶ",
    use: "このフォルダを使う",
  },
  settings: {
    paneTitle: {
      general: "設定",
      advanced: "詳細設定",
      models: "モデルとウェイト",
      update: "アップデート",
    },
    home: "ホーム",
    modelsDir: "モデルの場所",
    roots: "データセットのルート",
    rootsHelp:
      "キュレーションホームからの相対パスです; 各ツリーは同じ相対パスで対応づきます。ツールが書くのはワークスペースだけで、{0} は Export 専用です。",
    rootHelp: {
      src: "入力 — 元画像と手書きの master キャプション; 書き込みません",
      master: "ワークスペース — 修正した master キャプション",
      dst: "ワークスペース — リサイズ画像、derived キャプション、.variants.txt",
      masks: "ワークスペース — {stem}_mask.png、元のサブフォルダ構成をそのまま反映します",
      out: "出力 — Export の書き出し先; トレーナーが読むツリー",
    },
    rootMissing: "存在しません — ",
    stageDefaults: "ステージの既定値",
    stageDefaultsHelp:
      "これを受け取るすべてのステージに入るので、各フォームが訊き直しません。空にすると CLI 自身の既定値になります。{0} は意図的にここにありません: 各ステージが自動で判定します。{1} は自分のフラグを持たない唯一の値です: 各ステージがその下に自分のフォルダ（{2}、{3}）を持つので、ルートを動かせば全部が一緒に動きつつ、二つのステージが同じレポートを指すことはありません — キュレーション監査の適用もここから監査レポートを読み直します。",
    reportRootHint: "各ステージの report.json が置かれる場所 — 空なら dst ルートの隣",
    maskRootHint:
      "各ジェネレータ自身のマスクツリーが置かれる場所 — 空ならマージが埋める masks ルートの隣",
    preprocessHelp:
      "ステージが扱うのと同じ画像に対して走るので、画像単位の Apply はその画像だけをリサイズします。すでに最新の画像は飛ばすため、再実行はほぼ無料です。段は トレーナーの {0} と一致させてください。",
    hf: "Hugging Face",
    token: "トークン",
    tokenPlaceholder: "hf_…（huggingface_hub が保存し、二度と表示しません）",
    tokenHelp:
      "タガーのバックボーンと SAM3 の重みは Hub で承認制です — 初回実行に読み取り権限のトークンが要ります。",
    models: "モデル",
    modelsHelp:
      "各ステージは初回利用時に必要なものを自分で取得します — このボタンは、その待ち時間と承認制リポジトリの拒否を、あなたが選んだ瞬間に移すだけです。ダウンロードもジョブとして走ります: 一度に一つで、ステージバーではなくここに報告するので、このダイアログを開いたままにできます。",
    downloadAll: (n) => `未取得の ${n} 件をすべてダウンロード`,
    allInstalled: "すべてのモデルが揃っています",
    download: "ダウンロード",
    redownload: "再ダウンロード",
    downloadingRow: "ダウンロード中…",
    downloadPack: "パックをダウンロード",
    redownloadPack: "パックを再ダウンロード",
    gated: "承認制 — 上のトークンと同じアカウントで{0}してください。",
    gatedLink: "利用規約に同意",
  },
  stage: {
    panels: {
      Resize: "リサイズ",
      Autotag: "自動タグ",
      Curate: "キャプション整備",
      OCR: "OCR",
      Groups: "グループ",
      Masks: "マスク",
      Export: "エクスポート",
    },
    titles: {
      resize: "バケットサイズにリサイズ",
      autotag: "キャプションを自動タグ付け",
      position: "位置句を付ける",
      correct: "キャプション補正 + 左右反転",
      audit: "マルチビュー点検",
      ocr: "画像内の文字認識",
      groups: "近縁グループを作る",
      masks_sam: "SAM3 被写体マスク",
      masks_merge: "マスクを統合",
      export: "ワークスペースを書き出す",
    },
    shorts: {
      position: "位置",
      correct: "補正",
      audit: "点検",
      masks_sam: "被写体",
      masks_merge: "統合",
    },
    docs: {
      resize:
        "キャプション master を、すべてのステージが読むバケット解像度のツリーへリサイズします。\n\n" +
        "他のステージはどれも --dst の下を歩くので、--src にしかない画像はそれらから見えません。各画像は縮小がいちばん小さい --target_res の段に入り、その段のトークン帯の中で元の縦横比を保ちます。ジオメトリはトレーナーの make preprocess-resize と同じなので、どちらが先に走っても、もう片方はスキップします。\n\n" +
        "常に書き込みます。ドライランはなく、すでに目的のバケットにある画像は再デコードせずスキップします。glob は --src を基準に照合します。",
      autotag:
        "Anima Tagger でデータセットにタグを付け、revised キャプションを書きます。\n\n" +
        "リサイズ済みツリーを歩き、画像ごとにキャプションを提案します。--mode は三つ: missing(どのキャプションも語っていない画像、既定)、merge(句はそのままに新しいタグだけ足す)、overwrite(キャプションごと置き換える)。読むのは revised、無ければ master、書くのは常に revised で、そこで押し出されたテキストは {stem}.history.txt のバージョンとして残ります。\n\n" +
        "既定はドライラン、--apply が書きます。TE キャッシュは古びても見た目は最新のままなので、実際に適用したら make preprocess-te を続けてください。",
      position:
        "複数の被写体があるキャプションを位置句へ書き直します(SAM3 + Anima Tagger)。\n\n" +
        "リサイズ済みデータセットの上で 検出 → 並べ → 切り抜いて消し → タグ付け → 組み立て と進み、帰属できるタグを平らなタグの束からそれぞれの句へ移します(--flatten が逆向きのパスです)。句は --dst の下の revised キャプションに付き、キャプション master には触れません。docs/position_captions.md を参照。\n\n" +
        "既定はドライラン、--apply が書き、古い .variants.txt サイドカーを落とすので、続けて TE の再エンコードを回してください。",
      correct:
        "リサイズ済みの前処理画像の隣に、補正したキャプションを書きます。\n\n" +
        "revised キャプションをその場で補正し、revised がまだ無い画像についてだけ master を読みます。バリアントのサイドカーは任意です。常に書き込みます: ドライランもレポートもありません。",
      audit:
        "1girl のキャプションのうち、実は一人の複数視点である画像を点検します。\n\n" +
        "位置ステージが single-subject として飛ばした画像をさらい、girl プロンプトが被写体を二つ以上見つけたものをすべて報告します。docs/multiview_audit.md を参照。位置ステージは --multiview_audit でこれを自分の最初のフェーズとして走らせ、普通はそちらが入口です。\n\n" +
        "既定はドライラン。--apply は欠けたタグを --dst の下の revised キャプションへ書き、master には書きません。続けて TE の再エンコードを回してください。",
      ocr:
        "画像ごとに、その中の文字を読んで何と書いてあるかを記録します。\n\n" +
        "リサイズ済みツリーを歩き、AnimeText 検出器(anime_tools.ocr.animetext、torch 上の YOLO12。初回に 54 MB を取得し、重みは GPL-3.0 なので同梱しません)で文字ブロックごとに箱を作り、マンガ VL リーダー(anime_tools.ocr.sfx、手描きの効果音でファインチューンした PaddleOCR-VL-1.6。torch、初回に約 2.8 GB)で箱をすべて読み、{stem}.ocr.txt を OCR ツリーへリサイズ済みツリーと同じ形で書きます。キャプションは読みも書きもせず、あとで TE の再エンコードも要りません。\n\n" +
        "吹き出しの台詞と絵に描かれた効果音を、一つの検出器がまとめて拾います。箱ひとつの読みがそのまま一行なので、デコードガードが弾いた読みはその箱ごと落ちます。--mask_dir を渡すと、どの箱も覆っていないテキストマスクの成分も独立した行として読みます(det 0.000)。各行には検出器の箱の信頼度(det)とリーダーの平均トークン信頼度(score)が付きます。\n\n" +
        "既定はドライラン: ドライランは書くはずだった行をすべて載せた report.json を残し、--apply はサイドカーを書くだけで他には何もしません。",
      groups:
        "PE-Spatial の見た目の近さでデータセットの画像をまとめ、groups.json マニフェストを書きます。\n\n" +
        "学習の前処理ではなくキュレーションの道具です: 絵師ごとにほぼ同じ画像をクラスタし、GUI のデータセットタブでグループとして絞れるようにするだけで、ほかには何も書きません。二枚は、セルごとの下限 --cell-match-min のもとで match_frac が --match-frac-min 以上のときにまとまります。再実行は共有の PE-Spatial 特徴キャッシュを使い回すので、しきい値の詰め直しは安く済みます。",
      masks_sam:
        "SAM3 の被写体マスクを workspace/masks_sam/ に書きます。\n\n" +
        "--prompts はマスクで外すもの(損失で無視するもの)を、--focus-prompts は残すもの(それ以外をすべて外す)を指します。両方を与えると、残した領域から外す領域を引いた分が生き残ります。被写体プロンプトは既定では学習済みのソフトプロンプト(--prompt_embed)が担い、素のテキストプロンプトを使うなら none を渡します。",
      masks_merge:
        "複数の出どころのマスクを、画素ごとの最小値で統合します(マスク領域の和集合)。\n\n" +
        "(rel_dir, name) をキーに統合するので、入力をまたいで同じ相対パスにあるマスクどうしが出会います。入れ子の形は --output-dir の下にそのまま残ります。無い入力ディレクトリはエラーではなくスキップ — 既定の入力は生成器ひとつのツリーで、二つ目のツリー(手で塗ったマスク、別のツールの出力)はその隣に並べるだけです。",
      export:
        "ワークスペースを、トレーナーが読むパスへ公開します。\n\n" +
        "このパッケージで workspace/ の外に書く唯一の操作です: リサイズ済みの画像とマスクとキャプションが --out の下へコピーされます。\n\n" +
        "既定はドライラン。--apply は実際にコピーしつつ、一行ごとに宛先と照らし直すので、ドライラン後に編集されたファイルは上書きではなく報告されます。書き出しを取り消すのは、ここのフラグではなく GUI の Undo です。",
    },
    notes: {
      resize:
        "リサイズ済みツリーを読むステージすべての前で自動的に走ります。ここの既定値はそのすべてに効きます。",
      autotag:
        "revised キャプションをリサイズ済みツリーの下に書きます。master は無いときに読むだけで、編集しません。missing はすでにキャプションが語っている画像を飛ばします。",
      correct:
        "revised キャプションをその場で補正します。master は revised が無い画像でだけ読み、編集はしません。",
      ocr: "{stem}.ocr.txt を、リサイズ済みツリーと同じ形で OCR ツリーへ書きます。キャプションは読みも書きもしません。",
      export:
        "ワークスペースの外に書く唯一のステージです。実行するとトレーナーが読むツリーへコピーし、すでに同じものは飛ばします。Undo は上書きしたテキストを戻しますが、置き換わった画素は戻せません。",
    },
    run: "実行",
    runBatch: "一括実行",
    undo: "元に戻す",
    cancel: "中止",
    aim: (rel) => `${rel} だけ`,
    noImage: "先にサイドバーで画像を選んでください",
    batchHint: "設定の path_pattern が指すすべての画像",
    undoHint: "直前の実行が書いたキャプションを戻します",
    missingModels: (n) => `↓ モデル ${n} 件が未取得`,
    missingModelsHint: (names) =>
      `未取得: ${names}。最初の Run が自分で取得します — このボタンは待ち時間をあなたが選んだ瞬間に移すだけです。`,
    noStages: "ステージがありません。",
    unavailable: (title, error) => `${title} は使えません: ${error}`,
    reinstall: "再インストール:",
  },
  form: {
    options: "オプション",
    onePerLine: "1 行に 1 つ",
    reset: "既定値に戻す",
    advanced: (n) => `▸ 詳細 (${n})`,
    advancedHide: "▾ 詳細",
    advancedHint: "実行のたびに変えることはまずない項目",
    advancedDirty: (n) => `折りたたまれた項目のうち ${n} 個が既定値ではありません`,
  },
  runner: {
    nothingToUndo: "戻すものがありません",
    nothingApplied: "まだ実行していません — Undo は実行が書いた内容を戻します",
    following: (id) => `${id} を実行中`,
    exit: (code) => `終了コード ${code}`,
    changed: (n) => `ファイル ${n} 件を変更`,
    undoing: "戻しています…",
    undone: (restored, removed) => `戻しました: ${restored} 件復元、${removed} 件削除`,
    skipped: (what) => `— 飛ばした ${what}`,
    logClosed: "ログのストリームが閉じました",
  },
  guide: {
    title: "ガイドブック",
    loading: "ガイドブックを読み込み中…",
  },
  job: {
    log: "ログ",
    logHint: "ジョブが出力した内容すべて",
    title: "ジョブログ",
    empty: "まだ出力がありません",
    of: (done, total) => `${done} / ${total}`,
    step: (index, total, label) => `ステップ ${index}/${total} · ${label}`,
    tail: (n) => `最後の ${n} 行だけを表示しています`,
  },
  downloads: {
    starting: "開始しています…",
    finished: "ダウンロード完了",
  },
  update: {
    version: "バージョン",
    help: "パネルは GitHub に最新リリースを尋ね、その答えを6時間キャッシュします — 再読み込みのたびに問い合わせることはありません。アップデートはこのサーバーが動いている環境を置き換えます。データセットには何も書き込みませんが、完了後は GUI の再起動が必要です。",
    installed: "インストール済み",
    latest: "最新リリース",
    never: "未確認",
    viewRelease: "リリースページ ↗",
    checking: "確認中…",
    checkNow: "今すぐ確認",
    checkedAt: (when) => `${when} に確認`,
    checkFailed: (err) => `確認に失敗: ${err}`,
    state: {
      current: "最新です",
      available: "アップデートがあります",
      ahead: "最新リリースより新しい",
      unknown: "バージョン不明",
    },
    auto: "起動時にアップデートを確認する",
    autoHint: "オフにすると[今すぐ確認]を押したときだけ GitHub に問い合わせます。",
    releaseNotes: "リリースノート",
    install: "インストールの形",
    installKind: {
      "uv-tool": "install.sh が作った uv tool 環境です — ここから更新します。",
      checkout: "リポジトリの git チェックアウトで動いています。",
      other: "uv tool 以外の方法でこの環境に入っています。",
    },
    warning:
      "アップデートは GUI が動いている環境を置き換えます。anime-tools-gui を再起動するまで、その後に開始したステージはファイルを見つけられないことがあります。",
    restart: "アップデートを適用しました — anime-tools-gui を再起動してください。",
    updateTo: (tag) => `${tag} に更新`,
    upToDate: "すでに最新リリースです",
    nothingToDo: "インストールするものはありません",
    updating: "更新中…",
    starting: "開始しています…",
    installedOk: "アップデートを適用しました",
  },
};

export default ja;
