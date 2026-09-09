import type { Dict } from "./en";

const ko: Dict = {
  langName: "한국어",
  common: {
    cancel: "취소",
    save: "저장",
    close: "닫기",
    browse: "찾아보기 (홈 기준 상대 경로)",
    on: "켜짐",
    off: "꺼짐",
    running: "실행 중",
    downloading: "내려받는 중",
    installed: "설치됨",
    missing: "없음",
    present: "있음",
    dragToResize: "드래그해서 크기 조절",
  },
  header: {
    images: (n, root) => `${root}에 이미지 ${n}장`,
    noToken: "⚠ HF 토큰 없음",
    noTokenHint: "태거 백본과 SAM3는 허브에서 승인이 필요합니다 — 설정에서 토큰을 넣으세요",
    showSidebar: "데이터셋 사이드바 열기",
  },
  menu: {
    open: "설정과 상태",
    settings: "⚙ 설정…",
    advanced: "고급",
    models: "모델과 가중치",
    update: "업데이트",
    updateReady: "새 버전",
    hfToken: "HF 토큰",
    notSet: "⚠ 미설정",
    guidebook: "📖 가이드북",
    showHelp: "설명 전부 보이기",
    language: "언어",
  },
  help: {
    hide: "설명 숨기기",
    show: "설명 보기",
    showWarn: "설명 보기 (이 스테이지에는 경고가 있습니다)",
  },
  tree: {
    modeTree: "폴더",
    modeTreeHint: "데이터셋이 저장된 폴더 구조",
    modeGroups: "그룹",
    modeGroupsHint: "Groups 스테이지가 찾아낸 근접 쌍 묶음",
    rescan: "데이터셋 다시 훑기",
    collapse: "사이드바 접기",
    filter: "필터…",
    scanning: "훑는 중…",
    reading: "읽는 중…",
    noImages: "조건에 맞는 이미지가 없습니다.",
    noRoot: "큐레이션 홈에 {0}이(가) 없습니다. ⚙ 설정에서 루트를 데이터셋으로 지정하세요.",
    truncated: (shown, total) => `${total}개 중 ${shown}개 표시 — 필터로 좁히세요`,
    more: (n) => `+ ${n}개 더`,
    capMaster: "master — 손으로 쓴 캡션; 편집은 workspace/master 에 저장됩니다",
    capHistory: "history — .history.txt, revised 캡션이 예전에 뭐라고 했는지",
    capRevised: "revised — workspace/resized, 스테이지 출력",
    capVariants: "variants — .variants.txt, 자동 생성이라 읽기 전용",
    onDisk: "디스크에 있음",
    capMissing: "없음",
    flagPending: "마지막 실행이 이 이미지를 바꿨습니다",
    flagResized: "resized — workspace/resized에 이 이미지가 있습니다",
    flagMask: "마스크 있음",
    root: "(최상위)",
    group: (id) => `그룹 ${id}`,
    groupHint: (id, cos) => `묶음 ${id} — 쌍별 CLS 코사인 평균 ${cos}`,
    ungrouped: "묶이지 않음",
    noManifest: "아직 {0}이(가) 없습니다 — 독에서 {1}을(를) 실행한 뒤 다시 오세요.",
    buildGroups: "Groups › Build groups",
    staleLabel: "예전 매니페스트",
    staleHint: "— 지금의 그룹 판정 기준을 반영하려면 다시 만드세요.",
    clustersNothing: "{0}이(가) 이 목록에서는 아무것도 묶지 못합니다",
    builtFrom: " — {0}을(를) 대상으로 만들어졌습니다",
  },
  item: {
    loading: "불러오는 중…",
    pick: "왼쪽에서 이미지를 고르세요.",
    keys: "↑/↓ 또는 j/k로 이미지 이동 · ⌘/Ctrl+Enter로 캡션 저장",
    views: {
      image: "원본",
      mask: "마스크",
      overlay: "겹쳐 보기",
    },
    overlayHint: "이미지 위에 마스크를 40%로 겹칩니다",
    overlayNeeds: "이미지와 마스크가 모두 있어야 합니다",
    notGenerated: "생성되지 않음",
    noOverlay: "이 이미지에는 겹쳐 볼 것이 없습니다",
    noView: (view) => `이 이미지에는 ${view}이(가) 없습니다`,
    readOnly: "읽기 전용",
    /** {0}은 하한값, 예: "0.50 MP". 픽셀 수 칩에 붙습니다. */
    aboveFloor: (floor: string) => `리사이즈 하한 ${floor} 이상`,
    /** {0}은 하한값. 이 이미지에 스테이지를 돌려도 아무 일이 없는 이유입니다. */
    belowFloor: (floor: string) =>
      `리사이즈 하한 ${floor} 미만 — 이 이미지는 리사이즈되지 않으므로, workspace/resized를 걷는 스테이지들에게는 아예 보이지 않습니다. ⚙ 설정 › Preprocess에서 하한을 낮추세요.`,
    zoomHint: "⌘/Ctrl+스크롤로 확대 · 드래그로 이동 · 더블클릭하면 원래 크기",
  },
  ocr: {
    title: "이미지 속 텍스트",
    /** {0} is how many lines were recognized. */
    count: (n: number) => `${n}줄`,
    readOnly: "OCR 스테이지가 생성함",
    boxHint: "이미지 안에서의 좌측 상단 좌표(픽셀)",
  },
  caption: {
    master: "master",
    history: "history",
    revised: "revised",
    variants: "variants",
    where_master: "손으로 쓴 캡션; 스테이지는 읽기만 합니다",
    where_history: "이 캡션이 예전에 하던 말 — 덮어쓴 실행 직전의 내용입니다",
    where_revised: "스테이지 출력; 다음 실행이 고쳐 쓰면서 지금 이 글은 버전으로 남깁니다",
    where_variants:
      "생성된 파일 — v0이 손대지 않은 revised 캡션입니다. 여기서 손으로 고친 내용은 덮어써집니다",
    diffHere: "마지막 실행이 이 버전을 고쳐 썼습니다",
    diffElsewhere: (kind: string) => `마지막 실행은 ${kind}을(를) 고쳐 썼습니다 — 열기`,
    new: "신규",
    revert: "되돌리기",
    save: "저장",
    saveHint: "⌘/Ctrl+Enter",
    empty: "아직 캡션 파일이 없습니다 — 입력하고 저장하세요",
    saved: "저장됨 — 이전 글은 위쪽 버전으로 남았습니다; 이어서 트레이너의 TE 재인코딩을 돌리세요",
    savedStale:
      "저장됨 — .variants.txt가 오래되었습니다; correct와 트레이너의 TE 재인코딩을 다시 돌리세요",
    noCaption: "캡션 없음",
    tags: (n) => `태그 ${n}개`,
    clauses: (n) => `절 ${n}개`,
    unsaved: "저장하지 않은 미리보기",
    lookUpHint: "태그를 더블클릭하면 뜻을 찾아봅니다",
    bag: "태그 묶음",
  },
  diff: {
    written: "기록됨",
    by: (stage) => `${stage} 실행 결과`,
    lastRun: "마지막 실행",
    onDisk: "디스크에 있음",
    stale: "지나간 내용",
    staleHint: "이 실행 이후 캡션이 바뀌었습니다 — 지금 내용은 위 편집기에 있습니다",
    reordered: "태그는 같고 순서만 바뀌었습니다 — 아래 본문을 보세요.",
  },
  tag: {
    what: (tag) => `"${tag}"이(가) 무엇인가요?`,
    posts: (n) => `게시물 ${n}개`,
    close: "닫기 (Esc)",
    looking: "찾는 중…",
    notInstalled:
      "Danbooru 태그 KB가 아직 없습니다. 캡션 교정이 태그를 대조하는 기준이자, 이 패널이 읽는 자료입니다.",
    getIt: "설정 › 모델에서 받기",
    unknown: "Danbooru 태그가 아닙니다 — Anima 품질 태그, 위치 구문, 또는 오타입니다.",
    noDescription: "이 태그에는 위키 설명이 없습니다.",
    matchedAs: (name) => `“${name}”(으)로 찾았습니다.`,
  },
  picker: {
    title: "경로 고르기",
    use: "이 폴더 사용",
  },
  settings: {
    paneTitle: {
      general: "설정",
      advanced: "고급 설정",
      models: "모델 & 가중치",
      update: "업데이트",
    },
    home: "홈",
    modelsDir: "모델 폴더",
    roots: "데이터셋 루트",
    rootsHelp:
      "큐레이션 홈 기준 경로입니다; 트리들은 같은 상대 경로로 이어집니다. 도구는 워크스페이스에만 씁니다 — {0}은 Export만 건드립니다.",
    rootHelp: {
      src: "입력 — 원본 이미지와 손으로 쓴 master 캡션; 절대 쓰지 않습니다",
      master: "워크스페이스 — 수정한 master 캡션",
      dst: "워크스페이스 — 리사이즈된 이미지, derived 캡션, .variants.txt",
      masks: "워크스페이스 — {stem}_mask.png, 원본 하위 폴더 구조를 그대로 따릅니다",
      out: "출력 — Export가 내보내는 곳; 트레이너가 읽는 트리",
    },
    rootMissing: "없음 — ",
    stageDefaults: "스테이지 기본값",
    stageDefaultsHelp:
      "이 값을 받는 모든 스테이지에 채워지므로 각 폼이 다시 묻지 않습니다. 비워 두면 CLI 자체 기본값을 씁니다. {0}는 일부러 없습니다: 스테이지마다 알아서 잡습니다. {1}는 자기 플래그가 없는 유일한 값입니다: 스테이지마다 그 아래 자기 폴더({2}, {3})를 쓰므로, 루트를 옮기면 전부 함께 옮겨지면서도 두 스테이지가 한 리포트를 가리키는 일이 없습니다 — 큐레이션 감사 적용도 여기서 감사 리포트를 다시 읽습니다.",
    reportRootHint: "각 스테이지의 report.json이 떨어지는 곳 — 비우면 dst 루트 옆",
    maskRootHint: "각 생성기의 마스크 트리가 떨어지는 곳 — 비우면 병합이 채우는 masks 루트 옆",
    preprocessHelp:
      "스테이지가 다루는 바로 그 이미지들에 대해 돌아가므로, 이미지 하나만 Apply하면 그 이미지만 리사이즈됩니다. 이미 최신인 이미지는 건너뛰므로 다시 돌려도 거의 공짜입니다. 단계 값은 트레이너의 {0}와 맞아야 합니다.",
    hf: "Hugging Face",
    token: "토큰",
    tokenPlaceholder: "hf_… (huggingface_hub이 저장하며, 다시 보여주지 않습니다)",
    tokenHelp:
      "태거 백본과 SAM3 가중치는 허브에서 승인이 필요합니다 — 첫 실행에 읽기 권한 토큰이 필요합니다.",
    models: "모델",
    modelsHelp:
      "모든 스테이지는 처음 쓸 때 필요한 것을 스스로 받습니다 — 이 버튼은 그 기다림과 승인 거절을, 당신이 고른 순간으로 옮길 뿐입니다. 내려받기도 작업으로 돌아갑니다: 한 번에 하나씩이고, 스테이지 바가 아니라 여기에 보고하므로 이 창을 열어 둔 채로 둘 수 있습니다.",
    downloadAll: (n) => `없는 모델 ${n}개 모두 받기`,
    allInstalled: "모든 모델이 설치되어 있습니다",
    download: "받기",
    redownload: "다시 받기",
    downloadingRow: "받는 중…",
    downloadPack: "팩 받기",
    redownloadPack: "팩 다시 받기",
    gated: "승인 필요 — 위 토큰과 같은 계정으로 {0}.",
    gatedLink: "약관에 동의",
  },
  stage: {
    panels: {
      Resize: "리사이즈",
      Autotag: "자동 태깅",
      Curate: "캡션 정리",
      OCR: "OCR",
      Groups: "그룹",
      Masks: "마스크",
      Export: "내보내기",
    },
    titles: {
      resize: "버킷 크기로 리사이즈",
      autotag: "캡션 자동 태깅",
      position: "위치 절 붙이기",
      correct: "캡션 교정 + 좌우 반전",
      audit: "다중 시점 점검",
      ocr: "이미지 속 글자 인식",
      groups: "근접 그룹 만들기",
      masks_sam: "SAM3 인물 마스크",
      masks_merge: "마스크 합치기",
      export: "워크스페이스 내보내기",
    },
    shorts: {
      position: "위치",
      correct: "교정",
      audit: "점검",
      masks_sam: "인물",
      masks_merge: "합치기",
    },
    docs: {
      resize:
        "캡션 master를 모든 스테이지가 읽는 버킷 해상도 트리로 리사이즈합니다.\n\n" +
        "다른 스테이지는 모두 --dst 아래를 훑기 때문에, --src 에만 있는 이미지는 그들에게 보이지 않습니다. 각 이미지는 가장 적게 줄어드는 --target_res 단계로 들어가고, 그 단계의 토큰 대역 안에서 원래 비율을 지킵니다. 기하는 트레이너의 make preprocess-resize와 같으므로, 어느 쪽이 먼저 돌든 나머지 한쪽은 건너뜁니다.\n\n" +
        "항상 씁니다. 드라이런이 없고, 이미 목표 버킷에 있는 이미지는 다시 디코드하지 않고 건너뜁니다. glob은 --src 기준으로 맞춥니다.",
      autotag:
        "Anima Tagger로 데이터셋에 태그를 달아 revised 캡션을 씁니다.\n\n" +
        "리사이즈된 트리를 훑으며 이미지마다 캡션을 제안합니다. --mode는 셋 중 하나입니다: missing(어떤 캡션도 설명하지 않는 이미지, 기본값), merge(절은 그대로 두고 새 태그만 덧붙이기), overwrite(캡션을 통째로 교체). 읽는 캡션은 revised이고 없으면 master로 넘어가며, 쓰는 캡션은 언제나 revised입니다. 그때 밀려난 텍스트는 {stem}.history.txt 버전으로 남습니다.\n\n" +
        "기본은 드라이런이고 --apply가 씁니다. TE 캐시는 낡아도 겉보기에는 멀쩡하므로, 실제로 적용한 뒤에는 make preprocess-te를 이어서 도세요.",
      position:
        "여러 인물이 있는 캡션을 위치 절로 다시 씁니다 (SAM3 + Anima Tagger).\n\n" +
        "리사이즈된 데이터셋 위에서 검출 → 정렬 → 자르고 지우기 → 태깅 → 조합 순으로 돌며, 귀속되는 태그를 평평한 태그 뭉치에서 자기 절로 옮깁니다(--flatten이 그 역방향 패스입니다). 절은 --dst 아래 revised 캡션에만 붙고, 캡션 master에는 손대지 않습니다. docs/position_captions.md를 보세요.\n\n" +
        "기본은 드라이런이고 --apply가 씁니다. 이때 낡은 .variants.txt 사이드카를 지우므로, 뒤이어 TE 재인코딩을 도세요.",
      correct:
        "리사이즈된 전처리 이미지 옆에 교정한 캡션을 씁니다.\n\n" +
        "revised 캡션을 제자리에서 교정하고, revised가 아직 없는 이미지에 대해서만 master를 읽습니다. 변형 사이드카는 선택입니다. 항상 씁니다: 드라이런도 리포트도 없습니다.",
      audit:
        "1girl 캡션 중 사실은 한 인물의 여러 시점인 이미지를 점검합니다.\n\n" +
        "위치 스테이지가 single-subject로 건너뛴 이미지를 훑어, girl 프롬프트가 인물을 둘 이상 찾아내는 이미지를 모두 보고합니다. docs/multiview_audit.md를 보세요. 위치 스테이지는 --multiview_audit로 이 점검을 자기 첫 단계로 돌리며, 보통은 그쪽이 들어오는 길입니다.\n\n" +
        "기본은 드라이런이고, --apply는 빠진 태그를 --dst 아래 revised 캡션에 씁니다. master에는 쓰지 않습니다. 뒤이어 TE 재인코딩을 도세요.",
      ocr:
        "이미지마다 그 안의 글자를 읽어 무엇이라고 쓰여 있는지 기록합니다.\n\n" +
        "리사이즈된 트리를 훑으며 AnimeText 검출기(anime_tools.ocr.animetext, torch 위의 YOLO12. 첫 사용 때 54 MB를 받고, 가중치가 GPL-3.0이라 패키지에 넣지 않습니다)로 글자 블록마다 상자를 치고, 만화 VL 리더(anime_tools.ocr.sfx, 손으로 그린 효과음에 파인튜닝한 PaddleOCR-VL-1.6. torch, 첫 사용 때 약 2.8 GB)로 그 상자를 모두 읽어 {stem}.ocr.txt를 OCR 트리에 리사이즈된 트리와 같은 구조로 씁니다. 캡션은 읽지도 쓰지도 않으므로 뒤에 TE 재인코딩이 필요 없습니다.\n\n" +
        "말풍선 대사와 그림 위에 그려진 효과음을 검출기 하나가 함께 잡습니다. 상자 하나의 읽기가 곧 한 줄이므로, 디코드 가드가 물리친 읽기는 그 상자째 버려집니다. --mask_dir을 주면 어떤 상자도 덮지 않은 텍스트 마스크 성분까지 따로 한 줄로 읽습니다(det 0.000). 줄마다 검출기의 상자 신뢰도(det)와 리더의 평균 토큰 신뢰도(score)가 붙습니다.\n\n" +
        "기본은 드라이런입니다. 드라이런은 썼을 모든 줄을 담은 report.json을 남기고, --apply는 사이드카를 쓰는 것 말고는 아무것도 하지 않습니다.",
      groups:
        "PE-Spatial 시각 유사도로 데이터셋 이미지를 묶어 groups.json 매니페스트를 만듭니다.\n\n" +
        "학습 전처리가 아니라 큐레이션 도구입니다: 작가별로 거의 같은 이미지들을 묶어 GUI 데이터셋 탭에서 그룹으로 걸러 볼 수 있게 할 뿐, 다른 것은 쓰지 않습니다. 두 이미지는 셀별 하한 --cell-match-min에서 match_frac가 --match-frac-min 이상일 때 묶입니다. 다시 돌리면 공유 PE-Spatial 특징 캐시를 재사용하므로 임계값을 다시 맞추는 일은 쌉니다.",
      masks_sam:
        "SAM3 인물 마스크를 workspace/masks_sam/ 에 씁니다.\n\n" +
        "--prompts는 마스크로 가릴 것(손실에서 무시할 것)을, --focus-prompts는 남길 것(그 밖은 모두 가림)을 가리킵니다. 둘 다 주면 남긴 영역에서 가릴 영역을 뺀 만큼이 살아남습니다. 인물 프롬프트는 기본적으로 학습된 소프트 프롬프트(--prompt_embed)가 맡습니다. 평범한 텍스트 프롬프트를 쓰려면 none을 주세요.",
      masks_merge:
        "여러 곳의 마스크를 픽셀별 최솟값으로 합칩니다(가려진 영역의 합집합).\n\n" +
        "(rel_dir, name)을 키로 합치므로 입력마다 같은 상대 경로에 있는 마스크가 서로 만납니다. 중첩된 구조는 --output-dir 아래에 그대로 남습니다. 없는 입력 디렉터리는 오류가 아니라 건너뜀입니다 — 기본 입력은 생성기 한 곳의 트리이고, 두 번째 트리(손으로 칠한 마스크, 다른 도구의 출력)는 그 옆에 적어 두기만 하면 됩니다.",
      export:
        "워크스페이스를 트레이너가 읽는 경로로 내보냅니다.\n\n" +
        "이 패키지에서 workspace/ 바깥에 쓰는 유일한 작업입니다: 리사이즈된 이미지와 마스크와 캡션이 --out 아래로 복사됩니다.\n\n" +
        "기본은 드라이런입니다. --apply는 실제로 복사하면서 모든 줄을 목적지와 다시 견주므로, 드라이런 뒤에 편집된 파일은 덮이지 않고 보고됩니다. 내보낸 것을 되무르는 일은 여기 플래그가 아니라 GUI의 Undo입니다.",
    },
    notes: {
      resize:
        "리사이즈된 트리를 읽는 모든 스테이지 앞에서 자동으로 실행됩니다. 여기 기본값이 그 모두에 적용됩니다.",
      autotag:
        "revised 캡션을 리사이즈된 트리 아래에 씁니다. master는 없을 때 대신 읽을 뿐 편집하지 않습니다. missing은 이미 캡션이 설명하는 이미지를 건너뜁니다.",
      correct:
        "revised 캡션을 제자리에서 교정합니다. master는 revised가 없는 이미지에서만 읽고, 편집하지 않습니다.",
      ocr: "{stem}.ocr.txt를 리사이즈된 트리와 같은 구조로 OCR 트리에 씁니다. 캡션은 읽지도 쓰지도 않습니다.",
      export:
        "워크스페이스 바깥에 쓰는 유일한 스테이지입니다. 실행하면 트레이너가 읽는 트리로 복사하고, 이미 같은 것은 건너뜁니다. Undo는 덮어쓴 텍스트를 되돌리지만, 교체된 픽셀은 되돌릴 수 없습니다.",
    },
    run: "실행",
    runBatch: "일괄 실행",
    undo: "되돌리기",
    cancel: "중단",
    aim: (rel) => `${rel} 하나만`,
    noImage: "먼저 사이드바에서 이미지를 고르세요",
    batchHint: "설정의 path_pattern이 가리키는 모든 이미지",
    undoHint: "마지막 실행이 쓴 캡션을 되돌립니다",
    missingModels: (n) => `↓ 모델 ${n}개 없음`,
    missingModelsHint: (names) =>
      `아직 받지 않음: ${names}. 첫 실행이 알아서 받습니다 — 이 버튼은 그 기다림을 당신이 고른 순간으로 옮길 뿐입니다.`,
    noStages: "스테이지가 없습니다.",
    unavailable: (title, error) => `${title}을(를) 쓸 수 없습니다: ${error}`,
    reinstall: "다시 설치:",
  },
  form: {
    options: "옵션",
    onePerLine: "한 줄에 하나씩",
    reset: "기본값으로 되돌리기",
    advanced: (n) => `▸ 고급 (${n})`,
    advancedHide: "▾ 고급",
    advancedHint: "실행할 때마다 바꿀 일은 거의 없는 값들",
    advancedDirty: (n) => `접혀 있는 값 ${n}개가 기본값이 아닙니다`,
  },
  runner: {
    nothingToUndo: "되돌릴 것이 없습니다",
    nothingApplied: "아직 실행한 것이 없습니다 — Undo는 실행이 쓴 내용을 되돌립니다",
    following: (id) => `${id} 실행 중`,
    exit: (code) => `종료 코드 ${code}`,
    changed: (n) => `파일 ${n}개 변경`,
    undoing: "되돌리는 중…",
    undone: (restored, removed) => `되돌림: ${restored}개 복원, ${removed}개 삭제`,
    skipped: (what) => `— 건너뜀 ${what}`,
    logClosed: "로그 스트림이 닫혔습니다",
  },
  guide: {
    title: "가이드북",
    loading: "가이드북을 읽는 중…",
  },
  job: {
    log: "로그",
    logHint: "작업이 출력한 내용 전체",
    title: "작업 로그",
    empty: "아직 출력이 없습니다",
    of: (done, total) => `${done} / ${total}`,
    step: (index, total, label) => `단계 ${index}/${total} · ${label}`,
    tail: (n) => `마지막 ${n}줄만 표시합니다`,
  },
  downloads: {
    starting: "시작하는 중…",
    finished: "내려받기 완료",
  },
  update: {
    version: "버전",
    help: "패널은 GitHub에 최신 릴리스를 묻고 그 답을 여섯 시간 동안 캐시합니다 — 새로고침이 곧 요청이 되지는 않습니다. 업데이트는 이 서버가 실행 중인 환경을 교체합니다. 데이터셋에는 아무것도 쓰지 않지만, 끝난 뒤에는 GUI를 다시 시작해야 합니다.",
    installed: "설치됨",
    latest: "최신 릴리스",
    never: "아직 확인하지 않음",
    viewRelease: "릴리스 페이지 ↗",
    checking: "확인하는 중…",
    checkNow: "지금 확인",
    checkedAt: (when) => `${when} 확인함`,
    checkFailed: (err) => `확인 실패: ${err}`,
    state: {
      current: "최신 상태",
      available: "업데이트 있음",
      ahead: "최신 릴리스보다 앞섬",
      unknown: "버전을 알 수 없음",
    },
    auto: "시작할 때 업데이트 확인",
    autoHint: "끄면 [지금 확인]을 누를 때만 GitHub에 물어봅니다.",
    releaseNotes: "릴리스 노트",
    install: "설치 형태",
    installKind: {
      "uv-tool": "install.sh가 만든 uv tool 환경입니다 — 여기서 업데이트합니다.",
      checkout: "저장소의 git 체크아웃에서 실행 중입니다.",
      other: "uv tool이 아닌 다른 방법으로 이 환경에 설치되었습니다.",
    },
    warning:
      "업데이트는 GUI가 실행 중인 환경을 교체합니다. anime-tools-gui를 다시 시작하기 전에 시작한 스테이지는 파일을 찾지 못할 수 있습니다.",
    restart: "업데이트를 설치했습니다 — anime-tools-gui를 다시 시작하세요.",
    updateTo: (tag) => `${tag}(으)로 업데이트`,
    upToDate: "이미 최신 릴리스입니다",
    nothingToDo: "설치할 것이 없습니다",
    updating: "업데이트하는 중…",
    starting: "시작하는 중…",
    installedOk: "업데이트를 설치했습니다",
  },
};

export default ko;
