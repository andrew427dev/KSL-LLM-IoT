# HANDOFF — 2026-06-11

본 문서는 KSL-LLM-IoT 프로젝트 작업을 이어받는 팀원·차기 세션이 *현재 상태와 다음 단계*를 파악하기 위해 작성된다. 매번의 작업 마무리 시 갱신한다.

- 작성 시점: **2026-06-11** (Week 6 — 최종 제출 6/22)
- 작성자: 이성준 + Claude (Opus 4.8)
- 대상: 배진규(팀원), 이후 본인, 차기 세션

---

## 1. 현재 상태 (2026-06-11 기준)

### 1.1 코드

- 기본 브랜치: `main` (보호) ← `develop` ← feature 브랜치 (GitFlow 변형)
- **현재 브랜치: `develop`** — **PR #9 MERGED** (2026-06-11, CI 그린). 131차원 전환·서버 학습 파이프라인·물리 버튼·페르소나·시간 리샘플링·도메인 갭 정리(§1.6)가 develop에 통합됐다.
- 머지 전 전수 리뷰(9각도 finder + 후보별 검증, Claude Code) 결과 **15건이 PR #9 리뷰 코멘트에 기록** — develop 대비 회귀 없음, High 2건 포함 후속 수정은 §4.5.
- 주의: 머지 시 원격 `feat/131-feature-format`이 자동 삭제되어 재푸시(84af918)로 복구했다 — **GPU 재학습 체인이 이 브랜치를 pull하므로 체인 종료 전까지 삭제 금지**.
- 머지 완료 PR:

  | PR | 브랜치 | 내용 |
  |----|--------|------|
  | #2 | `feat/smoke-test-pipeline` | smoke-test 인프라, 비동기 파이프라인, genai SDK |
  | #5 | `docs/data-sources` | 데이터 소스 섹션 확장 |
  | #6 | `develop` → `main` | develop 통합 머지 |
  | #7 | `feat/two-hand-input` | 양손 입력 전환 (INPUT_SHAPE 63→126) |
  | #8 | `ci/static-checks` | 정적 검증 워크플로 추가 |
  | #9 | `feat/131-feature-format` | 131차원 전환, GPU 학습 파이프라인, 물리 버튼 4개, 시간 리샘플링, 도메인 갭 정리 |

- 머지 흐름: feature 브랜치 → `develop` → `main`.

### 1.2 검증된 동작

Raspberry Pi 4B (Raspberry Pi OS Trixie, aarch64) + Pi Camera v1 (OV5647) 환경:

- [x] `rpicam-vid` 서브프로세스 카메라 백엔드
- [x] MediaPipe 손 인식
- [x] 분류기 더미 모드 (`model/ksl_model.tflite` 부재 시)
- [x] SentenceBuilder 오프라인 모드 (`GEMINI_API_KEY` 부재 시)
- [x] 헤드리스 실행 (`KSL_HEADLESS=1`)
- [x] 비동기 파이프라인 — 인식 직후 카메라 루프 정지 없음
- [x] `Ctrl+C` 안전 종료

GPU 학습 서버 (`ssh -p 30007 root@cscloud.gpu3.hufs.ac.kr`, RTX 4000 Ada 20GB, CUDA 12.7, Python 3.10, 컨테이너 메모리 제한 24GB):

- [x] 종단 학습 파이프라인 — `scripts/deploy_code.sh` → `upload_dataset.sh`(MP4 제외 tar 스트림) → `server_setup.sh`(TF 2.15.1 GPU) → `run_training.sh`(변환→증강→학습→평가) → `fetch_model.sh`
- [x] 검증 실행 (2026-06-10, 예시 데이터셋 18단어×3,540샘플, `KSL_LABELS_OVERRIDE` 사용): GPU 학습 ~84 epoch, **TFLite 평가 정확도 0.98**, 추론 0.41ms, `ksl_model.tflite` 740KB (131차원 입력 (1,30,131)→(1,18) 검증 통과). 단 시연자 1명+슬라이딩 윈도우 데이터라 수치는 낙관적 — 파이프라인 동작 검증 목적
- [ ] 서버 confusion matrix의 한글 라벨이 □로 표시 (matplotlib 한글 폰트 부재 — 기능 무관)

RPi 실기 통합 (2026-06-10/11, USB 웹캠 + 운용 모델):

- [x] SSH 직접 운용: `ssh rpi` (~/.ssh/config, IP는 DHCP — ARP에서 MAC `d8-3a-dd` 검색으로 재발견)
- [x] USB 웹캠(OpenCV 백엔드, 카메라 단독 30FPS) / USB 전원 스피커(3.5mm) / LCD / 부저 / **물리 버튼 4개 결선·동작**
- [x] 운용 모델(30클래스) 실기 가동 — 종단 경로 실증: 카메라→인식→Gemini(실호출)→TTS(한국어)+LCD(영어)
- [x] 출력 이중화: 음성=한국어, LCD·GUI=영어 (LCD 한글 미지원)
- [x] tmux 세션 운용 (`tmux new -d -s ksl ...` — nohup은 와이파이 끊김에 취약)
- [x] 진단 도구 `tools/diagnose_live.py` — top-3·검출률·VNC 미리보기
- [ ] **실기 인식률 미달** — 진단: MediaPipe 추론 8-9.5FPS(병목), 시간 리샘플링으로 창 정합했으나 모델 과신(전 예측 ~1.00)·도메인 갭 잔존 → 16명 재학습 진행 중 (§4)

### 1.3 미완료

- [x] KSL_LABELS 30단어 확정 (2026-06-10) — AI Hub 표제어 기준 재구성, `config/settings.py` 반영 (§1.5 WORD 목록)
- [x] AI Hub 데이터셋: **16명(REAL01~16) 재다운로드 완료** (2026-06-12, 수정판 fail-fast 스크립트). 시연자 분포 직접 검수(2-W) 통과 — 각 225 dir=3,600, 30/30 단어 매칭, 잔재 0. chain_train.sh 재학습 완주
- [x] 하드웨어 결선 전부 완료 (LCD/부저/스피커/버튼 4개/USB 웹캠)
- [x] Gemini API 키 등록 (RPi .env) — 실호출 검증
- [x] **실기 인식률 확보** — 16명 재학습 모델 RPi 실기 **27/30 단어 정상·확신 0.3~0.9 건강(1.00 과신 병리 해소)**. 잔존 혼동 밥/배고프다/주다는 G7 스파이크로 위치 부재(G7) 아닌 런타임 손모양 갭으로 판명 → 데모 시나리오 회피 결정(§1.6 G7 검증). future work: 본인 데이터 수집 혼합(§4.2, collect_data 시간 리샘플 수정 선행 — 미착수)
- [x] PR #9 → develop 머지 (2026-06-11). main 릴리스는 제출 직전 수행
- [ ] 데모 영상 (`docs/final_report_outline.md` 샷 리스트) · 최종 보고서(PDF ≥10p) · PPT(영어 ≥15슬라이드) — 아웃라인 docs/에 완비
- [ ] AI Hub API 키 파일: 서버 `/mnt/data/ksl/.aihub_key` (비밀 — 커밋 금지)

### 1.4 입력 차원 정책 (2026-06-10 개정 — 131차원)

- KSL는 양손 언어이므로 단일 손(63차원) 입력으로는 의미 보존이 불가능.
- 입력 = **131차원 = [LEFT 21×3 | RIGHT 21×3 | 손목간 벡터 3 | presence flag 2]**. 레이아웃 단일 정의: `src/feature_format.py` (추론 `hand_tracker.py`·변환 `convert_aihub.py`가 공유 — train/serve skew 구조적 차단).
- 각 손은 손목(landmark 0) 상대좌표를 **intra-hand scale(‖landmark9−landmark0‖)로 나눠** 정규화 — 좌표계 단위(MediaPipe 0~1 vs AI Hub 미터)·카메라 거리·손 크기에 invariant. 이 scale 정규화가 없으면 AI Hub(손목상대 max≈0.09m)와 MediaPipe(max≈0.1~0.35) 분포가 어긋난다(실측).
- 손목간 벡터 = (우손목−좌손목)/두 손 scale 평균. 양손 단어의 두 손 상대 위치 신호 보존. 한 손이라도 부재면 0.
- 미감지 손은 zero + presence flag 0. 양손 모두 미감지된 프레임은 시퀀스 누락(`extract_landmarks()` None). 분류기는 연속 `NO_HAND_RESET_FRAMES`(10) 미검출 시 버퍼를 비운다(stale 추론 방지).
- **시간 정규화 (2026-06-11)**: 런타임 FPS는 환경 의존(RPi 4B + MediaPipe 실측 8-9.5FPS — 추론이 병목, 카메라는 30FPS). 분류기는 (timestamp, vector) 버퍼에서 **최근 1.0초(`SEQUENCE_LENGTH/TRAIN_FPS`)를 30포인트로 선형 보간**해 학습 분포(30fps 영상)와 시간 창을 정합한다 (`src/classifier.py:_resample_window`). 프레임 수 기반 버퍼는 저FPS에서 1초 동작을 ~4초 창으로 왜곡 — 실기 인식 실패의 1차 원인이었다.
- **잔여 FPS 영향 (보정 후에도 실재)**: ① 정보 밀도 — 1초 창에 실측 8~9프레임뿐, 보간은 형태만 복원하고 빠른 동작의 고주파 성분은 누락 ② 저가 웹캠 모션 블러 → 랜드마크 노이즈. 단, 보정 후 진단에서 출력이 불확실 분산이 아니라 **유사 의미 단어로 1.00 쏠림**(밥→배고프다·주다) — 입력 붕괴가 아닌 도메인 갭 패턴. 주범 판정은 16명 재학습의 held-out 시연자 정확도(=FPS 무관한 순수 일반화 측정)로 한다.
- **FPS 상승 백로그**: MediaPipe 신형 tasks API 전환 (legacy solutions 대비 RPi 고속화 보고 다수, 마이그레이션 비용 중간) — 16명 재학습 후에도 인식 부족 시 착수.
- MediaPipe handedness는 *입력이 거울 모드(selfie)임을 가정* — `main.py`/`collect_data.py`가 `cv2.flip(frame, 1)` 후 호출하므로 'Left' = 사용자의 해부학적 왼손. `extract_landmarks(frame, mirrored=False)`로 호출하면 라벨을 swap한다.
- 한 손 단어(예: "안녕", "주세요" 중 한 손 위주 동작)는 시연자가 자신의 **주손**으로 자연스럽게 수행한다 — 왼손잡이는 LEFT에, 오른손잡이는 RIGHT에 수집된다. `model/augment.py:flip_horizontal`이 슬롯·flag swap을 수행하므로 학습 데이터엔 양쪽 분포가 자연스럽게 채워진다. 단 flip은 **방향 의존 수어를 제외한 `FLIP_SAFE_LABELS`만 opt-in**. **억지로 비주손 시연을 강요하지 않는다**.
- handedness 신뢰도: `HANDEDNESS_SCORE_THRESHOLD`(settings.py, 기본 0.7, `.env` 오버라이드) 미만의 손은 라벨 대신 **x좌표 fallback**(거울모드에서 작은 x = 좌손)으로 슬롯 배정. 두 손이 동일 라벨로 분류되는 충돌 케이스는 score 낮은 쪽을 반대편으로 재배정하며 stderr에 warning을 출력한다.

### 1.5 AI Hub 수어 데이터셋 변환 파이프라인 (2026-05-27)

직접 촬영 외에 [AI Hub 수어 영상 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=103)의 3D 키포인트를 프로젝트 CSV로 변환하여 학습 데이터를 보강할 수 있다.

**데이터셋 구조:**

| 경로 | 내용 | 필요 여부 |
|------|------|-----------|
| `라벨링데이터/REAL/WORD/*_keypoint/` | 프레임별 3D 키포인트 JSON | **필요** |
| `수어 영상/.../morpheme/` | WORD번호 → 한국어 단어 매핑 | **필요** |
| `원천데이터/REAL/WORD/` (MP4) | 원본 영상 | 불필요 (키포인트 이미 추출됨) |

**AI Hub 3D 키포인트 → 프로젝트 호환성:**

| 항목 | AI Hub | 프로젝트 |
|------|--------|----------|
| 손 관절 수 | 21개 | 21개 (동일) |
| 3D 좌표 | (x, y, z, confidence) | (x, y, z) |
| 정규화 | 절대 좌표 (멀티카메라 3D 복원) | 손목 기준 상대 좌표 |

변환 시 confidence를 버리고 (x,y,z)만 추출한 뒤 손목 기준 정규화를 적용한다. `hand_tracker.py:_normalize_hand()`와 동일한 로직이다.

**KSL_LABELS 매칭 현황 (2026-06-10 라벨 전면 개정 후):**

- 30단어 전부 AI Hub 사전 표제어와 **정확 매칭** (30/30, WORD 패키지 45개).
- 변환은 반드시 `--exact` 모드 사용 — 어근 매칭은 의미 오염을 유발한다 (실측: "기다리세요"↔"다리", "맞다"↔"뺨맞다", "먹다"↔"얻어먹다"). 이 오염 때문에 먹다·기다리세요는 라벨에서 제외하고 생활명사·감정어로 교체했다.

**현재 보유 키포인트:** WORD1501-1520 (운전면허, 골키퍼 등) — 새 KSL_LABELS와 겹침 없음 (파이프라인 검증용으로만 사용됨). 아래 45개 WORD 패키지의 키포인트를 AI Hub에서 다운로드해야 한다:

```
나(1157)  당신(1353)  좋다(0738,1191)  싫다(1278,1385)  맞다(1174,2317,2318)
가다(0943,0944,0946,1345)  오다(1149)  서다(1148,1344)  자다(1377,1378,1544)
주다(2394,2395,2396)  배고프다(0953,1197)  목마르다(2036)  아프다(1152)
피곤하다(1158)  춥다(1248)  덥다(1382)  슬프다(0009)  화나다(1236)
행복(1169)  감사(1290)  부탁(1589)  돕다(2388,2389,2390)
밥(1534)  병원(1496)  의사(0163)  엄마(1528)  가족(1492)  친구(1204)
얼마(0436)  완료(0742)
```

**변환 스크립트 사용법:**

```bash
# 매칭 현황 확인 (변환 없이 스캔만) — --exact 필수 (어근 오염 차단)
python convert_aihub.py --dataset /path/to/aihub/dataset --scan --exact

# 정면 카메라만 변환 (실 사용 환경과 유사)
python convert_aihub.py --dataset /path/to/aihub/dataset --angles F --exact

# 전체 카메라 각도 사용 (데이터 5배, 다양성 확보)
python convert_aihub.py --dataset /path/to/aihub/dataset --exact

# 슬라이딩 윈도우 stride 조정 (작을수록 샘플 많음)
python convert_aihub.py --dataset /path/to/aihub/dataset --stride 10 --exact
```

출력은 `data/landmarks/<단어>/aihub_NNNN.csv`로 저장되며, 기존 직접 촬영 데이터와 동일 디렉터리에 병합된다. `train.py`가 자동으로 인식한다.

**좌표계 차이 — 실측 확정 (2026-06-10):** AI Hub는 멀티카메라 3D 복원 좌표(미터, 비거울), MediaPipe는 0~1 정규화 좌표(거울 영상)이다. `tools/verify_aihub_alignment.py`가 동일 영상의 MP4(런타임 경로)와 키포인트를 프레임별 비교한 결과:

- 필요한 변환 = **x 부호 반전** (`convert_aihub.py:AIHUB_AXIS_SIGNS = (-1, 1, 1)`), 좌우 슬롯 swap 불필요.
- 변환 적용 후 상관계수 x=0.954, y=0.982, z=0.577 (무변환이면 x=−0.954 — 좌우 반전 데이터로 학습하게 됨).
- z 상관 0.577은 MediaPipe 단안 추정 z의 한계 — 방향 일치하는 보조 신호로 유효. ToF 카메라 불필요.
- 스케일 차이는 feature_format의 intra-hand scale 정규화가 흡수. 데이터셋 버전 변경 시 검증 스크립트를 재실행한다.

### 1.6 학습 데이터 ↔ 런타임 입력 도메인 갭 전수 정리 (2026-06-11)

131차원 벡터 조립은 `src/feature_format.py` 단일 소스를 추론(`hand_tracker.py`)·변환(`convert_aihub.py`)이 공유하므로 **전처리 코드 차원의 train/serve skew는 구조적으로 차단**되어 있다(§1.4). 잔여 불일치는 전부 *동일한 함수에 들어가는 입력 값의 통계 분포 차이* — 즉 센서 도메인 갭이다. 실기 인식률 미달(§1.2)의 원인 후보를 전수 정리한다.

| # | 갭 | 학습 데이터 (AI Hub) | 런타임 (RPi) | 대응 상태 |
|---|----|---------------------|--------------|-----------|
| G1 | z축 품질 | 멀티뷰 3D 복원 z (정밀) | MediaPipe 단안 추정 z — 동일 영상 상관 **0.577** (x=0.954, y=0.982). z 성분은 131차원 중 43개(33%) | `z_jitter` 증강(`model/augment.py`) — 16명 재학습 반영 중. 단 jitter는 스케일·노이즈만 모사, 단안 z의 구조적 바이어스는 미커버 |
| G2 | 시간 정보 밀도 | 30fps 실측 30프레임 (독립 측정 30개) | 실측 8~9프레임을 30포인트로 **선형 보간**(`classifier.py:_resample_window`) — 저역 필터라 빠른 동작의 고주파 성분 소실, "구간별 직선" 궤적은 학습 분포에 없는 형태 | **대응 없음** — 백로그 증강 ① 참조. `time_warp`(±15% 속도 변화)는 이 변형을 모사하지 못함 |
| G3 | 검출 실패·handedness 노이즈 | 멀티뷰 복원이라 양손 상시 존재, 좌우 구분 ground-truth 수준 | 저FPS+모션 블러로 미검출 빈발(한 손만 잡히는 프레임), handedness 오분류 시 슬롯 스왑·x좌표 fallback 개입(`hand_tracker.py`) — presence flag 패턴 자체가 학습 분포 밖 | **대응 없음** — 백로그 증강 ② 참조 |
| G4 | 시연자 분포 | 전문 수어자 (1차 모델은 3명 과적합), 동작 크고 표준 속도 | 실사용자 — 동작 작고 속도·정확도 상이 | 16명 재학습 진행 중(§4.1). 잔여분은 본인 데이터 혼합(§4.2) |
| G5 | 과신 → threshold 무력화 | 좁은 시연자 분포 학습 → 전 예측 ~1.00 | `CONFIDENCE_THRESHOLD=0.85`가 오인식을 거르지 못함 | label smoothing 0.1(`train.py`) — 16명 재학습 반영 중 |
| G6 | x·y 등방성 (2026-06-11 리뷰 발견) | 멀티뷰 3D 복원 — 등방 미터 좌표 | MediaPipe 좌표는 축별 정규화(x÷너비, y÷높이) — 640×480에서 **y가 x 대비 4/3배 스케일**. intra-hand scale(혼합축 L2 norm)로는 소거 불가. 축별 상관계수는 스케일 불변이라 정합 도구가 탐지 못함(r_y=0.982와 공존) | **(w,h,w) 등방 보정 적용** (`hand_tracker.py:_build_feature_vector`, 2026-06-11) — 런타임을 학습 분포(등방)에 정합. 보정 코드 + 새 모델을 RPi에 **동시 배포**해 재진단 1회로 검증 |
| G7 | 조음 위치 결손 (2026-06-11 배진규 제기) | AI Hub 키포인트는 hand 21×2 외에 **pose 25관절·face 70점(2D 픽셀·3D 미터 모두, conf 포함)** 제공 — 손-신체 상대 위치 복원 가능 (실측: WORD1509 JSON, 코·목·어깨·골반 앵커 conf 1.00) | 131차원은 손목 상대화가 **조음 위치(손이 몸 어디에 있는가)를 전부 소거** — 손모양 유사·위치 구분 단어쌍(밥/배고프다 등)을 구조적으로 분리 불가. 1차 진단의 밥→배고프다·주다 1.00 혼동과 정합. MediaPipe Hands는 신체 앵커 미제공 | **재진단 게이트** — 위치 구분 쌍 혼동 잔존 시 2차 사이클로 적용. 설계: 얼굴 앵커 상대 손 위치 +4차원(131→135) — 학습=face 키포인트(중심·눈 간격), 런타임=MediaPipe Face Detection(BlazeFace 경량, N프레임 1회). Pose/Holistic은 8~9.5FPS 병목에서 비현실적. 적용 전 `verify_aihub_alignment` 확장으로 도메인 정합 실측 필수 |

**백로그 증강 (G2·G3 직접 공략 — 학습 데이터를 런타임처럼 열화):**

1. **low-FPS 시뮬레이션**: 30프레임 시퀀스를 랜덤 8~10포인트로 다운샘플 후 다시 30포인트로 선형 보간 — 런타임 입력의 smoothing 특성을 그대로 재현. `classifier._resample_window`와 동일 수식 재사용 가능.
2. **presence dropout**: 랜덤 구간에서 한 손 슬롯을 zero + flag 0으로 — MediaPipe 미검출 깜빡임 패턴 재현. feature_format 불변식(부재 손=zero, wrist_vec=0)은 기존 `time_warp` 복원 코드와 동일 규칙 적용.

**주범 역추적 신호 (16명 재학습 후 재진단 시):** 특정 *의미 유사 단어쌍* 혼동이 잔존하면 G1·G4(도메인·시연자 갭), *빠른 동작 단어만* 일관 오류면 G2(보간 smoothing)가 주범 — `tools/diagnose_live.py` 결과를 단어별 오류 패턴으로 분류해 판정한다. **G6은 전 단어 공통의 상수 기하 왜곡**이므로 held-out 정확도는 높은데 실기 전반이 미달하는 패턴이면 등방 보정+재학습(§4.5 ①)을 우선 적용한다.

**G7 검증 결과 (2026-06-12 스파이크 — 실측으로 G7 보류):** 16명 모델 RPi 실기에서 **27/30 단어 정상·확신 0.3~0.9 건강(1.00 병리 해소)**, 잔존 혼동은 밥/배고프다/주다(특히 배고프다→돕다 0.92). G7 적용 전 스파이크(MP4 부재로 도메인 정합 대신 AI Hub 키포인트의 *분리 가능성* 측정)에서 **손-앵커 위치가 이 3단어를 분리하지 못함**(face/pose 전 쌍 분리도 <0.6; 손 도달 최고높이로는 밥만 부분 시그니처 min‖y‖1.86, 배고프다·주다·돕다는 3~4로 겹침). 반면 이 단어들의 held-out f1는 0.96~0.98 — 모델은 *깨끗한 데이터에선* 손모양으로 구분함. ∴ 주범은 위치 부재(G7)가 아니라 **런타임이 손모양 미세차를 잃는 것(G2 저FPS 6.4·G3 검출노이즈·G4 사용자 실행)**. G7 풀 구현은 며칠 대비 밥 하나만 건질 위험 → 보류. 정공법 = §4.2 본인 데이터(런타임 파이프라인으로 수집→손모양을 RPi가 보는 그대로 학습). 단기 결정: **데모 시나리오로 회피**(잘 되는 27/30 사용).

---

## 2. 환경 제약 — 함정과 해결법

다음 항목들은 *학기 진행 중 실제로 부딪혀서 통과한* 사항이다. 작업 재개 시 같은 함정을 재현하지 않도록 한다.

| # | 증상 / 함정 | 원인 | 해결 |
|---|------------|------|------|
| 2-A | `pip install mediapipe`가 `No matching distribution found` | Python 3.13 + aarch64 조합에 휠 없음 (2026-05 시점) | `uv venv --python 3.11`로 Python 3.11 강제 |
| 2-B | RPi OS Trixie 기본 Python이 3.13 | 시스템 업그레이드 흐름 | `uv`가 prebuilt 3.11 다운로드 — 시스템 Python은 그대로 둠 |
| 2-C | `apt install python3.11` 실패 | Debian Trixie 기본 저장소에서 제외됨 | apt 사용 안 함, `uv` 사용 |
| 2-D | `python-prctl` 빌드 실패 (`libcap`) | picamera2 의존성 컴파일 시 헤더 필요 | `sudo apt install libcap-dev` |
| 2-E | `ModuleNotFoundError: libcamera` | `python3-libcamera` apt 패키지는 시스템 Python 3.13 ABI 전용 | picamera2 사용하지 않고 `rpicam-vid` 서브프로세스 백엔드로 우회 (이미 구현됨) |
| 2-F | `libcamerify python ...`로 OpenCV가 `read()` 시 빈 프레임 | LD_PRELOAD V4L2 셈과 OpenCV 호환성 한계 | 사용 금지. `rpicam-vid` 백엔드만 사용 |
| 2-G | `ValueError: No API key` at `genai.Client()` | google-genai 신 SDK가 생성 시점에 검증 | `SentenceBuilder.__init__`에서 `GEMINI_API_KEY` 빈 값 체크 → `_offline=True` (이미 구현됨, `345f11e`) |
| 2-H | 콘솔에 한글 깨짐 (`?덈뀞`) | PuTTY 기본 인코딩이 UTF-8 아님 | PuTTY → Window → Translation → Remote character set = **UTF-8** |
| 2-I | 인식 직후 카메라 정지 (1~3초) | Gemini API + LCD I2C + beep이 메인 루프에서 동기 실행 | 워커 스레드로 분리 (`cdb2ac2`) |
| 2-J | `gh pr edit` title 갱신 안 됨 | GitHub의 Projects(classic) deprecation 경로 충돌 | `gh api -X PATCH /repos/.../pulls/N`으로 직접 호출 |
| 2-K | 모델이 정확해도 항상 엉뚱한 단어 출력 | sklearn `LabelEncoder`가 라벨을 유니코드 정렬 → 학습 인덱스 ≠ `KSL_LABELS[label_idx]` 역매핑 | LabelEncoder 금지, `train.py:LABEL_TO_IDX` 원본 순서 인코딩 (smoke test가 회귀 가드) |
| 2-L | TFLite 변환 시 `LLVM ERROR: Failed to infer result type(s)` abort | TF 2.16+(Keras 3)의 LSTM 변환 MLIR 버그 | `tensorflow<2.16` 고정 (2.15.1 검증) |
| 2-L′ | TFLite 변환 시 `TensorListReserve ... element_shape to be static` 또는 OOM kill(exit 137) | ① dynamic batch에서 LSTM TensorList shape 미확정 ② float16 양자화 패스의 메모리 폭주 (24GB 컨테이너 실측) | ① batch_size=1 고정 모델로 변환(`train.py:convert_and_verify_tflite`) ② float16 양자화 미적용 — 모델이 float32로도 ~740KB |
| 2-M | `AttributeError: module 'mediapipe' has no attribute 'solutions'` | mediapipe 0.10.30대부터 legacy solutions API 제거 | `mediapipe<0.10.30` 고정 |
| 2-N | AI Hub 학습 모델이 실 카메라에서 좌우 반전 동작 인식 | AI Hub는 비거울 월드좌표, 런타임은 cv2.flip 거울 영상 | `convert_aihub.py:AIHUB_AXIS_SIGNS=(-1,1,1)` x 반전 (§1.5, 실측 확정) |
| 2-O | WSL `/mnt/c` 위 venv에서 TF 업/다운그레이드 후 `No module named 'tensorflow.core'` | NTFS에서 pip 파일 교체 잔재 | `uv pip uninstall tensorflow keras` 후 재설치 |
| 2-P | `RPi.GPIO add_event_detect` → `RuntimeError: Failed to add edge detection` (RPi 실기) | 레거시 sysfs 이벤트가 Trixie의 libgpiod 커널과 충돌 | 이벤트 디텍트 사용 금지 — `src/button_input.py`는 메인 루프 폴링(에지+디바운스 자체 구현)으로 동작 |
| 2-Q | 3.5mm 스피커에서 재생 무관 "툭툭" 잡음 | 증폭 스피커의 USB 전원을 RPi에서 공급 → 5V 부하 리플 증폭 (스피커 전원 분리로 실측 확정) | 스피커 전원은 별도 어댑터. PWM 음질은 `audio_pwm_mode=2`+`disable_audio_dither=1` (적용됨) |
| 2-R | 실기에서 모델이 자신 있게(~1.00) 엉뚱한 단어만 출력 | ① 저FPS(8-9.5)로 30프레임 버퍼가 1초 동작을 ~4초 창으로 왜곡 ② 시연자 3명 과적합의 병적 과신 | ① 시간 리샘플링(§1.4) ② label smoothing 0.1 + z_jitter 증강 + 시연자 16명 (재학습 중) |
| 2-S | aihubshell: zip이 최상위가 아닌 `004.수어영상/...` 하위에 저장, 컨테이너에 unzip·curl 없음 | 기본 이미지 미포함 + 데이터셋 트리 경로 보존 | `apt-get install unzip curl`, zip 탐색은 find 기반 (`server_download_aihub.sh` 사전 점검 포함) |
| 2-T | ssh로 RPi 원격 명령이 간헐 무출력 exit 255 | nohup & 백그라운드 + 와이파이 절전/전환 | 장기 실행은 `tmux new -d -s <name> "..."`로 분리, 실행과 확인을 별도 ssh로 |
| 2-U | GPU 체인이 exit 0인데 label smoothing 등 신규 코드가 미반영된 채 학습됨 (2026-06-11 16명 재학습에서 실제 발생) | `deploy_code.sh` tar 배포가 추적 파일을 덮어써 서버 작업트리가 전면 dirty → 체인의 `git pull`이 무음 실패 (ad-hoc 체인이 pull 종료코드 미검사, 출력도 미로깅) | 체인은 `scripts/chain_train.sh` 사용 — `fetch` + `reset --hard origin/develop`(dirty 면역) + 학습 전 코드 표지 grep 검증 + `set -euo pipefail`. run_training.sh도 시작 시 `git log -1`을 로그에 남김 |
| 2-V | GPU pod 재시작 후 ssh 키 거부 + `git`·`python3` command not found, venv 실행 불가 (2026-06-11 실제 발생) | overlay 루트 소실 — `/root/.ssh/authorized_keys`·apt 패키지(git·unzip·curl)·**시스템 python3.10**까지 사라짐. venv는 시스템 파이썬 심링크라 함께 무력화 | ① `ssh-copy-id -p 30007 -i ~/.ssh/id_ed25519_gpu.pub root@...`(비밀번호 1회) ② `apt-get install -y git unzip curl python3.10 python3.10-venv` ③ `python3.10 -m venv --upgrade /mnt/data/ksl/ksl-venv` — site-packages(4.8G)는 영구볼륨에 보존되어 **TF 재설치 불필요** ④ aihubshell도 소실됨 — 영구 보관본 복사: `cp /mnt/data/ksl/bin/aihubshell /usr/bin/` (보관본 없으면 `curl -o ... https://api.aihub.or.kr/api/aihubshell.do`) |
| 2-W | "시연자 16명" 학습인데 16160 샘플(=3명분 4040×4)로 변화 없음, held-out이 원본 64%를 점유 | **16명 다운로드 무음 실패 실증** (2026-06-12): 구버전 다운로드 스크립트가 unzip 오류를 마스킹한 채 zip 삭제·마커 무조건 touch — done 마커 16개 vs 실데이터 REAL01~03뿐. 다운로드 로그의 "누적 675개"가 시연자 4부터 평탄했으나 미검출 | 마커만 믿지 말 것 — **시연자 분포를 직접 검수**: `find <dataset> -type d -name "NIA_SL_WORD*" \| grep -oE "REAL[0-9]+" \| sort \| uniq -c`. 재다운로드는 가짜 마커 삭제 후 수정판(PR #10, fail-fast) 스크립트로 |
| 2-X | rm 가드가 정상 데이터셋에서 "키포인트 없음" 오탐 중단 (2026-06-11 재학습 런치 2회 연속) | ① find maxdepth 4가 시연자 중간 디렉터리(`WORD/<시연자>/NIA_SL_WORD*`, 깊이 5)를 못 봄 ② 수정 후에도 `set -o pipefail`에서 `find \| grep -q`는 grep이 파이프를 먼저 닫으면 find가 SIGPIPE(141)로 실패 — **매치가 많을수록 오탐** | maxdepth 6 + `find -print -quit` 명령 치환으로 교체 (873ed56). 교훈: pipefail 스크립트에서 `\| grep -q` 조합 금지 |
| 2-Y | 같은 가중치·같은 held-out인데 평가 경로마다 정확도가 다름 — train.py(Keras eval) 0.72 vs evaluate.py(TFLite) 0.94 (2026-06-12 실측, 재변환·CPU 강제로도 재현) | Keras Sequential LSTM 실행 vs TFLite fused LSTM 실행의 **계통적 수치 분기**(둘 다 CPU·동일 가중치에서 22점). held-out처럼 약간 OOD 입력에서 Keras 추론이 불안정하고 TFLite fused 커널이 더 안정적인 것으로 추정 | **RPi·evaluate.py는 TFLite를 돌리므로 배포·보고 정확도 = TFLite(0.94)**. train.py "Test Accuracy"(Keras)는 production에 없는 경로라 판정 지표로 부적합 — **evaluate.py(TFLite) per-class 리포트로 판정**. 향후 train.py가 변환 후 TFLite도 평가해 manifest에 함께 기록하면 일원화 |
| 2-Z | RPi 재기동 시 `GPIO ... already in use` 경고 + **완료/페르소나 버튼 무반응**·부저 이상 (2026-06-12 데모 리허설 실측) | main.py를 tmux kill/SIGKILL로 강제 종료하면 정상 종료 경로의 `GPIO.cleanup()`이 안 돌아 핀이 점유된 채 다음 인스턴스가 뜬다. 별개로 물리 버튼 결선 불량 가능성도 있음(사용자 제기 — 배선도 점검 대상) | **정상 종료**: GUI는 `q`, tmux는 `tmux send-keys -t <s> C-c` 후 종료(GPIO.cleanup 실행). 재기동 전 `pgrep -af src/main.py`로 **단일 인스턴스** 확인. 완료 버튼 무반응은 단독 점검(버튼 핀 폴링 스크립트)으로 결선 vs 소프트 원인 분리 필요. 모니터링 시 `PYTHONUNBUFFERED=1` — stdout 리다이렉트는 블록버퍼라 `[Classifier]` 인식 로그가 지연된다 |

---

## 3. 작업 재개 절차

### 3.1 PC (개발 + 학습)

```bash
git clone https://github.com/andrew427dev/KSL-LLM-IoT.git
cd KSL-LLM-IoT
# PR #2 머지 전: feat/smoke-test-pipeline / 머지 후: develop
git checkout feat/smoke-test-pipeline

# uv + Python 3.11
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv --python 3.11
source .venv/bin/activate

uv pip install -r requirements.txt
cp .env.example .env
```

상세는 README §A 참조.

### 3.2 RPi (배포 + 실행)

```bash
# 시스템 패키지
sudo apt update
sudo apt install -y i2c-tools rpicam-apps libcap-dev libcamera-dev \
                    pkg-config python3-dev build-essential

# 저장소 가져오기 (또는 PC에서 pscp 전송)
git clone https://github.com/andrew427dev/KSL-LLM-IoT.git ~/Desktop/KSL-LLM-IoT
cd ~/Desktop/KSL-LLM-IoT
git checkout feat/smoke-test-pipeline

# venv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements-rpi.txt

# 환경설정
cp .env.example .env
nano .env   # GEMINI_API_KEY는 공란 가능 (offline fallback)

# 카메라/I2C 검증
rpicam-hello --timeout 2000
sudo raspi-config       # → Interface Options → I2C → Enable → 재부팅
i2cdetect -y 1

# 실행
KSL_HEADLESS=1 python src/main.py
```

상세는 README §B 참조.

### 3.3 동작 상태 매트릭스

| `GEMINI_API_KEY` | `model/ksl_model.tflite` | 결과 |
|---|---|---|
| 설정됨 | 존재 | 정상 운영 |
| 설정됨 | 없음 | LLM 동작, 분류기 더미 모드 |
| 비어있음 | 존재 | 분류기 동작, 문장은 단어 나열 |
| 비어있음 | 없음 | 전 구간 fallback (현재 기본 상태) |

각 모드 모두 파이프라인 전체가 끊김 없이 실행된다. 부분 실패가 전체 정지로 이어지지 않는 구조다.

---

## 4. 다음 작업 (2026-06-11 기준, 마감 6/22)

### 4.1 즉시 — 시연자 4~16 재다운로드 → 재학습 사이클

경과: ① 2-U 체인(구코드 학습)은 무효 ② PR #10 머지 후 체인(873ed56)은 **정상 완주** (2026-06-11, exit 0 + held-out 평가 동작 확인) — held-out Test Accuracy **0.3622**. 단 함정 **2-W**로 실데이터가 시연자 3명뿐이라 train=REAL03 1명 / test=REAL01·02 구도 — **16명 판정치가 아닌 참고치** (1명 학습 모델의 타인 일반화 수준).

**상태 (2026-06-12 — 사이클 완료)**: 16명(REAL01~16, 각 225 dir=3,600) 재다운로드 완료·**3중 검증 통과**(시연자 분포·30/30 단어매칭·`.part`/zip 잔재 0). 체인 완주(a159f36, 표지검증 통과) → **held-out TFLite 0.94**(배포본)/Keras 0.72. RPi 배포 완료(develop@a159f36, G6 보정+가드 포함, 모델 md5 검증·classifier dummy=False). 실기 진단: **27/30 정상·확신 0.3~0.9 건강**, 밥/배고프다/주다 혼동만 잔존 → G7 스파이크로 보류·데모 시나리오 회피 결정(§1.6 G7 검증·§4.6.2). **⚠️ 판정·보고 지표는 evaluate.py(TFLite)이며 train.py(Keras) Test Accuracy가 아니다 — 함정 2-Y(둘이 0.94 vs 0.72로 22점 갈림).** 아래 명령은 재실행 시 참고용 이력:

```bash
# 0) (사전) GPU pod 재시작됐으면 함정 2-V 복구 절차 먼저
# 1) 가짜 done 마커 삭제 + 시연자 4~16 재다운로드 (수정판 — 치명 오류 fail-fast)
ssh gpu
rm /mnt/data/ksl/aihub_full/.signer_{4..16}_done
cd /mnt/data/ksl/KSL-LLM-IoT
nohup bash scripts/server_download_aihub.sh 16 > /mnt/data/ksl/download3.log 2>&1 &
# 완료 후 시연자 분포 검수 (마커 신뢰 금지 — 2-W):
find /mnt/data/ksl/aihub_full -type d -name "NIA_SL_WORD*" | grep -oE "REAL[0-9]+" | sort | uniq -c

# 2) 체인 실행 (fetch+reset → 표지 검증 → 변환→증강→학습→평가)
nohup bash scripts/chain_train.sh /mnt/data/ksl/aihub_full >> /mnt/data/ksl/training.log 2>&1 &
# 다운로드 PID 종료 대기 후 자동 시작하려면: bash scripts/chain_train.sh <dataset> <download_pid>

# 3) 완료 확인 — 판정 지표 = train.log "Test Accuracy" (held-out 시연자 원본 기준)
grep -E "\[Chain\]|Test Accuracy|Held-out" /mnt/data/ksl/training.log | tail -5

# 4) 모델 회수 → RPi 배포 — G6 보정 코드와 모델을 반드시 함께 배포
bash scripts/fetch_model.sh /tmp/m && scp /tmp/m/ksl_model.tflite rpi:Desktop/KSL-LLM-IoT/model/
ssh rpi 'cd ~/Desktop/KSL-LLM-IoT && git pull'
# 5) RPi 재진단 (VNC 미리보기 포함)
ssh rpi 'cd ~/Desktop/KSL-LLM-IoT && tmux kill-session -t diag 2>/dev/null; tmux new -d -s diag "DISPLAY=:0 .venv/bin/python tools/diagnose_live.py 300 > /tmp/ksl_diag.log 2>&1"'
```

판정: top-3에 수행 단어가 들어오고 확신 분포가 정상화(0.3~0.9 분포)되면 데모 시나리오 확정. 미달이면 §1.6 주범 역추적 신호로 잔여 갭을 판별한 뒤 — G2·G3이면 백로그 증강(low-FPS 시뮬레이션·presence dropout) 추가 재학습, G4이면 4.2 본인 데이터 혼합, **손모양 유사·위치 구분 쌍(밥/배고프다·주다 등) 혼동 잔존이면 G7 얼굴 앵커 특징(§1.6)을 2차 사이클로** 적용. G1·G5·G6은 이번 사이클에 반영됨. **판정 지표는 train.log의 Test Accuracy(=held-out 시연자 정확도) 사용**.

### 4.2 (조건부) 본인 데이터 혼합

1. `collect_data.py`를 classifier와 동일한 시간 리샘플 방식으로 수정 (미착수 — 현재 버전은 저FPS에서 시간 왜곡 데이터 저장)
2. RPi+VNC로 30단어 × 20~30샘플 수집 (~1.5h) → `scripts/upload_dataset.sh`형태로 서버 전송 → AI Hub와 혼합 재학습
3. 근거: 데모 화자가 학습에 포함 — 6/22 "works correctly" 채점 기준에 가장 확실

### 4.3 시스템 운용 (RPi)

```bash
ssh rpi
tmux new -d -s ksl "cd ~/Desktop/KSL-LLM-IoT && DISPLAY=:0 KSL_HEADLESS=0 .venv/bin/python src/main.py > /tmp/ksl_gui.log 2>&1"
# 종료: tmux kill-session -t ksl / 로그: tail -f /tmp/ksl_gui.log
```

- 스피커는 별도 USB 전원 필수 (함정 2-Q)
- IP 변경 시: Windows에서 `arp -a | findstr d8-3a-dd` → ~/.ssh/config HostName 갱신

### 4.4 제출물 (6/22)

- 데모 영상·보고서·PPT 아웃라인: `docs/final_report_outline.md`, `docs/presentation_outline.md` — [TBD] 수치만 채우면 집필 가능
- **데모 리허설 발견 (2026-06-12)**: 인식 양호(부탁87·서다88·아프다93·병원91·의사90·돕다91%). 단 ① **가다→의사 오인**(이 사용자 실행) — 데모 단어는 사용자별 사전 재검증 ② **완료·페르소나 버튼 전부 무반응** — `src/button_input.py`(GPIO5→`EVENT_COMPLETE` 폴링·하강에지·디바운스)와 `src/main.py` 라우팅(`poll()`→`trigger_sentence`) 코드는 **정상 존재 확인**(코드 누락 아님) → 원인은 GPIO 재점유(함정 2-Z) 또는 물리 결선. **단독 버튼 폴링 테스트로 결선 vs 소프트 분리 필요** ③ **`SILENCE_TRIGGER_SEC=3.0`이 멈춤마다 조기 발화 → 문장 단편화**("많이 아","의"). 버튼 정상화 후 RPi `.env`에서 `SILENCE_TRIGGER_SEC=0`(버튼 전용) 권장
- **배선도 갱신 필요 (`docs/wiring_diagram.png` / `docs/generate_wiring_diagram.py`)**: ① 카메라 CSI(Pi Camera v2) → **USB 웹캠**으로 교체 ② 완료 버튼 GND 표기 물리 30/34 → **물리 6**(사용자 실측). 핀 본체(완료=GPIO5 물리29, 페르소나=GPIO6/13/19, 부저=GPIO17, LCD 0x27)는 settings.py와 일치. GND는 전기적으로 동일(무반응 원인 아님). 재생성: `python docs/generate_wiring_diagram.py` (matplotlib 필요 — 로컬 부재, RPi/서버에서 실행)
- PR #9 머지 → develop→main 릴리스 → zip 패키징 (`IoT_YourName_StudentID_TermProject.rar`)

### 4.5 PR #9 리뷰 후속 수정 백로그 (2026-06-11 — 전체 15건은 PR #9 리뷰 코멘트)

`fix/review-backlog-pre-retrain` 브랜치에서 1~4 + 함정 2-U 대응을 일괄 수정 (2026-06-11):

1. [x] **G6 등방 보정** (리뷰 #1, High) — `hand_tracker` (w,h,w) 스케일. 재학습·재진단과 한 사이클로 적용.
2. [x] **다운로드 무결성** (리뷰 #2, High) — unzip exit 11/치명 구분, 치명 시 zip 보존·마커 미생성·중단. 16명 데이터 단어별 시퀀스 수 검수는 완료(30클래스 전수, 최소 65).
3. [x] 빠른 수정 묶음 — `deploy_code.sh --exclude='.env'`(리뷰 #8) / `run_training.sh` rm 전 데이터셋 가드(리뷰 #7) / `classifier.__init__` shape 검증 fail-fast(리뷰 #9).
4. [x] 평가 신뢰성 — held-out **시연자** 분리(리뷰 #6): `model/data_split.py` 신설, 변환·증강 파일명에 출처 보존, train.py가 `model/holdout.json` 기록, evaluate.py가 그 집합만 평가. train.log Test Accuracy = held-out 시연자 정확도.
5. [ ] CI 커버리지 — classifier 최상단 tflite import 구조 해소 후 test_classifier 등 CI 편입(리뷰 #13). 신규 테스트 4종(test_data_split·test_source_naming·test_server_download·shape 검증)도 함께. **별도 브랜치**.

### 4.6 접근성·입력 범위 백로그 (2026-06-11 배진규 지시)

1. **LCD 표시 확장 (접근성 — High)**: 주 사용자(수어 화자)는 농인·언어장애인이 대다수 —
   현재 페르소나 변경 피드백이 **부저 비프(청각)** 라 주 사용자에게 닿지 않는 설계 모순.
   LCD에 다음을 모두 표기하도록 확장한다: ① 페르소나 변경 시 현재 페르소나,
   ② 인식된 단어 입력(버퍼 누적 상태), ③ 생성된 문장, ④ 문장 완성(전송/완료) 상태.
   대상: `src/lcd_display.py`(20×4 줄 배치 설계)·`src/main.py`. 비프는 보조 신호로 유지.
   develop 위 별도 브랜치로 진행, USER_MANUAL §3 갱신 동반.
2. **인식 범위 ↔ 데이터 정합 (G7 연계)**: 현 입력(양손만)이 데이터셋이 제공하는 정보
   범위(pose 25관절·face 70점)보다 좁아 정확한 수어 인식에 불충분하다는 문제 제기 —
   **이번 16명 재학습을 먼저 테스트하고, 결과를 보고 입력 범위 확장(§1.6 G7 얼굴 앵커
   +α)을 결정**한다. §4.1 판정 분기의 G7 게이트와 동일 경로.
   → **결정 완료 (2026-06-12)**: 16명 실기 27/30 정상. G7 스파이크(§1.6 G7 검증)에서
   위치가 잔존 혼동(밥/배고프다/주다)을 분리 못 함이 실측됨 → **G7 보류**. 데모는 잘 되는
   27단어로 구성(밥/배고프다/주다는 1개만 사용·제외), 입력 범위 확장은 본인 데이터 보강
   (§4.2)을 우선하는 future work로 이관.

## 5. 책임 분담 (현재 기록)

| 영역 | 담당 | 비고 |
|------|------|------|
| 카메라·MediaPipe·분류기 인프라 | 이성준 | 완료 (PR #2) |
| 비동기 파이프라인·SDK 마이그레이션 | 이성준 | 완료 (PR #2) |
| 문서화 (README/USER_MANUAL/CLAUDE/HANDOFF) | 이성준 | 진행 중 |
| 데이터 수집 30단어 | 미정 | Week 2 회의에서 분담 확정 |
| 모델 학습 | 미정 | Week 3 |
| 하드웨어 결선 | 미정 | 병렬 진행 |
| Gemini 프롬프트 튜닝 | 미정 | Week 5 |
| 데모 영상 촬영 | 미정 | Week 6 |

배진규 담당 영역이 결정되면 본 표를 갱신한다.

---

## 6. 핵심 파일·문서

| 파일 | 내용 |
|------|------|
| [README.md](README.md) | 프로젝트 소개, 환경 설치(PC/RPi 분리), 하드웨어 결선표, 운영 모드 매트릭스 |
| [USER_MANUAL.md](USER_MANUAL.md) | 운용자용 한국어 매뉴얼. §5.1 PuTTY 파일 전송, §6 트러블슈팅, §9 changelog |
| [CLAUDE.md](CLAUDE.md) | AI 에이전트 가이드. §1.1 매뉴얼 갱신 트리거, §1.4 모든 문서 공통 작성 규칙 (객관 어조) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 기여 가이드 |
| [docs/wiring_diagram.png](docs/wiring_diagram.png) | 결선도 (재생성: `python docs/generate_wiring_diagram.py`) |
| `src/main.py` | 메인 루프, `CameraReader` 3단 폴백, 헤드리스 감지 |
| `src/sentence_builder.py` | google-genai 호출 + 오프라인 fallback + 비동기 큐 |
| `src/classifier.py` | TFLite 추론 + 더미 모드 |
| `src/lcd_display.py` | I2C LCD + 워커 스레드 큐 |
| `make_dummy_model.py` | 검증용 더미 TFLite 생성 (PC) |
| `collect_data.py` | 랜드마크 CSV 수집 — 직접 촬영 (PC) |
| `convert_aihub.py` | AI Hub 수어 키포인트 JSON → 프로젝트 CSV 변환 (§1.5 참조) |

---

## 7. 다음 핸드오프까지의 체크리스트

작업 마무리 시점에 다음을 갱신:

- [ ] §1.1 PR 번호·커밋·머지 상태
- [ ] §1.2 검증된 동작 (새 항목 체크박스)
- [ ] §1.3 미완료 (완료된 항목 제거 또는 체크박스 마크)
- [ ] §2 함정과 해결법 — 새로 부딪힌 함정 1행 추가
- [ ] §4 다음 작업 — 진척된 Week 섹션 갱신
- [ ] §5 분담 — 확정/변경 사항 반영
- [ ] §8 변경 이력 본 문서 한 줄 추가

---

## 8. 본 문서 변경 이력

| 날짜 | 작성자 | 내용 |
|------|--------|------|
| 2026-06-12 | 배진규 + Claude (Opus 4.8) | **16명 재학습 완주·RPi 배포** — held-out **TFLite 0.94**(배포본)/Keras 0.72. 데이터: 3,600 keypoint dir(16명×225)→21,710 시퀀스→증강 65,130, held-out=REAL01·02 통째분리(train 76,460/test 2,595). 함정 **2-Y**(Keras LSTM vs TFLite fused LSTM held-out 22점 계통차 — RPi·evaluate.py는 TFLite라 **배포정확도=0.94**, 판정·보고는 TFLite 사용) 추가. RPi 실기: **27/30 정상·확신 0.3~0.9 건강(1.00 병리 해소)**, 밥/배고프다/주다 혼동(배고프다→돕다 0.92), det 57~74%·6.4FPS. **G7 스파이크**: AI Hub서 손-앵커 위치가 3단어 분리 못 함(밥만 부분) → G7 보류, 주범=런타임 손모양(G2/G3/G4). 결정: **데모 시나리오로 회피**, 본인데이터 보강(§4.2) future work |
| 2026-06-12 | 배진규 + Claude (Fable 5) | 체인 완주 — held-out Test Accuracy 0.3622 (3명 데이터 참고치). 함정 **2-W**(16명 다운로드 무음 실패 실증 — 실데이터 REAL01~03뿐, 시연자 분포 직접 검수 규칙)·**2-X**(rm 가드 오탐 2건: maxdepth·pipefail SIGPIPE — 4273972·873ed56 수정) 추가, §4.1을 재다운로드→재학습 경로로 교체 |
| 2026-06-11 | 배진규 + Claude (Fable 5) | PR #10 머지(develop=cff0c32)·재학습 체인 가동(chain_train.sh, 코드 표지 검증 통과). 함정 **2-V**(pod 재시작 — 키·apt·시스템 python 소실, venv --upgrade 복구 절차) 추가 |
| 2026-06-11 | 배진규 + Claude (Fable 5) | 리뷰 백로그 일괄 수정(§4.5 1~4): G6 등방 보정, held-out 시연자 평가(`data_split.py`·`holdout.json`), 다운로드 무결성, 가드 3종. **함정 2-U 발견·기록** — 16명 체인이 dirty 트리 pull 무음 실패로 구코드 학습, `chain_train.sh` 신설로 차단. §4.1 재학습 경로 갱신 |
| 2026-06-11 | 이성준 + Claude (Fable 5) | PR #9 develop 머지 반영(§1.1·§1.3), 머지 전 전수 리뷰 15건(PR #9 코멘트), §1.6 **G6 좌표 비등방성 갭** 추가, §4.5 리뷰 후속 백로그 신설, §4.1 판정 지표를 train.log Test Accuracy로 명시 |
| 2026-06-11 | 이성준 + Claude (Fable 5) | §1.6 신설 — 학습 데이터↔런타임 도메인 갭 전수 정리(G1~G5), 백로그 증강 2종(low-FPS 시뮬레이션·presence dropout), 주범 역추적 신호. §4.1 판정 경로를 갭별 분기로 확장 |
| 2026-06-11 | 이성준 + Claude (Opus 4.8) | 전면 개정 — RPi 실기 통합(§1.2), 시간 리샘플링(§1.4), 실기 진단·과신 대응(§2 2-R~2-T), §4를 현 시점 크리티컬 패스로 교체 |
| 2026-06-10 | 이성준 + Claude (Opus 4.8) | 출력 이중화: Gemini가 (한국어, 영어) 두 줄 생성 — TTS=한국어/LCD·GUI=영어(KSL_LABELS_EN), LCD ASCII 안전망. 운용 모델(30클래스, 전수 0.95) 서버 학습·RPi 배포 완료 |
| 2026-06-10 | 이성준 + Claude (Opus 4.8) | 물리 버튼 4개(완료+페르소나 3) 도입 — src/button_input.py, 침묵 트리거 기본 비활성(SILENCE_TRIGGER_SEC=0), 페르소나 비프 1/2/3회 피드백, 결선도 재생성 |
| 2026-06-10 | 이성준 + Claude (Opus 4.8) + Codex 리뷰 | 131차원 전환(§1.4), AI Hub 축 변환 실측 확정(§1.5), LabelEncoder 버그 수정, GPU 서버 학습 파이프라인(scripts/) + 서버 E2E 검증, 함정 2-K~2-O 추가 |
| 2026-05-27 | 이성준 + Claude | §1.1 PR #7-#8 반영, §1.5 AI Hub 데이터셋 변환 파이프라인 추가, §6 convert_aihub.py 등록 |
| 2026-05-15 | 이성준 + Claude | 초안 작성 (PR #2 진행 중, Week 2 진입 시점, smoke-test 인프라 완성) |
