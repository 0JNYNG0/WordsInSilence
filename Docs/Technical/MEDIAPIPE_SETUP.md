# MediaPipeUnityPlugin 설치 가이드

## 대상 버전

**MediaPipeUnityPlugin v0.16.3** (MediaPipe 0.10.22 기반)
릴리스 날짜: 2025-11-08
저장소: https://github.com/homuler/MediaPipeUnityPlugin

## 사전 조건

- Unity 6000.3.14f1 (Unity 6.3 LTS)
- Windows Standalone Scripting Backend = Mono (D-01 결정 — 이미 적용됨)
- 인터넷 연결 (패키지 다운로드)

---

## 설치 절차

### 방법 A: UPM Git URL (권장)

1. Unity Editor에서 **Window > Package Manager** 열기
2. 좌측 상단 **+** 버튼 클릭 → **Add package from git URL** 선택
3. 다음 URL 입력 후 **Add** 클릭:
   ```
   https://github.com/homuler/MediaPipeUnityPlugin.git#v0.16.3
   ```
4. 패키지 다운로드 및 컴파일 완료 대기 (수 분 소요 가능)
5. Console 패널에서 컴파일 오류 없는지 확인

> **버전 태그 고정이 불가한 경우:** 방법 B로 전환한다.

### 방법 B: .unitypackage 파일 (대안)

1. https://github.com/homuler/MediaPipeUnityPlugin/releases/tag/v0.16.3 접속
2. `MediaPipeUnityPlugin-v0.16.3.unitypackage` 다운로드
3. Unity Editor에서 **Assets > Import Package > Custom Package** 선택
4. 다운로드한 .unitypackage 파일 선택 후 **Import All**

---

## 설치 후 확인

### 모델 파일 복사

Hand Landmarker 실행에는 모델 파일이 필요하다:

1. `Packages/com.github.homuler.mediapipe/Runtime/Resources/` 또는
   샘플 패키지 내 `StreamingAssets/` 폴더에서 모델 파일(`.bytes`) 확인
2. 해당 파일들을 `Assets/StreamingAssets/` 에 복사

> 필요한 파일 목록은 샘플 Scene README 또는 플러그인 공식 문서 참조.

### 샘플 Scene Import 및 실행

1. Package Manager에서 **Packages: In Project** → `MediaPipeUnityPlugin` 선택
2. **Samples** 탭에서 **Hand Tracking** 또는 **Hand Landmark Detection** → **Import** 클릭
3. Import된 샘플 폴더에서 Hand Landmarker Scene 열기
4. Play 버튼 클릭
5. 웹캠 영상과 손 랜드마크가 오버레이되는지 확인

---

## 알려진 이슈

| 이슈 | 설명 | 대처 방법 |
|---|---|---|
| UIToolkit 호환 #1415 | UI Toolkit 기반 샘플이 Unity 6에서 레이아웃 오류 | UGUI 기반 Scene 사용 |
| UIToolkit 호환 #1417 | VisualElement 이벤트 처리 오류 | 동일 |
| IL2CPP 미지원 | `DllNotFoundException: mediapipe_c` | Mono 백엔드 사용 (D-01) |

---

## 설치 결과 기록

> 아래 항목을 Phase 0 완료 시 채운다.

- **설치 방법:** [ ] 방법 A (Git URL) / [ ] 방법 B (.unitypackage)
- **설치 날짜:**
- **컴파일 오류:**
- **샘플 Scene 동작 여부:**
- **발견된 추가 이슈:**
- **D-04 결정 갱신:** `Docs/DECISIONS.md` D-04 항목 업데이트 필요

---

## 삭제 방법

### UPM 설치의 경우
1. **Window > Package Manager** 열기
2. **MediaPipeUnityPlugin** 선택
3. **Remove** 클릭
4. `Assets/StreamingAssets/` 내 모델 파일 수동 삭제

### .unitypackage 설치의 경우
1. `Assets/MediaPipeUnity/` (또는 Import된 폴더) 삭제
2. `Assets/StreamingAssets/` 내 모델 파일 수동 삭제
3. `Packages/manifest.json` 에서 관련 항목 수동 제거
