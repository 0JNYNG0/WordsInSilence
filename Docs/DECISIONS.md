# 주요 결정 사항 (Architecture Decision Records)

## D-01: Scripting Backend — Mono (Windows)

**날짜:** 2026-07-20
**상태:** 확정

### 결정
Windows Standalone 빌드의 Scripting Backend를 **Mono**로 설정한다.

### 근거
MediaPipeUnityPlugin은 Windows x86_64에서 IL2CPP를 지원하지 않는다.
IL2CPP 빌드 시 `DllNotFoundException: mediapipe_c` 오류가 발생한다.
플러그인 저자가 Windows IL2CPP 지원을 명시적으로 제외하였다.

### 변경 내용
`ProjectSettings/ProjectSettings.asset` 의 `scriptingBackend` 블록에 `Standalone: 0` 추가.
- 변경 전: `scriptingBackend` 블록에 `Standalone` 키 없음 (기본값 = Mono로 동작)
- 변경 후: `Standalone: 0` 명시 (0 = Mono, 1 = IL2CPP)

### 영향
- Windows 빌드 크기: Mono는 IL2CPP 대비 빌드 용량이 더 클 수 있음
- 성능: IL2CPP 대비 약간 낮을 수 있으나 기능성 게임 수준에서는 무관
- Android/iOS: 영향 없음 (Android는 IL2CPP 유지)

---

## D-02: MediaPipeUnityPlugin 버전 — v0.16.3

**날짜:** 2026-07-20
**상태:** 확정 (Phase 0 설치 후 재확인)

### 결정
MediaPipeUnityPlugin **v0.16.3** (최신 정식 릴리스, 2025-11-08)을 사용한다.
MediaPipe 0.10.22 기반.

### 근거
- 최신 정식 릴리스이므로 보안 패치 및 버그 수정이 포함됨
- Unity 6000.0.34f1에서 사용자 검증 보고 존재
- 비공식 DLL이나 main 브랜치 코드 사용 금지 원칙

### 주의 사항
- Unity 6000.3.x에서의 공식 확인은 없음 (Phase 0에서 검증)
- UIToolkit 관련 호환성 이슈(#1415, #1417) 보고됨 — 샘플 Scene 먼저 확인
- 문제 발생 시 즉시 기록하고 대안(UGUI, 다운그레이드) 검토

---

## D-03: Assembly Definition — Phase 1에서 도입 보류

**날짜:** 2026-07-20
**상태:** 확정

### 결정
Phase 1에서는 `.asmdef` 파일을 도입하지 않는다.

### 근거
MediaPipe 플러그인의 asmdef 구조와 호환성을 확인하기 전에 일괄 도입하면
컴파일 오류가 발생할 위험이 있다.

### 향후 계획
Phase 1 완료 후 MediaPipe 플러그인 asmdef 구조를 분석하여
Phase 2 또는 Phase 3에서 도입 여부를 결정하고 이 문서에 D-05로 추가한다.

---

## D-04: MediaPipe 설치 방법

**날짜:** 2026-07-20
**상태:** Phase 0 완료 후 기록 예정

### 결정
UPM Git URL 또는 .unitypackage 중 Unity Editor에서 직접 확인한 방법을 사용한다.

> **이 항목은 Phase 0 완료 후 실제 설치한 방법으로 갱신한다.**

### 우선 시도 순서
1. UPM Git URL: `https://github.com/homuler/MediaPipeUnityPlugin.git#v0.16.3`
2. 실패 시: GitHub Releases에서 `.unitypackage` 다운로드 후 Import

### 기록 예정
- 설치 성공 방법
- 설치 날짜
- 컴파일 오류 발생 여부
- 샘플 Scene 동작 여부

---

## D-05: Glog 초기화 공유 — MediaPipeRuntime 레퍼런스 카운터

**날짜:** 2026-07-21
**상태:** 확정

### 결정
`MediaPipeRuntime` 내부 정적 클래스(레퍼런스 카운터)를 통해 Glog/Protobuf 초기화를 공유한다.

### 근거
- `MediaPipeLandmarkProvider`(Hand)와 `PoseLandmarkProvider`(Pose)가 동시에 실행될 때
  `Glog.Initialize`가 중복 호출되면 크래시가 발생한다.
- `static bool s_mediaPipeInitialized` 방식은 단일 Provider 가정이므로 다중 Provider에 불안전하다.

### 구현
- `Assets/_Project/Scripts/Landmarks/MediaPipeRuntime.cs`
- `EnsureInitialized()` / `Release()` 쌍으로 사용
- `_refCount` 가 0→1이 될 때 초기화, 1→0이 될 때 종료

---

## D-06: Pose Landmarker 모델 — lite 우선

**날짜:** 2026-07-21
**상태:** 확정

### 결정
`pose_landmarker_lite.bytes` 를 기본 모델로 사용한다.

### 근거
- Windows CPU 추론에서 속도 우선 (목표 ≥15 FPS)
- full/heavy 모델은 정확도는 높으나 FPS 저하 위험
- Phase 3.5 목표는 어깨/팔꿈치/손목 감지 기반 구축이므로 lite로 충분

### 변경 방법
`PoseLandmarkProvider._initializeAsync()` 내 `ResolveModelPath("pose_landmarker_lite.bytes")`를
`"pose_landmarker_full.bytes"` 또는 `"pose_landmarker_heavy.bytes"` 로 변경.

---

## D-07: 픽셀 복사 2회 허용 (Phase 3.5)

**날짜:** 2026-07-21
**상태:** 임시 (Phase 4에서 재검토)

### 결정
Phase 3.5 에서는 Hand/Pose 두 Provider가 각각 `GetPixels32()` → `SetPixels32()` 를 호출하는 것을 허용한다.

### 근거
- 성능 측정 전 최적화는 과잉 설계
- FPS 목표(≥15)를 달성하지 못하면 Phase 4에서 공유 프레임 버퍼 도입

---

## D-08: Hand-Pose timestamp 통합 기준 — 100ms

**날짜:** 2026-07-21
**상태:** 확정 (Inspector에서 조정 가능)

### 결정
`HandPoseIntegrator._maxTimestampDiffMs = 100f` (기본값)

### 근거
- 60FPS 기준 프레임 간격 ≈ 16ms; 100ms 는 약 6프레임 여유
- Hand/Pose 처리 속도가 다를 수 있으므로 여유 있게 설정
- 너무 작으면 유효 통합 프레임 수가 감소함
