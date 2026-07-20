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
