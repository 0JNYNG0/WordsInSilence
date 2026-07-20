# 환경 감사 결과

**감사 날짜:** 2026-07-20

## 시스템 환경

| 항목 | 결과 |
|---|---|
| OS | Windows 11 Home (x86_64) Build 10.0.26200 |
| Unity 버전 | 6000.3.14f1 (Unity 6.3 LTS) |
| 렌더 파이프라인 | URP (Universal Render Pipeline) 17.3.0 |
| API Compatibility Level | 6 (.NET Standard 2.1) |
| Windows Build Support | 설치됨 |
| IL2CPP 모듈 | 설치됨 (단, MediaPipe Windows 빌드에는 사용 불가 — D-01 참조) |
| Unity Test Framework | 1.6.0 (설치됨) |
| Git 저장소 | Phase 0~1 구현 시작 시 초기화 (`git init`) |

## 프로젝트 상태 (감사 시점)

| 항목 | 결과 |
|---|---|
| MediaPipeUnityPlugin | 미설치 |
| 기존 .asmdef | 없음 |
| 기존 Scene | `Assets/Scenes/SampleScene.unity` (1개) |
| 기존 스크립트 | 2개 (TutorialInfo 전용, 수정 불필요) |
| StreamingAssets | 없음 (Phase 0~1 구현 시 생성) |
| WebCamTexture 사용 코드 | 없음 |

## 설치된 UPM 패키지 (주요)

Unity 6.3 URP Blank Template 기준:
- `com.unity.render-pipelines.universal`: 17.3.0
- `com.unity.inputsystem`: (설치됨)
- `com.unity.test-framework`: 1.6.0

## 주요 파일 경로

| 항목 | 경로 |
|---|---|
| 프로젝트 루트 | `C:\MyUnityProjects\WordsInSilence\` |
| ProjectSettings | `ProjectSettings/ProjectSettings.asset` |
| 기존 Scene | `Assets/Scenes/SampleScene.unity` |
| StreamingAssets | `Assets/StreamingAssets/` |
| 신규 스크립트 루트 | `Assets/_Project/Scripts/` |

## 감사 메모

- TutorialInfo 폴더는 Unity URP 템플릿의 기본 에셋이다. 수정하지 않는다.
- SampleScene.unity는 삭제하지 않는다.
- IL2CPP는 Android 빌드에 유지하고 Windows Standalone은 Mono로 변경한다 (D-01).
