# 개발 변경 이력

## [미배포] — 2026-07-20

### 추가
- 프로젝트 문서 구조 생성 (`Docs/` 하위 전체)
- Phase 1 WebCam 기술 스파이크 구현
  - `ICameraFrameSource` 인터페이스 및 데이터 클래스
  - `UnityWebCamFrameSource` 구현체
  - `WebCamSpikeController` / `WebCamSpikeUI` Scene 컨트롤러
  - `CaptureConfig` ScriptableObject
  - `CaptureDiagnostics` 진단 데이터 클래스
  - 핵심 인터페이스: `ILandmarkProvider`, `IMotionRecorder`, `IMotionEvaluator`, `ISignInteractionService`
  - EditMode / PlayMode 테스트 초안
- `Assets/StreamingAssets/` 폴더 생성 (MediaPipe 모델 파일 보관용)
- `.gitignore` 생성

### 변경
- `ProjectSettings/ProjectSettings.asset`: Windows Standalone Scripting Backend → Mono (D-01)

### 미완료 (사용자 수동 단계)
- MediaPipeUnityPlugin v0.16.3 설치 (Phase 0)
- `Assets/_Project/Scenes/Technical/01_WebCamSpike.unity` Scene 내 UI 계층 구성
- `Assets/_Project/Settings/DefaultCaptureConfig.asset` ScriptableObject 인스턴스 생성
- Phase 0 완료 체크리스트 갱신
