# 개발 변경 이력

## [미배포] — 2026-07-21

### 추가 (Phase 3.5 — Pose Landmarker 및 Hand-Pose 통합 기반)
- `Assets/_Project/Scripts/Landmarks/MediaPipeRuntime.cs` — Glog 공유 초기화 레퍼런스 카운터
- `Assets/_Project/Scripts/Pose/UpperBodyPoseFrame.cs` — Pose 프레임 데이터 모델 (33점, visibility)
- `Assets/_Project/Scripts/Pose/PoseLandmarkProvider.cs` — MonoBehaviour, VIDEO 모드 Pose 검출
- `Assets/_Project/Scripts/Pose/PoseBodyMath.cs` — 순수 정적 유틸 (어깨 중심/너비, BodyRelative, 팔 각도)
- `Assets/_Project/Scripts/Pose/HandPoseFrame.cs` — 통합 프레임 + FrameDataQuality 열거형
- `Assets/_Project/Scripts/Pose/HandPoseIntegrator.cs` — Hand+Pose timestamp 매칭, UnifiedFrameArrived 이벤트
- `Assets/_Project/Scripts/Pose/PoseOverlayRenderer.cs` — 상체 스켈레톤 오버레이 (7점)
- `Assets/_Project/Scripts/Pose/PoseIntegrationTestUI.cs` — 진단 패널 (FPS/품질/신체 좌표, 레이어 토글)
- `Assets/_Project/Scripts/Pose/HandPoseRecording.cs` — schemaVersion 2 녹화 데이터 모델 (정의만)
- `Assets/_Project/Tests/EditMode/PoseBodyMathTests.cs` — 16개 단위 테스트
- `Docs/Technical/POSE_INTEGRATION_AUDIT.md` — 감사 결과 + 성능 측정 템플릿

### 변경
- `Assets/_Project/Scripts/Landmarks/MediaPipeLandmarkProvider.cs`
  - `static bool s_mediaPipeInitialized` + 직접 Glog 초기화/종료 → `MediaPipeRuntime.EnsureInitialized()` / `Release()` 로 교체
  - 동작 동일, Glog 중복 초기화 안전성 확보

### 문서
- `Docs/ROADMAP.md` — Phase 3.5, Phase 4~9 계획 추가
- `Docs/PHASE_STATUS.md` — Phase 3.5 체크리스트 추가
- `Docs/DECISIONS.md` — D-05 ~ D-08 추가 (MediaPipeRuntime, Pose 모델, 픽셀 복사, timestamp)
- `Docs/CHANGELOG_DEV.md` — 이 항목

### 미완료 (사용자 수동 단계)
- `01_WebCamSpike.unity` 복제 → `02_PoseIntegration.unity` 생성
- SpikeMgr에 PoseLandmarkProvider / HandPoseIntegrator 컴포넌트 추가 및 배선
- Canvas에 Image_PoseOverlay + PoseOverlayRenderer + PoseIntegrationTestUI 추가
- POSE_INTEGRATION_AUDIT.md 성능 측정값 기입

---

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
