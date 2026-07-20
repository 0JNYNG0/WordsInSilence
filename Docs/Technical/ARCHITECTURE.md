# 아키텍처 개요

## 레이어 구조

```
┌─────────────────────────────────────────────────────┐
│                  Game Layer (Phase 4)                │
│  NPC 상호작용, 퀘스트, 수어 이벤트 소비               │
├─────────────────────────────────────────────────────┤
│              Sign Interaction Layer (Phase 3)        │
│  ISignInteractionService, IMotionEvaluator           │
│  수어 동작 → 게임 이벤트 변환                         │
├─────────────────────────────────────────────────────┤
│              Motion Recognition Layer (Phase 3)      │
│  IMotionRecorder, LandmarkFrame 시계열 분석           │
├─────────────────────────────────────────────────────┤
│              Landmark Layer (Phase 2)                │
│  ILandmarkProvider (MediaPipeLandmarkProvider)       │
│  WebCamTexture → 랜드마크 좌표                        │
├─────────────────────────────────────────────────────┤
│              Camera Layer (Phase 1) ← 현재 구현       │
│  ICameraFrameSource (UnityWebCamFrameSource)         │
│  WebCamTexture 추상화, 장치 관리, 오류 처리            │
└─────────────────────────────────────────────────────┘
```

## 핵심 원칙

### 1. 인터페이스 우선
게임 로직은 구체 구현체(WebCamTexture, MediaPipe)를 직접 참조하지 않는다.
모든 접근은 인터페이스(ICameraFrameSource, ILandmarkProvider 등)를 통해서만 한다.

### 2. 오류 전파 방식
예외(Exception)를 던지지 않고 결과 객체(CameraStartResult)의 ErrorCode로 오류를 전달한다.
ErrorCode 목록: `Docs/Technical/ERROR_CODES.md`

### 3. 리소스 해제
MonoBehaviour의 `OnDestroy`에서 WebCamTexture.Stop() + Destroy()를 반드시 호출한다.
Scene 전환 시 카메라 리소스 누수가 없어야 한다.

### 4. 좌표계 분리
화면 표시용 좌표계(거울 반전, 회전)와 분석용 좌표계를 분리한다.
Phase 1에서는 화면 표시만 구현한다.
Phase 2~3에서 분석 좌표계를 별도로 정의한다.
상세: `Docs/Technical/COORDINATE_SYSTEMS.md`

## 폴더 구조

```
Assets/_Project/
├── Scripts/
│   ├── Core/           # 공통 인터페이스 (ILandmarkProvider 등)
│   ├── Camera/         # ICameraFrameSource 및 구현체
│   │   └── WebCamSpike/  # Phase 1 스파이크 컨트롤러
│   └── Debugging/      # 진단 데이터 클래스
├── Settings/           # ScriptableObject 정의 및 인스턴스
├── Scenes/
│   └── Technical/      # 기술 스파이크 Scene
└── Tests/
    ├── EditMode/
    └── PlayMode/
```

## Phase별 구현 계획

| Phase | 추가되는 레이어 | 주요 파일 |
|---|---|---|
| 1 (현재) | Camera Layer | `UnityWebCamFrameSource`, `WebCamSpikeController` |
| 2 | Landmark Layer | `MediaPipeLandmarkProvider`, `LandmarkFrame` |
| 3 | Motion/Sign Layer | `MotionRecorder`, `SignEvaluator` |
| 4 | Game Layer | NPC, Quest, Sign Dictionary |
