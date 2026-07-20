# 데이터 형식 정의

## CameraStartRequest

카메라 시작 요청 데이터.

| 필드 | 타입 | 설명 | 기본값 |
|---|---|---|---|
| `DeviceName` | `string` | 사용할 카메라 장치 이름. `null`이면 기본 장치 | `null` |
| `RequestedWidth` | `int` | 요청 해상도 (가로) | 640 |
| `RequestedHeight` | `int` | 요청 해상도 (세로) | 480 |
| `RequestedFps` | `int` | 요청 프레임레이트 | 30 |
| `MirrorDisplay` | `bool` | 화면 표시 시 좌우 반전 여부 | true |

## CameraStartResult

카메라 시작 결과 데이터.

| 필드 | 타입 | 설명 |
|---|---|---|
| `IsSuccess` | `bool` | 성공 여부 |
| `ErrorCode` | `string` | 실패 시 오류 코드 (ERROR_CODES.md 기준) |
| `ErrorDetail` | `string` | 오류 상세 메시지 (사람이 읽을 수 있는) |
| `ActualWidth` | `int` | 실제 해상도 (가로) |
| `ActualHeight` | `int` | 실제 해상도 (세로) |
| `ActualFps` | `int` | 실제 프레임레이트 |
| `DeviceName` | `string` | 실제 사용된 장치 이름 |

## CameraDeviceInfo

카메라 장치 정보.

| 필드 | 타입 | 설명 |
|---|---|---|
| `Name` | `string` | 장치 이름 (OS 제공) |
| `Index` | `int` | 장치 배열 인덱스 |
| `IsFrontFacing` | `bool` | 전면 카메라 여부 |

## CaptureDiagnostics

실시간 진단 데이터. `DiagnosticsUpdated` 이벤트로 전달됨.

| 필드 | 타입 | 설명 |
|---|---|---|
| `DeviceName` | `string` | 현재 장치 이름 |
| `RequestedWidth` | `int` | 요청 해상도 (가로) |
| `RequestedHeight` | `int` | 요청 해상도 (세로) |
| `RequestedFps` | `int` | 요청 FPS |
| `ActualWidth` | `int` | 실제 해상도 (가로) |
| `ActualHeight` | `int` | 실제 해상도 (세로) |
| `ActualFps` | `int` | 실제 FPS (WebCamTexture.requestedFPS) |
| `FrameProcessingMs` | `float` | 프레임 처리 시간 (ms) |
| `LastErrorCode` | `string` | 마지막 오류 코드 |
| `IsRunning` | `bool` | 현재 실행 중 여부 |

## CaptureConfig (ScriptableObject)

카메라 캡처 설정. `Assets/_Project/Settings/DefaultCaptureConfig.asset` 인스턴스.

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `requestedWidth` | `int` | 640 | 요청 해상도 (가로) |
| `requestedHeight` | `int` | 480 | 요청 해상도 (세로) |
| `requestedFps` | `int` | 30 | 요청 FPS |
| `mirrorDisplay` | `bool` | true | 거울 표시 |
| `countdownSeconds` | `float` | 3.0 | 수어 인식 시작 전 카운트다운 (Phase 3용) |
| `maxRecordingSeconds` | `float` | 10.0 | 최대 녹화 시간 (Phase 3용) |
| `minValidFrameRatio` | `float` | 0.7 | 유효 프레임 최소 비율 (Phase 3용) |

> 경고: `countdownSeconds`, `maxRecordingSeconds`, `minValidFrameRatio`는 프로토타입 단계 임시 수치다.
> 실제 수어 인식 구현(Phase 3) 시 검증된 값으로 교체해야 한다.

## LandmarkFrame (Phase 2 예정)

손 랜드마크 데이터. Phase 2에서 정의 예정.

## SignMotionSample (Phase 3 예정)

수어 동작 시계열 데이터. Phase 3에서 정의 예정.
