# 오류 코드 목록

## Camera Layer (Phase 1)

| 코드 | 설명 | 발생 조건 |
|---|---|---|
| `CAMERA_NOT_FOUND` | 연결된 카메라 장치가 없음 | `WebCamTexture.devices.Length == 0` |
| `CAMERA_DEVICE_NOT_FOUND` | 요청한 장치 이름을 찾을 수 없음 | `DeviceName`이 장치 목록에 없음 |
| `CAMERA_TIMEOUT` | 카메라 초기화 타임아웃 | Play() 후 3초 내 실제 해상도를 얻지 못함 |
| `CAMERA_ALREADY_RUNNING` | 이미 실행 중인 상태에서 Start 재호출 | `IsRunning == true` 상태에서 StartAsync 호출 |
| `CAMERA_PERMISSION_DENIED` | 카메라 권한 없음 | OS 레벨 카메라 접근 거부 |
| `CAMERA_IN_USE` | 다른 앱이 카메라 점유 중 | 카메라 장치 잠금 상태 |
| `CAMERA_UNKNOWN_ERROR` | 분류되지 않은 오류 | 예외 메시지가 위 코드에 해당하지 않는 경우 |

## Landmark Layer (Phase 2 예정)

| 코드 | 설명 |
|---|---|
| `MEDIAPIPE_NOT_INITIALIZED` | MediaPipe 초기화 실패 |
| `MEDIAPIPE_MODEL_NOT_FOUND` | StreamingAssets 내 모델 파일 없음 |
| `LANDMARK_INFERENCE_ERROR` | 랜드마크 추론 중 오류 |

## Motion Layer (Phase 3 예정)

| 코드 | 설명 |
|---|---|
| `MOTION_RECORDING_TIMEOUT` | 녹화 시간 초과 |
| `MOTION_INSUFFICIENT_FRAMES` | 유효 프레임 부족 (`minValidFrameRatio` 미달) |
| `MOTION_EVALUATION_FAILED` | 수어 평가 실패 |

---

## 오류 코드 사용 규칙

1. 예외(Exception)를 게임 로직 레이어로 전파하지 않는다.
2. 오류는 항상 결과 객체(`CameraStartResult.ErrorCode`)로 반환한다.
3. `ErrorDetail`에는 사람이 읽을 수 있는 상세 설명을 포함한다.
4. UI에는 ErrorCode 코드 자체가 아닌 `ErrorDetail` 또는 한글 번역 메시지를 표시한다.
5. 새 오류 코드 추가 시 이 문서를 먼저 갱신한다.
