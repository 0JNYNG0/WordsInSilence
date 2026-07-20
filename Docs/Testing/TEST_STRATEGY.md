# 테스트 전략

## 테스트 계층

### 1. EditMode 테스트 (자동)
- 데이터 클래스, 순수 로직 단위 테스트
- 하드웨어(카메라, MediaPipe) 불필요
- CI/CD에서 실행 가능
- 위치: `Assets/_Project/Tests/EditMode/`

### 2. PlayMode 테스트 (자동, 부분)
- MonoBehaviour 생명주기, 비동기 흐름 테스트
- 실제 웹캠 의존 테스트는 `[Ignore]` 처리하거나 별도 분류
- 위치: `Assets/_Project/Tests/PlayMode/`

### 3. 수동 테스트
- 실제 웹캠 동작, 화면 표시 품질, 장시간 안정성
- 체크리스트: `Docs/Testing/MANUAL_TEST_CHECKLIST.md`

---

## Phase 1 테스트 범위

### EditMode (CameraDeviceInfoTests.cs)
| 테스트 | 유형 | 설명 |
|---|---|---|
| `CameraDeviceInfo_Fields_SetCorrectly` | 단위 | 필드 기본값 및 할당 확인 |
| `CaptureConfig_DefaultValues_AreValid` | 단위 | ScriptableObject 기본값 범위 확인 |
| `CameraStartResult_Success_Properties` | 단위 | 성공 결과 객체 상태 확인 |
| `CameraStartResult_Failure_HasErrorCode` | 단위 | 실패 결과 객체 오류 코드 확인 |
| `CaptureDiagnostics_CanBeInstantiated` | 단위 | 직렬화 가능 여부 확인 |

### PlayMode (WebCamFrameSourceTests.cs)
| 테스트 | 유형 | 설명 |
|---|---|---|
| `StartAsync_NoCameras_ReturnsCameraNotFound` | 통합 | 카메라 없을 때 올바른 오류 코드 반환 |
| `StartAsync_AlreadyRunning_IsSafeToCall` | 통합 | 중복 Start 호출 시 안전 처리 |
| `StopAsync_WhenNotRunning_DoesNotThrow` | 통합 | 미실행 상태에서 Stop 호출 안전성 |

> 실제 웹캠 의존 테스트: `[Ignore]` 처리 후 MANUAL_TEST_CHECKLIST.md에서 관리

---

## 테스트 실행 방법

1. **Unity Editor > Window > General > Test Runner** 열기
2. **EditMode** 탭: 모든 EditMode 테스트 실행
3. **PlayMode** 탭: 모든 PlayMode 테스트 실행

---

## 테스트 제외 항목 (Phase 1)

- 실제 웹캠 존재 여부에 의존하는 테스트 (환경 의존성)
- 화면 렌더링 품질 테스트 (수동으로만 확인 가능)
- MediaPipe 연동 테스트 (Phase 2에서 추가)
- 10분 이상 장시간 안정성 테스트 (수동으로만 확인 가능)
