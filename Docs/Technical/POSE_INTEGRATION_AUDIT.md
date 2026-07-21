# Phase 3.5 — Pose Integration Audit

최종 갱신: 2026-07-21

## 감사 결과 요약

### 구현 전 확인 사항

| 항목 | 결과 |
|------|------|
| MediaPipeUnityPlugin v0.16.3 PoseLandmarker API 존재 여부 | 확인 (`Mediapipe.Tasks.Vision.PoseLandmarker`) |
| 모델 파일 경로 (`pose_landmarker_lite.bytes`) | `Packages/com.github.homuler.mediapipe/PackageResources/MediaPipe/` |
| Glog 중복 초기화 위험 | `MediaPipeRuntime` 레퍼런스 카운터로 해결 |
| 기존 Hand 인식 (0.15 임계값) 영향 | 없음 — Recognition/ 전체 미수정 |
| asmdef 수정 필요 여부 | 불필요 — `Pose/` 폴더는 `WordsInSilence.Runtime` 하에 포함 |

### 핵심 설계 결정

| 결정 | 내용 |
|------|------|
| Glog 초기화 공유 | `MediaPipeRuntime.EnsureInitialized()` / `Release()` 레퍼런스 카운터 |
| 픽셀 복사 2회 허용 | Hand+Pose 각각 `GetPixels32()` 호출 — Phase 3.5에서 허용, Phase 4에서 최적화 검토 |
| PoseLandmarker 모델 | `pose_landmarker_lite.bytes` (속도 우선) |
| timestamp 통합 기준 | `_maxTimestampDiffMs = 100ms` (Inspector 에서 조정 가능) |
| 신체 기준 정규화 단위 | 어깨 너비 (`ShoulderWidth2D`) |

---

## 성능 측정 템플릿

Unity Editor Play 모드에서 `02_PoseIntegration` 씬을 실행하고 진단 UI에서 측정값을 아래에 기입한다.

### 측정 환경

| 항목 | 값 |
|------|----|
| 측정 날짜 | (미기입) |
| 기기 | (미기입) |
| 해상도 | (미기입) |
| 조명 조건 | (미기입) |

### FPS 측정

| 항목 | 측정값 | 목표 |
|------|--------|------|
| Combined FPS (Good 품질) | (미기입) | ≥ 15 FPS |
| Hand FPS | (미기입) | ≥ 15 FPS |
| Pose FPS | (미기입) | ≥ 15 FPS |
| HandPose Timestamp Diff (평균) | (미기입) | ≤ 50 ms |

### 품질 분포 (1분 측정)

| Quality | 프레임 수 | 비율 |
|---------|-----------|------|
| Good | (미기입) | (미기입) |
| HandOnly | (미기입) | (미기입) |
| PoseOnly | (미기입) | (미기입) |
| InsufficientPose | (미기입) | (미기입) |
| AssociationUncertain | (미기입) | (미기입) |

### 어깨 측정 안정성

| 항목 | 측정값 |
|------|--------|
| ShoulderWidth 평균 | (미기입) |
| ShoulderWidth 표준편차 | (미기입) |
| ShoulderCenter 이동량 (1분) | (미기입) |

### 신체 기준 손목 좌표 범위

| 항목 | 최솟값 | 최댓값 |
|------|--------|--------|
| LeftWristBodyRelative.x | (미기입) | (미기입) |
| LeftWristBodyRelative.y | (미기입) | (미기입) |

---

## 픽셀 복사 2회 성능 노트

Phase 3.5 에서는 Hand/Pose 두 Provider가 각각 `GetPixels32()` → `SetPixels32()` 를 호출한다.
이는 CPU 메모리 대역폭을 2배 사용한다.

**최적화 방안 (Phase 4 검토):**
1. 공유 `SharedFrameBuffer` MonoBehaviour 도입 → 1회 복사 후 두 Provider에 공유
2. `NativeArray<byte>` + GPU readback (복잡도 높음, Phase 5 이후 검토)

**Phase 3.5 판단:** FPS가 목표(≥15)를 만족하면 최적화 보류.

---

## 완료 체크리스트

- [ ] PoseBodyMathTests 16개 통과
- [ ] 기존 EditMode/PlayMode 테스트 18개 유지
- [ ] Unity Editor 02_PoseIntegration 씬 Play → 어깨/팔꿈치/손목 점 오버레이 확인
- [ ] Hand 미검출 시 앱 종료 안됨 확인
- [ ] 성능 측정값 이 문서에 기입
