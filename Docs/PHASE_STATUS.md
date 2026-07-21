# Phase 진행 상태

최종 갱신: 2026-07-21

## 현재 Phase

| Phase | 상태 | 비고 |
|---|---|---|
| Phase 0 — 플러그인 호환성 검증 | 진행 중 | 사용자 수동 설치 단계 대기 |
| Phase 1 — WebCam 기술 스파이크 | 구현 완료 | 검증 필요 |
| Phase 2 — Hand Landmarker 스파이크 | 구현 완료 | 씬 배선 + Editor 검증 필요 |
| Phase 3 — 동작 인식 파이프라인 | 구현 완료 | Hand 기반 |
| Phase 3.5 — Pose Landmarker 통합 | 구현 완료 | 씬 배선 + Editor 검증 필요 |
| Phase 4 이상 | 미착수 | |

## Phase 0 체크리스트

- [ ] MediaPipeUnityPlugin v0.16.3 설치 완료
- [ ] 설치 출처 기록 (`MEDIAPIPE_SETUP.md`)
- [ ] 라이선스 파일 확인 및 보존
- [ ] 샘플 Hand Landmarker Scene 동작 확인 (Editor)
- [ ] 본 프로젝트 컴파일 오류 없음
- [ ] 설치·삭제·복구 절차 문서화
- [x] Windows Scripting Backend = Mono 설정 (ProjectSettings.asset 편집)
- [ ] Windows 1차 빌드 성공 여부 기록

> Phase 0은 사용자가 Unity Editor에서 MediaPipe를 설치하고 샘플 Scene을 확인한 후 완료 판정한다.
> 상세 절차: `Docs/Technical/MEDIAPIPE_SETUP.md`

## Phase 1 체크리스트

- [ ] Unity Editor에서 노트북 웹캠 영상 표시
- [ ] 카메라 장치 선택 가능
- [ ] Stop/Start/Restart 반복 가능
- [ ] Scene 전환 후 카메라 리소스 정상 해제
- [ ] Windows Standalone Build에서 작동
- [ ] 10분 이상 연속 실행 시 중대한 메모리 증가 없음
- [ ] 카메라 없을 때 비정상 종료하지 않음
- [ ] 카메라 점유 시 명확한 오류 표시
- [ ] EditMode 테스트 통과
- [ ] PlayMode 테스트 통과 (웹캠 비의존 테스트)

> Phase 1 수동 체크리스트: `Docs/Testing/MANUAL_TEST_CHECKLIST.md`

## Phase 3.5 체크리스트

- [ ] `MediaPipeRuntime` 공유 초기화 — Hand+Pose 동시 실행 시 Glog 크래시 없음
- [ ] `PoseLandmarkProvider` — pose_landmarker_lite.bytes 로드 성공
- [ ] `PoseBodyMathTests` 16개 통과
- [ ] 기존 EditMode/PlayMode 테스트 18개 유지
- [ ] `02_PoseIntegration` 씬 생성 (01_WebCamSpike 복제)
- [ ] `PoseLandmarkProvider` / `HandPoseIntegrator` / `PoseOverlayRenderer` 씬 배선
- [ ] Editor Play → 어깨/팔꿈치/손목 점 오버레이 확인 (노란 점 7개)
- [ ] 어깨 중심 빨간 점 표시 확인
- [ ] 진단 UI — FPS / Quality / ShoulderWidth / WristBodyRelative 수치 확인
- [ ] Hand 미검출 (손을 화면 밖) → 앱 종료 안됨
- [ ] POSE_INTEGRATION_AUDIT.md 성능 측정값 기입
- [ ] Windows Standalone Build 검증 (가능하면)
