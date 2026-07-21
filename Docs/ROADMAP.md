# WordsInSilence — 개발 로드맵

## Phase 0 — 플러그인 호환성 검증
**목표:** MediaPipeUnityPlugin v0.16.3이 Unity 6.3 LTS + URP 환경에서 정상 동작하는지 확인한다.

**완료 조건:**
- [ ] 플러그인 버전(v0.16.3) 및 설치 출처 기록
- [ ] 라이선스 파일 보존
- [ ] 샘플 Hand Landmarker Scene 동작 확인 (Editor)
- [ ] 본 프로젝트 컴파일 오류 없음
- [ ] 설치·삭제·복구 방법 문서화
- [ ] Windows Scripting Backend = Mono 설정
- [ ] Windows 1차 빌드 성공 여부 기록

**상태:** 진행 중 (사용자 수동 단계 대기)

---

## Phase 1 — WebCam 기술 스파이크
**목표:** Unity WebCamTexture API를 추상화하여 카메라 영상을 안정적으로 표시한다.

**완료 조건:**
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

**상태:** 구현 완료 (검증 필요)

---

## Phase 2 — Hand Landmarker 기술 스파이크
**목표:** MediaPipe Hand Landmarker를 WebCamTexture 피드에 연결하고 랜드마크를 화면에 오버레이한다.

**상태:** 구현 완료 (씬 배선 + Editor 검증 필요)

---

## Phase 3 — 동작 인식 파이프라인
**목표:** 수어 동작을 인식하고 게임 이벤트로 변환하는 파이프라인 구축.

**상태:** 구현 완료 (Hand 기반 PoseTemplateEvaluator, SignDictionary, MotionRecorder)

---

## Phase 3.5 — Pose Landmarker 및 Hand-Pose 통합 기반
**목표:** 어깨/팔꿈치/손목 감지 → Hand+Pose 통합 프레임 → 신체 기준 손 위치 정규화 기반 완성.
기존 Hand 인식 기능(0.15 임계값, SignDictionary, MotionRecorder)은 일체 수정하지 않는다.

**완료 조건:**
- [ ] PoseBodyMathTests 16개 통과
- [ ] 기존 테스트 18개 유지
- [ ] 02_PoseIntegration 씬 Play → 어깨/팔꿈치/손목 오버레이 확인
- [ ] Hand 미검출 시 앱 종료 안됨
- [ ] 진단 UI에서 FPS/품질/신체 좌표 확인
- [ ] POSE_INTEGRATION_AUDIT.md 성능 측정값 기입

**상태:** 구현 완료 (씬 배선 + Editor 검증 필요)

---

## Phase 4 — Unified Motion Recording
**목표:** Hand+Pose 통합 녹화 저장/로드. HandPoseRecording ScriptableObject 래퍼. RecognitionTestUI 연동.

**상태:** 미착수

---

## Phase 5 — Multi-Component Evaluation
**목표:** HandShapeError / BodyRelativePositionError / ArmPoseError 분리. 표현별 EvaluationProfile.

**상태:** 미착수

---

## Phase 6 — Reference Set and Evaluation Profiles
**목표:** 기준 동작 3~5회 다중 녹화, 표현별 가중치, Pose 필수 여부 설정.

**상태:** 미착수

---

## Phase 7 — Sign Interaction Sandbox
**목표:** ISignInteractionService / IMotionEvaluator 기반 NPC 상호작용 테스트. 웹캠 없이 녹화 재생 테스트 가능.

**상태:** 미착수

---

## Phase 8 — 실제 한국수어 콘텐츠
**목표:** 공식 자료 기반, 전문가 검토 후 표현 등록. Face Landmarker 필요 시 별도 스파이크.

**상태:** 미착수

---

## Phase 9 — RPG Vertical Slice
**목표:** 소형 지역, NPC 3명, 퀘스트 3개, 검증된 표현 5~8개, Windows 빌드.

**상태:** 미착수

---

## 주의 사항
- 각 Phase 완료 조건 달성 후 다음 Phase로 진행한다.
- MediaPipe 호환성 이슈 발생 시 대안 라이브러리 검토 (Phase 0 판정 포함).
