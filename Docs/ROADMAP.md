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

## Phase 2 — Hand Landmarker 기술 스파이크 (미구현)
**목표:** MediaPipe Hand Landmarker를 WebCamTexture 피드에 연결하고 랜드마크를 화면에 오버레이한다.

**예정 작업:**
- MediaPipeLandmarkProvider 구현 (ILandmarkProvider)
- LandmarkFrame 데이터 모델
- 화면 랜드마크 오버레이
- 손별 검출률 측정

**상태:** 미구현 (Phase 1 완료 후 착수)

---

## Phase 3 — 동작 인식 파이프라인 (미구현)
**목표:** 수어 동작을 인식하고 게임 이벤트로 변환하는 파이프라인 구축.

**예정 작업:**
- IMotionRecorder 구현
- IMotionEvaluator 구현
- 수어 사전 데이터 모델
- Assembly Definition 도입 (D-03 결정 후)

**상태:** 미구현

---

## Phase 4 — 수직 슬라이스 데모 (미구현)
**목표:** 완전한 수어 상호작용 퀘스트 1개 포함 데모 빌드.

**상태:** 미구현

---

## 주의 사항
- 이 문서는 Phase 0~1 구현 이후 단계를 포함하며, Phase 2 이상은 아직 설계 단계다.
- 각 Phase 완료 조건 달성 후 다음 Phase로 진행한다.
- MediaPipe 호환성 이슈 발생 시 대안 라이브러리 검토 (Phase 0 판정 포함).
