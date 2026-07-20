# 수어 상호작용 흐름

## 인터랙션 상태 머신

```
[IDLE]
  │ NPC에게 상호작용 키 입력
  ▼
[NPC_SIGNING]          ← NPC가 수어 대화 표시 (애니메이션 또는 이미지)
  │ 플레이어 응답 요구
  ▼
[COUNTDOWN]            ← 3초 카운트다운 (준비 시간)
  │ 카운트다운 완료
  ▼
[RECORDING]            ← WebCamTexture 프레임 수집 + MediaPipe 분석
  │ maxRecordingSeconds 경과 또는 인식 완료
  ▼
[EVALUATING]           ← IMotionEvaluator가 수어 동작 판정
  │
  ├── 성공 → [SUCCESS] → 퀘스트/대화 분기 진행
  └── 실패 → [FAILURE] → 힌트 제공 후 [COUNTDOWN] 재시도 (최대 3회)
```

## Phase별 구현 계획

| Phase | 구현 내용 |
|---|---|
| 1 | WebCamTexture 표시 (IDLE → 카메라 표시만) |
| 2 | MediaPipe 랜드마크 오버레이 |
| 3 | RECORDING → EVALUATING 구현 |
| 4 | 전체 상태 머신 + NPC 연동 |

## 수어 인식 기준 (임시)

> 아래 수치는 프로토타입 단계 임시 값이다. 실제 사용자 테스트 후 조정 필요.

- 최소 유효 프레임 비율: 70% (`minValidFrameRatio = 0.7`)
- 최대 녹화 시간: 10초 (`maxRecordingSeconds = 10.0`)
- 카운트다운: 3초 (`countdownSeconds = 3.0`)
- 최대 재시도: 3회

## 피드백 표시

- 성공: 화면 효과 + 텍스트 "잘 했어요!"
- 실패: 수어 힌트 이미지 표시 + "다시 해봐요"
- 타임아웃: "시간이 초과됐어요. 다시 시도해봐요"
