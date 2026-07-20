# 좌표계 정의

## Phase 1 범위

Phase 1에서는 화면 표시 좌표계만 구현한다.
분석 좌표계(MediaPipe 입력, 랜드마크 출력)는 Phase 2~3에서 정의한다.

---

## 화면 표시 좌표계 (Phase 1 구현)

### uvRect 기반 거울 표시

```
기본 (거울 없음):
  uvRect = Rect(0, 0, 1, 1)
  → 좌상단이 (0,0), 우하단이 (1,1)

거울 표시 (수평 반전):
  uvRect = Rect(1, 0, -1, 1)
  → 좌상단이 (1,0), 우하단이 (0,1)
  → X 좌표가 반전됨
```

### videoRotationAngle 보정

```csharp
// RawImage의 RectTransform에 카메라 회전 보정 적용
rawImage.rectTransform.localRotation = Quaternion.Euler(0, 0, -webCamTexture.videoRotationAngle);
```

### videoVerticallyMirrored 보정

```
일부 웹캠에서 Y축 반전 발생:
  videoVerticallyMirrored == true 인 경우:
  uvRect.y = 1, uvRect.height = -1 로 조정 (거울 없을 때)
  uvRect.y = 1, uvRect.height = -1 + 기존 X 반전 조합
```

---

## 분석 좌표계 (Phase 2~3 예정)

MediaPipe 입력 이미지와 출력 랜드마크 좌표는 화면 표시와 독립적으로 처리해야 한다.

### 원칙
- 분석용 이미지는 **거울 반전 없이** MediaPipe에 전달한다.
  (화면 표시 거울 반전과 분석 이미지를 동기화하면 좌우 손 인식이 뒤바뀜)
- 랜드마크 좌표 (0~1 정규화)는 원본 이미지 공간 기준이다.
- 화면 오버레이 시 거울 변환을 별도로 적용한다.

### 좌표 변환 (Phase 2에서 구현 예정)
```
MediaPipe 랜드마크 (0~1 정규화, 원본 이미지 기준)
    │
    ├── 화면 오버레이용: mirrorDisplay 여부에 따라 X 변환 후 스크린 픽셀로 변환
    └── 분석용: 변환 없이 정규화 좌표 그대로 사용
```

---

## Unity 좌표계 참고

- Screen Space: 좌하단 (0,0), 우상단 (Screen.width, Screen.height)
- UV Space: 좌하단 (0,0), 우상단 (1,1)
- RawImage uvRect: UV 공간에서 텍스처의 표시 영역 정의
