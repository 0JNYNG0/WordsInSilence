# 카메라 파이프라인

## 개요

Phase 1에서 구현하는 카메라 파이프라인은 `WebCamTexture`를 추상화하여
게임 로직이 카메라 구현 세부사항에 의존하지 않도록 설계한다.

## 파이프라인 흐름

```
[물리 카메라 장치]
        │
        ▼
[WebCamTexture]          ← Unity API
        │
        ▼
[UnityWebCamFrameSource] ← ICameraFrameSource 구현체
        │
        ├──► [RawImage (화면 표시)]     ← uvRect로 거울/회전 보정
        │
        └──► [ILandmarkProvider]        ← Phase 2에서 연결 (분리된 좌표계)
```

## 장치 초기화 흐름

```
StartAsync(CameraStartRequest)
    │
    ├── WebCamTexture.devices 확인
    │       └── 없으면 → CameraStartResult { ErrorCode = "CAMERA_NOT_FOUND" }
    │
    ├── 요청된 DeviceName 또는 기본 장치 선택
    │       └── 없으면 → CameraStartResult { ErrorCode = "CAMERA_DEVICE_NOT_FOUND" }
    │
    ├── new WebCamTexture(deviceName, width, height, fps)
    │
    ├── webCamTexture.Play()
    │
    ├── 최대 3초 대기 (width > 16 될 때까지 — 초기화 완료 신호)
    │       └── 타임아웃 → CameraStartResult { ErrorCode = "CAMERA_TIMEOUT" }
    │
    └── CameraStartResult { IsSuccess = true, ActualWidth, ActualHeight, ActualFps }
```

## 해제 흐름

```
StopAsync() 또는 OnDestroy()
    │
    ├── webCamTexture.Stop()
    └── Object.Destroy(webCamTexture)
```

## 주의 사항

- `WebCamTexture.Play()` 호출 직후에는 `width`가 16으로 반환될 수 있다.
  실제 해상도는 몇 프레임 후에 확정된다. → 대기 루프로 처리.
- `videoRotationAngle`: 일부 기기에서 카메라가 회전된 상태로 전달된다.
  `RawImage.rectTransform.localRotation`으로 보정한다.
- `videoVerticallyMirrored`: 일부 웹캠에서 Y축이 반전된다.
  uvRect의 Y 시작점을 조정하여 보정한다.

## 오류 코드 목록

`Docs/Technical/ERROR_CODES.md` 참조.
