# 빌드 설정

## Windows Standalone

| 설정 항목 | 값 | 비고 |
|---|---|---|
| Scripting Backend | **Mono** | D-01 결정 — IL2CPP 사용 불가 |
| API Compatibility Level | .NET Standard 2.1 | |
| Architecture | x86_64 | |
| Build Type | Development Build 권장 (프로토타입 단계) | |

### Scripting Backend 변경 방법 (Unity Editor)

1. **File > Build Settings** 열기
2. **Standalone** 선택
3. **Player Settings** 클릭
4. **Other Settings > Configuration > Scripting Backend** → **Mono** 선택
5. 저장 후 빌드

> `ProjectSettings/ProjectSettings.asset` 에 `scriptingBackend: Standalone: 0` 이 기록된다.

### 변경 이력

- 2026-07-20: `scriptingBackend.Standalone` = 0 (Mono) 으로 설정
  - 이전 값: 미설정 (기본값 = Mono로 동작하나 명시적으로 기록)
  - 이유: MediaPipeUnityPlugin Windows IL2CPP 미지원 (D-01)

---

## Android

| 설정 항목 | 값 | 비고 |
|---|---|---|
| Scripting Backend | IL2CPP | Android는 Mono 사용 불가 |
| Target Architecture | ARM64 | |
| API Level (최소) | 25 | |

---

## 빌드 전 체크리스트

### 모든 플랫폼 공통
- [ ] Console에 컴파일 오류 없음
- [ ] `Assets/StreamingAssets/` 에 MediaPipe 모델 파일 있음

### Windows Standalone
- [ ] Scripting Backend = Mono 확인
- [ ] Development Build 체크 여부 결정
- [ ] `mediapipe_c.dll` 이 빌드 출력 폴더에 포함되는지 확인

---

## 알려진 빌드 이슈

| 이슈 | 증상 | 해결 |
|---|---|---|
| IL2CPP + MediaPipe | `DllNotFoundException: mediapipe_c` | Scripting Backend를 Mono로 변경 |
| 모델 파일 누락 | 런타임에 `FileNotFoundException` | StreamingAssets 확인 |
