# WordsInSilence — Project Overview

## 프로젝트 개요

**WordsInSilence**는 수어(手語, Sign Language) 인식 기능성 RPG 게임이다.
플레이어는 웹캠 앞에서 실제 수어 동작을 수행하여 NPC와 상호작용하고 퀘스트를 진행한다.

## 핵심 목표

- 실시간 손 랜드마크 인식(MediaPipe Hand Landmarker)을 통해 수어 동작 인식
- 배움의 장벽 없이 수어를 자연스럽게 경험할 수 있는 게임 환경 제공
- 접근성과 게임성을 동시에 갖춘 수직 슬라이스 데모 구현

## 기술 스택

| 항목 | 선택 |
|---|---|
| 엔진 | Unity 6000.3.14f1 (Unity 6.3 LTS) |
| 렌더 파이프라인 | URP (Universal Render Pipeline) 17.3.0 |
| 손 인식 | MediaPipeUnityPlugin v0.16.3 |
| 스크립팅 백엔드 (Windows) | Mono (IL2CPP 미지원 — D-01 참조) |
| API 호환성 | .NET Standard 2.1 |
| 테스트 프레임워크 | Unity Test Framework 1.6.0 |

## 관련 문서

- [로드맵](ROADMAP.md)
- [Phase 진행 상태](PHASE_STATUS.md)
- [주요 결정 사항](DECISIONS.md)
- [변경 이력](CHANGELOG_DEV.md)
- [환경 감사 결과](Technical/ENVIRONMENT_AUDIT.md)
- [아키텍처](Technical/ARCHITECTURE.md)
- [MediaPipe 설치](Technical/MEDIAPIPE_SETUP.md)
- [빌드 설정](Technical/BUILD_CONFIGURATION.md)
- [게임 컨셉](GameDesign/GAME_CONCEPT.md)
