import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

U = Path(__file__).resolve().parent
ROOT = U.parents[3]
STATE = U.parent / "STATE.json"
KEY = U.name
now = datetime.now(timezone(timedelta(hours=9))).isoformat()


def read(path):
    return json.loads(path.read_text("utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for path in [ROOT / "WordsInSilence_PROJECT_CONTEXT.md", ROOT / "NEW_CHAT_HANDOFF.md", STATE]:
    assert sha(path) == sha(U / "before" / path.name), f"Concurrent update: {path}"

audit = read(U / "FINAL_AUDIT.json")
verify = read(U / "revised/VERIFY.json")
delivery = read(U / "DELIVERY_CHECK.json")
assert delivery["status"] == "PASS"
assert audit["numeric_parameters_unchanged"]

next_action = (
    "손 영역·중심 분리 후보의 공개 TRAIN36 비교 결과를 사용자와 검토한다. "
    "관측/추정/영역만/근거없음 분리와 현재 관측 우선은 확인했지만 위치 복원 개선은 입증되지 않아 미채택. "
    "자연 지원9→중심1+영역1(본동작2→0), 같은 합성38시점 평균 오차는 사실상 동일. "
    "다음 제안은 보이는 손을 놓친 공개 장면에서 현재 이미지 손 재검출을 별도 조건으로 비교하는 것. "
    "추적 규칙 완화/계수 재탐색 대신 정상·교차 대조와 오배정/허위복원/지연을 함께 검증할 계획부터 확정한다. "
    "C·T3-F/운영DTW/웹 유지, 사용자·멘토 자료 학습/튜닝/선택 금지. "
    "현재 승인 단위는 완료; 결과 확인 전 추가 학습·다운로드·운영 반영 없음."
)
title = "2026-10-20 손 영역·중심 분리와 현재 관측 재확인 완료 — 복원 개선 미입증, 미채택"
entry = f"""### {title}
- 단위 `research/front_keyphase_v1/ops/{KEY}`. 보고서 `RESULT_KO.html`/`RESULT_KO.txt`, 종합 판정 `FINAL_AUDIT.json`. Codex 로컬 구현·실행·별도 저장 검산·시각 검토 완료. 학습/새 검출기/분류 재평가 없음.
- 같은 공개 TRAIN36영상·23단어·9촬영자·4641프레임, 저장 C 관측과 sensor 범위 픽셀만 사용. 사용자/멘토/DEV/보류14~18 미접근. 단어별 예외 없음.
- 현재 관측 중심/2D 영상 변환의 중심 추정/영상점 영역만/근거없음 네 상태로 분리. 영역은 손 전체 분할이 아님. 현재 관측 재개 시 그 좌표를 우선하고 현재 팔·손 연결로 추적 근거 재확인. 기존 좌우는 강제 수정하지 않음.
- 관측 4086손시점 그대로, 기존 정상42/42·교차54프레임 배정 보존. 알려진 좌우 오류12는 그대로. 손가락 관측 생성0. 정상 보존은 기존값 유지 설계의 검증이지 좌우 오류 해결 성과가 아님.
- 자연 누락의 이전9개 위치보조 → 새 중심1개+영역만1개, 나머지7보류. 주석 본동작 안 이전2개 지원도0. 모두 짧은 시점이며 긴 가림 복원 아님. 보이는 손을 보류하는 회귀가 있어 미채택.
- 검출만 숨긴 동일420손프레임: 이전61추정 → 새38추정/382보류. 이전 어깨너비10%초과4오차도 추정 보류로 사라졌으며 좌표를 고친 성과가 아님. 같은38시점 평균 proxy 오차 2.10525%→2.11137%(사실상 동일/소폭 악화),23개선/15악화. 숨긴 MP 중심은 임시 기준, 해부학적 정답/수어 정확도 아님.
- 현재 완전 손바닥 관측 재개195건, 2D 방향 변화367건을 별도 기록. 서로 겹칠 수 있고 손 전체 물리적 재등장/실제3D회전 횟수 아님. 현재 관측으로 재확인하는 계약은 통과했지만 정확한 중심 자체를 새로 검출한 것은 아님.
- 첫 후보 결과(합성33추정) 보존 후 현재 손바닥 안 유효 추적점 유지 수정본을 `revised/`에 별도 동결. 숫자 조건 동일;358현재손시점에서 영상점 유지, 합성38로 회복, 자연 지원 동일. 결과를 보고 같은 TRAIN에서 수정한 개발 비교이며 독립 시험 아님.
- 수정본 준비의 STATE 백업 경로 오류는 영상 실행 전 최초 증거 보존 후 경로만 복구(`revised/FIRST_ERROR.json`, `SETUP_RECOVERY.json`). 과거/첫 후보 결과 덮어쓰기 없음.
- 계약11개, 저장 추적2525건/중심 변환2161건 검산.22검토 시점의8이미지 시트 확인. 동일 Codex 별도 코드/시각 검토이며 다른 사람·에이전트 독립 검증은 아님. 전 프레임 정밀 좌표 정답 없음.
- C·T3-F6가중치 포함32보호파일/첫·수정 동결핀 해시 유지. 운영 모델/DTW/웹 불변. 보고서36영상·36그래프·17링크·JS구문 확인. 브라우저 file URL 정책으로 자동 미리보기 미확인, 우회 없음; 이미지 로컬 검토.
- 실행 종료·두 후보 잠금 없음. 다음은 결과 검토 및 현재 이미지 재검출 비교 조건 설계. 이번 중심 추정은 학습 입력에 적용하지 않음.

"""
path = ROOT / "WordsInSilence_PROJECT_CONTEXT.md"
context = path.read_text("utf-8-sig")
assert f"### {title}" not in context
anchor8 = "## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n"
anchor9 = "## 9. 다음 단일 작업\n\n"
assert context.count(anchor8) == context.count(anchor9) == 1
context = context.replace(anchor8, anchor8 + entry, 1)
context = context.replace(anchor9, anchor9 + f"### {title}\n{next_action}\n\n", 1)
path.write_text(context, encoding="utf8")

state = read(STATE)
state.update(
    updated_date_kst="2026-10-20", updated_at=now,
    current_stage="WORD100_REGION_CENTER_RECHECK_REVIEW",
    activity_state="AWAITING_USER_REVIEW",
    next_single_action=next_action, resume_command=next_action,
)
assert KEY not in state
state[KEY] = dict(
    status=audit["status"], at=now, unit=KEY,
    role="Codex local implementation/execution; separate saved arithmetic and visual checks by same Codex; no independent agent/human",
    scope=dict(videos=36, words=23, train_people=9, frames=4641),
    result=verify, final_audit="FINAL_AUDIT.json",
    visual_review="VISUAL_REVIEW.json", delivery_check=delivery,
    first_candidate_preserved=True, revised_candidate="revised/candidate.py",
    numeric_parameters_unchanged=True, production_adoption=False,
    existing_C_T3F_unchanged=True, user_recordings_used=False,
    training=0, new_detector_runs=0, classifier_inference=0,
    natural_regression="9old support ->1estimated center+1region-only+7abstain; annotation-core support2->0",
    paired_conclusion="Same38 proxy mean2.10525%->2.11137% shoulder; essentially unchanged/slightly worse. Lower error across different accepted subsets is not proven recovery.",
    next_single_action=next_action,
)
STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf8")

handoff = f"""# WordsInSilence 최신 인수인계

갱신 {now}

현재 목표: 관측 누락을 줄이되 추정한 위치를 실제 손가락 관측이나 올바른 수어로 오인하지 않기.
마지막 완료: 손 영역·중심 분리와 회전/재관측 확인 후보의 공개 TRAIN36 비교. 구조는 검증했지만 복원 개선 미입증으로 미채택.
다음 한 작업: 이번 결과 검토 후 현재 이미지 손 재검출의 별도 비교 조건 정리. 추가 학습은 아직 하지 않는다.

- 단위 `{U.as_posix()}`. `RESULT_KO.html`, `RESULT_KO.txt`, `FINAL_AUDIT.json`에 결론. `revised/VERIFY.json`의 pending visual은 검산 당시 상태; 이후 `VISUAL_REVIEW.json`과 종합 판정으로 시각 검토 완료.
- 공개 TRAIN36영상/23단어/9촬영자/4641프레임, 기존 C 저장 관측·동일 sensor 픽셀. 사용자/멘토/DEV/보류14~18 미접근.
- 현재 관측 중심·영상 변환 중심 추정·영상점 영역만·근거없음 분리. 현재 관측이 돌아오면 그 중심으로 재확인하며 이전 예측으로 끌어가지 않음. 영역은 손 전체 윤곽/손가락 관측이 아님. 좌우는 기존 C 보존.
- 기존 관측4086손시점, 정상42/42, 교차54프레임 그대로. 알려진 좌우오류12도 그대로. 없는 손가락 생성0.
- 자연 누락에서 이전 위치보조9 → 중심1+영역만1+보류7. 주석상 본동작 지원2→0. 대지/어린이날 등 영상에 손이 보이는데 지원을 잃은 회귀를 숨기지 않음.
- 동일 검출삭제420손프레임: 이전61추정→새38추정/382보류. 큰 오차4개도 새로 고친 것이 아니라 보류. 같은38개 평균 proxy 오차 어깨너비2.10525%→2.11137%,23개선/15악화. MP 임시 기준, 정밀 좌표 정답/인식률 아님.
- 첫 후보33추정 결과 보존. 현재 손바닥 안 유효 영상점만 유지하는 수정본을 별도 `revised/`에 고정, 수치 조건 변경0.358현재손시점 유지로 합성38, 자연 지원은 동일. 같은 TRAIN 결과 기반 개발 수정이며 독립 최종시험 아님.
- 재관측195/2D 방향변화367은 진단 사건 수, 실제 물리적 재등장/3D회전 정답 아님. 첫 등장/완전 가림/긴 공백/기존 좌우오류는 해결하지 못함.
- 계약11개, 저장추적2525건·중심변환2161건 검산,22시점8시트 시각검토. 같은 Codex 별도 검산이며 다른 사람/에이전트 독립 검토 아님.
- `revised/FIRST_ERROR.json`은 준비 백업 경로 오류(실험 전), `SETUP_RECOVERY.json`에 경로 수정. 첫 결과/수치조건/원본 유지. 완료 실험 재실행 금지.
- C·T3-F6가중치 포함32보호파일과 첫/수정 동결핀 유지. 학습/분류재평가/새검출기/다운로드/카메라/운영DTW·웹 변경0. 후보 채택 없음.
- 보고서36영상·36그래프·17상대링크·JS구문 검사 완료. 이미지8시트/이동그래프 로컬 검토. 브라우저 file URL 정책차단은 우회하지 않았으며 자동 미리보기 미확인.
- 명령 종료, 두 후보 RUNNING 잠금없음. 원본/W:/sources/과거 동결 결과 읽기전용. 사용자 촬영 학습/튜닝/선택 금지.

{next_action}
"""
(ROOT / "NEW_CHAT_HANDOFF.md").write_text(handoff, encoding="utf8")
for p in [STATE, U / "FINAL_AUDIT.json", U / "VISUAL_REVIEW.json"]:
    read(p)
receipt = dict(at=now, state_key=KEY, status=state[KEY]["status"],
    files={str(p): sha(p) for p in [
        path, ROOT / "NEW_CHAT_HANDOFF.md", STATE, U / "RESULT_KO.html",
        U / "RESULT_KO.txt", U / "FINAL_AUDIT.json", U / "DELIVERY_CHECK.json"]})
(U / "CLOSEOUT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
print(json.dumps(dict(status="CLOSED_NOT_ADOPTED", files=3, state_key=KEY), ensure_ascii=False))
