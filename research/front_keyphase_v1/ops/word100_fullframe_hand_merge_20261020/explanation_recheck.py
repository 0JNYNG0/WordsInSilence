"""Read saved evidence only; clarify conflict categories and future comparison."""
import collections
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

U = Path(__file__).resolve().parent
O = U.parent
ROOT = O.parents[2]


def read(p):
    return json.loads(p.read_text("utf-8-sig"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def exclusive(p, text):
    with p.open("x", encoding="utf-8") as fp:
        fp.write(text)


at = datetime.now(timezone(timedelta(hours=9))).isoformat()
docpaths = [ROOT / "WordsInSilence_PROJECT_CONTEXT.md",
            ROOT / "NEW_CHAT_HANDOFF.md", O / "STATE.json"]
docs = {p: p.read_text("utf-8") for p in docpaths}
pins = {}
categories = collections.Counter()
details = []
for c in read(U / "CONFLICTS.json"):
    p = U / "records" / (c["clip_id"] + ".json")
    pins[str(p)] = sha(p)
    frame = read(p)["frames"][c["index"]]
    decision = frame["decisions"][c["decision"]]
    stored = [frame["baseline_sides"][j] for j in decision["matched_old"]]
    assert not decision["accepted"]
    if any(s >= 0 and s != decision["side"] for s in stored):
        category = "opposite_assigned_side_matches"
    else:
        assert -1 in stored and decision["side"] in stored
        category = "assigned_same_side_plus_unassigned_candidate"
    categories[category] += 1
    details.append(dict(clip=c["clip_id"], label=c["label"], frame=frame["frame_id"],
                        category=category, matched_stored_sides=stored,
                        current_arm_side=decision["side"], accepted=False))
assert categories == {
    "assigned_same_side_plus_unassigned_candidate": 18,
    "opposite_assigned_side_matches": 3,
}

finite, unavailable = [], []
for r in read(U / "FEATURE_EFFECT.json")["rows"]:
    for s in r["hand_scales"]:
        row = dict(clip=r["clip_id"], label=r["label"], **s)
        if s["before"] is None or s["after"] is None:
            unavailable.append(row)
        else:
            row["relative_change_pct"] = abs(s["after"] / s["before"] - 1) * 100
            finite.append(row)
assert len(finite) == 70 and len(unavailable) == 2
for p in [U / "candidate.py", U / "run.py", U / "CONFLICTS.json",
          U / "FEATURE_EFFECT.json", U / "FROZEN.json",
          ROOT / "research/sign_model_pilot_v1/src/pilot/data.py"]:
    pins[str(p)] = sha(p)
audit = dict(
    at=at, status="SAVED_EVIDENCE_CLARIFICATION_AND_PLAN_REVIEW",
    conflict_categories=dict(categories), conflict_details=details,
    conflict_limit="Categories describe implementation evidence, not true hand identity. All21 withheld;18 are not opposite-side disagreements.",
    finite_scale_pairs=len(finite), unavailable_scale_pairs=unavailable,
    max_scale_change=max(finite, key=lambda x: x["relative_change_pct"]),
    first_exploratory_arithmetic_error="TypeError: unsupported operand type(s) for /: 'NoneType' and 'NoneType'; stdout-only query. Revised calculation preserves undefined scales separately; no source/output mutation or detector rerun.",
    scope="Code and saved TRAIN evidence review by same Codex; no new human/agent validation, detector run, classifier evaluation, training, user recording access or production change.",
    pins=pins,
)
exclusive(U / "EXPLANATION_RECHECK.json", json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

plan = """전체 영상 손 통합 후보: 설명 재검토 및 다음 비교안

현재 상태: 구현·입력 검증 완료 범위는 TRAIN36영상/23단어/9명이다. 전체1199 입력이나 인식 개선이 완료된 상태가 아니다. 이번에는 저장 근거와 코드를 읽었고 검출·학습·사용자 자료 접근은 하지 않았다.

처리:
1. 기존 원본 전체 영상 검출/좌우 배정 캐시를 재사용한다.
2. 같은 원본의 저장 대상 프레임을 IMAGE 모드로 새로 본다. 이번 비교에서는 전체4641프레임을 재검출했고, 통합은 기존에 빠진 손만 대상으로 했다. 빠진 쪽의 현재 pose 팔꿈치·손목이 유효하면 원본의 손목 주변을 따로 검사한다. 원본좌표로 되돌린 전체/부분 검사 결과가 맞아야 관절별로 추가한다. 부분 검사에서만 찾은 손을 단독으로 추가하지 않는다.
3. 기존 손은 보존하고 동일손/중복/좌우 근거 불일치를 분리한다. 새 팔 근거는 현재 pose 손목 거리와 팔꿈치 유효성이다. RGB 팔 윤곽 전체를 새로 이해하는 모델이라고 설명하지 않는다.
4. 통합 후 고정 상반신 ROI를 점별로 적용하고 좌표·관측 마스크·시간을 입력으로 만든다. 채택된 추정치이지 정밀 관절 정답은 아니다.

충돌 정정:21개 진단 중18개는 기존 배정손과 같은 쪽이지만 미배정 후보(-1)에도 대응된 경우다.3개는 반대쪽으로 이미 배정된 후보와 현재 팔 근거가 다르다(후원03 f113, 대지09 f58/59). 어느 쪽이 실제 정답인지는 이 집계로 확정하지 않는다. 모두 새 판단을 덮어쓰지 않았다.

크기 기준: 영상별·손별 손목~네 손가락 뿌리 거리를 모아 중앙값을 계산한다. 관측 추가로 표본이 바뀌어 분모도 바뀐다. 70개 계산가능 쌍 중67개 변화,36영상 중35영상.2개는 전후 모두 크기 계산 불가이며 임의 보충하지 않는다. 최대 마음에들다01 왼손48.8416→52.3541px(+7.1917%). 이미지 확대좌표 복원 오류가 원인이라는 증거는 없고, 현재 측정은 표본 변화에 따른 정규화 변화이다.

계획 판단: 전체검출 통합 전후 비교라는 방향은 유지한다. 즉시 학습 대신 입력 품질 및 정규화 대조 조건을 먼저 확정한다.
제안 A: 같은 전체검출·ROI, 통합 전 좌표, 통합 전 영상별 손 크기 기준.
제안 B: 통합 후 좌표, A와 동일한 해당 영상의 손 크기 기준. A→B는 관측 추가 영향을 분리한다.
제안 C: 통합 후 좌표, 추가 관측으로 재계산한 해당 영상의 손 크기 기준. B→C는 크기 기준 변화 영향을 분리한다.
이 A/B/C는 설명용 비교명이며 기존 C 후보의 이름과 혼동하지 않는다. 실행시 별도 full_before/merge_fixed_scale/merge_recomputed_scale 이름을 쓴다.

공통 조건은 동일 공개자료/촬영자 분리/ROI/시간/모델 구조/학습 예산/paired seed. 각3seed 총9회는 제안이며 아직 실행 승인이나 동결 완료가 아니다. 해당 영상 통합 전 크기가 계산불가면 fixed_scale 조건의 손로컬 채널은 미관측으로 유지하고 해당 수를 별도로 보고한다. 다른 사람·DEV·사용자 크기로 채우지 않는다. 새 검출의 몸 기준 좌표와 로컬 좌표를 구분해 집계한다.

학습 전에 확인할 것:
- 미배정후보 대응과 진짜 반대쪽 근거를 보고에서 분리. 기존 결과/후보를 소급 변경하지 않는다.
- 원본 손 위치의 근거, 손가락 모양의 근거를 분리한다. 두 크기에서 같은 모델의 일치만으로 가려진 관절 정답을 보증하지 않는다. 현재 RGB는 검출/재검출 입력이며 별도 독립 손모양 검증 모델은 없다.
- 범용 체크로 관측 추가, 관절 점프, 불가능한 기하, 정상 교차 보존, 관측부족 상태를 확인한다. 빠른 정상 동작/회전도 있으므로 단순 변화량으로 자동삭제·보간하지 않는다. 새 품질 규칙이 필요하면 별도 후보로 먼저 검증하고 계획을 재확인한다.
- 전체 입력 준비 전에 동일 규칙·예산·평가방식을 고정하고, 준비 후 전수 수치검산과 주요 의심 구간 시각검토를 한다. 일부36영상을 전체 입력 통과로 대체하지 않는다.
- 실제 웹캠 원본 화면 밖 손은 이용 불가하다. 학습에서 원본의 ROI 밖 단서로 얻는 이점과 실제 웹캠의 근거 부재 차이를 별도 공개자료 진단으로 확인한다. 이 진단을 crop-before-detection 학습자료 자동혼합으로 바꾸지 않는다.
- 기존 C·T3-F의 학습 경로는 이번 동일전체검출 대조와 다르므로 고정 외부 비교 기준으로 보고한다. 주효과는 새 동일조건 대조에서 측정한다. 이미 본 DEV/보류자료를 새 독립시험이라고 명명하지 않는다.

RGB 후속 방향: 보이는 윤곽/손 영역/영상 움직임과 관절이 모순되는지 진단하는 보조는 타당하다. 그러나 가려진 손가락의 실제각도 복원, 정답 수어 수행, 새 사용자 인식률 향상은 별도 문제이며 아직 검증되지 않았다. 손 위치는 비교 가능하지만 굽힘은 근거 부족인 상태를 허용한다. 독립 RGB 특징 모델은 이 비교에 한꺼번에 섞지 않고 후속 분리 실험으로 둔다.

사용자 계획 검토 후에만 다음 입력 준비와 학습 실행. 기존 C/T3-F/DTW/웹/사용자·멘토 기록 보존. 특정 단어별 예외 없음.
"""
exclusive(U / "PLAN_REVIEW_KO.txt", plan)

next_action = ("전체검출 통합 전후 비교 방향 유지. EXPLANATION_RECHECK.json/PLAN_REVIEW_KO.txt 사용자 검토 후 "
               "full_before / merge_fixed_scale / merge_recomputed_scale 세 조건과 공통 입력 품질·미관측 처리를 고정한다. "
               "각3seed는 제안이며 추가 학습 아직 미승인. 21충돌=미배정후보 포함18+반대배정근거3으로 구분; "
               "관측수 증가를 손가락 정답/인식 향상으로 해석하지 않는다. C·T3-F/웹 보존, 사용자자료 학습·튜닝·선택 금지.")
summary = """### 2026-10-20 처리 설명·좌우 충돌 세분 및 비교 계획 재검토 — 학습 미실행
- `ops/word100_fullframe_hand_merge_20261020/EXPLANATION_RECHECK.json`/`PLAN_REVIEW_KO.txt`. 기존 저장 근거 읽기만 수행.
- 21충돌 중18은 같은배정손+미배정후보 대응,3은 반대배정 근거 불일치(후원03 f113/대지09 f58·59).21개 실제좌우오류 아님. 모두기존손보존.
- 손크기70쌍 계산가능/2쌍 전후미계산,67쌍 변화. 최대48.8416→52.3541px(+7.1917%). 관측추가와정규화효과분리필요.
- 전체원본검출→같은이미지재검출/손목주변확인→원본좌표통합→고정ROI 유지. 전체/부분은같은검출기,정밀손가락정답검증아님.
- 다음대조는 통합전/통합후기존크기고정/통합후크기재계산 제안. 기존C·T3-F는고정참고. 전체입력·품질확인후사용자검토를거쳐학습; 이번학습0.

"""
for p, content in docs.items():
    assert p.read_text("utf-8") == content, f"Concurrent root edit: {p}"
p = docpaths[0]
text = docs[p]
for anchor, addition in [
    ("## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n", "\n" + summary),
    ("## 9. 다음 단일 작업\n", "\n### 2026-10-20 통합 효과와 크기 기준 분리 계획 검토\n" + next_action + "\n\n"),
]:
    assert anchor in text
    text = text.replace(anchor, anchor + addition, 1)
p.write_text(text, encoding="utf-8")
state = json.loads(docs[docpaths[2]])
state["updated_at"] = at
state["next_single_action"] = next_action
state[U.name]["explanation_recheck"] = {
    "at": at, "audit": str(U / "EXPLANATION_RECHECK.json"),
    "plan": str(U / "PLAN_REVIEW_KO.txt"),
    "training_executed": False, "next_single_action": next_action,
}
docpaths[2].write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
handoff = docs[docpaths[1]].splitlines()
handoff = [("갱신 " + at if x.startswith("갱신 ") else
            "다음 한 작업: " + next_action if x.startswith("다음 한 작업: ") else x)
           for x in handoff]
old_summary = next(i for i, x in enumerate(handoff) if x.startswith("### "))
handoff[old_summary:old_summary] = summary.splitlines()
docpaths[1].write_text("\n".join(handoff) + "\n", encoding="utf-8")
assert all(sha(Path(p)) == h for p, h in pins.items())
assert read(U / "EXPLANATION_RECHECK.json")["conflict_categories"] == dict(categories)
assert read(O / "STATE.json")["next_single_action"] == next_action
print(json.dumps({"status": "PASS", "categories": categories, "new_training": 0,
                  "preserved_read_inputs": len(pins)}, ensure_ascii=False))
