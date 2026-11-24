"""Publish the completed fixed comparison; never rerun models or alignment."""
from pathlib import Path
import datetime
import hashlib
import json
import statistics

U = Path(__file__).resolve().parent
R = U.parents[3]

def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))

def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf8")

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
ev = read(U / "EVALUATION.json")
verification = read(U / "review/RESULT_VERIFICATION.json")
method = read(U / "review/POSTEXECUTION_REVIEW.json")
diagnosis_review = read(U / "review/DIAGNOSIS_REVIEW.json")
assert verification["failure_count"] == 0
assert verification["all_candidate_results"] == 534
assert method["status"] == "PASS_FOR_REJECTED_CANDIDATE_COMPLETION"
assert not ev["decision"]["adoption"]
assert not (U / "RUNNING.json").exists()
assert not (U / "COMPLETION.json").exists(), "Do not overwrite completed result"
scores = ev["decision"]["dev_scores"]
mean = statistics.mean(scores)
baseline = ev["summary"]["development/global_dtw/None"]["same_kind_percent"]
wordrows = []
for wid, name in [("WORD0739", "순종"), ("WORD1133", "잘하다"), ("WORD1567", "조상"), ("WORD2523", "이집트")]:
    old = ev["byword"][f"development/{wid}/global_dtw/None"]["same_kind_percent"]
    vals = [ev["byword"][f"development/{wid}/joint/{seed}"]["same_kind_percent"] for seed in [42, 43, 44]]
    avg = statistics.mean(vals)
    wordrows.append(f"{name}: 기존 {old:.2f}% → 새 방식 평균 {avg:.2f}% ({avg-old:+.2f}%p)")

report = f"""손모양·양손 관계·움직임을 함께 보는 단계 정렬: 고정 비교 완료
작성: {now}

결론
세 정보를 함께 보고 영상 전체의 동작 순서를 맞추는 후보를 구현하고 검증했다.
일부 표현은 좋아졌지만 전체 결과는 기존보다 낮아 채택하지 않았다.
이번 결과를 보고 가중치나 기준을 바꿔 다시 시험하지 않았다. 기존 C·T3-F와 운영 웹/DTW는 그대로다.

무엇을 비교했나
공개 한국수어 4개 표현(순종·잘하다·조상·이집트), 16개 영상, 36개 영상 쌍이다.
기록 ID 1~3의 FIT 24쌍과 ID 4의 개발 비교 12쌍을 구분했다. 이미 사용한 개발 자료이며 새로운 독립 시험은 아니다.
손가락 각도·기존 손모양 판정값, 양손/팔/몸통의 관계, 손목의 이동량·방향을 세 묶음으로 계산했다.
단어 이름이나 사람별 예외를 정렬 계산에 넣지 않았다. 단계 정답도 결과를 만든 뒤 검산에만 썼다.
기존에도 관련 정보를 사용한 방식이 있었다. 따라서 단순히 처음 정보를 추가했다고 주장하지 않고,
기존 전체 정렬(global), 주요 구간을 따로 찾는 방식(core_first), 기존 관계 통합 정렬(relational)을 모두 대조했다.

새로 계산한 것은 534개 정렬 경로다.
36쌍 × 저장된 3개 seed 출력 = 108경로, 142개 기술 대조 × 3개 seed = 426경로다.
세 seed는 예전에 학습해 저장한 출력의 차이를 확인한 것이며, 이번에 3회 새 학습한 것이 아니다.
새 모델 추론 0회, 새 학습 0회다.

결과: 같은 관측 단계에 연결된 비율
기존 전체 정렬: {baseline:.2f}%
기존 구간별 검색: 78.29%
기존 관계 통합 정렬: 79.56%
새 결합 정렬: 평균 {mean:.2f}%, 범위 {min(scores):.2f}~{max(scores):.2f}%
기존 전체 정렬 대비 {mean-baseline:+.2f}%p다.

{chr(10).join(wordrows)}

이 숫자는 TCN 단어 인식률이나 수어를 올바르게 했는지의 정확도가 아니다.
관측 단계의 종류를 적어 둔 개발 비교 360개 참고 프레임 기회에서, 같은 종류로 연결된 비율이다.
서로 다른 단계뿐 아니라 경계/불확실 구간에 연결된 부분도 전체 분모에 남겼다.
반복의 정확히 몇 번째 회차인지까지 맞았다는 뜻도 아니다.
전체 하락은 경계/불확실 구간 연결이 늘어난 영향이 크다. 모두 명백한 다른 단계 오답으로 세지 않는다.
동결한 개선 기준(두 전체 정렬 기준보다 평균 +5%p, 모든 seed 개선, 표현별 큰 후퇴 없음)을 통과하지 못했다.

이번에 구체적으로 확인한 문제
1. 안 보이는 손 때문에 시간 정렬이 움직였다.
   같은 영상에서 오른손 정보만 가린 기술 시험은 원래 같은 시점을 맞추면 된다.
   그러나 16개 모두에서 새 방식은 미관측 벌점을 줄이기 위해 다른 시점으로 움직였다.
   보이는 다른 정보끼리의 차이는 오히려 커졌는데 총 계산 비용은 내려갔다.
   평균 약 2.89프레임의 대응 오차가 생겼다. 모든 프레임을 남기는 것만으로는 이를 막지 못했다.
   저장된 경로의 비용을 분해해 확인한 결과이며, 실제 가림 영상의 손을 복구했다는 뜻은 아니다.

2. 짧은 차이가 평균에서 작아질 수 있었다.
   5프레임의 오른손 각도를 바꾼 16개 시험 중 14개에서 참고 구간 기준의 차이가 줄었다.
   예를 들어 5개 변화 프레임이 참고의 2개 시점에 몰리고, 주변 참고 시점은 바뀌지 않은 이웃과 연결됐다.
   변화 프레임을 삭제하거나 구간 밖으로 빼지는 않았지만, 참고 시점별 평균을 내면 차이가 희석됐다.
   한 사례의 정규화 각도 차이는 0.9571에서 0.3242로 줄었고, 변경된 영상 쪽 기준 차이는 0.9572로 남았다.
   따라서 참고 쪽 평균뿐 아니라 실제 변화가 있는 영상 쪽 근거도 함께 보존해야 한다.
   왼손 대조 16개 중 실제 관측값이 바뀌지 않은 4개는 오류 탐지 성공으로 세지 않았다.

보존 및 한계
534개 모두 입력 양쪽의 전체 프레임을 유지했다. 36쌍의 참고 전체 4,380프레임과 확실 구간 1,080프레임도 유지했다.
미관측을 정상 손모양이나 실제 0 움직임으로 바꾸지 않았고 관절 좌표/좌우 배정 자체도 수정하지 않았다.
동일 영상 대조는 오차 0이었다. 속도 대조는 평균 원프레임 오차 약 0.02/0.23으로 작았지만 기존보다 소폭 나빠졌다.
0.75배 대조는 프레임 반복, 1.25배 대조는 일부 프레임 생략이며 실제 다른 속도로 촬영한 영상은 아니다.
각도 변형도 저장된 관측값을 바꾼 기술 시험이며 RGB나 손모양 모델을 다시 추론하지 않았다.
따라서 정상 손내림·교차에 잘못된 교육 피드백이 없다고 입증한 시험은 아니다. 정상 비교의 후퇴가 남아 미채택했다.
공개 정상 영상의 작은 표본과 기술 대조만으로 실제 학습자의 오류 진단 성능이나 모든 수어 적용 성능을 주장하지 않는다.

왜 추가 학습하지 않았나
관측 모델을 바꾸지 않은 상태에서도 정렬 비용이 잘못된 시점 이동을 유리하게 만드는 원인이 직접 확인됐다.
이를 먼저 고쳐야 손모양을 잘못 배운 문제와 비교 시점이 잘못된 문제를 구분할 수 있다.
현재 물리 단계 주석은 새 모델 학습 정답으로 검증된 자료가 아니고, 이 작은 묶음의 일부 손모양 양성 사례도 부족하다.
지금 추가 학습하면 확인된 정렬 결함을 해결한다는 근거가 없다. 필요한 학습을 생략한 것이 아니라 조건을 충족하지 못했다.

다음 한 작업의 방향
미관측 정도는 계속 기록하되, 미관측 벌점을 다른 시점으로 옮기는 것만으로 정렬이 좋아지는 계산을 제거한다.
또 참고 시점과 비교 영상 시점 양쪽에서 짧은 차이를 집계해, 몇 시점으로 몰렸다고 차이가 사라지지 않게 한다.
먼저 같은 영상의 가림/짧은 변화 대조에서 계산 성질을 확인한 뒤 한 후보만 고정해 기존과 비교한다.
이번 결과만으로 그 수정이 성공한다고 보장하지 않으며, 후속 후보는 아직 구현·실행하지 않았다.
특정 단어 예외, 점수에 맞춘 재학습, 새 모델 순회보다 확인된 두 계산 문제를 먼저 해결한다.

검증과 산출물
구현: Codex semantic_feedback / 실행·통합·기록: Codex root
방법·원인 검토: Codex semantic_model / 저장 결과·산술 검산: Codex semantic_data_audit
자체 합성 검사 174개 통과. 별도 저장 결과 검사 16,001개, 오류 0. 원인 진단 별도 검토도 완료했다.
이 검산은 같은 Codex의 별도 문맥 검토이며 한국수어 전문가의 독립 정답 평가와 다르다.
명확한 성능 개선이 없어 HTML은 만들지 않았다. 사용자·멘토 촬영 접근/학습/튜닝/후보 선택 없음.
다운로드·외부 전송·Git·카메라·운영 교체 없음. 기존 모델/운영 관련 32개 보호 핀과 397개 동결 핀 검산 완료.
실행 세션 45771은 종료 코드 0으로 완료했고 진행 중 작업은 없다.

근거 파일(이 보고서와 같은 폴더)
PLAN_KO.txt / FROZEN.json / RESULT_MANIFEST.json / EVALUATION.json
review/RESULT_VERIFICATION.json / review/POSTEXECUTION_REVIEW.json / review/DIAGNOSIS_REVIEW.json
review/MISSING_COST_DIAGNOSIS.json / review/SHORT_SHAPE_DIAGNOSIS.json
"""
(U / "RESULT_KO.txt").write_text(report, encoding="utf8")
status = "COMPLETE_JOINT_STAGE_ALIGNMENT_NOT_ADOPTED"
next_action = "Separate missingness accounting from alignment reward; retain both query-side and reference-side short-change evidence. One contract-first comparison; no existing result rerun or ungrounded training."
stable_files = ["PLAN_KO.txt", "AUTHORIZATION.json", "FROZEN.json", "TASKS.json", "SELFTEST.json", "RESULT_MANIFEST.json", "EVALUATION.json", "RESULT_KO.txt", "review/RESULT_VERIFICATION.json", "review/POSTEXECUTION_REVIEW.json", "review/FINAL_METHOD_REVIEW.json", "review/DIAGNOSIS_REVIEW.json", "review/MISSING_COST_DIAGNOSIS.json", "review/SHORT_SHAPE_DIAGNOSIS.json"]
completion = dict(status=status, at_kst=now, executed_results=534, new_training=0, new_model_inference=0,
                  adopted=False, html_created=False, active=None, process_exit_code=0,
                  development_stage_match=dict(baseline=baseline, candidate_mean=mean, candidate_scores=scores, delta_pp=mean-baseline),
                  independent_result_checks=verification["checks"], private_user_mentor_used=False,
                  production_changed=False, next_action=next_action,
                  artifacts={str(U / f): sha(U / f) for f in stable_files})
write(U / "COMPLETION.json", completion)
write(U / "CURRENT_EXECUTION.json", dict(status=status, active=None, at_kst=now, completion=str(U / "COMPLETION.json"), completed_session=45771, exit_code=0))

statepath = U.parent / "STATE.json"
state = read(statepath)
state.update(updated_date_kst=now, current_stage=status, next_single_action=next_action)
state[U.name] = dict(status=status, at=now, unit=str(U), result=str(U / "RESULT_KO.txt"), completion=str(U / "COMPLETION.json"),
                    active=None, new_training=0, new_model_inference=0, production_changed=False, user_mentor_used=False,
                    clear_improvement=False, development_stage_match=completion["development_stage_match"],
                    results=534, verification=str(U / "review/RESULT_VERIFICATION.json"), next=next_action)
write(statepath, state)

ctx = R / "WordsInSilence_PROJECT_CONTEXT.md"
s = ctx.read_text(encoding="utf-8-sig")
entries = [
    ("## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인", """### 2026-11-01 세 정보군 동시 단계정렬 비교 완료 — 미채택, 두 계산 원인 확인

`word100_joint_stage_alignment_20261101` 완료. 공개 4단어/16영상/36쌍, 142개 기술 대조를 저장 3seed 출력과 연결해 534경로를 한 번 계산했다. 새 학습/모델 추론 0. 기존 global/core_first/relational 저장 결과 대조. DEV 같은 관측단계 대응률 기존 global 82.16% → 새 평균 80.23%(79.86~80.46), −1.93%p로 동결 개선 기준 FAIL·미채택. 조상 +5.67%p, 잘하다 −8.33%p(각 global 대비); 단어 인식률·교육 정확도 아님. 전체 프레임/참고 4380·확실1080 분모 유지.

저장 경로 원인 분석: 오른손 미관측 대조 16개 모두 미관측 벌점 감소가 실제 공통관측 차이 증가를 상쇄해 오정렬(평균 약2.89프레임). 짧은 오른손 각도 변화16중14는 변화가 일부 참고 시점에 집중돼 참고 평균에서 희석됨; 프레임 삭제/구간 밖 이동은 아님. 같은 모델 출력에서 정렬 결함이 직접 확인돼 추가학습 근거 없음. 자체174/별도저장검산16001 오류0·원인별도검토 완료. C/T3F·운영·사용자/멘토·원본 보존, HTML없음. RESULT_KO.txt/COMPLETION.json 참조. 모든 실행 종료, 재실행 금지.

"""),
    ("## 9. 다음 단일 작업", """### 2026-11-01 다음 단일 작업 — 미관측 비용과 짧은 차이 집계의 분리

완료한 joint_stage_alignment의 점수/학습/정렬을 재실행하지 않는다. 다음 후보는 미관측/coverage 기록을 유지하면서 그 벌점 재배치만으로 경로가 유리해지지 않는 계산 계약부터 정한다. 짧은 변화는 참고 시점과 실제 비교 영상 시점 양쪽 근거를 남겨 집중/희석을 감춘 성공으로 만들지 않는다. 같은 영상의 가림·짧은 변화 대조에서 먼저 성질을 검증하고 한 후보만 동결해 정상 속도·손내림·교차와 기존 공개 동일 분모로 비교한다. 후속 수정은 아직 미실행. 이번 공개 자료는 이미 사용한 개발 자료이므로 새 평가라고 부르지 않는다. 교육 정오 주석 없이 정렬/미관측 오류를 학습으로 덮지 않는다. 충분한 올바른 대응·양음성 근거가 있는 특징분류 실패가 확인될 때만 한정 학습. 단어 예외/새 모델 순회/개인 튜닝 금지. 목표·변천사 Docs/PROJECT_DIRECTION_HISTORY_KO.txt 유지.

"""),
]
for heading, entry in entries:
    assert heading in s
    s = s.replace(heading, heading + "\n\n" + entry.rstrip(), 1)
ctx.write_text(s, encoding="utf8")

handoff = f"""# WordsInSilence 최신 인수인계

현재 목표: 지원하는 목표 표현의 여러 공개 참고와 비교해 어느 손·모양·관계·움직임이 어떤 순서로 달랐는지 설명한다. 단어 인식, 시간 정렬, 교육적 교정 성과를 구분한다.

## 마지막 완료
`{U.name}`: 사용자 승인 고정 결합정렬 후보 비교 완료·미채택. 4단어(순종·잘하다·조상·이집트)16영상36쌍/FIT24·DEV12와142기술대조, 저장seed42/43/44=534경로. 새 학습/모델 추론0. DEV 대응률 global82.16 → 평균80.23%(79.86~80.46),−1.93%p. 조상+5.67/잘하다−8.33%p(global대비). 단어 인식률/교육 정확도 아님. 참고전체4380/확실1080분모 유지. 초기 계획 기준FAIL, 점수후 조건변경/후보선택X.

## 확인한 원인과 다음 한 작업
미관측 대조16개에서 벌점 감소가 실제관측 차이 증가를 상쇄해 시점 이동(평균2.89프레임)을 보상했다. 짧은 오른손각도 대조16중14에서 변화가 일부 참고시점에 몰려 참고평균차이가 희석됐다. 삭제/구간외이동은 아님. left4개는 관측변화0의 no-op이므로 탐지성공X. 저장경로 산술분해이며 새DP/모델추론X.
다음은 미관측 비용을 경로선택의 이득에서 분리하는 계약 + 양쪽시간축 짧은차이 집계다. 작은 가림/짧은변화 반례부터 성질검증, 한후보 동결후 기존자료 공통분모 대조. 다음후보 아직 구현/실행X. 현재 학습으로 해결할 근거없음; PHYSICAL은 학습정답아니고 작은묶음 양음성부족. 같은시점 특징오류/정답지원 확인때만 한정학습. 특정단어 예외/새모델순회/미학습단어를 제품선결관문으로강제X.

## 근거와 실행
결과 `{U / 'RESULT_KO.txt'}`
완료 `COMPLETION.json`; 평가 `EVALUATION.json`; 별도검산 `review/RESULT_VERIFICATION.json` 16001오류0; 방법/원인 `review/POSTEXECUTION_REVIEW.json`·`DIAGNOSIS_REVIEW.json`. 자체174통과. 397동결핀/32보호핀 보존. 같은Codex별도검산≠독립수어전문가정답. 과거결과/현재완료경로 재실행금지.
exec45771 exit0, 현재진행중작업없음, CURRENT_EXECUTION active null. ROOT 문서갱신 완료. HTML생략.

## 보존
기존 C·T3-F/운영 웹·DTW·공개원본·이전동결결과 보존. 사용자/멘토촬영 접근·학습·튜닝·선택0, 카메라·외부전송·다운로드·Git·운영교체0. 전체 목표와 변천사는 `Docs/PROJECT_DIRECTION_HISTORY_KO.txt`.
"""
(R / "NEW_CHAT_HANDOFF.md").write_text(handoff, encoding="utf8")
history = R / "Docs/PROJECT_DIRECTION_HISTORY_KO.txt"
history_text = history.read_text(encoding="utf-8-sig")
history_text += f"""

2026-11-01 손모양·관계·움직임 결합정렬: 성능 향상 없이 원인 구체화
같은 손모양이지만 다른 동작 단계가 연결되는 문제를 줄이려고, 모양·관계·움직임 세 묶음을 공통 계산에 함께 넣었다. 공개4단어16영상36쌍과기술대조142개, 저장3seed출력으로534경로계산/검산 완료. DEV 같은 관측단계 대응률82.16→평균80.23%(−1.93%p), 조상은+5.67%p지만잘하다는−8.33%p여서미채택. 새학습0,운영미적용,HTML없음.
근본목표는 그대로다. 추가정보를 섞으면 자동으로 좋아지지 않는 이유를 구체적으로 확인했다: 미관측벌점을 피하는 시점이 실제동작대응보다 계산상유리해질 수 있고, 짧은변화가 일부참고시점에 몰려 평균차이에서 희석될 수 있었다. 프레임전체보존만으로충분하지않다. 다음우선순위는새모델학습보다 이두계산문제를분리해검증하는것. 실제사용자 교육오류 개선이나 모든수어 보편성은아직입증하지않았다. 별도수치검산16001오류0은구현/산술검증이며독립수어정답과다르다.
현재 최신 단위: {U}
근거: RESULT_KO.txt / EVALUATION.json / review/RESULT_VERIFICATION.json / review/POSTEXECUTION_REVIEW.json
"""
history.write_text(history_text, encoding="utf8")

doc_paths = [ctx, statepath, R / "NEW_CHAT_HANDOFF.md", history]
assert read(statepath)["current_stage"] == status
assert read(U / "CURRENT_EXECUTION.json")["active"] is None
assert read(U / "STATUS.json")["done"] == 534
assert len(read(U / "RESULT_MANIFEST.json")["results"]) == 534
for p, digest in completion["artifacts"].items():
    assert sha(Path(p)) == digest, p
write(U / "FINAL_DOCUMENT_VERIFICATION.json", dict(status="PASS", at_kst=now, report_metrics_match_evaluation=True,
    completed_artifact_hashes_verified=len(completion["artifacts"]), documents={str(p): sha(p) for p in doc_paths},
    completion_sha256=sha(U / "COMPLETION.json"), active=None, new_computation="documentation only"))
print(json.dumps(dict(status=status, mean=mean, difference_pp=mean-baseline, report=str(U / "RESULT_KO.txt")), ensure_ascii=False))
