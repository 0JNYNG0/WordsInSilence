"""Close this fixed public comparison without re-training or re-inference."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import collections
import hashlib
import json
import shutil

U = Path(__file__).resolve().parent
O = U.parent
R = O.parents[2]
G = O / 'word100_web_completion_graph_20261111'
P = O / 'word100_terminal_relation_targets_20261114'
read = lambda p: json.loads(p.read_text('utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', 'utf-8')


assert not (U / 'RECORDS_SYNCED.json').exists(), 'Already closed; inspect existing records.'
now = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
v = read(U / 'VERIFIED.json')
e = read(U / 'EVALUATION.json')
assert v['passed'] and not v['model_admission_passed']
execution = read(U / 'EXECUTION_FROZEN.json')
assert sha(Path(execution['code'])) == execution['sha256']
for p, h in read(G / 'PROTECTIONS.json').items():
    assert sha(Path(p)) == h

# Frozen variant='source' includes explicitly constant-posture controls.
# Keep the original records intact and publish a finer, non-overlapping account.
groups = collections.defaultdict(list)
for row in e['rows']:
    kind = ('constant_posture_control' if row['freeze'] else
            'time_reverse_control' if row['reverse'] else 'actual_source_interval')
    groups[(row['split'], kind)].append(row)
counts = [dict(split=s, kind=k, conditions=len(rows), correct=sum(x['correct'] for x in rows),
               source_clips=len({x['clip_id'] for x in rows}))
          for (s, k), rows in groups.items()]
probe = [r for r in e['rows'] if r['split'] == 'exposed_probe']
assert len(probe) == 24 and sum(r['correct'] for r in probe) == 22
assert len(groups[('exposed_probe', 'actual_source_interval')]) == 16
assert len(groups[('exposed_probe', 'constant_posture_control')]) == 2
assert len(groups[('exposed_probe', 'time_reverse_control')]) == 6
save(U / 'CONDITION_ACCOUNTING.json', dict(
    counts=counts, passed=True,
    note='Frozen source scope also includes constant-posture controls; use freeze/reverse flags for this finer account. No frozen prediction or metric is overwritten.',
    independent_examples=False, real_learner_error_accuracy=None))

# Keep the project record in Korean, with task success distinct from admission.
summary = (
    '공개 RGB와 RTMW 대응9장면을 확인해 손/프레임/좌표변환 오류는 배제했다. '
    '굽은 자세와 접는 움직임을 다른 정답으로 분리하고 별의 실제 펴고 접는 FIT6구간을 추가했다. '
    '기존 끝 손모양 모델을 고정한 변화 판정부3seed 학습 완료. '
    '노출 개발자료6영상에서 파생한24조건 중22일치: 실제 접기6/6, 유지10/12(고정자세 대조2 포함), 역재생 펼치기6/6. '
    '정상 손모양 유지2조건을 변화로 오판해 미채택. 기존 관측과 충돌한 변화는 교정으로 승격하지 않는 연구 연결부를 검증했다. '
    'C·T3-F/공개100/웹 유지. G2 보완 및 G3 통합은 여전히 남음.'
)
next_ko = (
    '손모양 유지인데 접기/펴기로 오판한 두 조건을 바탕으로, 공개 FIT에서 '
    '같은 손모양을 유지하며 손바닥 방향·팔 방향만 바뀌는 경우와 실제 손가락을 굽히는 경우를 짝지어 직접 주석한다. '
    '노출probe08/09는 학습하지 않고 진단 한계를 기록한다. 입력의 투영 변화와 관절 오검출을 RGB 대조로 먼저 구분하고, '
    '구분 가능한 관측 근거가 있으면 현재 시작·끝만 보는 변화 모델과 구간 내부 흐름/손 방향을 함께 보는 한 후보를 조건 동결 후 비교한다. '
    '정상 유지·교차·짧은 정상·중간 잘림과 원래 끝 손모양을 함께 보존해야 통과이며, 임계값 조정이나 유지시간 강제는 하지 않는다. '
    '통과 부품만 설명의 올바른 행동 구간에 연결해 G2/G3를 확인한다. 관측만으로 구분 불가능하면 더 학습해 해결된다고 주장하지 않는다. '
    '이번3학습/직전9학습과 완료 추론은 재실행하지 않는다. Final8과 웹 승격은 전체 통과 전 보류한다.'
)
report = f'''손모양 변화와 자세 유지 — 공개 자료 보완 결과
작성: {now}

이번 결론
‘손가락이 굽어 있다’와 ‘손을 접고 있다’를 따로 가르치는 실험을 완료했다.
접는 변화는 찾았지만, 손모양을 그대로 둔 정상 장면 2조건을 변화로 잘못 판단했다.
따라서 새 모델은 웹에 적용하지 않았다. 짧아도 끝까지 한 동작과 중간에 멈춘 동작을 안정적으로 구분하는 전체 기능은 아직 미완료다.

이전과 무엇이 달라졌나
이전: 한 장에서 손가락이 굽어 보이면 ‘접는 중간 상태’인지 판단하려 했다.
현재: ① 지금 손모양이 무엇인지 ② 앞뒤로 실제 손모양이 변했는지 ③ 설명에서 요구하는 끝 상태까지 했는지를 분리한다.
예: 팔을 쓸어내리며 손가락을 약간 굽힌 채 유지해도, 그것만으로 ‘주먹을 쥐다 말았다’고 해석하지 않는다.
현재 모델이 만든 것은 ②의 제한된 관측 판정부다. ③의 수어 완료/오답을 대신하는 모델이 아니다.

실제로 한 작업
1. 이전 오판·회귀의 공개 원본 9장면을 RTMW 점과 겹쳐 확인했다. 손의 좌우, 프레임, 정규화 좌표를 원본으로 되돌리는 계산은 맞았다.
   이는 관절점의 실제 위치가 전부 정확하다는 뜻이 아니다. 주먹/손날처럼 겹쳐 보이는 관절은 RGB만으로 정확한 3D 위치를 확정할 수 없다.
2. 다른 표현 ‘별’의 공개 FIT 영상 2개에서 펴기·접기·유지 6구간을 추가 확인했다. 설명만 보고 정답을 추측하지 않았다.
3. 기존 끝 손모양 모델과 100단어 공통 모델은 그대로 두고, 관절의 상대적 변화로 펴기/유지/접기를 구분하는 작은 신경망을 3개 초기값으로 학습했다.
   입력은 손 크기로 정규화한 관절 각도·길이 비율이다. 단어 이름/촬영 ID/동작 시간은 입력하지 않았다.
   이것은 707개 가중치의 별도 관측 모델이며, 전체 TCN+GRU를 다시 학습하거나 교체한 작업은 아니다.
4. 원래 손모양 관측과 새 변화 판정이 충돌하면 불확실한 근거로 남기는 연구 연결부를 만들었다.
   기존 결과와 구간 내부의 펴기→접기→펴기 기록도 지우지 않는다. 이 연결부는 웹에서 동작하는 교정 기능으로 승격하지 않았다.

자료와 결과
전체 공개 원본 26영상에서 만든 100조건: FIT 20영상·76조건 / 노출 개발비교 6영상·24조건.
FIT는 기록 ID02~07, 개발비교는08~09다. 기록 ID를 외형으로 사람 신원이라고 단정하지 않았다.
개발비교24조건의 구성: 원본 실제 구간16 + 역재생 대조6 + 중간 굽은 자세를 고정한 대조2.
같은 영상에서 여러 조건을 만든 것이므로 독립 영상24개나 사용자 오류24개가 아니다.

비교 항목                            결과
실제로 접는 변화                     6조건 중6 일치
같은 손모양 유지                     12조건 중10 일치 (고정자세 대조2 포함)
접기를 거꾸로 재생한 펼치기 대조       6조건 중6 일치

남은 오판
- 순종(WORD0739)/기록08 오른손: 주먹을 쥔 채 팔을 올리는데, 손을 더 접는다고 판단.
- 잘하다(WORD1133)/기록08 왼손: 주먹을 유지하는데, 손을 편다고 판단.
원본에서 손모양 유지가 보이는데 관측값이 변화처럼 나타나는 문제다. 방향·투영 변화와 검출 오차를 더 구분해야 한다.
이 두 결과는 모델의 높은 출력 확률만 믿어도 해결되지 않았다. 숫자 문턱을 낮추거나 정상에 예외를 붙이지 않았다.

이 결과의 정확한 의미
- 이번 정답은 ‘변화 방향’이고 이전 정답은 ‘순간 손모양’이어서, 이전1/4와 이번6/6을 직접 성능 향상으로 비교할 수 없다.
- 자연스러운 실제 펼치기의 추가 FIT 예시는 있지만, 개발비교의 펼치기6조건은 역재생이다. 실제 펼치기 일반화 검증이 더 필요하다.
- 새로운 입력은 주로 구간의 처음·끝 차이를 사용한다. 사이의 반복·잠깐 바뀜을 이 모델 하나가 이해한다고 주장할 수 없다. 별도 기존 시점별 관측은 보존했다.
- 비교할 구간을 확인한 뒤 시험했다. 새 영상에서 구간을 스스로 찾고 설명과 연결하는 성능은 이 시험으로 확인하지 않았다.
- 부분 접기와 완전 접기는 모두 ‘접는 변화’일 수 있다. 이 결과만으로 끝까지 수행했다고 말하지 않는다.
- 기존보다 실제 피드백 정확도나 처리시간이 개선됐다는 결론은 아직 없다.

검증과 보존
저장된 확률을 별도 코드로 재계산하고 자료 분리·해시·불확실성 처리 {v['total']}검사를 통과했다. 이는 프로그램/기록 검사 수이며 정답률이 아니다.
기존246개 저장 결과의 원래 필드가 새 연구 필드 때문에 바뀌지 않는지 확인했다. 246개를 새 모델로 재추론한 성능 검증은 아니다.
학습/구현과 검산 모두 Codex 로컬 수행이다. 별도 스크립트 검산을 독립 사람·수어 전문가의 정답 검수라고 부르지 않는다.
기존 C·T3-F/공개100/끝 손모양 모델/웹 및 보호 대상477경로를 보존했다. 사용자·멘토 촬영은 접근·학습·튜닝·선택에 쓰지 않았다.
Final8은 열지 않았다. 새 다운로드·외부 전송·카메라·Git·서버 변경이 없고 새 학습3회는 종료했다.
EVALUATION의 source 분류에 고정자세 대조2개도 포함된 점은 CONDITION_ACCOUNTING에서 별도 구분했다. 동결 결과는 고치지 않았다.

다음 한 작업
{next_ko}

전체 진행 위치
원본/좌표 대조 → 정답 구분 → 제한 학습·비교 → [정상 유지 오판 보완] → 설명 구간 연결·완료 판단 → 공개 최종평가 → 통과 기능 웹 적용
앞의 세 단계는 이번 단위에서 완료. 대괄호부터 전체 기능의 검증이 남아 있다. G2는 REWORK, G3 및 이후 전체 통과는 대기다.

주요 근거 파일
AUDIT_INPUTS.json / OBSERVATION_REVIEW.json / observation_review_1~3.jpg
CROSSWORD_LABELS.json / crossword_record02.jpg / crossword_record03.jpg
FROZEN.json / PAIR_LABELS.json / TRAINING_COMPLETE.json / EVALUATION.json
RECONCILIATION.json / VERIFIED.json / CONDITION_ACCOUNTING.json
'''
(U / 'RESULT_KO.txt').write_text(report, 'utf-8')
evidence = {str(U / n): sha(U / n) for n in [
    'AUDIT_INPUTS.json', 'OBSERVATION_REVIEW.json', 'PAIR_LABELS.json', 'CROSSWORD_LABELS.json',
    'EXECUTION_FROZEN.json', 'TRAINING_COMPLETE.json', 'EVALUATION.json', 'RECONCILIATION.json',
    'VERIFIED.json', 'CONDITION_ACCOUNTING.json', 'reconcile.py', 'verify.py']}
status = 'OBSERVATION_CHANGE_SEPARATED_TESTED_REWORK'
save(U / 'RESULTS_REVIEW.json', dict(
    status=status, at=now, summary=summary, evidence=evidence, new_training_runs=3,
    model_admission_passed=False, whole_graph_complete=False, next=next_ko,
    web_changed=False, private_data_used=False, final8_unopened=True,
    implementation='Codex local', verification='same Codex separate saved-output calculation',
    independent_human_validation=False, real_user_error_accuracy=None,
    old_endpoint_reinference=False, common100_retraining=False,
    runtime_speed_improvement_measured=False, counts=counts))
save(U / 'NEXT_PLAN.json', dict(
    basis=['Two false changes under maintained shape', 'Input only describes projected start/end differences',
           'Current experiment does not establish automatic interval association or whole completion'],
    steps=[
        dict(id='A', task='FIT RGB/keypoints: separate shape change from pose/view change with direct paired labels',
             gate='Unclear hidden joints remain unknown; no probe08/09 fit labels'),
        dict(id='B', task='Test whether available input distinguishes these paired conditions before choosing a model',
             gate='No detector-confidence-as-truth or word-specific thresholds; report non-identifiability'),
        dict(id='C', task='If supported, freeze one comparison using within-interval shape/orientation evidence',
             gate='Keep old endpoints and source intervals; preserve normal/short/speed/crossing and partial controls'),
        dict(id='D', task='Only admitted evidence connects to the corresponding description action in G2/G3',
             gate='Completion requires the action and its goal, not just a closing prediction; Final8/web later')],
    text=next_ko, no_repeat_completed_runs=True))
save(U / 'GRAPH.json', dict(nodes=[
    dict(id='A', title='원본과 관절 대응 확인', status='COMPLETED'),
    dict(id='B', title='굽은 자세와 굽히는 변화 분리·공개 FIT 보강', status='COMPLETED'),
    dict(id='C', title='고정 학습·개발 비교·검산', status='COMPLETED_ADMISSION_FAILED'),
    dict(id='D', title='방향만 바뀐 정상 유지의 오판 보완', status='NEXT'),
    dict(id='E', title='설명 연결·짧은 완료/부분 수행 통합 검증', status='PENDING'),
    dict(id='F', title='봉인평가 후 웹 적용', status='PENDING')],
    edges=[['A', 'B'], ['B', 'C'], ['C', 'D'], ['D', 'B'], ['C', 'E'], ['E', 'F']],
    whole_graph_complete=False, report=str(U / 'RESULT_KO.txt')))

# Complete the evidence first, then synchronize the current entry points.
paths = [O / 'STATE.json', G / 'TASK_GRAPH.json', R / 'WordsInSilence_PROJECT_CONTEXT.md', R / 'NEW_CHAT_HANDOFF.md']
backup = U / 'before_final_records'
backup.mkdir(exist_ok=True)
for p in paths:
    if not (backup / p.name).exists():
        shutil.copy2(p, backup / p.name)
s = read(O / 'STATE.json')
assert s['current_followup'] == P.name
s['current_followup'] = U.name
s['next_single_action'] = next_ko
s[U.name] = dict(status=status, summary=summary, next=next_ko, unit=str(U), updated_at=now,
                 training_running=False, evaluation_running=False, web_changed=False, whole_graph_complete=False,
                 report=str(U / 'RESULT_KO.txt'), graph=str(U / 'GRAPH.json'))
save(O / 'STATE.json', s)
g = read(G / 'TASK_GRAPH.json')
g2 = next(n for n in g['nodes'] if n['id'] == 'G2')
g2['status'] = 'REWORK'
g2['latest_followup'] = str(U / 'RESULT_KO.txt')
g2.setdefault('attempts', []).append(dict(at=now, status='REWORK', evidence=[dict(
    path=str(U / 'RESULTS_REVIEW.json'), sha256=sha(U / 'RESULTS_REVIEW.json'))],
    note='Static shape and physical change targets separated; 3 fixed runs completed. Two maintained-shape probe conditions misclassified as change. No web promotion; endpoint predictions preserved.'))
g3 = next(n for n in g['nodes'] if n['id'] == 'G3')
g3['shape_change_followup'] = dict(status='DIAGNOSTIC_ONLY_REWORK', path=str(U / 'RESULTS_REVIEW.json'))
g['updated'] = now
g['next_task'] = next_ko
save(G / 'TASK_GRAPH.json', g)
p = R / 'WordsInSilence_PROJECT_CONTEXT.md'
text = p.read_text('utf-8-sig')
title = '### 2026-11-14 손모양 유지·변화 분리 학습 및 원본 관측 감사 완료 / 정상 유지 오판 보완 필요'
assert title not in text
for header, body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인', summary), ('## 9. 다음 단일 작업', next_ko)]:
    marker = header + '\n\n'
    assert marker in text
    text = text.replace(marker, marker + title + '\n\n' + body + '\n근거: `research/front_keyphase_v1/ops/' +
                        U.name + '/RESULT_KO.txt`, `RESULTS_REVIEW.json`.\n\n', 1)
p.write_text(text, 'utf-8')
(R / 'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — 손모양 변화/유지 비교 완료 · G2/G3 미완료
갱신 {now}

{summary}

현재 단위: {U}
실행 중 학습/평가 없음. 서버·카메라 변경 없음. 현재 전체 그래프: G0/G1 PASS, G2 REWORK, G3 이후 전체 통과 대기.

완료 결과 재실행 금지
- 이번3seed/800step 작은40→16→3 변화판정부 종료. 26공개영상(20 FIT/6 노출probe), 100파생조건(76/24). 기존100 TCN+GRU를 재학습한 것이 아니다.
- 노출24=실제구간16+역재생6+고정굽은자세2; 22일치. EVALUATION의source는 고정자세를 포함하므로 CONDITION_ACCOUNTING을 같이 읽는다.
- 접기6/6, 유지10/12, 역재생펴기6/6. 유지오판:0739/08/R74~84 closing,1133/08/L70~92 opening. 모델 출력확률이 높아도 정상 오판. 학습 대상에 probe를 넣지 않는다.
- 기존 손모양 결과는 그대로. RECONCILIATION은 연구 진단이고 모든 교육적오답/전체완료는 null, 웹 미적용. 알려진 구간 처음·끝 차이 시험이지 자율 행동 이해/완료 성능 아님.
- VERIFIED {v['total']} 수치/계약검사 통과. 같은Codex 별도 검산이며 독립 사람/교육자 정답이 아니다. 기존246필드 보존은 인터페이스 검사이지 새모델246추론 아님.
- 직전 {P.name}의9학습/첫JSON저장실패·준비실패 및 복구기록 보존. 새 모델 목표가 달라 직전1/4와 이번6/6을 성능개선으로 직접 비교하지 않는다.
- goal_binding_integration의 후보 전체구간·관측/소유충돌 보호도 유지. Final8 미개봉. 새 HTML 생략.

다음 한 작업
{next_ko}

현재 목표: 짧아도 충분히 수행한 동작 / 도중에 멈춘 동작 / 관측 부족을 구분해 해당 설명 구간의 교정으로 연결.
이번에는 손모양 관측의 혼동을 분리했다. 실제 사용자 미완료 판정과 전체 웹 완성은 주장하지 않는다.
C·T3-F/공개100/운영웹/원본 읽기전용. 사용자·멘토 자료 접근·학습·튜닝·선택 금지. 외부전송·다운로드·Git 없음.
''', 'utf-8')
save(U / 'CURRENT_EXECUTION.json', dict(status='COMPLETED_FIXED_COMPARISON_REWORK_REMAINS', at=now,
    training_running=False, evaluation_running=False, process_ids=[], web_changed=False,
    model_admission_passed=False, next=next_ko, no_background_job_claim=True))

for p, h in read(G / 'PROTECTIONS.json').items():
    assert sha(Path(p)) == h
assert read(O / 'STATE.json')['current_followup'] == U.name
assert title in (R / 'WordsInSilence_PROJECT_CONTEXT.md').read_text('utf-8')
assert str(U) in (R / 'NEW_CHAT_HANDOFF.md').read_text('utf-8')
for p, h in evidence.items():
    assert sha(Path(p)) == h
save(U / 'RECORDS_SYNCED.json', dict(passed=True, at=now, all_protected_unchanged=True,
    files={str(p): sha(p) for p in paths + [U / 'CURRENT_EXECUTION.json', U / 'RESULT_KO.txt']},
    verification='Record synchronization only; no repeated inference/training.'))
print(json.dumps(dict(records_synced=True, new_runs=3, model_admitted=False, web_changed=False,
                      probe_conditions=24, probe_correct=22, report=str(U / 'RESULT_KO.txt')), ensure_ascii=False))
