"""Close this experiment from verified saved results; no inference/training."""
from pathlib import Path
import json, hashlib
from datetime import datetime, timezone, timedelta

U = Path(__file__).resolve().parent
O = U.parent
ROOT = O.parents[2]

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def write(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

r = read(U / 'RESULT.json')
v = read(U / 'VERIFICATION.json')
assert v['passed'] and v['result_sha256'] == sha(U / 'RESULT.json')
assert not r['changed_clip_displays']
assert r['new_components']['42']['normal']['True']['matched'] == 17
stamp = datetime.now(timezone(timedelta(hours=9))).isoformat()

report = '''실제 동작의 시작·방향 전환·끝을 가르친 결과

결론: 동작 구간을 찾는 성능은 개선됐습니다. 최종 교정 피드백은 아직 이전과 같습니다.
새 연구 모델과 정답 자료를 보존했으며, 기존 웹 기본 모델은 교체하지 않았습니다.

1. 무엇을 바꿨나
이전에는 이동 구간을 일정 비율로 시작·진행·끝으로 나눈 약한 정답이 있었습니다.
이번에는 공개 영상의 실제 모습을 보고, 움직이기 시작하는 순간·방향을 바꾸는 순간·끝나는 순간을 확인했습니다.
한 장면으로 확정하기 힘든 경계는 몇 프레임의 범위로 표시했습니다. 화면에서 이미 움직이고 있었다면 시작 순간을 지어내지 않았습니다.
시작과 끝을 각각 예측하도록 신경망의 보조 판정부를 학습했습니다. 한 동작의 끝이 다음 동작의 시작이 될 수 있게 했습니다.
기존 손모양·방향 판정부는 그대로 두었습니다. 특정 단어 번호에 따라 답을 바꾸는 예외는 넣지 않았습니다.

2. 어떤 자료로 학습했나
공통 학습: 공개 100단어, 899영상, 기록 수행자 ID 01~09. 세 번의 고정 학습에서 매 학습 회차마다 899영상을 모두 사용했습니다.
새 세부 정답: 10표현의 공개 학습 영상 18개에서 이동 29구간과 유지 17구간을 표시했습니다.
이는 100단어 전체에 세밀한 시간 정답이 모두 생겼다는 뜻은 아닙니다.
실제 RGB 검토는 같은 Codex가 모델 예측을 보지 않고 2프레임 간격의 원본 화면을 보고 수행했습니다. 독립된 한국수어 전문가의 정답 검수는 아닙니다.
6개의 매우 짧은 내부 구간은 기존 좌표 방향 계산에 필요한 4프레임보다 짧았습니다. 이를 틀린 정답으로 버리거나 모델 답에 맞추지 않았고, 이동 정답 29개는 유지했습니다.
학습은 이전 공통 신경망에서 시작해 시작·진행·끝 보조 출력을 추가 학습했습니다. 3개의 고정 난수값으로 각각 12회 학습했으며, 대표값 42는 결과 확인 전에 정했습니다.

3. 이전과 현재 비교
아래 21개는 학습 영상 수가 아닙니다. 기존 개발 비교용 공개 4단어×기록 ID 11~13의 12영상에 있는 이동 구간 수입니다.
같은 기존 구간 정답과 비교했고, 정답과 예측 구간의 겹침이 절반 이상인지 확인했습니다.

검사항목                         이전       현재(대표 학습)
이동 구간 찾기(부분 구간 포함)     14/21      19/21
시작과 끝이 연결된 구간 찾기        4/21      17/21
느리게 바꾼 영상의 시작·끝 연결      2/21      14/21
빠르게 바꾼 영상의 시작·끝 연결      7/21      17/21
손 관절 2프레임 가림의 시작·끝 연결  4/21      17/21

17/21은 양쪽 경계가 한 프레임도 안 틀렸다는 뜻이 아닙니다. 시작·끝 근거가 연결된 예측 구간이 정답 구간과 충분히 겹쳤다는 뜻입니다.
다른 두 고정 학습에서도 정상 영상의 시작·끝 연결은 각각 17/21, 19/21이었습니다. 가장 높은 학습만 골라 보고한 것이 아닙니다.

더 엄격하게 이번에 직접 확인한 7개 경계(4영상, 기록 ID 11 하나)와 비교하면:
- 구간을 찾음: 이전 1/7 → 현재 6/7.
- 시작과 끝이 모두 관측한 허용 범위 안: 이전 0/7 → 현재 3/7.
- 다른 두 학습의 양쪽 경계 일치는 각각 1/7입니다. 아직 정밀한 경계 시각이 안정적이라고 말할 수 없습니다.
- 대표 모델은 교차 동작의 중간 이동 하나를 놓쳤습니다. 교차의 모든 세부 동작을 해결했다는 뜻이 아닙니다.

4. 최종 피드백은 어떻게 됐나
원래 비교 경로에 새 보조 결과를 붙인 78조건을 세 학습에서 확인했습니다. 234개 결과의 최종 표시가 모두 이전과 같았습니다.
아래는 대표 학습 결과이며, 교차 3개는 정상 12개 안에 포함되는 부분집합입니다.

공개 시험 조건                  이전 → 현재
정상 영상 12개                  잘못된 경고 0 → 0; 비슷함 9, 판단 보류 3 유지
정상 속도 차이 24개             잘못된 경고 0 → 0; 비슷함 17, 판단 보류 7 유지
정상 교차 3개                   3개 모두 비슷함 유지
손모양 오류/방향 반전/둘 다 9개  9개 모두 차이 경고 유지
마지막 동작 삭제 6개            6개 모두 판단 보류 유지; 정상으로 통과시키지 않음
마지막 동작의 관절 가림 6개      6개 모두 판단 보류 유지
전체 손 관절 미관측 12개         12개 모두 판단 보류 유지
주요 동작 순서를 뒤집은 9개      경고 3, 잘못된 비슷함 1, 판단 보류 5 그대로

이 오류 시험은 공개 자료의 시간·관절값을 변형한 시험입니다. 새 사용자가 실제로 틀리게 수행한 영상에서 검증한 결과가 아닙니다.
관측 부족을 정상으로 넘기지 않는 것은 유지했지만, 누락을 확실하게 설명하는 능력이 개선됐다고 주장할 수는 없습니다.

5. 왜 최종 경고는 그대로인가
움직임이 언제 있었는지 찾는 것과, 그것이 설명의 어느 행동인지 알아내는 것은 별개입니다.
예를 들어 이동을 찾았어도 이것이 첫 번째 행동인지, 다음 행동인지 연결을 못 하면 순서 오류를 확정할 수 없습니다.
저장된 9개 역순 사례를 확인하니 비교할 설명 구간 자체가 없는 경우, 한쪽 행동에만 후보가 몰린 경우, 두 행동 후보가 겹친 경우가 남았습니다.
반대로 새 보조 판정이 역순을 순서 일치로 잘못 본 경우도 있었습니다. 기존 오류 경고를 지우지 못하게 해 최종 결과는 보호했습니다.
새 모델만으로 유지 구간을 판정하면 9/11 → 8/11로 떨어지는 진단 결과도 있어, 기존 유지 판단을 대체하지 않았습니다.

6. DTW와 속도
DTW 시간 정렬과 기존 비교는 유지했습니다. 이번 비교에서는 저장된 기존 DTW 대응 경로를 고정해 새 구간 판정부의 영향을 확인했습니다.
실제 시작·끝 정답은 DTW로 추정한 시점이 아니라 원본 영상 시점에 붙였습니다. 동작을 안 했는데 정렬만으로 완료했다고 만들지 않습니다.
추가 구간·순서·부분방향 계산은 중앙값 약 0.025초였습니다. 관절 검출, 기존 특징 계산, TCN, DTW, 화면 표시는 제외한 값입니다. 웹 전체 처리시간이 0.025초라는 뜻은 아닙니다.

7. 검증과 한계
학습 전 1,866검사, 경계 연결 규칙 4검사, 저장 결과 별도 산술검사 4,640검사를 통과했습니다.
작성·실행·검산 담당은 같은 Codex입니다. 별도 검사 코드로 확인했지만 독립된 사람의 검증과는 다릅니다.
기존 C·T3-F, 현재 TCN, 손모양·방향 근거, DTW 경로와 프레임 목록, 웹 기본 파일을 보존했습니다. 사용자·멘토 촬영은 접근하거나 학습·튜닝에 사용하지 않았습니다.
평가는 반복 사용된 과거 개발 자료입니다. 새 사람에 대한 독립 성능 확인으로 해석하면 안 됩니다.
정답·출력층·구간 연결 방법을 함께 바꿨으므로, 개선 중 어느 한 변경만의 기여인지는 분리 측정하지 않았습니다.
첫 자료 준비에서 개발 영상의 구간 정보가 없는 오류가 있었고, 최초 실패와 코드를 보존한 뒤 공개 개발 자료의 기존 구간 정보로 복구했습니다. 오류 상태로 학습하지 않았습니다.

8. 다음 한 작업
이번에 찾은 실제 이동 구간을 설명의 행동 특징(어느 손·어떤 모양·어느 방향·어떤 신체 관계)과 연결하는 공통 판정을 보완합니다.
학습 자료에서 연결 정답을 보강하고, 기존 비교창이 없거나 후보가 서로 겹칠 때도 관측된 동작이 비교에서 사라지지 않게 해야 합니다.
단순히 순서대로 끼워 맞추지 않고, 틀린 방향·누락·역순 근거가 남는지 함께 검증합니다. 불확실한 것은 완료나 정상으로 바꾸지 않습니다.
새 평가 자료의 분리 가능성을 먼저 확인하고, 최종 오류 경고가 개선되면서 정상·속도·교차가 유지될 때 웹 반영 여부를 판단합니다.

근거: RESULT.json, VERIFICATION.json, OBSERVED_BOUNDARIES.json, FROZEN.json,
TRAINING_COMPLETE.json, EVALUATION_COMPLETE.json, FULLPATH_COMPLETE.json.
이번 단위는 학습·비교·검산·기록 완료이며, 연구용 보완을 웹 기본에 적용한 상태는 아닙니다.
'''
(U / 'RESULT_KO.txt').write_text(report, encoding='utf-8')
write(U / 'CAUSE_AUDIT.json', {
    'source_result_sha256': sha(U / 'RESULT.json'),
    'source': 'saved automatic outputs; no new inference',
    'rows': r['order_linkage_audit'],
    'conclusion': 'Improved boundary detection has not resolved guide-stage identity, missing query windows, overlapping candidates, or false supported order. Old warnings remain protected.',
    'current_unit_fix': 'Start/end branches, reviewed ranges and shared-turn decoder completed; unresolved semantic linkage is the next distinct unit, not silently retuned on DEV.'})
# The next action is prospective, not an assertion of completed work.
next_text = '실제 관측 사건과 설명 행동의 연결을 공통 특징으로 보강한다. 먼저 공개FIT 연결 정답·새 평가 분리 가능성을 감사하고, 누락된 비교창/겹친 후보/역순에도 근거를 보존하는 조건을 동결한다. 기존 모양·방향·DTW·100단어공통은 유지하며 역사DEV 튜닝·단어예외 금지. 완료 학습·평가 재실행 금지.'
write(U / 'NEXT_PLAN.json', {
    'status': 'PROPOSED_NEXT_UNIT_NOT_STARTED', 'next': next_text,
    'steps': [
        'Audit FIT event-to-guide links and availability of separate public evaluation records; preserve actual boundary ranges.',
        'Supervise common side/shape/relative-location/direction and guide event identity jointly; retain unmatched/contradicting events.',
        'Freeze missing-window and overlapping-candidate handling without expected-order forcing; preserve DTW and old evidence.',
        'Compare saved baseline against normal/speed/crossing/shape/direction/omission/order conditions; adopt only with downstream gain and no regression.'
    ], 'not_done': ['new semantic training', 'new independent evaluation', 'default web replacement'],
    'forbidden': ['user/mentor training,tuning,selection', 'word-specific overrides', 'test score threshold tuning', 're-running completed baseline inference']})
summary = '공개 10표현18FIT의 실제 RGB 경계 이동29·유지17을 검토하고100단어899공통을 유지해 보조 신경망3seed×12회 학습 완료. 대표 이동21개 중 검출14→19, 시작·끝 연결4→17. 별도 세밀경계7개(4영상/기록ID11)는 검출1→6·양경계일치0→3으로 아직불안정. 78자동조건×3seed 최종표시234개 모두동일: 정상·속도·교차·기존손모양/방향경고 유지, 역순9의3경고/1잘못된비슷함/5보류 잔존. 새유지진단9→8/11로기존판정부미교체. 1866사전·4규칙·4640저장검산PASS(같은Codex). 역사DEV·독립전문가정답아님. 연구구간검출개선이며웹/C/T3F/기존모양방향DTW 보존·개인자료0.'
title = '### 2026-11-10 실제 관측 경계 학습·구간 검출 개선 검증 완료'
context_path = ROOT / 'WordsInSilence_PROJECT_CONTEXT.md'
context = context_path.read_text(encoding='utf-8-sig')
oldtitle = '### 2026-11-10 실제 동작 경계 정답 보강 진행'
assert context.count(oldtitle) == 2
for section, text in [('## 8.', summary), ('## 9.', next_text)]:
    s = context.index(oldtitle, context.index(section))
    e = context.index('\n### ', s + len(oldtitle))
    context = context[:s] + title + '\n\n' + text + '\n근거 ' + str(U / 'RESULT_KO.txt') + '\n\n' + context[e:]
context_path.write_text(context, encoding='utf-8')
state_path = O / 'STATE.json'
state = read(state_path)
state.update(current_stage=U.name, next_single_action=next_text, last_updated=stamp)
state[U.name] = dict(status='COMPLETE_BOUNDARY_GAIN_NO_DEFAULT_ADOPTION', summary=summary, next=next_text,
                    report=str(U / 'RESULT_KO.txt'), result=str(U / 'RESULT.json'),
                    verification=str(U / 'VERIFICATION.json'), execution=str(U / 'CURRENT_EXECUTION.json'),
                    private_access=False, default_change=False)
write(state_path, state)
(ROOT / 'NEW_CHAT_HANDOFF.md').write_text(
    '# WordsInSilence 최신 인수인계\n\n' + summary + '\n\n다음 한 작업: ' + next_text +
    '\n\n현재 단위: ' + str(U) + '\n실행: 학습·평가·저장검산 종료. 진행 중인 현재 단위 프로세스 없음. 웹 서버 새 실행/교체 없음.\n' +
    '주요 근거: RESULT_KO.txt, RESULT.json, VERIFICATION.json, CAUSE_AUDIT.json, NEXT_PLAN.json.\n', encoding='utf-8')
history = ROOT / 'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
with history.open('a', encoding='utf-8') as f:
    f.write('\n\n' + title + '\n' + summary + '\n' +
            '변경 이유: 시간비율로 나눈 약한 단계 정답을 실제 RGB에서 관측한 경계 범위로 보강. 한 방향 전환 시점이 앞 동작의 끝과 다음 동작의 시작을 함께 나타내도록 분리 학습.\n' +
            '성과의 범위: 구간 검출은 개선됐지만 설명 속 행동 연결·순서 판정은 해결되지 않음. DTW 정렬은 그대로 유지하고 새 부품의 영향만 비교.\n' +
            '최초 준비 오류는 SETUP_FAILURE_01.txt와 setup_review_v0.py.txt에 보존. 독립 사람/전문가 검증 없음.\n' +
            '다음: ' + next_text + '\n근거: ' + str(U / 'RESULT_KO.txt') + '\n')
write(U / 'CURRENT_EXECUTION.json', dict(status='COMPLETE_BOUNDARY_GAIN_NO_DEFAULT_ADOPTION',
      pid=None, completed_at=stamp, verification=str(U / 'VERIFICATION.json'),
      default_changed=False, ongoing_training=False, next_unit_started=False))
with (U / 'EVENTS.jsonl').open('a', encoding='utf-8') as f:
    f.write(json.dumps(dict(at=stamp, event='REPORT_AND_PROJECT_RECORDS_CLOSED',
                           result_sha256=sha(U / 'RESULT.json')), ensure_ascii=False) + '\n')
print(json.dumps(dict(report=str(U / 'RESULT_KO.txt'), result_unchanged=sha(U / 'RESULT.json') == v['result_sha256']), ensure_ascii=False))
