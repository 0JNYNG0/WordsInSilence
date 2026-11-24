"""Record the user's guide-grounded feedback goal; no data/model execution."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json

R = Path(__file__).resolve().parents[4]
U = Path(__file__).resolve().parent
O = U.parent
PLAN = R / 'Docs/FEEDBACK_GOAL_AND_PLAN_KO.txt'
RECORD = U / 'GUIDE_DTW_CONTRACT_RECORD.json'
assert PLAN.is_file() and not RECORD.exists(), 'Do not overwrite a recorded revision.'
now = datetime.now(timezone(timedelta(hours=9))).isoformat()
status = 'GUIDE_GROUNDED_LEARNED_DTW_PLAN_DEFINED_NOT_TRAINED'
summary = (
    '사용자가 설명 문구와 실제 동작 구간을 함께 배우고, 새 사용자+목표 단어를 학습된 기준 흐름과 DTW로 정렬해 '
    '문구·시간·기대 행동·관측 차이가 연결된 피드백을 받는 목표를 확정했다. '
    '절대 좌표보다 자기 신체 부위 관계와 정상 변이를 학습하며 9개 원영상 개별 대조를 최종 판단 경로로 삼지 않는다. '
    '현재 8804는 9참고 연구 데모이고 SignRep+7특징 판정부는 전체 문구/순서 평가 모델이 아니다. '
    '직전 DTW 준비/설명 보조 명세를 수정했다. 새 성능 결과가 아닌 목표·설계 기록 완료다.'
)
next_action = (
    'Docs/FEEDBACK_GOAL_AND_PLAN_KO.txt의 1단계: 기존 지원 4표현의 공개 다인 자료·출처 설명·시간 주석을 재사용하여 '
    '문구-구간-특징-사람-확실성 지원표와 최초 학습/개발/평가 계약을 준비한다. '
    '그 뒤 동결 SignRep 기반 전체 시간 판정부 학습→학습된 기준 흐름과 DTW→다른 사람 검증→실제 피드백/속도→범위 확대 순서다. '
    '기본 관측은 목표 단어로 덮어쓰지 않고, 기준 흐름은 9영상 단순 평균이나 런타임 9인 대조로 대체하지 않는다. '
    '사용자·멘토 자료 학습/튜닝/선택 금지, C·T3-F와 기존 데모 보존. 이번은 계획/기록이며 새 학습·추론은 없다.'
)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
old_contract = U / 'GOAL_CONTRACT_KO.txt'
previous_record = json.loads((U / 'TARGET_CONTRACT_RECORD.json').read_text(encoding='utf8'))
assert sha(old_contract) == previous_record['contract_sha256'], 'Previous contract changed.'
record = {
    'at': now, 'status': status, 'summary': summary, 'next_action': next_action,
    'plan': str(PLAN), 'plan_sha256': sha(PLAN),
    'supersedes_dtw_role_in': str(old_contract), 'previous_contract_sha256': sha(old_contract),
    'runtime_inputs': ['new_user_video', 'target_word'],
    'runtime_alignment': 'observed temporal features vs learned guide-linked reference flow; not nine raw references',
    'learning': 'transfer of frozen video features + supervised guide/feature/temporal heads; not reinforcement learning',
    'current_implementation': 'nine-reference demo, frozen SignRep, limited seven attributes; planned evaluator not implemented',
    'new_training_runs': 0, 'new_inferences': 0, 'user_mentor_inputs_read': 0,
    'downloads': 0, 'live_code_changed': False, 'c_t3f_changed': False,
    'review': 'Codex source/report/primary-document review and documentation consistency check; no new model evaluation or independent KSL labels',
    'routine_read_errors': ['PowerShell Python stdout cp949 failed on an em dash; retried with UTF-8, no input mutation.',
                            'Instruction search included a nonexistent directory; corrected to explicit existing parent paths.'],
}

context = R / 'WordsInSilence_PROJECT_CONTEXT.md'
t = context.read_text(encoding='utf-8-sig')
sections = [
    ('## 2. 프로젝트 정체성과 최종 목표', '가이드 문구·학습된 흐름·DTW 피드백 목표 재설정',
     '현재 피드백 목표와 실행 계획의 기준은 `Docs/FEEDBACK_GOAL_AND_PLAN_KO.txt`다. '
     '여러 사람의 공개 영상과 해당 표현의 설명 문구를 함께 배워, 사용자 영상+목표 단어를 학습된 기준 흐름과 DTW로 맞추고 '
     '설명의 어느 문구에 해당하는 사용자 시간대에서 손모양·신체 관계·움직임·순서가 중요한 차이를 보이는지 알려준다. '
     '정상 체형/거리/속도 차이는 몸 기준 관계와 다인 변이로 다룬다. TCN 단어 인식과 수행 피드백을 구분한다. '
     '9개 원영상의 실행 시 개별 비교나 설명 글 표시만으로 이 목표를 달성했다고 부르지 않는다. '
     '이는 전체 목적을 버린 변경이 아니라 사용자가 원한 학습 방식/DTW 역할/문구 기반 출력을 구체화한 최신 명세다.'),
    ('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인', '설명 문구와 학습된 기준 흐름에 근거하는 피드백 목표 정리', summary),
    ('## 9. 다음 단일 작업', '문구-구간 지원표와 전체 시간 학습 계약 준비', next_action),
]
for marker, title, body in sections:
    assert t.count(marker) == 1, marker
    block = '\n\n### 2026-11-02 ' + title + '\n\n' + body
    assert title not in t
    t = t.replace(marker, marker + block, 1)

state_path = O / 'STATE.json'
state = json.loads(state_path.read_text(encoding='utf-8-sig'))
assert U.name in state
state['current_stage'] = status
state['updated_date_kst'] = now[:10]
state['next_single_action'] = next_action
state['current_feedback_goal_contract'] = str(PLAN)
state[U.name]['guide_dtw_contract'] = record
state[U.name]['status'] = status
state[U.name]['summary'] = summary
state[U.name]['next_action'] = next_action
state[U.name]['updated_at'] = now

handoff = (
    '# WordsInSilence 최신 인수인계\n\n' + now + '\n\n' + summary +
    '\n\n현재 목표·계획: ' + str(PLAN) + '\n\n다음 한 작업: ' + next_action +
    '\n\n이번 변경은 문서뿐이다. 사용자의 추가 SVM 질문에 맞춰 지도학습/SignRep 전이 사용/DTW 계산의 역할도 정리했다. '
    '현재 SignRep 동결·작은 특징 판정부와 향후 전체 가이드 흐름 평가 학습을 혼동하지 않는다. '
    '기존 GOAL_CONTRACT_KO.txt는 이력으로 보존하며 DTW 역할은 최신 문서가 대체한다. '
    '이번에 에이전트 위임/독립 KSL 정답 검증은 하지 않았고 자체 코드·기록·문서 일관성 점검이다.\n'
    '\n8804/8802 서버는 변경하지 않았으며 이번에는 프로세스 생존을 확인하지 않았다. '
    '과거 기록 PID31864/26000을 현재 실행 상태로 단정하지 말고 서버 작업이 필요할 때 먼저 확인한다. '
    '새 학습/추론/촬영접근/다운로드/운영교체/Git 작업 없음.\n'
)
with RECORD.open('x', encoding='utf8') as f:
    json.dump(record, f, ensure_ascii=False, indent=2)
context.write_text(t, encoding='utf8')
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf8')
(R / 'NEW_CHAT_HANDOFF.md').write_text(handoff, encoding='utf8')
with (R / 'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a', encoding='utf8') as f:
    f.write('\n\n' + now + ' — 가이드 문구와 배운 동작 흐름으로 사용자 평가 목표 구체화\n' +
            summary + '\n이전: 관측 특징/9참고 DTW/부분 설명 연결을 통합해 데모를 만들었으나 사용자는 '
            '여러 사람에게서 미리 배운 수행 지식으로 평가받기를 원했다. '
            '정정: 9참고 대조 자체를 최종 AI라고 부르지 않고 문구-구간-특징-순서 학습을 중심으로 둔다. '
            '직전 계획의 DTW 준비/설명 보조 역할도 최신 사용자 요구에 따라 실행 시 학습된 기준 흐름과의 정렬로 바꾼다. '
            '부위 상대 관계/정상 변이/문구와 사용자 시간에 근거한 안내가 목표다. '
            '새 성능 수치나 교육적 정오 입증은 없으며 현재 모델/서버는 보존했다.\n' +
            next_action + '\n근거: ' + str(PLAN) + '\n')

assert sha(old_contract) == record['previous_contract_sha256']
assert json.loads(state_path.read_text(encoding='utf8'))[U.name]['guide_dtw_contract']['new_training_runs'] == 0
print(status)
print(PLAN)
