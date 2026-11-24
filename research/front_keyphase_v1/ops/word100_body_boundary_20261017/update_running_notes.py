"""Record preparation progress; never controls input or training."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json

U = Path(__file__).resolve().parent
R = U.parents[3]
now = datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
status = read(U / 'PREP_STATUS.json')
if (U / 'training_PROGRESS.json').exists():
    raise RuntimeError('Preparation-only notes cannot overwrite training progress')
lock = read(U / 'PREP_RUNNING.json') if (U / 'PREP_RUNNING.json').exists() else None
n = len(list((U / 'data/records').glob('*.json')))
execution = read(U / 'CURRENT_EXECUTION.json') if (U / 'CURRENT_EXECUTION.json').exists() else {'exec_session': 33402, 'description': '첫 추가100개 청크'}
entry = {
    'status': status['status'], 'updated_kst': now,
    'unit': 'ops/' + U.name,
    'summary': '공통 하단경계 동결. 기존 성공 경계 보존, 초기정지 부재에는 TRAIN889개에서 정한 단일 어깨기준 비율 적용. 기존1199관측 사전검산과 실제 조기30입력 독립검산 성공. 고정100대표/기존실패10/우선5 시각표본 검토 완료. 전수 실제입력 준비 및 신규 관측 검토 대기, 학습 미시작.',
    'latest_user_direction': '계속 진행하고 입력 검증에서 문제없으면 학습과 성능 비교까지 멈추지 말고 마무리. 중요한 문제 즉시 보고, 완료 후 기존 모델 대비 개선·회귀 보고.',
    'prepared_records_at_update': n, 'requested_records': 1199,
    'early_independent_records': 30, 'early_matched_channel_values': 6189400,
    'preflight_legacy_records': 1199, 'preflight_preserved_boundaries': 1187,
    'preflight_fallback_boundaries': 12,
    'coefficient': 1.477334276053632,
    'preparation_context_sha256': '9cc72d7ecbcb97732fe9dcb6412e71dc7809023df5c780fa81517b85984f100d',
    'process': lock, 'exec_session': execution['exec_session'] if lock else None,
    'training': 0, 'candidate_comparisons': 0, 'production_changes': 0,
    'next_single_action': '전체1199 실제 입력 준비 후 verify_full_inputs.py 독립검산·actual_input_triage.py 신규관측 검토. 완료된 표본 시각검토 근거 재사용. 문제없으면 READY 근거 연결→A/B/C각3seed 비교학습→저장점수·학습추적 독립검산까지 이어간다. 입력오류/필수동작잘림 즉시보고.',
    'evidence': ['AUTHORIZATION.json', 'FROZEN_PROTOCOL.json', 'PREP_CONTEXT.json',
                 'LEGACY_PREFLIGHT.json', 'BOUNDARY_RECORDS_INDEPENDENT.json',
                 'evidence_VISUAL_REVIEW.json', 'PARALLEL_EXECUTION_CONTEXT.json'],
    'limitations': '조기30개 검산과 표본 장면 검토는 전수 실제입력/언어학적 의미보존 또는 새사용자 인식개선 증명이 아님.'
}
state_path = U.parent / 'STATE.json'
state = read(state_path)
state.update(updated_date_kst=now, current_stage='WORD100_BODY_BOUNDARY_INPUT_PREPARATION',
             next_single_action=entry['next_single_action'])
state[U.name] = entry
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

p = R / 'WordsInSilence_PROJECT_CONTEXT.md'
s = p.read_text(encoding='utf-8-sig')
heading = '### 2026-10-17 초기정지 부재 공통경계 검증·전수입력 준비 진행'
paragraph = (f'{heading}\n갱신 {now}. `ops/{U.name}`의 경계를 동결했다. 기존 성공1187개 경계는 보존하고 실패12개에는 TRAIN889개에서 추정한 어깨기준 단일비율1.477334276 적용. 기존1199관측 전수사전점검·독립계산 성공. 실제 조기30입력(기존20+이전실패TRAIN10) 6,189,400채널값 일치·영역밖134,599점 교란불변·5151보호핀 불변. 현재 실제입력 {n}/1199 저장, 2프로세스 준비 중. 고정100TRAIN·이전실패10·우선5 시각표본 검토 완료, 신규 실제입력 변화분은 전수 준비 후 검토한다. 아직 TCN학습0, 성능개선 판정없음. 사용자 최신 재확인에 따라 입력검증 통과 후 학습·비교까지 연속 진행하며 중요문제만 즉시보고한다.\n\n')
marker = '## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'
if heading in s:
    start = s.index(heading); end = s.index('\n### ', start + len(heading))
    s = s[:start] + paragraph.rstrip() + '\n' + s[end:]
else:
    s = s.replace(marker, marker + paragraph, 1)
next_heading = '### 2026-10-17 다음 — 동결 입력 전수준비·보존검토 후 9회 비교학습'
next_p = f'{next_heading}\n`ops/{U.name}/PREP_RUNNING.json`·PREP_STATUS·CURRENT_EXECUTION 및 실행세션부터 확인해 중복실행을 막는다. 별도환경에서 PID가 안보이는 것만으로 잠금해제하지 않는다. 원본/조건을 유지한1199입력 준비→verify_full_inputs 독립검산·actual_input_triage 신규관측 검토→기존 시각표본근거를 연결한 READY→A/B/C3seed 학습·M0비교→저장점수/학습추적 독립검산을 이어간다. 사용자 최신 재확인으로 이 범위의 추가승인 불필요. TRAIN실패 자료 조용한 제외·특정단어 예외·DEV점수 기반 경계수정 금지. 사용자자료/보류촬영자/운영모델 교체 없음.\n\n'
if next_heading in s:
    start = s.index(next_heading); end = s.index('\n### ', start + len(next_heading))
    s = s[:start] + next_p.rstrip() + '\n' + s[end:]
else:
    s = s.replace('## 9. 다음 단일 작업\n\n', '## 9. 다음 단일 작업\n\n' + next_p, 1)
p.write_text(s, encoding='utf-8')

(R / 'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

갱신 {now}. 사용자 최신 재확인: 계속 진행하고 입력검증에 문제없으면 학습·성능비교까지 멈추지 말고 완료. 중요문제 즉시보고, 최종 기존모델 대비 개선·회귀 보고. Codex 직접 구현/검증 승인 유지.

## 현재 목표와 마지막 완료
현재단위 `research/front_keyphase_v1/ops/{U.name}`. 전체영상검출→상반신마스크와 실제RGB잘림→독립검출 입력을 준비 중이다. 기존 안정자세 경계는 정확히 유지하고, 보정불가 영상은 첫1초 안정 어깨구간의 중심+어깨너비×1.477334276으로 공통하단선을 정한다. 계수는 TRAIN 안정성공889개만 사용, DEV/사용자/단어별 계수 없음. 추가휴식손목필터OFF.

- 기존1199관측 사전검사:1187경계보존/12몸기준대안, 입력불가0. 이것은 새검출1199완료가 아니다.
- 실제 조기30입력: 기존20+이전실패TRAIN10, 독립6,189,400채널값일치/134,599숨긴점교란불변/5151핀불변. 초기18검출조건 새실행, 나머지검증된캐시재사용.
- 고정100TRAIN대표459표본프레임·이전실패TRAIN10·추가위험5 시각검토 완료(집합 중복가능). 명백한 새 주요동작손 잘림은 확인하지 못했으며, 영역 안 검출누락과 하단으로 내려가는 전환은 남음. EARLY_REVIEW_RESULT/clarification 근거 재사용. 표본만으로 모든수어 의미보존 보장 금지.
- 현재 {n}/1199 records, 상태 `{status['status']}`, 실행 `{json.dumps(lock, ensure_ascii=False)}`. exec{execution['exec_session']}, {execution['description']}. `PREP_STATUS`/`PREP_RUNNING`/`CURRENT_EXECUTION.json`/해당log 실제최신 확인. 첫추가100개청크(exec33402)는787초에오류0으로종료했고 원본핀검증완료. 별도exec에서 PID열거가 다른실행을보지못할수있으므로 PID없음만으로잠금삭제하지않는다.
- 학습0/성능비교0. 전수실제입력과시각검토 통과 전 학습하지 않는다.

## 다음 한 작업
2프로세스 준비를 완료하고 신규실패/경계잘림/영역내실제검출누락을 구분한다. 전체1199·잠금해제 후 `verify_full_inputs.py`로 독립검산, `actual_input_triage.py`로 TRAIN 실제입력 변화분집계·장면검토. 기존 `EARLY_REVIEW_RESULT.json`/100대표/10fallback/5우선장면/clarification 검토 재사용. numeric PASS만으로 준비완료를선언하지않고 최종시각검토근거를묶어 READY 작성·학습driver동결→9회학습→`verify_training_scores.py --verify` 저장점수독립검산. 실질문제는 최초증거를 보존해 즉시 보고한다. 이미 승인된 일반후속·역할승인 재질문하지 않는다. 후속자동확인 `st-gcn`을현재TCN작업으로갱신했으며15분간격·일상진행무알림·최종비교보고후삭제한다.

## 유지 조건
AIHub100단어 TRAIN01~09 899개/DEV11~13 300개. ID10미배정/14~18보류. 280채널TCN37636params, A전체/B마스크/C마스크+실제센서50:50, seeds20261013/14/15. whole/context교대,899원본/epoch, max80/patience12. 세부 `ops/word100_roi_training_plan_20261017/PROPOSED_EXPERIMENT.json`. 준비경계만 최신FROZEN_PROTOCOL로 대체. C성능개선 자동보장/운영교체 없음.
M0 원래개발300 290/300=96.67% 저장검산 재사용. M0 SHA e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f. 사용자/멘토/NIKL/KSL/말뭉치 학습·튜닝·후보선택/카메라/다운로드/외부전송/Git/운영수정 금지.

## 담당과 근거
root 경계/준비wrapper, roi_data_budget TRAIN계수·fallback10/위험5시각검토·2process실행기. roi_missing_protocol 독립36합성검사/계수/전수사전점검/실제30입력계약/실행기·학습driver검토. roi_training_protocol 학습driver 순수10검사·TRAIN619기하flag triage·고정100시각검토. `training_INTERFACE_KO.txt`에 READY형식/명령. 준비·학습·운영적용 완료는 서로 다르다.
''', encoding='utf-8')
print(json.dumps({'status': status['status'], 'records': n, 'updated_kst': now}, ensure_ascii=False))
