import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

U = Path(__file__).resolve().parent
LAB = U.parents[1]
ROOT = LAB.parents[1]
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
write = lambda p, value: p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
now = datetime.now(timezone(timedelta(hours=9))).isoformat()
verification = read(U / 'VERIFICATION.json')
web = read(U / 'WEB_CHECK.json')
assert verification['status'] == web['status'] == 'PASS'
assert web['ready']['cameraOff'] and not web['ready']['overflow']
assert web['ready']['participant'] == 'self_eval_20261014'
guard = read(U.parent / 'word100_multiref_20261014/ORIGINAL_GUARD.json')
for p, h in guard.items():
    assert sha(ROOT / p) == h, p
changed = {str(p.relative_to(ROOT)): sha(p) for p in [LAB/'web'/name for name in ['practice.html', 'practice.js', 'practice.css', 'dtw-display.js']]}
next_action = '사용자가 현재12표현을3라운드 직접 촬영한 후, 확인/미확인·오인식·취소를 포함한 실제 모든 시도를 집계해 기존 발표 웹에 반영한다. 모델·DTW 기준·척도는 고정하고 사용자 자료로 변경하지 않는다.'
result = dict(at_kst=now, status='PRETEST_CLARITY_VERIFIED_AWAITING_USER_TRIALS', implementation='Codex local', verification='Codex saved-value/preservation arithmetic checks and real browser verification; no external expert correctness validation', changed_files=changed, saved_values_unchanged=27, original_hashes_preserved=17, summary_cases=7, no_training=True, no_camera_opened=True, reference_rule_unchanged=True, original_test_boundary_preserved=True, servers={'practice':65612,'presentation':58776}, evidence=['VERIFICATION.json','WEB_CHECK.json','SCREEN.png'], next=next_action)
write(U/'COMPLETION.json',result)
(U/'RESULT_KO.txt').write_text('TCN 1위와 DTW 목표 단어를 구분해 표시하고, 양손 종합이 범위 안이어도 각 손의 범위 초과/보류를 먼저 표시한다. 기존 감사 오인식에서 종합60.0은 참고 이내이나 왼손 초과가 분명히 보임을 실제 브라우저에서 확인했다. 수치27개 및 원본17파일/모델/기준/계산코드 보존, 수치 재계산·학습·사용자 튜닝 없음.\n단어를 바꾸면 이전 촬영 상태·화면이 남는 표시도 초기화했다. 기존 확인미저장 오인식의 결과 파일이 보존됨을 확인. 촬영 안내에 자동기록·실수 메모를 명시.\n새 사람 일반화 또는 정상/오류 판정 정확도 검증이 아닌 화면 명확화 완료. 실제 사용자 촬영과 그 결과의 발표 반영은 대기.\n'+next_action+'\n',encoding='utf-8')
state_path=LAB/'ops/STATE.json'
state=read(state_path)
state['updated_date_kst']=now
state['current_stage']='WORD100_PRETEST_CLARITY_READY_AWAITING_USER_TRIALS'
state['next_single_action']=next_action
state['word100_pretest_clarity_20261014']={**result,'unit':'ops/word100_pretest_clarity_20261014'}
write(state_path,state)
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md'
s=context.read_text(encoding='utf-8')
status='''### 2026-10-14 실사용 전 TCN/DTW 비교 대상·좌우 차이 표시 보완 완료
사용자 요청으로 TCN 1위/DTW 선택 목표를 분명히 분리, 오인식 때도 DTW는 선택한 단어와 비교한다는 안내를 표시했다. 각 항목 맨 위에 좌우 각각의 참고 범위 초과·판단 보류를 요약하여 양손 종합값에 가려지지 않도록 했다. 원거리/표시공식/학습참고상한/모델은 변경하지 않았다. 단어 선택 후 이전 촬영 상태가 남던 표시도 초기화했다.
기존 감사 오인식에서 TCN 유성구주민지원센터 / DTW 감사, 굽힘 종합60.0 참고이내·왼손초과를 실제 화면 확인.27저장값·17원본해시·모델/기준/계산코드 보존,7좌우요약검사PASS. 확인미저장 오인식도 결과 파일 보존 확인. 카메라 시작0, 새학습0, 외부전문가검증아님. 근거 ops/word100_pretest_clarity_20261014/COMPLETION.json.
현재 연습8765 탭4 self_eval_20261014/다시보기체크/수어선택/촬영전. 서버65612·발표58776실제확인. 실사용 결과 미집계, 이전 TEST_READY 경계/TEST_PROTOCOL 유지. 촬영 뒤 결과 반영은 아직 대기.

'''
next_block='''### 2026-10-14 다음 — 비교 대상·좌우 표시가 보완된 화면에서 사용자 시험
'''+next_action+''' 기존12표현×3회36회는 권고이며 실제 시도 수만 보고한다. 인식이 틀려도 수행 자체를 따라했다면 ‘따라 했어요’, 실제 동작 실수면 해당 메모를 남긴다. 사용자 완료 통보 뒤 TCN Top1/3/5와 DTW 각손 참고 범위/미관측을 구분 집계한다. 발표의 기존 AIHub개발96.67%와 본인실사용 결과를 분리하며, 본인실사용을 독립 새사람 일반화 증명으로 확대하지 않는다.

'''
assert status.splitlines()[0] not in s
s=s.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+status,1)
s=s.replace('## 9. 다음 단일 작업\n\n','## 9. 다음 단일 작업\n\n'+next_block,1)
context.write_text(s,encoding='utf-8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

갱신 {now}. AGENTS / CONTEXT 최신 §3·§8·§9 / WORKFLOW 우선.

## 현재 요구와 완료
사용자는 시험 전 보완을 완료하고 직접 테스트한 뒤 기존 발표 웹을 수정하라고 요청했다. ops/word100_pretest_clarity_20261014/COMPLETION.json 상태 PRETEST_CLARITY_VERIFIED_AWAITING_USER_TRIALS.
TCN이 고른1위와 DTW가 비교하는 선택 목표를 분리 표시. 각 항목에서 좌우 각각의 참고 범위 초과/판단 보류를 우선 요약, 종합값과 기존 모든 좌우 수치는 유지. 단어를 바꿀 때 이전 촬영 상태가 남던 표시도 초기화. 자동기록/실수메모 안내 추가. backend·모델·DTW계산·기준·척도는 변경하지 않음.
기존 tri_9171304d585b(10/13 두번째전체촬영/첫감사):TCN유성구주민지원센터51.1점/감사44.9점,DTW감사9학습참고.굽힘종합60.0/상한60.9참고이내지만 왼손62.6/상한57.7초과,오른손57.5/상한63.9이내. 새 표시에서 왼손초과를 강조. 수어 정답을 보장하는 기준 아님.

## 검증
Codex 로컬 구현 및 별도 저장값/해시 검산, 실제 브라우저. 외부전문가 정확성 검증 아님.7좌우요약 사례,27저장수치 동일,17원본해시·모델/BASELINE·계산코드 보존. 실제오인식에서 비교단어/9그래프/좌우요약/가로넘침없음 확인. 기존3시도 모두 결과파일 존재, 확인미저장 오인식도 보존. 새촬영/카메라시작/학습/튜닝0.
학습 참고 규칙은 기존 training_observed_max9_v1: 단어/항목/손별9학습자 본인제외비교 최댓값. 차이표시100r/(1+r),50은대표차이. 범위이내=수어정답 또는 범위초과=오류로 간주하지 않음. 원정상/오류라벨검증 미완료. 이전108분포검증/개발36영상 범위이내 굽힘30·위치35·궤적33 근거는 ops/word100_dtw_reference_range_20261014 유지, 반복미실행.

## 다음 한 작업
{next_action}
ops/word100_readable_dtw_20261014/TEST_PROTOCOL.json 및 word100_multiref_20261014/TEST_READY.json(20:29KST 경계/과거3trial제외) 유지. 현재상태표시 explicit_target_hands_v1. 촬영권고12표현 한 번씩3라운드36회, 계획을 실제결과로 쓰지 않음. 모든발급trial/session trial_ids/timing trial_closed/result/self-report 함께 확인. UI최근30개나확인메모있는행만집계금지. TCN정답과 DTW참고범위 초과를 구분. 학습미포함 개발참여자1인 실사용이며 독립새사람일반화증명 아님. 사용자자료학습/튜닝/상한/척도/후보선택금지.

## 서버와 화면
연습 http://127.0.0.1:8765/practice.html 일반환경PID65612와 모델READY 실제확인. 기존탭4 self_eval_20261014/다시보기체크/수어선택/촬영전/카메라꺼짐/기록정보접힘 준비. 카메라는 사용자 연결 버튼으로만 시작. 새로고침하면 설정 초기화 가능.
발표 http://127.0.0.1:8766/#s7 정적PID58776확인, 이번발표파일변경없음. 실사용촬영후 실제결과로 수정할것. 기존62.2/82.6/80.5예시는10/13과거감사촬영이며 새시험결과아님.

## 고정 기준
모델 runs/word100_tcn_v1/whole_context/best.pt SHA e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f.
AIHub100단어899학습01–09/300개발11–13,개발96.67%.최종보류10/14–18미사용.
BASELINE SHA 0af5e66be355ab0a781fc0b7bfcbcee551fd92f9f1cf3bf2097f4c87d34440cf.12표현108학습영상/864사람간비교.학습자는본인제외8명,사용자/개발은9참고.손별유효참고최소5개.
학습/다운로드/운영모델교체/Git변경없음. 말뭉치 등 다른후속을자동재개하지않음.
''',encoding='utf-8')
assert read(state_path)['word100_pretest_clarity_20261014']['status']==result['status']
assert (U/'SCREEN.png').is_file()
assert status.splitlines()[0] in context.read_text(encoding='utf-8')
print(json.dumps({'status':result['status'],'next':next_action},ensure_ascii=False))
