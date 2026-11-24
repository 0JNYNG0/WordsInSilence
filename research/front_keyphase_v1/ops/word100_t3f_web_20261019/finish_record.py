"""Synchronize completed opt-in web work; no model or recordings read/written."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,shutil

U=Path(__file__).resolve().parent
LAB=U.parents[1]
ROOT=LAB.parents[1]
stamp=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
def read(p):return json.loads(p.read_text('utf-8-sig'))
for name in ('VERIFICATION.json','FRONTEND_CHECK.json','BROWSER_CHECK.json'):
    assert read(U/name)['status']=='PASS',name
http=read(U/'HTTP_READY.json');server=read(U/'SERVER.json')
assert http['camera'] is None and all(x['model_status']=='READY' for x in http['models'])
assert http['default']=='word100_tcn_context_v1'
changed=['src/wis_lab/service/word100_t3f.py','src/wis_lab/service/word100.py',
 'src/wis_lab/service/word100_web_analysis.py','src/wis_lab/service/app.py',
 'web/practice.js','web/practice.html','web/practice.css']
hashes={p:hashlib.sha256((LAB/p).read_bytes()).hexdigest() for p in changed}
result=dict(at=stamp,status='READY_OPT_IN_T3F_WITH_PAIRED_C',unit='ops/'+U.name,
 url=server['url'],server=server,changed_files=hashes,model_id='word100_tcn_t3f_v1',
 T3F_checkpoint_sha256='b0faa0df5055204295c60b78c596a0b490cd63ef14f30225582a85ea39855846',
 C_checkpoint_sha256='283a54bd9866c97959f9d615ebc38d126ee25688821960b208cf9412389eeaab',
 seed=20261013,T3F_epoch=69,C_epoch=77,
 same_capture=True,channel_builds_per_take=1,input_hash_saved=True,logits_per_take=200,
 requested_word_used_for_inference=False,existing_C_M0_checkpoints_and_C_adapter_preserved=True,
 default_model_changed=False,DTW_calculation_changed=False,
 camera_opened_by_agent=False,user_recording_reads_for_development=0,new_training=0,heldout_reinference=0,
 data_use='evaluation_only_never_training_tuning_or_model_selection',
 verification={'runtime_and_transport_checks':7,'frontend_fixture_checks':7,'browser_widths':[1440,390],
  'independent_formula_max_abs_error':9.5367431640625e-7,'reviewer':'Codex local, not independent person/agent'},
 limitations=['Actual live capture performance awaits user test','Softmax scores are not execution correctness',
  'Same person repeated tests are paired practical checks, not population-wide generalization'],
 recovery='QA bundled browser absent: used installed Chrome; no application/model fix required')
(U/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
(U/'RESULT_KO.txt').write_text(f'''T3-F 웹 선택 및 동일 촬영 C 비교 준비 완료
갱신: {stamp}

주소: {server['url']}
기존 C: http://127.0.0.1:8765/practice.html?model=c
기본 M0 유지: http://127.0.0.1:8765/practice.html?model=m0

1. 인식 모델에서 ‘후보 T3-F · 기존 C와 함께 비교’를 선택한다.
2. 카메라를 연결하고 원하는 표현을 한 번 촬영한다. 시작 후 3초 준비·촬영 중 표시를 확인하고 동작한다. 동작 완료 시 촬영 종료(최대10초).
3. 같은 촬영의 동일 C 관절 입력을 한 번 준비하고, T3-F와 기존 C에 각각 넣는다. 두 번 촬영할 필요 없다.
4. 두 모델의 1위, 목표 표현의 100개 중 순위, 상위5개 후보와 점수를 함께 표시한다. 두 모델100개 점수·모델SHA·동일입력SHA·관절파일을 함께 자동 저장한다.
5. 확인 기록은 시도 상태/메모다. 틀린 결과도 자동 기록되며 성공만 선별하지 않는다. 다시 촬영하면 별도 시도로 남는다.

T3-F: 고정 seed20261013/epoch69/SHA b0faa0df. 기존 C: 동일 seed/epoch77/SHA283a54bd. 재학습·seed재선택0. 기존 C/M0 파일과 C 입력처리·촬영코드·DTW 계산코드 보존.
T3-F는 C가 배운 부분을 고정한 채 동작의 앞·중간·뒤 흐름을 요약하는 보조 부분을 더한 후보이다. 이번 연결은 완성된 모델의 추론만 수행한다.
DTW는 선택한 표현의 기존 참고 동작과 비교하는 공통 분석이다. 모델별로 달라지는 DTW 점수나 새 교정 기준을 만들지 않았다.
사용자 촬영은 평가 기록 전용. 학습·튜닝·임계값 조정·후보선택에 쓰지 않는다. 웹캠 개선률은 이번 작업에서 측정하지 않았다.

검증: 공개 TRAIN1개로 C 기존 출력 완전동일·동일 입력·T3-F 별도산술검산(최대오차9.54e-7)·상태불변, 합성API 저장/재열람/변조거절 등7검사 PASS. 프런트엔드7검사 PASS. 실제Chrome headless 1440/390폭, 비교2칸/후보10행/JS오류0/카메라요청0 PASS. 비교스크린샷의 수치는 합성시험용으로 실제 성능이 아니다.
서버: 일반실행 PID{server['pid']}, 127.0.0.1:8765. 카메라는 사용자 연결 버튼 때만 열림. 참고 영상 HTTP200 video/mp4 확인(Range요청은서버가전체응답하여3277974bytes).
환경 메모: 첫UI검사의 번들Chromium 부재로 설치된Chrome을 사용했다. 임의추가다운로드0. 첫참고영상HTTP프로브의 잘못된URL404는 정확한라우트로정정해200확인. 코드/모델문제아님.

다음: 사용자가 직접 촬영한 뒤 같은 시도의 C/T3-F 정답순위,상위5포함,관측실패,개선/회귀를 저장결과로 집계한다. 사용자1명의 반복측정을 모든 새 사람 성능으로 해석하지 않는다.
''','utf-8')
next_action='T3-F 웹 선택 모드에서 사용자가 직접 시험한다. 한 촬영의 동일 관절 입력 C/T3-F 저장결과를 짝지어 집계한다. 성공/실패/재시도 모두 보존하고 사용자·멘토 촬영을 학습·튜닝·임계값·모델선택에 쓰지 않는다. 기존 C/M0/DTW 유지. 새 학습이나 추가 heldout 재평가 없음.'
paths=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',LAB/'ops/STATE.json']
for p in paths:
    dest=U/'before'/p.name
    if not dest.exists():shutil.copy2(p,dest)
context=paths[0].read_text('utf-8')
entry=f'''### 2026-10-19 T3-F 웹 선택·같은 촬영 C 비교 연결 완료
갱신 {stamp}. 최신 사용자 승인에 따라 T3-F(seed20261013/epoch69/b0faa0df)를 웹에 명시 선택으로 추가. 기존 C(epoch77/283a54bd)와 M0 및 기본M0 유지. 한 촬영에서 C의 동결 관절입력을1회 생성해 T3-F/C 두 모델을 비교하고 동일입력SHA·각100점수·모델SHA를 저장한다. 웹에 두 모델1위/목표순위/상위5후보 표시, 과거C 기록 재열람 유지. C 입력/검출/카메라 및 DTW 계산코드 불변. 공개TRAIN1개 산술·합성저장/실패경로7검사, 프런트7검사, Chrome두폭 QA PASS. 사용자촬영접근/새학습/보류자료재추론/에이전트카메라0. 일반실행PID{server['pid']} 포트8765 READY, 실제웹캠성능은 사용자시험대기. 근거 ops/{U.name}/RESULT.json, VERIFICATION.json, BROWSER_CHECK.json. 구현/검산 Codex로컬이며 별도사람/agent검토아님.

'''
marker='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n'
assert marker in context
context=context.replace(marker,marker+'\n'+entry,1)
marker9='## 9. 다음 단일 작업\n'
assert marker9 in context
context=context.replace(marker9,marker9+'\n### 2026-10-19 T3-F/C 사용자 동일촬영 비교\n'+next_action+'\n주소 http://127.0.0.1:8765/practice.html?model=t3f . 서버상태부터 확인해 중복실행하지 않는다.\n\n',1)
paths[0].write_text(context,'utf-8')
state=read(paths[2]);state.update(current_stage='WORD100_T3F_WEB_READY_PAIRED_C',updated_date_kst='2026-10-19',updated_at=stamp,next_single_action=next_action)
state[U.name]={**result,'next_single_action':next_action}
paths[2].write_text(json.dumps(state,ensure_ascii=False,indent=2),'utf-8')
paths[1].write_text(f'''# WordsInSilence 최신 인수인계

갱신: {stamp}

현재 목표: T3-F를 웹에서 선택해 동일 촬영 C와 비교하도록 준비 완료. 사용자 실사용 시험 대기.

- 단위: research/front_keyphase_v1/ops/{U.name}
- URL: {server['url']} . 기본M0·기존C 그대로. C선택기존모드는C/M0비교, T3-F선택모드는T3-F/C비교.
- T3-F 고정 seed20261013/epoch69/SHA b0faa0df5055204295c60b78c596a0b490cd63ef14f30225582a85ea39855846. C 동일seed/epoch77/SHA283a54bd9866c97959f9d615ebc38d126ee25688821960b208cf9412389eeaab.
- 같은촬영에서 동결C 채널1회생성·동일tensor/padding으로 두모델추론. 원본관절·candidate_c.npz·동일입력SHA·모델별100logits·순위 자동저장. 목표단어는추론입력아님. 저장실패시FEEDBACK미발행/입력불일치차단/모델실패시다른점수대체없음.
- 화면: 모델선택, 두모델1위/목표100개중순위/상위5개후보와모델점수. DTW는기존선택단어참고비교공통분석. 새로운교정기준추가없음.
- 보존: C/M0/T3-F checkpoint, C adapter/카메라/DTW계산코드/공개평가결과. 사용자촬영학습·튜닝·후보선택금지. 이번개발중사용자기록읽기0/학습0/heldout재추론0/카메라0.
- 검증: VERIFICATION7/7(공개TRAIN1개기존C출력동일, T3-F별도산술최대9.54e-7, 입력1회·무변경·실패차단·합성API저장/재열람/변조거절), FRONTEND7검사, 실제Chrome1440/390폭 PASS. 독립수식/합성검산은Codex로컬; 다른사람/agent검토아님. 스크린샷비교값은합성이다.
- 서버: 일반실행(require_escalated) PID{server['pid']} 포트8765. 이전기록PID34640와리스너없음확인후신규시작. HTTP_READY모델3개READY/카메라null. 서버만시작했으며카메라는사용자연결버튼으로열림.
- 실행환경첫오류: Playwright번들브라우저부재로설치Chrome사용, QA_ENVIRONMENT_NOTE보존. 실제참고영상HTTP200확인. 현재미해결연결오류없음.
- 직전공개평가: ops/word100_t3f_heldout_20261019 완료. 200새영상(ID17/18)×3고정모델쌍, sensor평균C99.6667→T3-F100%, masked99.3333→99.6667%(각+0.3333%p). sensor2동률1상승으로사전2/3상승기준미충족; masked3개선1회귀. 모델새선택없음. ID17/18은이미평가사용; ID14~16정상299개미사용유지. 과거EVAL_EXIT1은점수저장후상태파일WinError5로별도복구·재추론없음.
- 한계: 웹연결검증은실사용개선증명아님. 사용자본인1명반복은실용적인동일촬영비교이며모든사람일반화아님.

다음 한 작업: {next_action}
''','utf-8')
assert read(paths[2])['current_stage']=='WORD100_T3F_WEB_READY_PAIRED_C'
assert all((LAB/p).exists() for p in changed)
print(json.dumps(dict(status='COMPLETE',unit=str(U),server_pid=server['pid'],docs_updated=[str(p) for p in paths]),ensure_ascii=False))
