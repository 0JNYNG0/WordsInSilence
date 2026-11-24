from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, hashlib

U=Path(__file__).resolve().parent;OPS=U.parent;ROOT=U.parents[3]
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
assert ROOT.name=='WordsInSilence' and not (U/'FINAL_AUDIT.json').exists()
at=datetime.now(timezone(timedelta(hours=9))).isoformat()
f=read(U/'FROZEN.json');s=read(U/'SUMMARY.json');d=read(U/'DIAGNOSTICS.json');dec=read(U/'DECISION.json')
for name in ('SELFTEST.json','PREPARED.json','VERIFICATION.json','BROWSER_QA.json'):assert read(U/name)['status']=='PASS'
for k in ('protected','previous','code'):
    for p,h in f[k].items():assert sha(p)==h,p
assert s['stress_metric_hands']==7092 and s['query_views']==600
assert len(d['previous_escaped'])==8 and sum(r['new_missing_expected_s']>1e-8 for r in d['previous_escaped'])==4
assert dec['deployment'] is False
assert not list(U.glob('FIRST_ERROR*'))
write(U/'VISUAL_REVIEW.json',dict(status='PASS',at=at,reviewer='Codex inspected rendered local report screenshots',
    screenshots=['QA_DESKTOP.png','QA_RESULTS.png','QA_MOBILE.png'],
    checks=['Korean text legible','partial success and non-deployment explicit','chart not called recognition accuracy','no horizontal overflow in390px and1440px report']))
write(U/'CURRENT_EXECUTION.json',dict(phase='public_reference_extent_validation',session_id=79130,status='COMPLETE',exit_code=0,
    verifier_session_id=68730,verifier_exit_code=0,at=at,duplicate_execution='Do not rerun completed extraction/comparison/verification'))
next_action='이번 공개 다중참고 범위 후보의 부분 개선과 반복 한계를 사용자와 검토한다. 다음 후보는 전체 시퀀스의 한 번/반복 구간 대응을 함께 고려하되, 손 미검출로 범위를 자르지 않고 미확인 시간을 보존하는 별도 비교다. 조건을 먼저 고정해 정상 손내림·한 손·반복·짧은 구간 및 팔 근거 누락을 검증한다. 운영 적용과 추가 학습은 아직 하지 않는다. C·T3-F 유지, 사용자 촬영 학습·튜닝·선택 금지.'
title='2026-10-20 다중 공개 참고 범위 검증 완료 — 손 누락 범위 보존, 반복 한계로 운영 미적용'
entry=f'''### {title}
갱신 {at}. `ops/{U.name}`. Codex 로컬 별도 후보 구현·공개 검증·독립 구현 검산 완료. C·T3-F/운영 DTW/웹은 유지했다.
- TRAIN899(01~09), DEV300(11~13), 공개100단어의 legacy/sensor 두 관측 경로. 1199×2 입력이며2398개 서로 다른 영상 아님. DEV 주석·full 원본 상태는 사후 검증 전용. 기존개발자료이며새최종시험아님.
- 전체96시점의 어깨 기준 팔꿈치·pose손목으로 정렬한 뒤, TRAIN8~9명 중5명 이상의 주석/core·관측·내림 근거를 이전. query 손가락21점이나 hand active_span으로 범위를 자르지 않음. 동작 오답/전체수행 합격 판정 없음, 근거 부족 시간 보존. 새 arm 정렬비용만 계산했고 기존 DTW 손차이/TCN 재채점 없음(CLARIFICATION).
- 2398입력 손21점 제거시 팔 입력 동일. 7092항목·손·가림조건 비교범위 동일/미관측 단조성 검산. 이전8범위축소 누락조건 중4포착(같은순종2항목×2가림), 자석·백십육4조건 미포착. 차이감소3조건 중2포착. 순종peak10에서 제거0.233초 중0.067초만 포착; 전체구별단계 보장 금지.
- 알려진3한손단어9DEV×2경로의18대조/54항목에서 반대손 무영향. 안정내림 추정·미관측599영상손(위치945.88초)과 expected 겹침0초; 전문가 의미판정아님. 시간축만0.75/1.5배1200대조 동일. 역할미확인88단어는손별진단만.
- sensor 원래300영상 주석시간포착 중앙값94.35%, projected시간 주석포함97.60%. 합성전체2회연결시48.92%/50.00%로 하락. 반복을 오답이라하지는않지만 반복비교 안정성 미통과→운영 미적용. 이 수치는 동작시간 중첩으로 인식률 아님. pose손목도 가린591조건 중156범위변화, 팔근거 의존 한계.
- 자체9/별도scalar5394경로·투표/7092조건/4535해시 PASS. 구현자와 별도 검산자는 같은 Codex, 다른 agent·전문가 아님. 보고서desktop/mobile QA PASS. C/T3-F6가중치 포함32보호파일·과거동결결과 불변. 학습/분류추론/사용자멘토자료/보류14~18/카메라/다운로드/Git/운영변경 없음.
- RESULT_KO.html/txt, FROZEN/CLARIFICATION/PREPARED/INDEX/BASELINE/STRESS/CONTROLS/LOWERING/ARM_LOSS/DIAGNOSTICS/VERIFICATION/DECISION/FINAL_AUDIT. 모든 연구 실행 종료0(session79130/검산68730),중복재실행금지.

'''
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=context.read_text('utf-8-sig')
assert title not in text
for marker,new in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',entry),('## 9. 다음 단일 작업',f'### {title}\n{next_action}\n\n')]:
    assert marker in text;text=text.replace(marker,marker+'\n\n'+new,1)
context.write_text(text,'utf-8')
state=read(OPS/'STATE.json')
state.update(updated_date_kst='2026-10-20',updated_at=at,current_stage='WORD100_REFERENCE_EXTENT_REVIEW',activity_state='AWAITING_USER_REVIEW',next_single_action=next_action,resume_command=next_action)
state[U.name]=dict(status=dec['status'],at=at,unit=str(U),report=str(U/'RESULT_KO.html'),summary=s,
    diagnostics=str(U/'DIAGNOSTICS.json'),decision=str(U/'DECISION.json'),verification=read(U/'VERIFICATION.json'),
    protected_files_unchanged=32,model_weights_unchanged=6,production_changes=0,new_training=0,private_data_read=0,
    execution='All preparation, comparison and verification completed exit0; no new research process remains.',next_single_action=next_action)
write(OPS/'STATE.json',state)
handoff=f'''# WordsInSilence 最新 인수인계

갱신 {at}

현재 목표: 손이 사라져도 비교할 동작 범위를 보존하고, 관측 부족을 낮은 차이값과 구분한다. 마지막 완료: 다중 공개 참고 기반 별도 후보 검증. 다음 한 작업: 반복 정렬 한계를 포함한 결과를 사용자와 검토한다.

- 단위 `{U}`. 보고 RESULT_KO.html/txt. DECISION/DIAGNOSTICS/VERIFICATION/FINAL_AUDIT 참조.
- 공개100단어 TRAIN899(01~09)만 참고, DEV300(11~13)평가. 기존관측+sensor2경로=2398입력. DEV주석과 전체원본 손상태는 후보입력금지/오프라인 검증만. 기존DEV로최종일반화수치아님.
- 전체영상96시점 어깨기준 팔꿈치·pose손목 DTW정렬, 참고8~9명 중5명 이상 구간투표. 손21점/active_span으로 비교범위 축소안함. reference annotation은필수단계정답아님. arm정렬비용은새계산하나 기존DTW차이값·TCN재채점없음(CLARIFICATION).
- 2398입력 손21점제거에도팔입력동일,7092항목손가림범위고정. 이전누락8조건 중4포착(순종같은영상2항목×2가림),자석/백십육4조건미포착. 순종손실0.233초중0.067초만포착. 차이감소3조건중2포착. 모든누락해결아님.
- 기존역할3한손단어9DEV×2경로18대조/54항목 반대손무영향,원본안정내림추정·미관측599영상손(위치945.88초)과expected겹침0초. 언어적정상판정아님. 시간축0.75/1.5배1200대조동일;실제고속검출검증아님. 역할미확인88단어는손별진단.
- sensor300영상 주석시간포착중앙값94.35%/후보시간주석포함97.60%; 전체2회합성연결48.92%/50.00%로저하. 단어인식률아님. 반복오류판정은만들지않았지만비교범위유지실패→운영미적용. pose손목동시누락591중156범위변화.
- 자체9,별도scalar5394정렬경로·투표/7092가림/4535해시PASS. 같은Codex별도구현검산,다른agent아님. 보고서desktop/mobile시각QA PASS.
- C/T3-F6가중치 포함32보호파일/과거동결결과 보존. 사용자/멘토/보류14~18미접근,학습·분류추론·운영교체·카메라·다운로드·외부전송·Git없음. 웹 재시작/접근없음; 기존서버마지막기록62068은이번재확인안함.
- 모든연구실행종료0: 본실험79130,검산68730. 완료된추출/비교/검산재실행금지. FIRST_ERROR없음. 후보/동결계수 DEV결과에맞춰수정금지.

{next_action}
'''
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff.replace('最新','최신'),'utf-8')
write(U/'STATUS.json',dict(status='COMPLETE_OFFLINE_REVIEW',at=at,production_applied=False,next_single_action=next_action))
for p,h in f['protected'].items():assert sha(p)==h,p
artifacts={p.name:sha(p) for p in U.iterdir() if p.is_file() and p.name!='FINAL_AUDIT.json'}
docpaths=[context,ROOT/'NEW_CHAT_HANDOFF.md',OPS/'STATE.json']
write(U/'FINAL_AUDIT.json',dict(status='PASS',at=at,scope_complete='Implementation and frozen public evaluation complete; repetition acceptance failed, candidate not deployed.',
    protected_files_unchanged=len(f['protected']),model_weights_unchanged=6,previous_results_unchanged=len(f['previous']),
    production_changes=0,private_data_read=0,new_training=0,classifier_inference=0,review_required_before_training=True,
    artifact_sha256=artifacts,document_sha256={str(p):sha(p) for p in docpaths}))
assert read(OPS/'STATE.json')[U.name]['status']==dec['status']
assert title in context.read_text('utf-8') and U.name in (ROOT/'NEW_CHAT_HANDOFF.md').read_text('utf-8')
print('FINALIZED',at)
