"""Document-only closeout. No model imports, inference, or evaluation rerun."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json

U=Path(__file__).resolve().parent
O=U.parent
ROOT=O.parents[2]
def rd(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def wr(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
r=rd(U/'FINAL_RESULT.json');v=rd(U/'FINAL_VERIFICATION.json')
assert v['integrity_passed'] and r['steps_completed']==[1,2,3,4]
assert not r['web_adopted'] and not r['goal_complete']
assert [r['gain'][str(s)]['correct_omission_after'] for s in [42,43,44]]==[5,4,6]
for p,h in rd(U/'PROTECTIONS.json').items():assert sha(p)==h,p

summary=('승인1~4단계 완료. 공개54영상/10표현의 기존주석 재사용,20영상107시점 원본검토,역할/구간품질/조각관계 판정부3초기값 학습. '
         '100단어899공통부·기존모양방향 동결. 최종 행동삭제6조건 누락안내0→5/4/6,역재생9 경고5→6/5/5,거짓비슷함1잔존. '
         '정상12+속도24·기존오류9 유지,관측부족18 정상/누락승격0. 역할시점64/67유지. '
         '추가6표현 후보21개10→14/14/13이나 완전이동8→6퇴행. 충돌보호 후처리·첫결과보존. '
         '과거개발재사용/합성오류이며독립최종평가아님. 웹미채택·개인자료0·HTML생략.')
next_action=('방향을 읽을 수 있는 부분시도와 실제 행동완료를 분리한 FIT정답부터 보강; '
             '양손 주체/상대손 공통연결과 불연속 근거를 명시하고 자동후보→설명→최종교정 비교. '
             '상세 NEXT_PLAN_KO.txt. 현재 완료 학습·평가 재실행금지; 후속학습 미착수.')
title='2026-11-11 후보 역할·설명 연결 4단계 완료·제한적 누락 개선'
report=str(U/'RESULT_KO.txt');plan=str(U/'NEXT_PLAN_KO.txt')

scope=dict(reviewed_at=now,code=str(U/'integration.py'),
 findings=[
 'No word-ID-specific corrective branches; declarative guide contracts select action families.',
 'New role-generated proposals use moving=R and newly bound direction checks filter R; opposite=L. General left-hand independent direction correction is not established.',
 'Paired-hand relation features and crossing regression do not prove arbitrary hand-role generalization.',
 'Current acceptance criteria fail; this unit remains research only.'
 ],changed_scored_code=False,next_plan_includes_side_contract=True)
wr(U/'SCOPE_REVIEW.json',scope)

p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text(encoding='utf-8')
assert title not in s,'Already synchronized; do not duplicate'
for heading,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',next_action)]:
 assert heading in s
 addition=f'\n\n### {title}\n\n{body}\n근거 {report}\n후속 {plan}\n'
 s=s.replace(heading,heading+addition,1)
p.write_text(s,encoding='utf-8')

sp=O/'STATE.json';state=rd(sp)
state['current_stage']=U.name;state['next_single_action']=next_action;state['last_updated']=now
state[U.name]=dict(status=r['status'],summary=summary,execution=str(U/'CURRENT_EXECUTION.json'),
 report=report,next_plan=plan,final_result=str(U/'FINAL_RESULT.json'),verification=str(U/'FINAL_VERIFICATION.json'),
 first_result=str(U/'RESULT.json'),scope_review=str(U/'SCOPE_REVIEW.json'),
 author='Codex',verifier='Same Codex separate saved arithmetic/source audit; not independent human',
 web_adopted=False,goal_complete=False,common100_frozen=True,private_data=0,
 primary_omission=dict(before=0,after=5,total=6),all_seed_omission=[5,4,6],
 all_seed_reverse_warnings=[6,5,5],reverse_total=9,remaining_false_similar=1,
 historical_development_replay=True,post_score_safety_reconciliation=True)
wr(sp,state)

handoff=f'''# WordsInSilence 최신 인수인계

전체 목표: 공개 한국수어의 설명과 여러 수행자의 행동을 배워 새 사용자의 모양·방향·순서·누락을 짧고 근거 있게 알려준다.

현재 단위: {U}
상태: {r['status']}. 1~4단계 실행·학습·검산 완료, 실행 중 학습/평가 PID 없음.
{summary}

현재 웹/TCN 기본과 C·T3-F는 그대로다. 새 후보를 적용 완료로 설명하지 않는다.
핵심 한계: 역재생 잘하다 기록13 거짓비슷함1,완전이동 판정 퇴행,새 방향 연결 오른손 중심. 자연오류/독립전문가 검증 미완료.
추가 연결부 중앙0.035초는 RTMW/기존특징/TCN/DTW/UI를 제외한 시간이다.
첫 결과 후 누락을 순서오류로 설명한 사례를 발견해 기존 충돌보호를 복원했다. 재학습/추론0,첫 결과 보존. 후처리 결과를 새 독립평가라고 부르지 않는다.
검증: 같은 Codex 별도 저장산술·원본RGB 확인. 독립 사람 검수 아님.

다음 한 작업: {next_action}
근거: {report}
세부 계획: {plan}
최종 결과: {U/'FINAL_RESULT.json'}
최종 검사: {U/'FINAL_VERIFICATION.json'}
범위 확인: {U/'SCOPE_REVIEW.json'}

완료한3학습·702비교·234저장후처리 재실행금지. 새모델다운로드/카메라/외부전송/Git/운영교체 없음. 사용자·멘토자료 학습·튜닝·선택 금지,원본/이전동결 읽기전용.
'''
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')

hp=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt';history=hp.read_text(encoding='utf-8')
history+=f'''\n\n[{title}]
이유: 자동으로 움직임을 찾는 부품 개선이 최종 교정으로 이어지지 않아 준비·주요행동·복귀 역할과 후보-설명 대응을 함께 학습했다.
{summary}
학습부는 기존 시간합성곱+GRU 특징을 쓰는 역할 어댑터와 품질/쌍 판정부이며 공통 인코더 전체를 재학습한 것은 아니다. 새로운20영상으로 부풀리지 않고 기존노출을 감사했다.
첫 누락조건에 생긴 잘못된 순서 안내를 원본RGB로 발견했다. 경고수가 늘었다는 사실만으로 성공을 주장하지 않고 기존단계와 새역할이 충돌하는13개 새경고 성분을 보류했다. 일부 맞는새경고도 줄어드는 대가와 후처리의 개발성격을 기록했다.
현재 개선은 인위적으로 마지막 동작을 삭제한 시험의 누락안내에 국한된다. 방향을 읽을 수 있는 부분과 동작전체 완료를 분리해야 함을 후속 핵심으로 확정했다. 새연결의 오른손 중심 제약도 범위감사에서 기록; 양손관계 입력과 모든손에 대한 교정을 혼동하지 않는다.
전체목표는 유지. 입력/TCN/DTW/화면의 기반과 자동 최종교정 완성 여부를 분리한다. 현재 웹은 불변,새 후보 전체채택 실패. 동일Codex 검산을 독립사람정답으로 부르지 않는다.
다음: {next_action}
근거 {report}
계획 {plan}
'''
hp.write_text(history,encoding='utf-8')

wr(U/'CURRENT_EXECUTION.json',dict(status=r['status'],pid=None,completed_at=now,
 training_completed=True,evaluation_completed=True,verification_completed=True,
 web_adopted=False,goal_complete=False,report=report,next_plan=plan))
paths=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',sp,hp,
 U/'RESULT_KO.txt',U/'NEXT_PLAN_KO.txt',U/'CURRENT_EXECUTION.json',U/'FINAL_RESULT.json',
 U/'FINAL_VERIFICATION.json',U/'SCOPE_REVIEW.json']
for p in paths:
 assert p.is_file()
 if p.suffix=='.json':rd(p)
assert all(str(U) in p.read_text(encoding='utf-8') for p in [ROOT/'NEW_CHAT_HANDOFF.md',hp])
for p,h in rd(U/'PROTECTIONS.json').items():assert sha(p)==h,p
wr(U/'DOCUMENT_CHECK.json',dict(passed=True,completed_at=now,files={str(p):sha(p) for p in paths},
 frozen_files_unchanged=len(rd(U/'PROTECTIONS.json')),training_or_inference_rerun=False))
print(json.dumps(dict(status=r['status'],documents=len(paths),frozen_files_unchanged=346),ensure_ascii=False))
