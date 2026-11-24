"""Record the explicit research resume without changing frozen experiments."""
from pathlib import Path
import json, datetime, hashlib

HERE=Path(__file__).resolve().parent
G=HERE.parent; OPS=G.parent; ROOT=OPS.parents[2]
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');q.replace(p)
at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
note='사용자가 2026-11-13 새벽 그래프 자율 연구 재개를 요청. G2에서 측정 대상·시간문맥·행동 부재를 분리 진단하고 공개 FIT 근거로 보완한다. 기존 완료 실험 재실행, 사용자·멘토·최종8 접근, C·T3-F·공통100·웹 변경 없음.'
next_action='공개 FIT에서 방향 학습의 반전 입력과 실행 입력 일치 여부부터 검증한다. 손끝/손목/팔방향 정답을 구분하고, 실제 관측 시퀀스 반전에 따른 시간문맥을 재계산한 후보를 동결 비교한다. 행동 부재 감독은 원본 연속 구간을 확인한 근거만 사용한다. 정상/속도/교차/모양/방향/순서/누락을 함께 통과하기 전 G3·최종평가·웹 교체로 넘어가지 않는다.'
assert not (HERE/'RESUME.json').exists()
write(HERE/'RESUME.json',dict(at=at,request='사용자: 아까 그래프 작업 계획대로 새벽 내내 자율 재개',scope=note,next=next_action,previous_execution=read(G/'CURRENT_EXECUTION.json'),process_check='Get-Process python/pythonw returned no active process. Win32_Process CIM query was unavailable; no process assumed alive from stale PID.',reviewers=['root Codex','motion_contract_review_1113','absence_contract_review_1113'],not_independent_human_truth=True))
e=read(G/'CURRENT_EXECUTION.json');e.update(status='ACTIVE_G2_TYPED_MOTION_AUDIT',at=at,training_running=False,process_running=False,next=next_action,current_command=None,session_id=None,resume_authorized=True,active_unit=str(HERE));write(G/'CURRENT_EXECUTION.json',e)
s=read(OPS/'STATE.json');s['updated_at']=at;s['next_single_action']=next_action;s['word100_web_completion_graph_20261111'].update(status='ACTIVE_G2_TYPED_MOTION_AUDIT',at=at,training_running=False,active_unit=str(HERE),remaining=next_action);write(OPS/'STATE.json',s)
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8-sig')
for heading,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',note),('## 9. 다음 단일 작업',next_action)]:
    assert heading in t;t=t.replace(heading,heading+'\n\n### 2026-11-13 사용자 승인 새벽 연구 재개\n'+body+'\n근거: `'+str(HERE.relative_to(ROOT)).replace('\\','/')+'/RESUME.json`.\n',1)
p.write_text(t,'utf-8')
p=ROOT/'NEW_CHAT_HANDOFF.md';t=p.read_text('utf-8-sig');p.write_text('# 최신 인수인계 — 2026-11-13 G2 연구 재개\n\n'+note+'\n'+next_action+'\n현재 실행: 중복 학습 없음, 입력 계약 감사부터 진행.\n\n'+t.split('\n',1)[1],'utf-8')
print('Research resume recorded',at)
