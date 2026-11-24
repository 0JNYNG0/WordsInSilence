from pathlib import Path
import json,hashlib,shutil,collections
from datetime import datetime
from zoneinfo import ZoneInfo
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2];P=O/'word100_clause_finger_diagnostics_20261114'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not(U/'PLAN.json').exists()
now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
plan=dict(at=now,authorization='User approves remaining finger error audit, additional public finger-combination labels, supervised learning/comparison, and only passing functions connected to web.',steps=['A Inspect wrong finger evidence and available public label diversity','B Review/add missing finger combinations with separate FIT/DEV identities and unknown masks','C Fixed learned candidates and paired old/new comparison, preserve normal/source timing/other evidence','D Integrate only passing evidence, public web verification, record limitations'],constraints=['C/T3F and common100 weights frozen','No private/user/mentor recordings or camera','No LLM/3D work','No word-specific prediction exception','No target-state or test-variant flags in observer inputs','Never move DEV labels to FIT','Unknown, bent and folded must not be silently interchanged','No promotion of untested per-finger claims'],gates=['New known finger error rate not worse and old correct evidence retained','Independent finger combinations represented and tested before detailed claims','Normal/shape/direction/order/missing regressions preserved','Uncertain observations never emitted as errors','Model/input provenance and precise intervals retained'],implementation='Codex',verification='Separate saved-output/invariance audit by same Codex; not an independent expert',HDD_present=Path('W:/WordsInSilence/data/AIHub_103/videos').exists(),web_promotion_conditional=True)
write(U/'PLAN.json',plan);b=U/'before';b.mkdir()
for p in [R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,b/p.name)
pins=read(O/'word100_web_completion_graph_20261111/PROTECTIONS.json');pins.update(read(P/'PROTECTED.json'))
web=O/'web_tutorial_live_integration_20261113';g6=O/'word100_live_g6_20261114'
for root in [web,g6]:
 for rel in (['server.py','site/live.js','site/feedback.js','site/index.html']if root==web else['engine.py','g6_runtime_adapter.py','g6_routes.py']):
  p=root/rel;pins[str(p)]=sha(p);d=b/root.name/rel;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
write(U/'BEFORE_PINS.json',pins)
state=read(O/'STATE.json');state['current_followup']=U.name;state['next_single_action']='손가락 오판 원인·공개 조합 정답 보강→고정 학습/비교→통과 기능만 웹 연결. 사용자 자료 금지, 기존 C·T3-F 유지.';state[U.name]=dict(status='IN_PROGRESS',plan=str(U/'PLAN.json'),at=now);write(O/'STATE.json',state)
title='### 2026-11-15 손가락 오판·부족 조합 보강 및 조건부 웹 연결 승인'
doc=(R/'WordsInSilence_PROJECT_CONTEXT.md').read_text('utf-8-sig')
for h in ['## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 9. 다음 단일 작업']:doc=doc.replace(h+'\n\n',h+'\n\n'+title+'\n\n'+state['next_single_action']+' 근거: `research/front_keyphase_v1/ops/'+U.name+'/PLAN.json`.\n\n',1)
(R/'WordsInSilence_PROJECT_CONTEXT.md').write_text(doc,'utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text('# 손가락 오판 보강 진행 중\n\n'+state['next_single_action']+'\n현재 '+str(U)+' / PLAN.json, CURRENT_EXECUTION.json 확인. W: 미연결 확인, C:에 보존된 공개 원본·관절·주석부터 재사용. 사용자 실제 웹캠 확인 보류, 자동카메라 금지.\n','utf-8')
write(U/'CURRENT_EXECUTION.json',dict(status='A_AUDIT',at=now,pid=None))
print(str(U))
