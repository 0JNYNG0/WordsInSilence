from pathlib import Path
import json,hashlib,copy,time
import numpy as np
import torch
from runtime_goal_evidence import ResearchGoalExtension
U=Path(__file__).resolve().parent;O=U.parent;D=O/'word100_positive_direction_20261109'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with (U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
cases=sorted(p for p in(D/'final_decisions').glob('*_full_normal.json'))
local={r['case']:r for r in read(O/'word100_action_linkage_repair_20261113/LOCAL_DIRECTION_COMPARISON_V2.json')['cases']}
write('RUNTIME_EXTENSION_FROZEN.json',dict(code_sha256=sha(U/'runtime_goal_evidence.py'),cases=[p.stem for p in cases],
 scope='12 cached public normal video observations; frozen goal heads only. A saved-output fixture, not webcam/end-to-end benchmark.',
 checks=['All original payload fields exact','Native frame/time correspondence','No blanket completion','No empty hands positive','Reject role11 as web14'],
 production_web_changed=False))
torch.set_num_threads(2);model=ResearchGoalExtension();checks=[];rows=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
for p in cases:
 raw=read(p)['after'];result=copy.deepcopy(raw);cur=local[p.stem]
 for m in cur['motion']:
  next(c for c in result['cards']if c['id']==m['card'])['motion_evidence']=m['evidence']
 result['coaching']=copy.deepcopy(cur['current_cues'])
 z=np.load(D/'predictions'/(p.stem+'.npz'));t=time.perf_counter()
 out=model.attach(result,z['x'],z['timestamp_us'],z['frame_id'],z['phase']);dt=time.perf_counter()-t
 ev=out.pop('goal_observation');ck('existing_payload_unchanged',out==result)
 ck('disabled_until_gate',ev['display_enabled']is False and ev['whole_action_complete']is None)
 for c in ev['components']:
  for e in c['events']:
   ck('uniform_hint_schema',isinstance(e['hint'].get('text_ko'),str))
   ck('no_new_normal_difference',e['joint_evidence']['state']!='component_difference_observed')
 rows.append(dict(case=p.stem,seconds=dt,evidence=ev))
 if ev['components']:
  xx=z['x'].copy();xx[:,:102].reshape(-1,51,2)[:,9:51]=0;xx[:,195:237]=0;xx[:,237:279]=0
  masked=model.attach(result,xx,z['timestamp_us'],z['frame_id'],z['phase'])['goal_observation']
  ck('masked_no_positive_goal',not any(e['evidence']['goal_observed']for c in masked['components']for e in c['events']))
 try:model.attach(result,z['x'],z['timestamp_us'],z['frame_id'],z['phase'][:,:11])
 except ValueError:ck('role11_rejected',True)
 else:ck('role11_rejected',False)
write('RUNTIME_EXTENSION_VERIFIED.json',dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows,
 failures=[c for c in checks if not c['passed']],production_changed=False,server_started=False,camera_opened=False,
 scope='Isolated extension fixture, not deployed feedback accuracy or total user waiting time',reviewer='same root Codex separate runtime contract checks'))
print(json.dumps(dict(passed=all(c['passed']for c in checks),checks=len(checks),failures=[c for c in checks if not c['passed']],normal_public_inputs=len(cases)),ensure_ascii=False))
