from motion_core import *
import ast
checks=[]
def check(k,v):assert v,k;checks.append(k)
for pred in [-1,0,1,2]:
 for eligible in [False,True]:
  od=dict(prediction=pred,eligible=eligible,confidence=.8)
  for new in [0,1,2]:
   nd=dict(prediction=new,eligible=True,confidence=.99);r=fused(od,nd)
   expected=eligible and pred==1 and new!=1
   check(f'gate_{pred}_{eligible}_{new}',r['rescue_used']==expected and (expected or r['prediction']==pred))
net=MotionNet().eval();zero=np.zeros((20,280),np.float32);ts=np.arange(20)*33333
check('missing_no_direction',not direction(net,zero,ts,2,10,0)['eligible'])
g=completion_guard(net,dict(a=2,b=10,axis=0,moving='R'),zero,ts)
check('missing_no_completion',g['state']=='unobserved_after' and not g['semantic_complete'])
g=completion_guard(net,dict(a=2,b=19,axis=0,moving='R'),zero,ts)
check('recording_end_no_completion',g['state']=='unobserved_after' and not g['semantic_complete'])
for r in read(U/'FIT_LABEL_AUDIT.json')['rows']:check('fit_sign_'+r['clip_id']+'_'+str(r['event'])+'_'+r['kind'],not r['sign_conflict'])
for p in [U/'motion_core.py',U/'evaluate.py',U/'train.py']:ast.parse(p.read_text('utf-8'))
check('runtime_no_word_argument','word' not in __import__('inspect').signature(direction).parameters)
for p,h in read(U/'FROZEN.json')['protections'].items():check('protected_'+p,sha(p)==h)
write(U/'CONTRACT_CHECK.json',dict(passed=len(checks),failed=0,checks=checks))
assert not (U/'EVAL_PINS.json').exists()
write(U/'EVAL_PINS.json',{str(p):sha(p) for p in [U/'motion_core.py',U/'evaluate.py',U/'EVALUATION_LABELS.json',U/'EVAL_PROTOCOL.json',U/'CONTRACT_CHECK.json',*[U/'runs'/str(s)/'model.pt' for s in [42,43,44]]]})
print('contracts passed',len(checks))
