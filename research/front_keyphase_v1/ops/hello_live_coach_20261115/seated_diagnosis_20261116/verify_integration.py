"""Synthetic/public verification, explicitly no personal diagnostic input."""
from pathlib import Path
import sys,json,copy,runpy,torch,numpy as np,hashlib
V=Path(__file__).resolve().parent;U=V.parent;sys.path.insert(0,str(U))
from diagnostic_ledger import DiagnosticLedger
from coach import Coach
from action_runtime import ActionObserver
from candidate_compare import VerticalObserver
checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)));assert v,n

l=DiagnosticLedger();l.begin('public_fixture',1)
for i,stage in enumerate([0,0,1,1,2]):
 ob=dict(assessed_timestamp_us=1000000+i*100000,stage_index=stage,
  diagnostic=dict(category='motion_uncertain',code='action_uncertain',message='test'),
  action_direction_check=[dict(moving='L',axis=0,eligible=True,state=1,reason=None,probabilities=[.1,.8,.1],evidence={'trajectory':[123456]}),dict(moving='R',axis=0,eligible=False,state=-1,reason='insufficient_observed_geometry')],assessed_xy=[[234567,1]])
 l.observe(ob,{})
sn=l.snapshot()['session'];ck('stage_counts_and_first_times',sn['stage_counts']['0']['motion_uncertain']==2 and sn['stage_first_seen_ms']=={'0':0.,'1':200.,'2':400.})
ck('hold_vs_geometry_diagnosed',sn['subcause_counts']['stage0.motion.L.axis0.model_hold']==2 and sn['subcause_counts']['stage1.motion.R.axis0.insufficient_observed_geometry']==2)
ck('raw_outputs_not_retained',not any(q in json.dumps(sn)for q in ['probabilities','trajectory','123456','234567']))
l.observe(dict(assessed_timestamp_us=1600000,stage_index=1,diagnostic=dict(category='shape_uncertain',code='required_shape_uncertain'),shape_diagnostic={'R':dict(unavailable_fingers=['little'],unknown_fingers=['ring'],contrary_fingers=['index'])}),{})
ck('finger_causes_separate',all(k in l.snapshot()['session']['subcause_counts']for k in ['stage1.shape.R.little.unavailable','stage1.shape.R.ring.model_unresolved','stage1.shape.R.index.predicted_contrary']))
# Re-run old scalar ledger/service safeguards without rewriting old frozen reports.
source=(U/'usability_repair_20261116/check_ledger.py').read_text('utf-8')
source=source.replace("(V/'LEDGER_CHECK.json').write_text", "(Path(r'"+str(V)+"')/'LEDGER_REGRESSION.json').write_text")
exec(compile(source,str(U/'usability_repair_20261116/check_ledger.py'),'exec'),{'__file__':str(U/'usability_repair_20261116/check_ledger.py'),'__name__':'isolated_old_contracts'})
ck('previous_ledger_and_stale_service_guards',json.loads((V/'LEDGER_REGRESSION.json').read_text('utf-8'))['passed'])

# Check non-mutating deployed implementation against the tested scoped candidate.
a=ActionObserver(vertical_candidate_path=V/'MOTION_CANDIDATE.pt');b=VerticalObserver()
torch.manual_seed(42);h=torch.randn(1,8,64);a.h=h;b.h=h.clone()
for axis in [0,2,1,3,4]:
 for i in range(3):
  g=np.random.default_rng(i).normal(size=58).astype(np.float32)
  with torch.inference_mode():
   got=a._direction_logits(g,1,6,axis)
   old=b.net.motion_head
   if axis==0:b.net.motion_head=b.vertical
   try:expected=b.net.movement(b.h,torch.from_numpy(g[None]),torch.tensor([[0,1,6]],dtype=torch.long))
   finally:b.net.motion_head=old
  ck(f'axis{axis}_routing_equivalence_{i}',torch.allclose(got,expected,atol=1e-6))
ck('original_head_not_mutated',all(torch.equal(x,b.net.motion_head.state_dict()[k])for k,x in a.net.motion_head.state_dict().items()))
# Learned public integration on both original and re-extracted upper-body pixels.
c=Coach(action=a)
for name,path in [('native',U/'source/reference.npz'),('upper',V/'public_video_upper_body_640.npz')]:
 c.reset();d=DiagnosticLedger();d.begin('public_replay',1)
 with np.load(path)as z:r={k:z[k]for k in z.files}
 for i,t in enumerate(r['timestamp_us']):
  ob=c.update(r['xy_px'][i],r['score'][i],int(t),int(r['source_width']),int(r['source_height']));d.observe(ob,{})
 ck(name+'_actual_runtime_two_stages',c.completed==[True,True]);ck(name+'_new_stage_diagnostics',set(d.snapshot()['session']['stage_counts'])=={'0','1','2'})
 ck(name+'_shape_reason_available',any('.shape.'in k for k in d.snapshot()['session']['subcause_counts']))

report=dict(passed=all(x['passed']for x in checks),checks=checks,physical_camera=False,private_data=False,scope='Own synthetic/runtime verification, plus previously independently authored regression contracts; no new independent reviewer this turn.',hashes={f:hashlib.sha256((U/f).read_bytes()).hexdigest()for f in ['coach.py','action_runtime.py','diagnostic_ledger.py']})
(V/'INTEGRATION_CHECK.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=report['passed'],checks=len(checks))),flush=True)
