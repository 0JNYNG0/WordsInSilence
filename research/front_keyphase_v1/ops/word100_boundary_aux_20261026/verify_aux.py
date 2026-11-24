"""Separate saved-checkpoint/cache/trace/score arithmetic audit. No training/inference."""
from pathlib import Path
from collections import Counter
import json,sys,hashlib,math
import numpy as np
import torch
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
R=Path('C:/MyUnityProjects/WordsInSilence');O=R/'research/front_keyphase_v1/ops'
U=O/'word100_boundary_aux_20261026';B=O/'word100_body_boundary_20261017';T=O/'word100_frozen_c_temporal_20261019'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=read(U/'FROZEN.json');assert (U/'DONE.json').exists() and not (U/'RUNNING.json').exists()
for p,h in f['pins'].items():assert sha(p)==h,p
ctx=read(B/'training_CONTEXT.json');tr=[r for r in ctx['ordered_rows'] if r['role']=='train_candidate'];dv=[r for r in ctx['ordered_rows'] if r['role']=='development_only']
train_ids={r['clip_id'] for r in tr};assert len(train_ids)==899 and len(dv)==300
seeds=f['seeds'];vv=('masked','sensor');kk=('clean','start_spike','end_spike','both_spikes')
totals={v:{k:dict(before=0,after=0,before_top5=0,after_top5=0,gains=0,losses=0,n=0) for k in kk} for v in vv}
seed_reports=[];all_gains=[];all_losses=[];trace_batches=0;score_rows=0;scalar_checks=0;maxerr=0.
for seed in seeds:
 r=read(U/f'BA_{seed}_RESULT.json');cm=read(U/f'features_{seed}.json')
 assert sha(U/f'features_{seed}.npz')==cm['cache_sha256']==r['cache_sha256']
 assert cm['protocol_sha256']==r['protocol_sha256']==sha(U/'FROZEN.json')
 assert cm['train_ids']==[q['clip_id'] for q in tr] and cm['dev_ids']==[q['clip_id'] for q in dv]
 assert cm['frozen_state_before']==cm['frozen_state_after'] and cm['cache_model_probe_error']==0
 assert sha(U/f'BA_{seed}_best.pt')==r['checkpoint_sha256']
 ck=torch.load(U/f'BA_{seed}_best.pt',map_location='cpu',weights_only=True)
 old=torch.load(T/f'T3F_{seed}_best.pt',map_location='cpu',weights_only=True)
 assert all(torch.equal(v,ck['state_dict'][k]) for k,v in old['state_dict'].items() if k!='phase_weight')
 assert ck['epoch']==r['selected_epoch'] and ck['protocol_sha256']==r['protocol_sha256']
 w=ck['state_dict']['phase_weight'].numpy().astype(np.float64)
 with np.load(U/f'features_{seed}.npz',allow_pickle=False) as z:dev=z['dev']
 history=read(U/f'BA_{seed}_history.json');best_epoch=0;best_ce=r['initial_ce']
 for i,h in enumerate(history,1):
  assert h['epoch']==i
  if h['clean_dev_ce']<best_ce-1e-6:best_epoch=i;best_ce=h['clean_dev_ce']
  assert h['selected_epoch']==best_epoch
 assert len(history)==40 and best_epoch==r['selected_epoch'] and abs(best_ce-r['selected_ce'])<1e-10
 trace=U/f'BA_{seed}_TRACE.jsonl';assert sha(trace)==r['trace_sha256'];epochrows={i:[] for i in range(1,41)}
 for line in trace.read_text(encoding='utf-8').splitlines():
  q=json.loads(line);e=q['epoch'];assert 1<=e<=40
  for cid,v,t,kind in q['clips']:
   assert cid in train_ids
   temporal='x_context' if (e+int(hashlib.sha256(cid.encode()).hexdigest(),16)%2)%2==1 else 'x_whole'
   spatial='sensor' if ((e-1)//2+int(hashlib.sha256(('sensor_view:'+cid).encode()).hexdigest(),16)%2)%2==1 else 'masked'
   ki=1+int(hashlib.sha256(f'{seed}:{e}:{cid}'.encode()).hexdigest(),16)%3
   assert (v,t,kind)==(spatial,temporal,kk[ki]);epochrows[e].append(cid)
  trace_batches+=1
 for e,ids in epochrows.items():
  assert len(ids)==899 and set(ids)==train_ids
  expected=[tr[i]['clip_id'] for i in np.random.default_rng(seed+e).permutation(899)]
  assert ids==expected
 assert r['batches']==1160 and r['original_clip_exposures']==35960 and r['training_examples']==71920
 byview={};ces=[]
 for vi,v in enumerate(vv):
  byview[v]={}
  for ki,k in enumerate(kk):
   pp=r['predictions'][v][k];base={q['clip_id']:q for q in r['baseline'][v][k]}
   assert len(pp)==300 and len(base)==300
   expected=dev[:,vi,0,ki,:100].astype(np.float64)+dev[:,vi,0,ki,100:].astype(np.float64)@w.T
   predbyid={q['clip_id']:q for q in pp};hit=0;oldhit=0;byid={}
   for i,row in enumerate(dv):
    p=predbyid[row['clip_id']];before=base[row['clip_id']];log=np.array(p['logits']);err=float(np.abs(expected[i]-log).max());maxerr=max(maxerr,err)
    assert np.allclose(log,expected[i],atol=5e-5,rtol=5e-5)
    order=sorted(range(100),key=lambda j:(-float(log[j]),j));ok=order[0]==row['label_index'];top5=row['label_index'] in order[:5]
    assert (p['prediction'],p['correct'],p['top5_correct'])==(order[0],ok,top5)
    m=float(log.max());ce=m+math.log(sum(math.exp(float(x)-m) for x in log))-float(log[row['label_index']])
    assert abs(ce-p['cross_entropy'])<1e-6
    if k=='clean':ces.append(ce)
    item=totals[v][k];item['n']+=1;item['before']+=before['correct'];item['after']+=ok
    item['before_top5']+=before['top5_correct'];item['after_top5']+=top5
    gain=ok and not before['correct'];loss=not ok and before['correct'];item['gains']+=gain;item['losses']+=loss
    if gain:all_gains.append(dict(seed=seed,view=v,kind=k,clip_id=row['clip_id']))
    if loss:all_losses.append(dict(seed=seed,view=v,kind=k,clip_id=row['clip_id']))
    hit+=ok;oldhit+=before['correct'];bb=byid.setdefault(str(row['signer_id']),dict(before=0,after=0,n=0));bb['before']+=before['correct'];bb['after']+=ok;bb['n']+=1
    if i in (0,149,299):
     for cl in (0,row['label_index'],99):
      scalar=float(dev[i,vi,0,ki,cl])+sum(float(dev[i,vi,0,ki,100+j])*float(w[cl,j]) for j in range(96))
      assert abs(scalar-expected[i,cl])<1e-10;scalar_checks+=1
    score_rows+=1
   byview[v][k]=dict(before=oldhit,after=hit,delta_pp=(hit-oldhit)/3,by_signer=byid)
 assert abs(sum(ces)/len(ces)-r['selected_ce'])<2e-6
 seed_reports.append(dict(seed=seed,selected_epoch=r['selected_epoch'],scores=byview))
for v in vv:
 for k,m in totals[v].items():
  m.update(before_percent=100*m['before']/m['n'],after_percent=100*m['after']/m['n'],delta_pp=100*(m['after']-m['before'])/m['n'])
accept={}
accept['normal_top1_no_regression']=all(totals[v]['clean']['after']>=totals[v]['clean']['before'] for v in vv)
accept['normal_top5_no_regression']=all(totals[v]['clean']['after_top5']>=totals[v]['clean']['before_top5'] for v in vv)
deltas={v:sum(totals[v][k]['after']-totals[v][k]['before'] for k in kk[1:]) for v in vv}
accept['synthetic_improves_without_other_view_regression']=any(d>0 for d in deltas.values()) and all(d>=0 for d in deltas.values())
accept['frozen_C_preserved']=True
out=dict(status='PASS',candidate_acceptance=accept,candidate_pass=all(accept.values()),totals=totals,
 seeds=seed_reports,gains=all_gains,losses=all_losses,synthetic_net=deltas,
 trace_batches_verified=trace_batches,original_clip_exposures=107880,training_examples=215760,
 scores_verified=score_rows,scalar_logit_checks=scalar_checks,maximum_logit_float64_difference=maxerr,
 unchanged_pin_files=len(f['pins']),implementer='Codex',verifier='Codex separate arithmetic; not independent human validation',
 limits=['Repeated public DEV; synthetic one-point edge stress, not measured webcam accuracy.',
         'Normal coordinates/crossing input unchanged by construction; this is not hand/finger recovery.',
         'No deployment, user/mentor input, fresh199/known200 access or new independent test.'])
with (U/'VERIFIED.json').open('x',encoding='utf-8') as fp:json.dump(out,fp,ensure_ascii=False,indent=2)
print(json.dumps({k:out[k] for k in ('status','candidate_acceptance','candidate_pass','totals','synthetic_net','scores_verified','trace_batches_verified')},ensure_ascii=False))
