"""FIT-only fixed test: physical wrist translation with same-hand forearm context.
Neither clause identity, semantic movement, completion nor sign correctness.
"""
from pathlib import Path
import importlib.util,json,hashlib,sys,time
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent
P=O/'word100_feedback_recovery_20261113/motion_recovery'
A=O/'word100_actionable_feedback_20261113'
C=O/'word100_web_completion_graph_20261111/typed_motion_recovery_20261113/body_context_generalization'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
spec=importlib.util.spec_from_file_location('invariant_features_frozen',A/'motion_invariant.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def features(w,wv,e,arm,av):
 av=np.asarray(av,bool)&e
 ids=np.flatnonzero(e)
 return np.r_[old.invariant_features(w,wv,e),old.invariant_features(arm,av,e),av.sum()/max(1,e.sum()),float(av[ids[0]] and av[ids[-1]])].astype(np.float32)

def prepare():
 z=np.load(P/'PHYSICAL_INPUTS.npz');rows=read(P/'PHYSICAL_INDEX.json')['rows'];c=np.load(C/'INPUTS.npz')
 raw=c['absolute_context'].reshape(-1,3,10,5)[:,1];xx=[];arms=[];avs=[]
 for i,r in enumerate(rows):
  axis=2 if r['side']=='R' else 3;indices=r['source_indices'];positions=r['positions'];a=np.zeros((9,2),float);v=np.zeros(9,bool)
  q=raw[indices,axis];a[positions]=q[:,:2]*q[:,2:3];v[positions]=q[:,4]>.5
  # In the saved body contract: forearm y is sign-inverted, x is anatomical-side standardized.
  expected=raw[indices,0 if r['side']=='R' else 1,:2]
  assert np.allclose(z['q'][i,positions],expected,atol=1e-6)
  f=features(z['q'][i],z['valid'][i],z['exists'][i],a,v)
  rev=features(z['q'][i,::-1],z['valid'][i,::-1],z['exists'][i,::-1],a[::-1],v[::-1])
  assert np.array_equal(f,rev)
  assert np.isfinite(f).all() and f.shape==(26,)
  arms.append(a);avs.append(v);xx.append(f)
 x=np.stack(xx);folds=read(P/'PHYSICAL_FROZEN.json')['folds']
 for fold in folds:
  w=np.load(fold['weights'])['weights'];tr=w>0;te=z['person']==fold['person']
  assert not (tr&te).any() and not (tr&(z['y']<0)).any()
  assert set(r['source_sha256'] for r,t in zip(rows,tr) if t).isdisjoint({r['source_sha256'] for r,t in zip(rows,te) if t})
 np.savez_compressed(U/'ARM_INPUTS.npz',x=x,arm=np.stack(arms),arm_valid=np.stack(avs),eligible=z['eligible'],y=z['y'],person=z['person'])
 pins={str(p):sha(p) for p in [Path(__file__),U/'ARM_INPUTS.npz',P/'PHYSICAL_INPUTS.npz',P/'PHYSICAL_INDEX.json',P/'PHYSICAL_SOURCE_LABELS.json',P/'PHYSICAL_ADOPTION_CONTRACT.json',C/'INPUTS.npz',A/'motion_invariant.py']}
 for f in folds:
  for k in ['weights','normalization_rule']:pins[f[k]]=sha(f[k])
 cfg=dict(condition='wrist_plus_same_hand_forearm_invariant',seeds=[42,43,44],steps=1200,lr=.001,weight_decay=.001,hidden=16,folds=folds,pins=pins,
  target='window_contains_wrist_translation',no_direction_or_clause_prediction=True,training_scope='Previously exposed public FIT leave-record-out diagnostic',
  adoption='Every previous physical preservation stratum + no old20 correct source cell lost; report 12-feature comparison; no gate waiver',
  unknown_target_excluded=True,word_person_inputs=False,rule_and_old_predictions='Saved only, no rerun',web_applied=False)
 put(U/'ARM_FROZEN.json',cfg)
 put(U/'ARM_PRETRAIN_VERIFIED.json',dict(passed=True,exact_reverse_invariance=True,source_sha_disjoint=True,known_cells=228,unknown_cells=2,rows=len(rows),frozen_sha256=sha(U/'ARM_FROZEN.json')))
 print('Prepared fixed26-input candidate; source and exact reversal checks passed.',flush=True)

def train():
 import torch
 from torch import nn
 cfg=read(U/'ARM_FROZEN.json');ver=read(U/'ARM_PRETRAIN_VERIFIED.json');assert ver['passed'] and ver['frozen_sha256']==sha(U/'ARM_FROZEN.json')
 for p,h in cfg['pins'].items():assert sha(p)==h,p
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 z=np.load(U/'ARM_INPUTS.npz');x=z['x'];y=torch.tensor(z['y'],dtype=torch.float32);done=[]
 for fold in cfg['folds']:
  pid=fold['person'];w=np.load(fold['weights'])['weights'];tr=w>0;mean=x[tr].astype(float).mean(0);std=np.maximum(x[tr].astype(float).std(0),1e-4);xt=torch.tensor((x-mean)/std,dtype=torch.float32);wt=torch.tensor(w[tr])
  for seed in cfg['seeds']:
   d=U/'arm_runs'/f'{pid}_{seed}';d.mkdir(parents=True,exist_ok=True)
   if (d/'COMPLETE.json').exists():done.append(read(d/'COMPLETE.json'));continue
   assert not(d/'STARTED.json').exists(),'Audit interrupted run first'
   put(d/'STARTED.json',dict(person=pid,seed=seed,frozen_sha256=sha(U/'ARM_FROZEN.json')))
   start=time.perf_counter();torch.manual_seed(seed);net=nn.Sequential(nn.Linear(26,16),nn.GELU(),nn.Linear(16,1));opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);lossfn=nn.BCEWithLogitsLoss(reduction='none')
   for step in range(cfg['steps']):
    opt.zero_grad(set_to_none=True);loss=(lossfn(net(xt[tr]).squeeze(1),y[tr])*wt).sum();assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
   net.eval()
   with torch.inference_mode():out=net(xt).squeeze(1).numpy()
   np.savez_compressed(d/'predictions.npz',logits=out,held=z['person']==pid,eligible=z['eligible'])
   torch.save(dict(state_dict=net.state_dict(),mean=mean,std=std,task=cfg['target'],input_contract=cfg['condition']),d/'model.pt')
   v=dict(person=pid,seed=seed,loss=float(loss.detach()),seconds=time.perf_counter()-start,prediction_sha256=sha(d/'predictions.npz'),model_sha256=sha(d/'model.pt'))
   put(d/'COMPLETE.json',v);done.append(v);print(f'Arm context {len(done)}/18 complete',flush=True)
 put(U/'ARM_TRAINED.json',dict(runs=done,frozen_sha256=sha(U/'ARM_FROZEN.json')))

def compare():
 cfg=read(U/'ARM_FROZEN.json');z=np.load(U/'ARM_INPUTS.npz');rows=read(P/'PHYSICAL_INDEX.json')['rows'];gates=read(P/'PHYSICAL_ADOPTION_CONTRACT.json');ys=z['y'];ok=z['eligible'];stats=[];details=[];passed=True
 for seed in cfg['seeds']:
  pred={k:np.full(len(ys),-1,int) for k in ['candidate','previous','invariant','rule']}
  for f in cfg['folds']:
   pid=f['person'];ii=(z['person']==pid)&ok
   for name,folder in [('candidate',U/'arm_runs'),('previous',P/'physical_runs'),('invariant',A/'motion_runs')]:
    zz=np.load(folder/f'{pid}_{seed}'/'predictions.npz');assert np.array_equal(zz['held'],z['person']==pid);pred[name][ii]=zz['logits'][ii]>=0
    if name=='previous':pred['rule'][ii]=zz['rule_prediction'][ii]
  for view in ['full','upper','actual_crop']:
   for rev in [False,True]:
    base=np.array([r['view']==view and r['reverse']==rev for r in rows])&(ys>=0)
    groups={'all':base,'M':base&(ys==1),'H':base&(ys==0)}
    for side in ['L','R']:
     side_mask=np.array([r['side']==side for r in rows]);groups[side]=base&side_mask
     for name,value in [('M',1),('H',0)]:groups[side+name]=base&side_mask&(ys==value)
    cyc=np.array([r['word_id']=='WORD1567' and r['kind'].startswith('main_') for r in rows]);groups['cycle']=base&cyc
    for side in ['L','R']:groups['cycle'+side]=base&cyc&np.array([r['side']==side for r in rows])
    groups['short']=base&np.array([r['source_key'] in gates['short_change']['source_hand_keys'] for r in rows])
    for group,m in groups.items():
     counts={k:int((p[m]==ys[m]).sum()) for k,p in pred.items()};pass0=counts['candidate']>=max(counts['previous'],counts['rule']);passed&=pass0
     stats.append(dict(seed=seed,view=view,reverse=rev,group=group,total=int(m.sum()),**counts,passed=pass0))
  loss=(ys>=0)&(pred['previous']==ys)&(pred['candidate']!=ys);passed&=not loss.any()
  for i,r in enumerate(rows):details.append(dict(**r,seed=seed,truth=int(ys[i]),eligible=bool(ok[i]),**{k:int(p[i]) for k,p in pred.items()}))
 put(U/'ARM_COMPARISON.json',dict(adoption_gates_passed=bool(passed),statistics=stats,whole_G2_pass=False,web_applied=False))
 put(U/'ARM_PREDICTIONS.json',details)
 print(json.dumps(dict(adoption_gates_passed=bool(passed),primary=[r for r in stats if r['view']=='full' and not r['reverse'] and r['group']=='all']),ensure_ascii=False),flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
