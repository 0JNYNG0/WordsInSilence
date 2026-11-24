"""Matched-1200-step annotation-coverage comparison. Frozen upstream and old runs."""
from pathlib import Path
import sys,importlib.util,hashlib,time,traceback
import numpy as np,torch
W=Path(__file__).resolve().parent;U=W.parent;P=U/'episode_stage';O=U.parent
sys.path.insert(0,str(U))
from graph import read,write,stamp
sp=importlib.util.spec_from_file_location('prior_stage',P/'run.py');prior=importlib.util.module_from_spec(sp);sp.loader.exec_module(prior)
F=prior.FAMILIES
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def freeze():
 assert not (W/'FROZEN.json').exists();meta=read(W/'INPUTS_COMPLETE.json');old=read(P/'INPUTS_DENSE_COMPLETE.json');oldmap={(r['clip_id'],r['view']):r for r in old['rows']};src={r['clip_id']:r for r in read(U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json')['rows']};contrast={r['clip_id']:r for r in read(W/'RGB_CONTRAST_LABELS.json')['rows']};counts={k:np.zeros((5,3),int) for k in ['train','probe']};checks=[]
 for p,h in meta['pins'].items():assert sha(p)==h
 for r in meta['rows']:
  assert sha(r['path'])==r['sha256'] and sha(r['cache'])==r['cache_sha256'];a=np.load(r['path']);cache=np.load(r['cache'])['z'];assert np.array_equal(a['x'],cache[a['frame_id']]);assert np.isfinite(a['x']).all()
  sr=src[r['clip_id']];direct={f for s in sr['sheets'] for f in s['frames']};assert set(a['frame_id'])<=direct
  labels={(t['frame'],t['family']):t for t in r['labels']};oldr=oldmap[r['clip_id'],r['view']]
  for t in oldr['labels']:assert labels[t['frame'],t['family']]==t
  expected=np.full_like(a['y'],-1);pos={int(f):i for i,f in enumerate(a['frame_id'])}
  for t in r['labels']:expected[pos[t['frame']],F.index(t['family'])]=t['label'];counts[r['split']][F.index(t['family']),t['label']]+=1
  assert np.array_equal(expected,a['y']);ob=contrast[r['clip_id']]
  for t in r['new_labels']:
   assert t['label']==0 and t['family'] in [g['family'] for g in sr['groups']]
   if t['basis']=='different_reviewed_action_terminal_hold':assert t['frame'] in sr['hold_sample_frames'] and t['family']!=sr['groups'][-1]['family']
   else:
    role=t['basis'].removeprefix('direct_reviewed_');assert t['frame'] in ob[role]
    assert not any(g['onset'][0]<=t['frame']<=g['offset'][1] for g in sr['groups'])
  checks.append(dict(clip=r['clip_id'],view=r['view'],old_labels_preserved=len(oldr['labels']),new_labels=len(r['new_labels'])))
 train={r['source_sha256'] for r in meta['rows'] if r['split']=='train'};probe={r['source_sha256'] for r in meta['rows'] if r['split']=='probe'};assert len(train)==16 and len(probe)==8 and not train&probe
 supported=counts['train']>0;assert supported.tolist()==read(P/'FROZEN.json')['supported']
 write(W/'PRETRAIN_VERIFICATION.json',dict(at=stamp(),status='PASS_ROOT_SECOND_ARRAY_AND_LABEL_AUDIT',checks=checks,counts={k:v.tolist() for k,v in counts.items()},training_ready=True,independent_human=False))
 paths=[Path(__file__),W/'prepare.py',W/'INPUTS_COMPLETE.json',W/'RGB_CONTRAST_LABELS.json',W/'PRETRAIN_VERIFICATION.json',P/'run.py',P/'FROZEN.json',P/'TRAINING_COMPLETE.json',P/'EVALUATION_COMPLETE.json',P/'SCORES.json']
 pins={str(p):sha(p) for p in paths};pins.update(read(P/'FROZEN.json')['pins'])
 write(W/'FROZEN.json',dict(at=stamp(),seeds=[42,43,44],steps=1200,batch=64,lr=.0003,weight_decay=.01,gradient_clip=2,supported=supported.tolist(),train_counts=counts['train'].tolist(),pins=pins,selection='Final step1200 only, matching old optimizer step count. More supervised contexts is the intentional difference; not a claim of equal label exposure.',initialization='Same original seed role hidden/family weights as stage baseline, not previously fitted stage weights.',loss='Same masked class balanced CE, recompute class weights on expanded train only.',comparison='Old340 labels retained; new128 probe labels on same8 earlier-exposed FIT sources. Separate old and new. Existing82 public exposed DEV regression once. No final-test access.',pass_requirement='No claim of G2 pass from sampled classification; preserve old movement witnesses/normal spans and reduce wrong-action support before integration.',proposal='Exactly prior stage probability threshold0.5/body proposal; hold and hand-qualification separate. No new runtime postfilter or threshold tuning.',web_changed=False))
 print('Verified old labels preserved; froze matched1200-step coverage comparison', {k:int(v.sum()) for k,v in counts.items()})

def train():
 assert not (W/'TRAINING_COMPLETE.json').exists();cfg=read(W/'FROZEN.json')
 for p,h in cfg['pins'].items():assert sha(p)==h
 torch.set_num_threads(4);torch.set_num_interop_threads(1);rows=read(W/'INPUTS_COMPLETE.json')['rows'];sets={k:[r for r in rows if r['split']==k] for k in ['train','probe']}
 def load(k,key):return np.concatenate([np.load(r['path'])[key] for r in sets[k]])
 x=torch.tensor(load('train','x'));y=torch.tensor(load('train','y'));px=torch.tensor(load('probe','x'));py=load('probe','y');counts=np.array(cfg['train_counts']);weights=np.where(counts>0,counts.sum(1,keepdims=True)/(np.maximum(counts,1)*np.maximum((counts>0).sum(1,keepdims=True),1)),0);weights=torch.tensor(weights,dtype=torch.float32);done=[]
 oldrows=[r for r in read(P/'INPUTS_DENSE_COMPLETE.json')['rows'] if r['split']=='probe'];oldloc={};offset=0
 for r in oldrows:
  ids=np.load(r['path'])['frame_id']
  for i,f in enumerate(ids):oldloc[r['clip_id'],r['view'],int(f)]=offset+i
  offset+=len(ids)
 mapping=[];oldmask=np.zeros_like(py,dtype=bool);offset=0
 for r in sets['probe']:
  d=np.load(r['path']);oldr=next(a for a in oldrows if (a['clip_id'],a['view'])==(r['clip_id'],r['view']));labels={(t['frame'],t['family']) for t in oldr['labels']}
  for i,f in enumerate(d['frame_id']):
   mapping.append(oldloc.get((r['clip_id'],r['view'],int(f)),-1))
   for j,fam in enumerate(F):oldmask[offset+i,j]=(int(f),fam) in labels
  offset+=len(d['frame_id'])
 mapping=np.array(mapping);assert oldmask.sum()==340 and (py>=0).sum()==468
 for seed in cfg['seeds']:
  dst=W/'runs'/str(seed);assert not dst.exists();torch.manual_seed(seed);net=prior.StageNet(seed,cfg['supported']);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);start=time.perf_counter();hist=[];step=0;epoch=0
  while step<cfg['steps']:
   order=np.random.default_rng(seed*1000+epoch).permutation(len(x));epoch+=1
   for a in range(0,len(order),cfg['batch']):
    ii=order[a:a+cfg['batch']];logit=net(x[ii]);target=y[ii];mask=target>=0;safe=target.clamp_min(0);raw=torch.nn.functional.cross_entropy(logit.reshape(-1,3),safe.reshape(-1),reduction='none').reshape(-1,5);cw=weights[torch.arange(5)[None],safe];loss=(raw*cw*mask).sum()/(cw*mask).sum();assert torch.isfinite(loss)
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip']);opt.step();step+=1
    if step%100==0:hist.append(dict(step=step,loss=float(loss.detach())))
    if step==cfg['steps']:break
  dst.mkdir(parents=True);torch.save(net.state_dict(),dst/'model.pt');net.eval()
  with torch.inference_mode():p=net(px).softmax(-1).numpy()
  # Reuse all old saved probe outputs; infer only genuinely new sample inputs once.
  bp=np.empty_like(p);known=mapping>=0;bp[known]=np.load(P/'runs'/str(seed)/'PROBE.npz')['p'][mapping[known]]
  base=prior.StageNet(seed,cfg['supported']);base.load_state_dict(torch.load(P/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));base.eval()
  with torch.inference_mode():bp[~known]=base(px[~known]).softmax(-1).numpy()
  np.savez_compressed(dst/'PROBE.npz',y=py,p=p,baseline_p=bp,old_mask=oldmask)
  result={}
  for name,mask in [('old',oldmask),('added',(py>=0)&~oldmask),('all',py>=0)]:
   target=np.where(mask,py,-1);result[name]=dict(baseline=prior.metrics(target,bp),candidate=prior.metrics(target,p))
  rec=dict(seed=seed,path=str(dst/'model.pt'),sha256=sha(dst/'model.pt'),seconds=time.perf_counter()-start,steps=step,history=hist,probe=result,new_baseline_inputs=int((~known).sum()),old_saved_inputs_reused=int(known.sum()))
  write(dst/'TRAINING.json',rec);done.append(rec);print(seed,{k:(v['baseline']['correct'],v['candidate']['correct'],v['candidate']['total']) for k,v in result.items()},flush=True)
 write(W/'TRAINING_COMPLETE.json',dict(at=stamp(),runs=done,web_changed=False,private_used=False))

def evaluate():
 # Reuse the already-audited evaluator with only the isolated output directory changed.
 prior.W=W;prior.evaluate()

if __name__=='__main__':
 try:{'freeze':freeze,'train':train,'evaluate':evaluate}[sys.argv[1]]()
 except BaseException:
  (W/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
