from pathlib import Path
import json,hashlib,time,collections,sys
import numpy as np,torch
from metric_core import MetricHead
from sequence_core import decide,CLASSES
D=Path(__file__).resolve().parent;U=D.parent;P=U.parent/'word100_transition_observation_audit_20261114'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(D/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
assert not(D/'TRAINING_COMPLETE.json').exists()
cfg=dict(seeds=[42,43,44],steps=1000,lr=.003,weight_decay=.01,aux_weight=.5,contrast_weight=.1,
 point_contrast_temperature=.15,confidence=.8,no_checkpoint_selection=True,
 pass_condition='All original24 probe conditions correct; original endpoint unchanged; then G2 controls',
 pins={str(p):sha(p)for p in [D/'LABELS.json',D/'point_features.npy',D/'metric_core.py',Path(__file__),U/'sequence_core.py',U/'training_arrays.npz',U/'LABELS.json']})
write('FROZEN.json',cfg)
meta=read(U/'LABELS.json');rows=meta['rows'];pr=read(D/'LABELS.json')['rows'];x=torch.tensor(np.load(U/'training_arrays.npz')['sequence']);px=torch.tensor(np.load(D/'point_features.npy'))
def targets(rr):
 fit=[i for i,r in enumerate(rr)if r['split']=='fit'and r['available']]
 n=collections.Counter((rr[i]['target'],rr[i]['clip_id'])for i in fit);c=collections.Counter(k[0]for k in n)
 w=np.array([1/(len(c)*c[rr[i]['target']]*n[rr[i]['target'],rr[i]['clip_id']])for i in fit]);w/=w.sum()
 return fit,torch.tensor([rr[i]['target']for i in fit]),torch.tensor(w,dtype=torch.float32)
fi,y,w=targets(rows);pi,py,pw=targets(pr);torch.set_num_threads(2);ppall=[];trained=[];(D/'models').mkdir(exist_ok=True)
for seed in cfg['seeds']:
 path=D/'models'/f'{seed}.pt';assert not path.exists();torch.manual_seed(seed);m=MetricHead(x.shape[-1]);opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);st=time.perf_counter()
 for step in range(cfg['steps']):
  out,_=m(x[fi]);pose=m.pose(m.embed(px[pi]));loss=(torch.nn.functional.cross_entropy(out,y,reduction='none')*w).sum()+cfg['aux_weight']*(torch.nn.functional.cross_entropy(pose,py,reduction='none')*pw).sum()+cfg['contrast_weight']*m.contrast(px[pi],py).mean()
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
 m.eval();torch.save(m.state_dict(),path)
 with torch.no_grad():pred,_=m(x);pp=pred.softmax(-1).numpy();pose=m.pose(m.embed(px)).softmax(-1).numpy()
 ppall.append(pp);np.savez_compressed(D/'models'/f'{seed}_predictions.npz',change=pp,pose=pose)
 trained.append(dict(seed=seed,path=str(path),sha256=sha(path),seconds=time.perf_counter()-st,fit_indices=fi,point_fit_indices=pi,parameters=sum(p.numel()for p in m.parameters())))
 print(json.dumps(dict(seed=seed,done=True,seconds=trained[-1]['seconds'])),flush=True)
write('TRAINING_COMPLETE.json',dict(rows=trained));old={r['index']:r for r in read(P/'EVALUATION.json')['rows']};out=[]
for i,r in enumerate(rows):
 pp=np.stack([p[i]for p in ppall]);pred=decide(pp,r['available']);out.append(dict(**r,new_prediction=pred,old_prediction=old[i]['prediction'],
   new_correct=pred==CLASSES[r['target']],old_correct=old[i]['correct'],probabilities=pp.tolist()))
probe=[r for r in out if r['split']=='exposed_probe'];summary=dict(probe_n=len(probe),old_correct=sum(r['old_correct']for r in probe),new_correct=sum(r['new_correct']for r in probe),
 errors=[dict(index=r['index'],clip_id=r['clip_id'],hand=r['hand'],frames=r['source_frames'],truth=CLASSES[r['target']],prediction=r['new_prediction'])for r in probe if not r['new_correct']],
 preliminary_gate=all(r['new_correct']for r in probe),whole_G2_passed=False)
write('EVALUATION.json',dict(rows=out,summary=summary,old_endpoint_unchanged=True,whole_completion_claim=False));print(json.dumps(summary,ensure_ascii=False))
