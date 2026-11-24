from pathlib import Path
import json,hashlib,time,collections
import numpy as np,torch
from sequence_core import SequenceHead,decide,CLASSES
U=Path(__file__).resolve().parent;P=U.parent/'word100_transition_observation_audit_20261114'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
cfg=read(U/'FROZEN.json');meta=read(U/'LABELS.json');assert not(U/'TRAINING_COMPLETE.json').exists()
for p,h in cfg['pins'].items():assert sha(Path(p))==h
for p,h in meta['pins'].items():assert sha(Path(p))==h
write('EXECUTION_FROZEN.json',dict(path=str(Path(__file__)),sha256=sha(Path(__file__))))
z=np.load(U/'training_arrays.npz');x=torch.tensor(z['sequence']);px=torch.tensor(z['point']);rows=meta['rows'];pr=meta['point_rows']
def targets(rr):
 fit=[i for i,r in enumerate(rr)if r['split']=='fit'and r['available']]
 n=collections.Counter((rr[i]['target'],rr[i]['clip_id'])for i in fit);c=collections.Counter(k[0]for k in n)
 w=np.array([1/(len(c)*c[rr[i]['target']]*n[rr[i]['target'],rr[i]['clip_id']])for i in fit]);w/=w.sum()
 return fit,torch.tensor([rr[i]['target']for i in fit]),torch.tensor(w,dtype=torch.float32)
fi,y,w=targets(rows);pi,py,pw=targets(pr);torch.set_num_threads(2);probs=[];pose=[];trained=[];(U/'models').mkdir(exist_ok=True)
for seed in cfg['seeds']:
 path=U/'models'/f'{seed}.pt';assert not path.exists();torch.manual_seed(seed);m=SequenceHead(x.shape[-1]);initial={k:v.clone()for k,v in m.state_dict().items()}
 opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);st=time.perf_counter()
 for step in range(cfg['steps']):
  change,_=m(x[fi]);point=m.pose(m.embed(px[pi]))
  loss=(torch.nn.functional.cross_entropy(change,y,reduction='none')*w).sum()+cfg['aux_weight']*(torch.nn.functional.cross_entropy(point,py,reduction='none')*pw).sum()
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
 m.eval();torch.save(m.state_dict(),path)
 with torch.no_grad():a,_=m(x);pp=a.softmax(-1).numpy();qp=m.pose(m.embed(px)).softmax(-1).numpy()
 probs.append(pp);pose.append(qp);np.savez_compressed(U/'models'/f'{seed}_predictions.npz',change=pp,pose=qp)
 trained.append(dict(seed=seed,path=str(path),sha256=sha(path),seconds=time.perf_counter()-st,fit_indices=fi,point_fit_indices=pi,
   weights_changed=any(not torch.equal(initial[k],v)for k,v in m.state_dict().items()),parameters=sum(p.numel()for p in m.parameters())))
 print(json.dumps(dict(seed=seed,training_finished=True,seconds=trained[-1]['seconds'])),flush=True)
write('TRAINING_COMPLETE.json',dict(rows=trained));old={r['index']:r for r in read(P/'EVALUATION.json')['rows']};out=[]
for i,r in enumerate(rows):
 pp=np.stack([p[i]for p in probs]);pred=decide(pp,r['available']);out.append(dict(**r,new_prediction=pred,old_prediction=old[i]['prediction'],
   new_correct=pred==CLASSES[r['target']],old_correct=old[i]['correct'],probabilities=pp.tolist()))
probe=[r for r in out if r['split']=='exposed_probe'];summary=dict(probe_n=len(probe),old_correct=sum(r['old_correct']for r in probe),new_correct=sum(r['new_correct']for r in probe),
 errors=[dict(index=r['index'],clip_id=r['clip_id'],hand=r['hand'],frames=r['source_frames'],truth=CLASSES[r['target']],prediction=r['new_prediction'])for r in probe if not r['new_correct']],
 preliminary_gate=all(r['new_correct']for r in probe),whole_G2_passed=False)
write('EVALUATION.json',dict(rows=out,summary=summary,old_endpoint_unchanged=True,whole_completion_claim=False))
print(json.dumps(summary,ensure_ascii=False))
