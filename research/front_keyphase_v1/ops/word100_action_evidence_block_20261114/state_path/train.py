from pathlib import Path
import json,hashlib,time,collections
import numpy as np,torch
from core import StatePath
D=Path(__file__).resolve().parent;U=D.parent;P=U.parent/'word100_transition_observation_audit_20261114'
read=lambda p:json.loads(p.read_text('utf-8-sig'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(D/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
assert not(D/'TRAINING_COMPLETE.json').exists()
cfg=dict(seeds=[42,43,44],steps=1000,lr=.003,weight_decay=.01,pose_weight=2.,confidence=.8,
 fixed_gate='All original24 change conditions correct and original endpoints untouched; subsequent path/partial/normal controls required',
 no_checkpoint_selection=True,pins={str(p):sha(p)for p in [D/'arrays.npz',D/'LABELS.json',D/'PLAN.json',D/'core.py',Path(__file__)]})
write('FROZEN.json',cfg);mta=read(D/'LABELS.json');rows=mta['rows'];pr=mta['point_rows'];z=np.load(D/'arrays.npz');x=torch.tensor(z['pairs']);px=torch.tensor(z['points']);torch.set_num_threads(2)
def fit(rr):
 ids=[i for i,r in enumerate(rr)if r['split']=='fit'and r['available']];cnt=collections.Counter((rr[i]['target'],rr[i]['clip_id'])for i in ids);nc=collections.Counter(k[0]for k in cnt)
 w=np.array([1/(len(nc)*nc[rr[i]['target']]*cnt[rr[i]['target'],rr[i]['clip_id']])for i in ids]);w/=w.sum()
 return ids,torch.tensor([rr[i]['target']for i in ids]),torch.tensor(w,dtype=torch.float32)
fi,y,w=fit(rows);pi,py,pw=fit(pr);py=torch.tensor([0.,1.,.5])[py];probs=[];states=[];trained=[];(D/'models').mkdir(exist_ok=True)
for seed in cfg['seeds']:
 path=D/'models'/f'{seed}.pt';assert not path.exists();torch.manual_seed(seed);m=StatePath();opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);st=time.perf_counter()
 for step in range(cfg['steps']):
  logits,_=m(x[fi]);pose=m.position(px[pi]);loss=(torch.nn.functional.cross_entropy(logits,y,reduction='none')*w).sum()+cfg['pose_weight']*((pose-py)**2*pw).sum()
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
 m.eval();torch.save(m.state_dict(),path)
 with torch.no_grad():
  logits,pp=m(x);pred=logits.softmax(-1).numpy();pp=pp.numpy();point=m.position(px).numpy();seq={f'path{i}':m.position(torch.tensor(z[f'path{i}'])).numpy()for i in range(len(rows))}
 probs.append(pred);states.append(pp);np.savez_compressed(D/'models'/f'{seed}_predictions.npz',change=pred,endpoint_positions=pp,point_positions=point,**seq)
 trained.append(dict(seed=seed,path=str(path),sha256=sha(path),seconds=time.perf_counter()-st,fit_indices=fi,point_fit_indices=pi,parameters=sum(p.numel()for p in m.parameters()),
  learned_scale=float(torch.nn.functional.softplus(m.scale_raw).detach()),learned_margin=float(torch.sigmoid(m.margin_raw).detach())))
 print(json.dumps(dict(seed=seed,done=True,seconds=trained[-1]['seconds'])),flush=True)
write('TRAINING_COMPLETE.json',dict(rows=trained));old={r['index']:r for r in read(P/'EVALUATION.json')['rows']};out=[];names=['opening','shape_maintained','closing']
for i,r in enumerate(rows):
 pp=np.stack([p[i]for p in probs]);vote=pp.argmax(-1);pred=names[int(vote[0])]if r['available']and len(set(vote.tolist()))==1 and pp.max(-1).min()>=.8 else None
 out.append(dict(**r,new_prediction=pred,old_prediction=old[i]['prediction'],new_correct=pred==names[r['target']],old_correct=old[i]['correct'],probabilities=pp.tolist(),positions=[p[i].tolist()for p in states]))
probe=[r for r in out if r['split']=='exposed_probe'];summary=dict(probe_n=len(probe),old_correct=sum(r['old_correct']for r in probe),new_correct=sum(r['new_correct']for r in probe),
 errors=[dict(index=r['index'],clip_id=r['clip_id'],hand=r['hand'],frames=r['source_frames'],truth=names[r['target']],prediction=r['new_prediction'])for r in probe if not r['new_correct']],
 preliminary_gate=all(r['new_correct']for r in probe),whole_G2_passed=False)
write('EVALUATION.json',dict(rows=out,summary=summary,old_endpoint_unchanged=True,whole_completion_claim=False));print(json.dumps(summary,ensure_ascii=False))
