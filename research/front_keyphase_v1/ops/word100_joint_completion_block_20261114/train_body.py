from pathlib import Path
import json,hashlib,time,collections
import numpy as np,torch
from body_core import BodyHeads,CLASSES,consensus
U=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text('utf-8-sig'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
cfg=read(U/'BODY_FROZEN.json');rows=read(U/'BODY_LABELS.json')['rows'];assert not(U/'BODY_TRAINED.json').exists()
for p,h in cfg['pins'].items():assert sha(Path(p))==h
torch.set_num_threads(2);data={};out=[];saved=[];(U/'body_models').mkdir()
for d in CLASSES:
 ids=[i for i,r in enumerate(rows)if r['domain']==d];fit=[i for i in ids if rows[i]['split']=='fit'and rows[i]['target']>=0 and rows[i]['available']]
 count=collections.Counter((rows[i]['target'],rows[i]['clip_id'])for i in fit);nc=collections.Counter(k[0]for k in count)
 weights=np.array([1/(nc[rows[i]['target']]*count[rows[i]['target'],rows[i]['clip_id']])for i in fit]);weights/=weights.sum()
 data[d]=(ids,fit,torch.tensor([rows[i]['feature']for i in fit]),torch.tensor([rows[i]['target']for i in fit]),torch.tensor(weights,dtype=torch.float32))
for seed in cfg['seeds']:
 torch.manual_seed(seed);m=BodyHeads(cfg['dims']);opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);start=time.perf_counter()
 for step in range(cfg['steps']):
  loss=sum((torch.nn.functional.cross_entropy(m(x,d),y,reduction='none')*w).sum()for d,(_,_,x,y,w)in data.items());opt.zero_grad();loss.backward();opt.step()
 m.eval();path=U/'body_models'/f'{seed}.pt';torch.save(m.state_dict(),path);pp=np.zeros((len(rows),2))
 with torch.no_grad():
  for d,(ids,_,_,_,_)in data.items():pp[ids]=m(torch.tensor([rows[i]['feature']for i in ids]),d).softmax(-1).numpy()
 np.save(U/'body_models'/f'{seed}_prob.npy',pp);saved.append(pp);out.append(dict(seed=seed,path=str(path),sha256=sha(path),seconds=time.perf_counter()-start,fit_indices={d:v[1]for d,v in data.items()}))
 print(json.dumps(dict(seed=seed,seconds=out[-1]['seconds'])),flush=True)
write('BODY_TRAINED.json',dict(rows=out));summary=[];rr=[]
for i,r in enumerate(rows):
 p=np.stack([x[i]for x in saved]);vote=p.argmax(-1);pred=CLASSES[r['domain']][int(vote[0])]if r['available']and(vote==vote[0]).all()and(p.max(-1)>=.8).all()else None
 rr.append(dict(index=i,clip_id=r['clip_id'],frame=r['frame'],hand=r['hand'],domain=r['domain'],split=r['split'],truth=r['label'],predicted=pred,correct=pred==r['label']if r['label']else None,probabilities=p.tolist()))
for s in ['fit','exposed_probe']:
 for d in CLASSES:
  rs=[r for r in rr if r['split']==s and r['domain']==d and r['truth']is not None];summary.append(dict(split=s,domain=d,n=len(rs),correct=sum(r['correct']for r in rs),unknown=sum(r['predicted']is None for r in rs),wrong=sum(r['predicted']is not None and not r['correct']for r in rs)))
write('BODY_EVALUATION.json',dict(rows=rr,summary=summary,preliminary_pass=all(r['correct']for r in rr if r['split']=='exposed_probe'and r['truth']is not None)))
print(json.dumps(dict(summary=summary,failures=[r for r in rr if r['split']=='exposed_probe'and r['truth']is not None and not r['correct']]),ensure_ascii=False))
