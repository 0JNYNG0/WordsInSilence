from pathlib import Path
import json,hashlib,collections,time
import numpy as np,torch
from region_extent import features,RegionExtent
U=Path(__file__).resolve().parent;O=U.parent;S=O/'word100_goal_state_learning_20261114'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
assert not(U/'EXTENT_FROZEN.json').exists()
rows=[dict(r)for r in read(U/'BODY_LABELS.json')['rows']if r['domain']=='region']
arr=np.load(S/'source_arrays.npz');face=np.load(U/'face_arrays.npz')
for r in rows:
 cid=r['clip_id'];f,ok=features(arr[cid+'_x'],face[cid+'_face'],face[cid+'_face_ok'],r['hand']);i=r['array_index']
 r.update(feature=f[i].tolist(),available=bool(ok[i]))
write('EXTENT_LABELS.json',dict(rows=rows,labels_unchanged=True))
cfg=dict(seeds=[42,43,44],steps=1200,lr=.015,weight_decay=.01,confidence=.8,
 rationale='Whole hand to observed face extent, reducing palm-height ambiguity. Linear model limits fitting correlated source poses. Forearm candidate rejected; original branch preserved.',
 gate='All 27 original known probe region labels correct; no unknown/wrong replacement. No new probe labels. FIT records02-07 only.',
 pins={str(p):sha(p)for p in[Path(__file__),U/'region_extent.py',U/'EXTENT_LABELS.json',U/'face_arrays.npz']})
write('EXTENT_FROZEN.json',cfg)
fit=[i for i,r in enumerate(rows)if r['split']=='fit'and r['target']>=0 and r['available']]
count=collections.Counter((rows[i]['target'],rows[i]['clip_id'])for i in fit);nc=collections.Counter(k[0]for k in count)
w=np.array([1/(nc[rows[i]['target']]*count[rows[i]['target'],rows[i]['clip_id']])for i in fit]);w/=w.sum()
x=torch.tensor([r['feature']for r in rows]);y=torch.tensor([rows[i]['target']for i in fit]);w=torch.tensor(w,dtype=torch.float32)
torch.set_num_threads(2);saved=[];models=[];(U/'extent_models').mkdir()
for seed in cfg['seeds']:
 torch.manual_seed(seed);m=RegionExtent();opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);start=time.perf_counter()
 for _ in range(cfg['steps']):
  loss=(torch.nn.functional.cross_entropy(m(x[fit]),y,reduction='none')*w).sum();opt.zero_grad();loss.backward();opt.step()
 m.eval()
 with torch.no_grad():p=m(x).softmax(-1).numpy()
 path=U/'extent_models'/f'{seed}.pt';torch.save(m.state_dict(),path);np.save(U/'extent_models'/f'{seed}_prob.npy',p);saved.append(p)
 models.append(dict(seed=seed,path=str(path),sha256=sha(path),seconds=time.perf_counter()-start,fit_indices=fit))
write('EXTENT_TRAINED.json',dict(rows=models))
pp=np.stack(saved);rr=[]
for i,r in enumerate(rows):
 p=pp[:,i];vv=p.argmax(-1);known=r['available']and(vv==vv[0]).all()and(p.max(-1)>=.8).all();pred=['face_front','upper_body'][int(vv[0])]if known else None
 rr.append(dict(index=i,clip_id=r['clip_id'],frame=r['frame'],hand=r['hand'],split=r['split'],truth=r['label'],predicted=pred,correct=pred==r['label']if r['label']else None))
summary=[]
for split in ['fit','exposed_probe']:
 rs=[r for r in rr if r['split']==split and r['truth']is not None];summary.append(dict(split=split,n=len(rs),correct=sum(r['correct']for r in rs),unknown=sum(r['predicted']is None for r in rs),wrong=sum(r['predicted']is not None and not r['correct']for r in rs)))
write('EXTENT_EVALUATION.json',dict(rows=rr,summary=summary,preliminary_pass=all(r['correct']for r in rr if r['split']=='exposed_probe'and r['truth']is not None)))
print(json.dumps(dict(summary=summary,failures=[r for r in rr if r['split']=='exposed_probe'and r['truth']is not None and not r['correct']]),ensure_ascii=False))
