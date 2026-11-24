"""Same frozen linear architecture; add FIT-only visually reviewed boundary examples."""
from pathlib import Path
import json,hashlib,collections,time
import numpy as np,torch
from region_extent import RegionExtent,features
from body_core import face_geometry
U=Path(__file__).resolve().parent;O=U.parent;S=O/'word100_goal_state_learning_20261114';V=U/'extent_augmented';V.mkdir(exist_ok=False)
read=lambda p:json.loads(p.read_text('utf-8-sig'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(V/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
rows=read(U/'EXTENT_LABELS.json')['rows'];rev=read(U/'EXTENT_FIT_REVIEW.json')['rows'];arr=np.load(S/'source_arrays.npz');added=[]
for k,r in enumerate(rev):
 # Hand lies below/beside the facial silhouette at collar/upper-torso level.
 # The remaining seven borderline face/chin/shoulder examples stay unknown;
 # their ambiguity must not become forced training labels.
 label='upper_body' if k<=16 else None
 rec=read(O/'word100_rtmw_joint_learning_20261108'/'records'/f'{r["clip_id"]}.json')
 assert sha(Path(rec['raw_path']))==rec['raw_sha256']and sha(Path(rec['channel_path']))==rec['channel_sha256']
 raw=np.load(rec['raw_path']);ch=np.load(rec['channel_path']);face,ok=face_geometry(raw['xy_px'],ch['point_available'])
 f,ok=features(arr[r['clip_id']+'_x'],face,ok,r['hand']);i=r['frame']
 a=dict(r,domain='region',label=label,target=1 if label else-1,feature=f[i].tolist(),available=bool(ok[i]),array_index=i,origin='direct_RGB_FIT_boundary_review',review_index=k)
 added.append(a);rows.append(a)
write('LABELS.json',dict(rows=rows,added=added,unknown_excluded=True,reviewer='Codex source-image review, not independent KSL educator'))
cfg=read(U/'EXTENT_FROZEN.json');cfg.update(reason='FIT-only direct review adds near-chin upper-torso variation. Same architecture, optimizer, confidence and original probe labels.',pins={str(p):sha(p)for p in[U/'region_extent.py',V/'LABELS.json',U/'EXTENT_FIT_REVIEW.json',Path(__file__)]})
write('FROZEN.json',cfg)
fit=[i for i,r in enumerate(rows)if r['split']=='fit'and r['target']>=0 and r['available']]
count=collections.Counter((rows[i]['target'],rows[i]['clip_id'])for i in fit);nc=collections.Counter(k[0]for k in count)
w=np.array([1/(nc[rows[i]['target']]*count[rows[i]['target'],rows[i]['clip_id']])for i in fit]);w/=w.sum()
x=torch.tensor([r['feature']for r in rows]);y=torch.tensor([rows[i]['target']for i in fit]);w=torch.tensor(w,dtype=torch.float32)
torch.set_num_threads(2);saved=[];models=[]
for seed in cfg['seeds']:
 torch.manual_seed(seed);m=RegionExtent();opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);start=time.perf_counter()
 for _ in range(cfg['steps']):
  loss=(torch.nn.functional.cross_entropy(m(x[fit]),y,reduction='none')*w).sum();opt.zero_grad();loss.backward();opt.step()
 m.eval()
 with torch.no_grad():p=m(x).softmax(-1).numpy()
 path=V/f'{seed}.pt';torch.save(m.state_dict(),path);np.save(V/f'{seed}_prob.npy',p);saved.append(p)
 models.append(dict(seed=seed,path=str(path),sha256=sha(path),seconds=time.perf_counter()-start,fit_indices=fit))
write('TRAINED.json',dict(rows=models));pp=np.stack(saved);rr=[]
for i,r in enumerate(rows):
 p=pp[:,i];vv=p.argmax(-1);known=r['available']and(vv==vv[0]).all()and(p.max(-1)>=.8).all();pred=['face_front','upper_body'][int(vv[0])]if known else None
 rr.append(dict(index=i,clip_id=r['clip_id'],frame=r['frame'],hand=r['hand'],split=r['split'],truth=r['label'],predicted=pred,correct=pred==r['label']if r['label']else None))
summary=[]
for split in ['fit','exposed_probe']:
 rs=[r for r in rr if r['split']==split and r['truth']is not None];summary.append(dict(split=split,n=len(rs),correct=sum(r['correct']for r in rs),unknown=sum(r['predicted']is None for r in rs),wrong=sum(r['predicted']is not None and not r['correct']for r in rs)))
write('EVALUATION.json',dict(rows=rr,summary=summary,preliminary_pass=all(r['correct']for r in rr if r['split']=='exposed_probe'and r['truth']is not None)))
print(json.dumps(dict(summary=summary,failures=[r for r in rr if r['split']=='exposed_probe'and r['truth']is not None and not r['correct']]),ensure_ascii=False))
