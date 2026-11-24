from split_model import *
import collections
assert not(U/'TRAINED.json').exists()
allnew=read(U/'LABELS.json')['rows'];new=[r for r in allnew if r['role']=='fit'];oldrows=read(P/'INPUTS.json')['rows'];orig=np.load(P/'INPUTS.npz');oldids=[i for i,r in enumerate(oldrows)if r['role']=='fit'];qm={}
for r in read(Q/'FIT_LABELS_V2.json')['rows']+new:qm[r['clip_id'],r['hand'],r['frame']]=r
sx=[];sy=[];meta=[];masks=[];cache={}
def seq(r):
 key=r['channel_path'],r['hand']
 if key not in cache:
  x,fid=load_channels(r['channel_path']);s,a=shape_features(x,r['hand']);q,_=quality_features(x,r['hand']);cache[key]=(s,q,a,fid)
 return cache[key]
for i in oldids:
 r=oldrows[i];y=orig['y'][i].copy();mask=(y>=0)&orig['availability'][i];key=r['clip_id'],r['side'],r['frame'];review=qm.get(key)
 if review:mask&=np.array(review['quality'])==1
 sx.append(orig['x'][i]);sy.append(y);masks.append(mask);meta.append(r|dict(origin='original_fit',source_index=i))
seen={(r['clip_id'],r['side'],r['frame'])for r in meta}
for r in new:
 key=r['clip_id'],r['hand'],r['frame'];s,q,a,fid=seq(r);j=int(np.flatnonzero(fid==r['frame'])[0]);y=np.array(r['shape']);mask=(y>=0)&(np.array(r['quality'])==1)&a[j]
 if key in seen:
  i=next(i for i,m in enumerate(meta)if(m['clip_id'],m['side'],m['frame'])==key)
  # Explicit latest reviewed source targets replace only resolved cells, never multiply duplicates.
  known=y>=0;sy[i][known]=y[known];masks[i][known]=mask[known];continue
 sx.append(windows(s,[j])[0]);sy.append(y);masks.append(mask);meta.append(r|dict(side=r['hand'],origin='new_fit'))
qx=[];qy=[];qmask=[];qmeta=[]
for r in qm.values():
 assert r['person']<=9 and r['role']=='fit'
 s,q,a,fid=seq(r);j=int(np.flatnonzero(fid==r['frame'])[0]);y=np.array(r['quality']);qx.append(windows(q,[j])[0]);qy.append(y);qmask.append((y>=0)&a[j]);qmeta.append(r)
sx=np.array(sx,np.float32);sy=np.array(sy);masks=np.array(masks);qx=np.array(qx,np.float32);qy=np.array(qy);qmask=np.array(qmask)
mu=qx.mean((0,1,2));sd=qx.std((0,1,2)).clip(.05)
np.savez_compressed(U/'TRAIN_INPUTS.npz',shape_x=sx,shape_y=sy,shape_mask=masks,quality_x=qx,quality_y=qy,quality_mask=qmask,quality_mean=mu,quality_std=sd)
write(U/'TRAIN_INPUTS.json',dict(shape_rows=meta,quality_rows=qmeta,old_fit=len(oldids),new_fit=len(meta)-len(oldids),shape_known=int(masks.sum()),quality_known=int(qmask.sum()),quality_bad=int(((qy==0)&qmask).sum()),quality_bad_by_finger=((qy==0)&qmask).sum(0),no_dev_inputs=True,mask_rule='new wrong/unknown chains masked; reviewed matching original cells masked too; other original manually labeled FIT retained'))
pins={str(p):sha(p)for p in [U/'LABELS.json',U/'TRAIN_INPUTS.npz',U/'split_model.py',U/'PLAN_CLARIFICATION.json']+[P/'shared_finger_chain'/f'{s}.pt'for s in [42,43,44]]}
cfg=dict(seeds=[42,43,44],shape_steps=250,quality_steps=350,shape_lr=.0002,quality_lr=.0005,weight_decay=.0001,confidence=.85,shape_new_weight=.3,distillation_weight=.3,shape_initialization='admitted P shared-finger Conv+GRU per seed, original normalization',quality_initialization='new independent shared-finger Conv+GRU; no posture gradient sharing',selection='three-seed consensus; fixed steps/no early stopping; diagnostic10 plus oldDEV before reserve16',candidates=['shape_only','quality_only','combined','compatible_shape'],pins=pins)
write(U/'TRAIN_CONFIG.json',cfg);print(json.dumps({k:v for k,v in read(U/'TRAIN_INPUTS.json').items()if not k.endswith('rows')}))
