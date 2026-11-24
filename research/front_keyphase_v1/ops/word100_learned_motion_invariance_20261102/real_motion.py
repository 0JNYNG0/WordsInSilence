import sys,numpy as np,torch
from prepare_real_fit import N,M,P,V,O,read
sys.path.insert(0,str(M))
from motion_model import MotionNet,features,normalize,predict,batch,public_arrays
def filter_input(x,condition):
 if condition=='real_variants':return x
 assert condition=='real_variants_motion_inputs';x=np.array(x,copy=True);keep=read(N/'TRAINING_CONTRACT.json')['motion_value_indices'];drop=np.setdiff1d(np.arange(73),keep);x[...,drop]=0;x[...,drop+73]=0;return x
def records_and_sequences():
 labels=[r for r in read(M/'MOTION_TARGETS.json')['rows'] if r['role']=='fit'];old={r['clip_id']:r for r in read(P/'INPUT_MANIFEST.json')['rows']};stats=read(M/'NORMALIZATION.json')
 arrays={};variants=[]
 for cid in sorted({r['clip_id'] for r in labels}):
  a,z,m=public_arrays(old[cid]);arrays[cid+'_cached']=(a,z,m);variants.append(dict(key=cid+'_cached',clip_id=cid,kind='cached',source_frame_indices=list(map(int,a['frame_id']))))
 for r in read(N/'INPUT_MANIFEST.json')['rows']:
  key=r['clip_id']+'_'+r['kind'];a=dict(np.load(r['input']));z,m=features(a,a['retained_probability'],a['relation_probability']);arrays[key]=(a,z,m);variants.append(dict(key=key,**r))
 sequences=[];records=[];ys=[];masks=[];pair_groups={}
 def add(l,r,q,y,m,kind,indices=None):
  a,z,valid=arrays[r['key']];x=normalize(z[q],valid[q],stats)
  if indices is not None:x=x[indices]
  sequences.append(np.ascontiguousarray(x));ys.append(y.copy());masks.append(m.copy());records.append(dict(episode=l['id'],variant=r['kind'],person=l['person'],word_id=l['word_id'],kind=kind,source_key=r['key'],frame_ids=a['frame_id'][q].tolist(),synthetic_feature_indices=None if indices is None else np.asarray(indices).tolist()))
 for r in variants:
  mapping=np.asarray(r['source_frame_indices'])
  for l in labels:
   if l['clip_id']!=r['clip_id']:continue
   q=np.flatnonzero((mapping>=l['frames'][0])&(mapping<=l['frames'][1]));assert len(q)>=3
   y=np.array(l['y'],np.float32);m=np.array(l['mask'],bool);i=len(records);add(l,r,q,y,m,'native_'+l['kind']);pair_groups.setdefault(l['id'],[]).append(i)
   if l['kind']!='directed':continue
   rev=y.copy();rev[:5]=0
   if y[2]:rev[4]=1
   if y[4]:rev[2]=1
   add(l,r,q,rev,m,'feature_reverse',np.arange(len(q)-1,-1,-1))
   k=max(1,int(np.floor(len(q)*.25)));indices=np.r_[np.arange(k),np.arange(len(q)-k,len(q))];cy=np.zeros(6,np.float32);cm=np.zeros(6,bool);cm[5]=True
   add(l,r,q,cy,cm,'feature_middle_omitted',indices)
   # Frozen positions cannot be a directed translation, even if their pose is right.
   hold=y.copy();hold[:5]=0
   for at in [0,len(q)-1]:add(l,r,q,hold,m,'feature_stopped',np.full(len(q),at))
 return sequences,np.array(ys),np.array(masks),records,list(pair_groups.values())
