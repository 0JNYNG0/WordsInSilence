"""A: exact source-index feature binding, source split, mask and shortcut audit."""
from core import *
import collections,time,traceback

def annotation_axes():
 result={}
 for r in read(U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json')['rows']:
  for e in r['events']:
   for s in r['sheets']:
    for f in s['frames']:
     if e['onset'][1]<f<e['offset'][0]:result[r['clip_id'],f]=[e['axis']*2]
 for r in read(L/'REPEAT_EVIDENCE_EXPANSION.json')['rows']:
  for e in r['events']:
   for f in r['viewed_frames']:
    if e['onset'][1]<f<e['offset'][0]:result[r['clip_id'],f]=[8] # relative wrists: observed source relation
 for r in read(L/'DISTINCT_ACTION_WITNESSES.json')['rows']:
  for j,g in enumerate(r['groups']):
   for f in g['sample_frames']:
    # Annotation-dependent observability only; no word ID enters model features.
    result[r['clip_id'],f]=[0] if r['word_id']=='WORD0185' and j==0 else [6]
 return result

def main():
 assert not (W/'INPUTS_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 target=read(L/'MULTILEVEL_RELATION_TARGETS.json');need={r['clip_id'] for r in target['rows']};assert len(need)==30
 raw={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows'] if r['role']=='fit' and r['clip_id'] in need};assert set(raw)==need
 cached={}
 for path in [O/'word100_candidate_role_learning_20261111/TRAIN_INPUTS_COMPLETE.json',U/'context_role_training/INPUTS_COMPLETE.json']:
  for r in read(path)['rows']:
   if r['clip_id'] in need and not r.get('reverse',False):cached[r['clip_id'],r['view']]=r
 axes=annotation_axes();feature_rows=[];cache={};observer=temporal=rc=role=None
 for cid in sorted(need):
  r=raw[cid];assert sha(r['path'])==r['sha256'];d=np.load(r['path']);frame=d['frame_id'];ts=d['timestamp_us'];assert len(set(frame.tolist()))==len(frame) and np.all(np.diff(ts)>0)
  for view in ['full','upper']:
   found=cached.get((cid,view));new=False
   if found:
    assert found['source_sha256']==r['source_sha256'] and sha(found['path'])==found['sha256'];path=Path(found['path']);z=np.load(path)['z']
   else:
    if observer is None:
     rc=load('_hier_req',O/'word100_guide_requirement_learning_20261111/requirement_core.py');role=load('_hier_role',O/'word100_candidate_role_learning_20261111/role_core.py');observer=rc.frozen_observer();temporal=role.frozen_model()
    x=rc.prior.physical(d['x_'+view]);fh,_=role.lc.be.ev.frozen_forward(observer,x,r['word']);ob=role.features(temporal,x,ts,fh);z=ob['z'];path=W/'feature_cache'/f'{cid}_{view}.npz';path.parent.mkdir(exist_ok=True);assert not path.exists();np.savez_compressed(path,z=z,ts=ts,frame_id=frame);new=True
   assert z.shape==(len(frame),684) and np.isfinite(z).all();cache[cid,view]=(z,frame,ts)
   feature_rows.append(dict(clip_id=cid,person=r['person'],view=view,source_sha256=r['source_sha256'],raw=r['path'],raw_sha256=r['sha256'],path=str(path),sha256=sha(path),new_encoder_pass=new,frames=len(frame)))
  print('Bound',cid,'new feature views',sum(v['new_encoder_pass'] for v in feature_rows if v['clip_id']==cid),flush=True)
 rows=[];xx=[];yy=[];mm=[];eligible=[]
 for idx,t in enumerate(target['rows']):
  r=raw[t['clip_id']];assert r['source_sha256']==t['source_sha256'] and r['person'] in ([5,8,9] if t['split']=='probe' else [1,2,3,4,6,7])
  for view in ['full','upper']:
   z,frame,ts=cache[t['clip_id'],view];points=[int(np.flatnonzero(frame==f)[0]) for f in t['frames']];a,b=points;assert frame[a]==t['frames'][0] and frame[b]==t['frames'][1];q=z[:,292:392].reshape(-1,10,10)
   obs=[all(q[i,k,8]>.5 for k in axes[t['clip_id'],f]) for i,f in zip(points,t['frames'])];ok=all(obs)
   parent=t['parent_same'];atomic=t['atomic_same'];y=[parent,0 if atomic is None else atomic];mask=[1,int(parent==1 and atomic is not None)]
   xx.append(pair_features(z,a,b));yy.append(y);mm.append(mask);eligible.append(ok)
   rows.append(dict(target_index=idx,clip_id=t['clip_id'],person=t['person'],word_id=t['word_id'],source_sha256=t['source_sha256'],view=view,split=t['split'],source_frames=t['frames'],indices=points,dt_us=int(ts[b]-ts[a]),source_frames_total=len(frame),observed=obs,eligible=ok,required_relation_slots=[axes[t['clip_id'],f] for f in t['frames']],y=y,mask=mask))
 x=np.asarray(xx,np.float32);y=np.asarray(yy,np.float32);mask=np.asarray(mm,np.float32);good=np.asarray(eligible,bool);train=np.array([r['split']=='train' for r in rows]);assert len(rows)==786 and int(train.sum())==318
 path=W/'pairs.npz';np.savez_compressed(path,x=x,y=y,mask=mask,eligible=good,train=train)
 write(W/'INPUTS_COMPLETE.json',dict(at=stamp(),rows=rows,feature_rows=feature_rows,arrays=str(path),arrays_sha256=sha(path),counts=dict(views=len(rows),source_pairs=len(target['rows']),train_views=int(train.sum()),probe_views=int((~train).sum()),unobserved_train=int((train&~good).sum()),unobserved_probe=int((~train&~good).sum()),new_encoder_views=sum(r['new_encoder_pass'] for r in feature_rows),reused_encoder_views=sum(not r['new_encoder_pass'] for r in feature_rows)),pins={str(p):sha(p) for p in [Path(__file__),W/'core.py',L/'MULTILEVEL_RELATION_TARGETS.json',L/'LABEL_VERIFICATION.json']},training_allowed=False,limits=['Only public FIT, per-head source/person split; upstream exposed people.','Full/upper correlated, frame pairs within video correlated.','Body relation evidence does not establish finger/contact correctness.','No word ID/frame index/elapsed gap/source length in x.','Parent/atomic relation is not boundary or semantic-completion supervision.']))
 print('Prepared',len(rows),'pair views; unobserved',int((~good).sum()),flush=True)

if __name__=='__main__':
 try:main()
 except BaseException:(W/f'PREP_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
