"""Direction-independent candidate proposal continuation, source FIT only."""
from core import *
import copy,time,traceback
V=W/'direction_neutral_candidates';R=W/'boundary_direction_repair'
def inputs(z):
 T=len(z);q=z[:,292:392].reshape(T,10,10);h=z[:,228:292];fh=z[:,392:456];query=np.c_[np.eye(5)[np.arange(10)//2],np.eye(2)[np.arange(10)%2]]
 return np.concatenate([np.repeat(h[:,None],10,axis=1),q,np.repeat(fh[:,None],10,axis=1),np.repeat(query[None],T,axis=0)],-1).astype(np.float32)
def main():
 assert not (R/'INPUTS_COMPLETE.json').exists();R.mkdir(exist_ok=True);torch.set_num_threads(4);torch.set_num_interop_threads(1)
 role=load('_boundary_repair_role',O/'word100_candidate_role_learning_20261111/role_core.py');net=role.frozen_model();sources=read(U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json')['rows'];contrast={r['clip_id']:r for r in read(U/'episode_stage_contrast/RGB_CONTRAST_LABELS.json')['rows']};supplement=read(W/'short_action_review/REVIEW.json')['rows']+read(W/'short_action_review/FOREARM_REVIEW.json')['rows'];normal={(r['clip_id'],r['view']):r for r in read(U/'context_role_training/INPUTS_COMPLETE.json')['rows']};reverse={(r['clip_id'],r['view']):r for r in read(V/'GENERATED.json')['rows']};out=[];xx=[];yy=[];mm=[];teacher_x=[];teacher_p=[];teacher_meta=[]
 for s in sources:
  cid=s['clip_id'];split='probe' if s['person'] in [8,9] else 'train'
  for item in supplement:
   if item['clip_id']==cid:
    e=next(e for e in s['events'] if e['event_id']==item['original_event']['event_id']);e['progression_witness_frames']=sorted(set(e['progression_witness_frames']+item['additional_progression_frames']))
  groups,bg=source_witnesses(s,contrast[cid])
  for view in ['full','upper']:
   for rev in [False,True]:
    source=(reverse if rev else normal)[cid,view];assert sha(source['path'])==source['sha256'];z=np.load(source['path'])['z'];x=inputs(z);T=len(z);q=z[:,292:392].reshape(T,10,10)
    if split=='train' and not rev:
     with torch.inference_mode():p=net.boundary(torch.tensor(x)).sigmoid().numpy()
     teacher_x.extend(x.reshape(-1,145));teacher_p.extend(p.reshape(-1,3));teacher_meta.extend([cid]*T*10)
    for family,g in groups.items():
     k=AX[family]*2;sg=next(v for v in s['groups'] if v['family']==family);onset=set(range(sg['onset'][0],sg['onset'][1]+1));offset=set(range(sg['offset'][0],sg['offset'][1]+1));motion=set(g['motion']);negative=set(bg+g['held'])|{f for f0,g0 in groups.items() if f0!=family for f in g0['motion']+g0['held']};points=sorted(onset|offset|motion|negative)
     for sf in points:
      f=T-1-sf if rev else sf
      if not q[f,k,8]>.5:continue
      y=np.zeros(3,np.float32);m=np.zeros(3,np.float32);kind=[]
      if sf in motion:y[0]=1;m[0]=1;kind.append('reviewed_progression')
      elif sf in negative and sf not in onset|offset:m[0]=1;kind.append('reviewed_nonmotion_or_other_action')
      for h,band in [(1,offset if rev else onset),(2,onset if rev else offset)]:
       if sf in band:y[h]=1;m[h]=1;kind.append('annotated_boundary_band_membership')
       elif sf in motion|negative and sf not in onset|offset:m[h]=1
      if not m.any():continue
      xx.append(x[f,k]);yy.append(y);mm.append(m);out.append(dict(clip_id=cid,person=s['person'],view=view,reverse=rev,split=split,family=family,axis=AX[family],moving='R',source_frame=sf,query_frame=f,source_video_sha256=s['source_sha256'],cache=source['path'],cache_sha256=source['sha256'],label_basis=kind,exact_physiological_boundary=False,correct_sign_label=None))
  print('boundary source',cid,flush=True)
 p=R/'inputs.npz';np.savez_compressed(p,x=np.array(xx),y=np.array(yy),mask=np.array(mm),train=np.array([r['split']=='train' for r in out]));tp=R/'preservation.npz';np.savez_compressed(tp,x=np.array(teacher_x),p=np.array(teacher_p));write(R/'PRESERVATION_META.json',dict(clip_ids=teacher_meta,scope='Original normal FIT all10axis/side channels; model probability preservation, not new truth'))
 write(R/'INPUTS_COMPLETE.json',dict(at=stamp(),rows=out,arrays=str(p),arrays_sha256=sha(p),teacher=str(tp),teacher_sha256=sha(tp),source_clips=24,positive_boundary_meaning='Within previously recorded uncertainty band, not exact physical onset/offset. Band supervision is weak interval membership. Direct moving/nonmoving labels only at reviewed witnesses.',body_mask_applied=True,train_people=[2,4,5,6],probe_people=[8,9],source_models_frozen=True,pins={str(p):sha(p) for p in [Path(__file__),W/'core.py',W/'short_action_review/REVIEW.json',W/'short_action_review/FOREARM_REVIEW.json',U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',V/'GENERATED.json',W/'shadow_components/RESULTS.json',V/'shadow_components/RESULTS.json']}))
 print('prepared boundary rows',len(out),'teacher',len(teacher_x),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:R.mkdir(exist_ok=True);(R/f'PREP_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
