from core import *
import itertools,time,traceback,copy
V=W/'direction_neutral_candidates'
def main():
 assert not (V/'INPUTS_COMPLETE.json').exists();V.mkdir(exist_ok=True);torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rc=load('_neutral_rc',O/'word100_guide_requirement_learning_20261111/requirement_core.py');role=load('_neutral_role',O/'word100_candidate_role_learning_20261111/role_core.py');observer=rc.frozen_observer();temporal=role.frozen_model()
 old=read(W/'INPUTS_COMPLETE.json');oldz=np.load(old['arrays']);xx=list(oldz['x']);yy=list(oldz['y']);mm=list(oldz['mask']);rows=[dict(r,variant='source_normal') for r in old['rows']];sources=read(U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json')['rows'];con={r['clip_id']:r for r in read(U/'episode_stage_contrast/RGB_CONTRAST_LABELS.json')['rows']};sup=read(W/'short_action_review/REVIEW.json')['rows']+read(W/'short_action_review/FOREARM_REVIEW.json')['rows'];need={r['clip_id'] for r in sources};raw={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows'] if r['role']=='fit' and r['clip_id'] in need};generated=[]
 write(V/'PREPARATION_CONTRACT.json',dict(at=stamp(),cause='Frozen stage has_motion excludes some opposite-direction attempts; candidate classifier learned only original direction contexts.',change='Add whole-source time reversal with recomputed frozen features and remapped source witness labels. It supervises related comparison-axis attempt, not normal/correct sign. No splice inside reversed source. Preserve all original labels and frozen shape/direction judge.',limits='Synthetic counterfactual is not filmed learner error. No inferred missing pose targets. No original-time relation evidence is overwritten.',source_people=dict(train=[2,4,5,6],probe=[8,9]),training_not_started=True))
 for s in sources:
  cid=s['clip_id'];r=raw[cid];assert r['source_sha256']==s['source_sha256'] and sha(r['path'])==r['sha256'];d=np.load(r['path']);T=len(d['frame_id']);ii=np.arange(T-1,-1,-1);ts=d['timestamp_us'];assert np.all(np.diff(ts)>0)
  for item in sup:
   if item['clip_id']==cid:
    e=next(e for e in s['events'] if e['event_id']==item['original_event']['event_id']);e['progression_witness_frames']=sorted(set(e['progression_witness_frames']+item['additional_progression_frames']))
  normal_groups,bg=source_witnesses(s,con[cid]);groups={f:dict(motion=sorted(T-1-i for i in v['motion']),held=sorted(T-1-i for i in v['held'])) for f,v in normal_groups.items()};background=sorted(T-1-i for i in bg);teacher=set()
  for f,g in groups.items():
   pts=g['motion'];teacher.add((min(pts),max(pts)))
   if g['held']:teacher.add((min(pts+g['held']),max(pts+g['held'])))
   for a,b in itertools.combinations(pts,2):teacher.add((a,b))
   for h in groups.values():
    if h is not g:teacher.add((min(pts+h['motion']),max(pts+h['motion'])))
   for part in ['entry','other_held_presentation','returning']:
    neg=[T-1-i for i in con[cid][part]]
    if len(neg)>=2:teacher.add((min(neg),max(neg)));teacher.add((min(pts+neg),max(pts+neg)))
  for view in ['full','upper']:
   x=rc.prior.physical(d['x_'+view])[ii].copy();fh,_=role.lc.be.ev.frozen_forward(observer,x,r['word']);ob=role.features(temporal,x,ts,fh);z=ob['z'];qn=z[:,292:392].reshape(T,10,10);candidates=role.jc.candidate_spans(ob['boundary'],ob['q'][0].numpy(),[])
   p=V/'inputs'/f'{cid}_{view}_whole_reverse.npz';p.parent.mkdir(exist_ok=True);assert not p.exists();np.savez_compressed(p,z=z,x=x,fh=fh,ts=ts,source_indices=ii);generated.append(dict(clip_id=cid,view=view,path=str(p),sha256=sha(p),source=r['path'],source_sha256=r['sha256'],source_video_sha256=r['source_sha256'],candidates=candidates,groups=groups,other_witnesses=background))
   spans=teacher|{(c['a'],c['b']) for c in candidates if c['moving']=='R' and c['axis'] in [AX[f] for f in groups]}
   for a,b in sorted(spans):
    y=np.zeros((5,3),np.float32);m=np.zeros_like(y);basis={}
    for f in groups:
     v,vm,why=labels(a,b,f,groups,background,qn[:,AX[f]*2,8]>.5);y[F.index(f)]=v;m[F.index(f)]=vm;basis[str(f)]=why
    if not m.any():continue
    xx.append(features(z,a,b));yy.append(y);mm.append(m);rows.append(dict(clip_id=cid,person=s['person'],word_id=s['word_id'],source_sha256=s['source_sha256'],view=view,split='probe' if s['person'] in [8,9] else 'train',span=[a,b],origin='teacher' if (a,b) in teacher else 'model',basis=basis,cache=str(p),cache_sha256=sha(p),raw=r['path'],raw_sha256=r['sha256'],variant='whole_reverse_axis_attempt',normal_sign_label=None))
  print('neutral input',cid,flush=True)
 arr=V/'inputs.npz';np.savez_compressed(arr,x=np.array(xx),y=np.array(yy),mask=np.array(mm),train=np.array([r['split']=='train' for r in rows]));write(V/'GENERATED.json',dict(rows=generated,new_frozen_encoder_calls=len(generated),no_spliced_sequence=True))
 write(V/'INPUTS_COMPLETE.json',dict(at=stamp(),rows=rows,arrays=str(arr),arrays_sha256=sha(arr),original_rows=len(old['rows']),source_clips=24,source_person_split=old['source_person_split'],counterfactual_axis_only=True,heads=old['heads'],pins={str(p):sha(p) for p in [Path(__file__),W/'core.py',W/'INPUTS_COMPLETE.json',V/'GENERATED.json',V/'PREPARATION_CONTRACT.json']}))
 print('Neutral input complete',len(rows),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:V.mkdir(exist_ok=True);(V/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
