from source_boundary_core import *
import time,traceback

def main():
 B.mkdir(exist_ok=True);assert not (B/'INPUTS_COMPLETE.json').exists();old=read(R/'INPUTS_COMPLETE.json');a=np.load(old['arrays']);rows=old['rows'];sources=read(U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json')['rows'];co={r['clip_id']:r for r in read(U/'episode_stage_contrast/RGB_CONTRAST_LABELS.json')['rows']};sup=read(W/'short_action_review/REVIEW.json')['rows']+read(W/'short_action_review/FOREARM_REVIEW.json')['rows'];raw={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows'] if r['role']=='fit'}
 cfg=dict(at=stamp(),primary='direct_boundary',control='body_window',reason='Candidate presence/coverage labels do not directly locate source onset/end; handshape differences should not prevent locating a body movement attempt.',inputs='Only normalized body-wrist relation4values+availability per10side-axis, before/current/after133333us. No fingers, hand-root branch, wordID, wholeclip latent, absolute clock, expected order.',source_labels='Previously reviewed onset/offset uncertainty BANDS + sparse progression and nonmotion; not exact frame gold or educational completeness',scope='Public24 sources; full/upper/actualcrop views; same16FIT/8probe split; upstream common100 saw probe',seeds=[42,43,44],steps=1200,lr=.001,weight_decay=.001,reverse='Full source time reversal for movement-attempt localization. Start/end bands swapped; never normal-sign target.',boundary_loss='Interval-censored logsumexp likelihood across observed times; sparse presence BCE balanced per source/family/class. No dense boundary pseudo-label.',selection=dict(conditions=['baseline','body_window','direct_boundary']),gate='Existing G2 normal/speed/shape/direction/cross/order/omission gates; report allseeds, no DEV model/epoch/threshold search',semantic_completion=None,no_final8=True,no_private=True,no_web=True,pins={str(p):sha(p) for p in [Path(__file__),W/'source_boundary_core.py',R/'INPUTS_COMPLETE.json',U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',U/'episode_stage_contrast/RGB_CONTRAST_LABELS.json',W/'short_action_review/REVIEW.json',W/'short_action_review/FOREARM_REVIEW.json']});write(B/'FROZEN.json',cfg)
 seq=[];pins={};audit=[];cache={};xf=np.zeros((len(rows),600),np.float32);xr=np.zeros_like(xf)
 for s in sources:
  cid=s['clip_id']
  for rr in sup:
   if rr['clip_id']==cid:
    e=next(e for e in s['events'] if e['event_id']==rr['original_event']['event_id']);e['progression_witness_frames']=sorted(set(e['progression_witness_frames']+rr['additional_progression_frames']))
  groups,bg=source_witnesses(s,co[cid]);src=np.load(raw[cid]['path']);ts=src['timestamp_us'];assert np.array_equal(src['frame_id'],np.arange(len(ts)))
  for g in s['groups']:
   pts=groups[g['family']]['motion'];assert g['onset'][0]<=min(pts)<=max(pts)<=g['offset'][1];assert g['onset'][1]<g['offset'][0]
   audit.append(dict(clip_id=cid,family=g['family'],onset=g['onset'],offset=g['offset'],progression=pts,semantic_completion=None))
  for view in ['full','upper','actual_crop']:
   rr=next(r for r in rows if r['clip_id']==cid and r['view']==view);p=Path(rr['cache']);assert sha(p)==rr['cache_sha256'];z=np.load(p);q=z['q'].reshape(-1,10,10) if view=='actual_crop' else z['z'][:,292:392].reshape(-1,10,10);assert len(q)==len(ts);pins[str(p)]=sha(p)
   feat=body_context(q,ts);rev_q=q[::-1].copy();rev_ts=ts[-1]-ts[::-1];revfeat=body_context(rev_q,rev_ts);cache[cid,view]=feat;ii=[i for i,r in enumerate(rows) if r['clip_id']==cid and r['view']==view];sp=np.array([rows[i]['span'] for i in ii]);xf[ii]=pool_windows(feat,sp);rsp=np.c_[len(ts)-1-sp[:,1],len(ts)-1-sp[:,0]];xr[ii]=pool_windows(revfeat,rsp)
   py=np.zeros((len(ts),5),np.float32);pm=np.zeros_like(py);bands=np.zeros((len(ts),5,2),bool);valid=np.zeros_like(bands)
   for g in s['groups']:
    family=g['family'];j=F.index(family);good=q[:,AX[family]*2,8]>.5;motion=set(groups[family]['motion']);other=set(bg)|{f for k,v in groups.items() if k!=family for f in v['motion']+v['held']}
    for i in motion:
     if good[i]:py[i,j]=1;pm[i,j]=1
    for i in other:
     if good[i]:pm[i,j]=1
    for h,key in enumerate(['onset','offset']):
     aa,b=g[key];bands[aa:b+1,j,h]=good[aa:b+1]
     if bands[:,j,h].any():valid[:,j,h]=good
   assert (bands<=valid).all()
   for reverse in [False,True]:
    arrays=dict(x=revfeat if reverse else feat,y=py[::-1].copy() if reverse else py,mask=pm[::-1].copy() if reverse else pm,bands=bands[::-1,:,::-1].copy() if reverse else bands,valid=valid[::-1,:,::-1].copy() if reverse else valid)
    dest=B/'sequences'/f'{cid}_{view}_{int(reverse)}.npz';dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,**arrays);seq.append(dict(clip_id=cid,person=s['person'],view=view,reverse=reverse,split=rr['split'],path=str(dest),sha256=sha(dest),source_sha256=s['source_sha256']))
  print('boundary truth',cid,flush=True)
 # Exact body-input invariance on the same cropped source, handshape-only edits.
 role=load('_boundary_qnorm',O/'word100_candidate_role_learning_20261111/role_core.py');net=role.frozen_model();pairs=read(O/'word100_positive_direction_20261109/COMPARISON.json')['rows'];checks=[]
 for person in [11,12,13]:
  cid=f'NIA_SL_WORD1133_REAL{person}_F';values=[]
  for kind in ['crop_normal','right_closed_hand_main']:
   r=next(r for r in pairs if r['clip_id']==cid and r['kind']==kind);assert sha(r['prediction'])==r['sha256'];z=np.load(r['prediction']);q=role.old.relation_features(z['x']);qn=(q-net.qm.numpy())/net.qs.numpy();qn=np.concatenate([qn[...,:4]*q[...,8,None],qn[...,4:8]*q[...,9,None],q[...,8:]],-1);values.append(body_context(qn,z['timestamp_us']))
  assert np.array_equal(*values);checks.append(dict(clip_id=cid,exact_same_features=True))
 assert np.isfinite(xf).all() and np.isfinite(xr).all()
 for r in seq:
  z=np.load(r['path']);assert all(np.isfinite(z[k]).all() for k in z.files)
 p=B/'windows.npz';np.savez_compressed(p,x=xf,x_reverse=xr,y=a['y'],mask=a['mask'],train=a['train'],crop=a['crop']);write(B/'INPUTS_COMPLETE.json',dict(at=stamp(),rows=rows,sequences=seq,arrays=str(p),arrays_sha256=sha(p),pins=pins,boundary_audit=audit,shape_invariance=checks,old_windows=old['old_windows'],source_clips=24,views=3,expert_gold=False,recovery_of=str(W/'source_boundary_candidate/INPUT_REJECTED.json')));print('Source boundary input verified',len(seq),'sequences')
if __name__=='__main__':
 try:main()
 except BaseException:B.mkdir(exist_ok=True);(B/f'INPUT_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
