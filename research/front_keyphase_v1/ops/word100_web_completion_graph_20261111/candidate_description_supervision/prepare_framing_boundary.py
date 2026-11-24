from framing_boundary_core import *
import time,traceback

def main():
 assert not (R/'INPUTS_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 meta=read(N/'INPUTS_COMPLETE.json');old=np.load(meta['arrays']);rows=list(meta['rows']);yy=list(old['y']);mm=list(old['mask']);prior=read(TQ/'INPUTS_COMPLETE.json');orig=np.load(prior['arrays']);crops=read(R/'CROP_INPUTS_COMPLETE.json')['rows'];rc=load('_frame_bound_role',O/'word100_candidate_role_learning_20261111/role_core.py');jn=rc.frozen_model();base=encoder();need={r['clip_id'] for r in rows};raw={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows'] if r['role']=='fit' and r['clip_id'] in need}
 cfg=dict(at=stamp(),conditions=CONDITIONS,reference='Saved transfer_local_timefix: no crop, no boundary halo; never retrained',hypotheses=['Real cropped detections differ from full-view/masked observations; same source labels need crop observation masks.','A candidate must see its immediate continuation at boundaries to distinguish a portion from the observed occurrence.'],context_us=HALO_US,context_basis='Same133333us as existing role feature; no search',encoder='frozen common100 observation encoder, no word or guide order; pool only candidate interior',head='1160->96->15 unchanged',seeds=[42,43,44],steps=1200,lr=.001,weight_decay=.001,train_people=[2,4,5,6],probe_people=[8,9],probe_upstream_exposed=True,labels='Existing reviewed progression witnesses, not educational completion; recompute availability masks for crop',reverse='Whole extended window reversal for same attempt axis; never normal-sign correctness',selection=dict(conditions=['baseline','timefix_local']+CONDITIONS),runtime_thresholds='unchanged presence .5; sum log three heads; no expected-order enforcement',frame_subsampling=False,adopt='G2 normal/speed/cross/shape/direction preserved across seeds, false order/omission reduced with positive evidence; no final8 before gates',pins={str(p):sha(p) for p in [Path(__file__),W/'framing_boundary_core.py',R/'CROP_PLAN.json',R/'CROP_INPUTS_COMPLETE.json',N/'INPUTS_COMPLETE.json',TQ/'INPUTS_COMPLETE.json',TQ/'VERIFICATION.json']})
 assert not (R/'FROZEN.json').exists();write(R/'FROZEN.json',cfg)
 seq={};pins={};xb=np.zeros((len(rows),1160),np.float32);xbr=np.zeros_like(xb)
 for cid,view in sorted({(r['clip_id'],r['view']) for r in rows}):
  ii=[i for i,r in enumerate(rows) if (r['clip_id'],r['view'])==(cid,view)];r=rows[ii[0]];assert sha(r['cache'])==r['cache_sha256'];sr=raw[cid];assert sha(sr['path'])==sr['sha256'];z=np.load(r['cache'])['z'];a=np.load(sr['path']);pose=a['x_'+view];q=z[:,292:392];ts=a['timestamp_us'];seq[cid,view]=(pose,q,ts);pins[sr['path']]=sr['sha256'];pins[r['cache']]=r['cache_sha256']
  for off in range(0,len(ii),128):
   ids=ii[off:off+128];sp=np.array([rows[i]['span'] for i in ids]);xb[ids]=encode_context(base,pose,q,ts,sp);xbr[ids]=encode_context(base,pose,q,ts,sp,True)
  print('boundary inputs',cid,view,flush=True)
 # Actual crop detections are new views of the SAME source, never extra people.
 sys.path.insert(0,str(O/'word100_observation_guard_20261108'));guard=load('_fb_guard',O/'word100_observation_guard_20261108/guard.py');ad=load('_fb_adapter',O/'word100_rtmw_shared_input_20261108/adapter.py')
 truths=read(U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json')['rows'];co={r['clip_id']:r for r in read(U/'episode_stage_contrast/RGB_CONTRAST_LABELS.json')['rows']};sup=read(W/'short_action_review/REVIEW.json')['rows']+read(W/'short_action_review/FOREARM_REVIEW.json')['rows'];truths={s['clip_id']:s for s in truths};xl=list(orig['x']);xlr=list(orig['x_reverse']);xb=list(xb);xbr=list(xbr);audits=[]
 for cr in crops:
  cid=cr['clip_id'];assert sha(cr['path'])==cr['sha256'];a=np.load(cr['path']);s=truths[cid]
  for r in sup:
   if r['clip_id']==cid:
    e=next(e for e in s['events'] if e['event_id']==r['original_event']['event_id']);e['progression_witness_frames']=sorted(set(e['progression_witness_frames']+r['additional_progression_frames']))
  groups,bg=source_witnesses(s,co[cid]);clean=guard.quarantine(a['xy_px'],a['score'],a['timestamp_us'],640,480);x=ad.convert(clean['xy_px'],a['score'],a['timestamp_us'],a['frame_id'],640,480)['tcn280'];q=rc.old.relation_features(x);qn=(q-jn.qm.numpy())/jn.qs.numpy();qn=np.concatenate([qn[...,:4]*q[...,8,None],qn[...,4:8]*q[...,9,None],q[...,8:]],-1).reshape(len(q),100);pose=normalized_pose(x);ts=a['timestamp_us'];oldsrc=np.load(raw[cid]['path']);assert np.array_equal(ts,oldsrc['timestamp_us']) and np.array_equal(a['frame_id'],oldsrc['frame_id'])
  dst=R/'observations'/f'{cid}.npz';dst.parent.mkdir(exist_ok=True);np.savez_compressed(dst,x=x,pose=pose,q=qn,ts=ts,frame_id=a['frame_id']);pins[str(dst)]=sha(dst);pins[cr['path']]=cr['sha256']
  # Use existing full-view window placements, with new observation masks.
  chosen=[]
  for r in [r for r in meta['rows'] if r['clip_id']==cid and r['view']=='full']:
   aa,b=r['span'];y=np.zeros((5,3),np.float32);m=np.zeros_like(y);basis={}
   for family in groups:
    valid=q[:,AX[family]*2,8]>.5
    if not valid[aa] or not valid[b] or valid[aa:b+1].mean()<.8:continue
    y0,m0,why=labels(aa,b,family,groups,bg,valid);j=F.index(family);y[j]=y0;m[j]=m0;basis[str(family)]=why
   if m.any():chosen.append((r,y,m,basis))
  spans=np.array([r['span'] for r,_,_,_ in chosen]);nf=[];nr=[];hf=[];hr=[]
  for off in range(0,len(spans),128):
   sp=spans[off:off+128];nf.extend(encode_windows(base,pose,qn,sp));nr.extend(encode_windows(base,pose,qn,sp,True));hf.extend(encode_context(base,pose,qn,ts,sp));hr.extend(encode_context(base,pose,qn,ts,sp,True))
  xl.extend(nf);xlr.extend(nr);xb.extend(hf);xbr.extend(hr)
  for r,y,m,basis in chosen:rows.append(dict(r,view='actual_crop',basis=basis,cache=str(dst),cache_sha256=sha(dst)));yy.append(y);mm.append(m)
  audits.append(dict(clip_id=cid,frames=len(x),windows=len(chosen),new_masks=True,time_match=True,hand_quarantined=int(clean['hand_quarantined'].sum())))
  print('crop feature',cid,len(chosen),flush=True)
 arrays=dict(x_local=np.asarray(xl),x_local_reverse=np.asarray(xlr),x_boundary=np.asarray(xb),x_boundary_reverse=np.asarray(xbr),y=np.asarray(yy),mask=np.asarray(mm),train=np.array([r['split']=='train' for r in rows]),crop=np.array([r['view']=='actual_crop' for r in rows]))
 assert all(np.isfinite(v).all() for v in arrays.values());assert np.array_equal(arrays['x_local'][:len(meta['rows'])],orig['x']);assert np.array_equal(arrays['y'][:len(meta['rows'])],old['y'])
 # Zero halo must reproduce the corrected existing feature extraction exactly.
 pose,q,ts=next(iter(seq.values()));sp=np.array([[10,19],[20,50]]);delta=float(abs(encode_context(base,pose,q,ts,sp,halo_us=0)-encode_windows(base,pose,q,sp)).max());assert delta<1e-5
 p=R/'inputs.npz';np.savez_compressed(p,**arrays);write(R/'INPUTS_COMPLETE.json',dict(at=stamp(),rows=rows,arrays=str(p),arrays_sha256=sha(p),pins=pins,crop_audits=audits,zero_halo_max_difference=delta,old_windows=len(meta['rows']),no_private=True,no_final8=True));print('prepared',len(rows),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:(R/f'INPUT_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
