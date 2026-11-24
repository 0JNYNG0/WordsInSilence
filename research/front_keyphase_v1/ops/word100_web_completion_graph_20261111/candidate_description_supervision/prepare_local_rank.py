"""Controlled hand-information ablation, with target-independent local common encoder."""
from local_rank_core import *
import time,traceback
def main():
 L.mkdir(exist_ok=True);assert not(L/'INPUTS_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1);meta=read(K/'INPUTS_COMPLETE.json');bags=[r for r in meta['bags']if not r['skipped']];seq=meta['sequences'];ready={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows']if r['role']=='fit'};crop={r['clip_id']:r for p in[R/'CROP_INPUTS_COMPLETE.json',D/'CROP_INPUTS_COMPLETE.json']for r in read(p)['rows']};rc=load('_local_rank_role',O/'word100_candidate_role_learning_20261111/role_core.py');jn=rc.frozen_model();base=encoder();sys.path.insert(0,str(O/'word100_observation_guard_20261108'));guard=load('_lr_guard',O/'word100_observation_guard_20261108/guard.py');ad=load('_lr_ad',O/'word100_rtmw_shared_input_20261108/adapter.py');pins={};out=[];pose_cache={}
 cfg=dict(at=stamp(),primary='local_full',control='local_no_hands',reference='Saved body-only rank_extended; no old training/inference rerun',reason='Body-only locator lost order distinction although handshape/direction checks improved. Is local hand information needed to distinguish action identity without excluding wrong-handshape attempts?',inputs='Frozen common100 temporalConv+GRU recomputed inside candidate only, plus relation/hand/mask temporal-third features. No target word/order or whole-video latent. Hand ablation removes both hand coordinates and masks, retains body wrists.',rank_loss=read(K/'FROZEN.json')['rank_loss'],seeds=[42,43,44],steps=1200,lr=.001,weight_decay=.001,training_scope='Same32 public source set,24FIT/8probe, same candidate bags and labels as rank_extended',counterfactual_warning='No-hands is an input ablation, not a new correct-sign video. Candidate ranking is not educational completion.',runtime_gate='Frozen body_window presence>=.5; never require correct target handshape for admission',selection=dict(conditions=['baseline','rank_extended','local_full','local_no_hands']),no_frame_subsampling=True,semantic_completion=None,gate='Original G2 allseed normal/speed/shape/direction/cross/order/omission regression; no final8 or web adoption unless passed',pins={str(p):sha(p)for p in[Path(__file__),W/'local_rank_core.py',W/'transfer_local_timefix_core.py',K/'INPUTS_COMPLETE.json',K/'VERIFICATION.json',O/'word100_broad_factorized_learning_20261110/runs/coupled_42/model.pt']});write(L/'FROZEN.json',cfg)
 for si,s in enumerate(seq):
  cid=s['clip_id'];view=s['view'];key=(cid,view)
  if key not in pose_cache:
   rr=ready[cid];assert sha(rr['path'])==rr['sha256']and sha(rr['channel'])==rr['channel_sha256'];z=np.load(rr['path']);ch=np.load(rr['channel']);ts=z['timestamp_us'];pins[rr['path']]=rr['sha256'];pins[rr['channel']]=rr['channel_sha256']
   if view=='actual_crop':
    cr=crop[cid];assert sha(cr['path'])==cr['sha256'];a=np.load(cr['path']);clean=guard.quarantine(a['xy_px'],a['score'],a['timestamp_us'],640,480);x=ad.convert(clean['xy_px'],a['score'],a['timestamp_us'],a['frame_id'],640,480)['tcn280'];pins[cr['path']]=cr['sha256'];assert np.array_equal(ts,a['timestamp_us'])
   else:x=ch['full']if view=='full'else ch['feedback']if'feedback'in ch.files else ch['upper_mask']
   q=rc.old.relation_features(x);qn=(q-jn.qm.numpy())/jn.qs.numpy();qn=np.concatenate([qn[...,:4]*q[...,8,None],qn[...,4:8]*q[...,9,None],q[...,8:]],-1).reshape(-1,100);pose_cache[key]=(normalized_pose(x),qn,ts)
  pose,q,ts=pose_cache[key];pose=pose.copy();q=q.copy()
  if s['reverse']:
   dt=pose[1:,279][::-1].copy();pose=pose[::-1].copy();pose[:,279]=np.r_[0,dt];q=q[::-1].copy()
  ii=[i for i,r in enumerate(bags)if(r['clip_id'],r['view'],r['reverse'])==(cid,view,s['reverse'])]
  if not ii:continue
  arrays={i:np.load(bags[i]['path'])for i in ii};union=sorted(set(tuple(v)for d in arrays.values()for v in d['spans']));sp=np.array(union,np.int32);lookup={s:i for i,s in enumerate(union)};features={}
  for mode in['local_full','local_no_hands']:
   px,qq=(pose,q)if mode=='local_full'else drop_hand_information(pose,q);ff=[]
   for off in range(0,len(sp),128):ff.extend(encode_windows(base,px,qq,sp[off:off+128]))
   features[mode]=np.asarray(ff);assert np.isfinite(features[mode]).all()
  for i in ii:
   idx=[lookup[tuple(s)]for s in arrays[i]['spans']];p=L/'bags'/f'{i}.npz';p.parent.mkdir(exist_ok=True);np.savez_compressed(p,**{m:v[idx]for m,v in features.items()},y=arrays[i]['y'],spans=arrays[i]['spans']);out.append(dict(bags[i],original_bag_index=i,path=str(p),sha256=sha(p),original_path=bags[i]['path'],original_sha256=bags[i]['sha256']))
  write(L/'PREPARATION_PROGRESS.json',dict(completed_sequences=si+1,total_sequences=len(seq),bags=len(out)));print('local rank inputs',si+1,len(seq),len(union),flush=True)
 out.sort(key=lambda r:r['original_bag_index']);assert len(out)==len(bags);write(L/'INPUTS_COMPLETE.json',dict(at=stamp(),bags=out,pins=pins,source_clips=32,fit_clips=24,probe_clips=8,encoder_frozen=True,word_or_order_inputs=False));print('Local ranking features ready')
if __name__=='__main__':
 try:main()
 except BaseException:L.mkdir(exist_ok=True);(L/f'PREP_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
