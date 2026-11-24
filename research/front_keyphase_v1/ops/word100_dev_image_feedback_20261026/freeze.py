import prepare as R
from pathlib import Path
import sys,numpy as np,traceback
from collections import Counter
Z=R.O/'word100_core_image_evidence_20261026';M=R.O/'word100_multipeople_feedback_integration_20261026';N=R.O/'word100_source_separate_multiref_20261026';V=R.O/'word100_recovery_dev300_20261026'
sys.path.insert(0,str(N));S=R.mod('dev_frozen_corroboration',N/'integrate.py')
def main():
 assert not(R.U/'FROZEN.json').exists();mf=R.read(M/'FROZEN.json');mr=R.read(M/'RESULT.json');mv=R.read(M/'VERIFIED.json');vf=R.read(V/'FROZEN.json');zf=R.read(Z/'FROZEN.json');assert R.read(Z/'FEEDBACK_V2_INDEPENDENT.json')['pairs']==200
 R.check(mf['pins']);R.check(mf['protected']);R.check(mr['pins']);R.check(vf['pins']);R.check(zf['protected'])
 meta={r['clip_id']:r for r in vf['rows']};base_meta={r['clip_id']:r for r in R.read(R.B/'PREP_CONTEXT.json')['ordered_rows']}
 assert len(mf['rows'])==300 and len(mf['pairs'])==600 and {r['signer_id'] for r in mf['rows']}=={11,12,13};assert not set(meta)&{r['clip_id'] for r in zf['rows']}
 for d in ['baseline','outputs','records','starts']:(R.U/d).mkdir()
 counts=Counter();rows=[];pins={};recognition={}
 for k,p in mf['scores'].items():
  saved=R.read(p);pred=saved['predictions']['masked'];assert len(pred)==300
  scores={x['clip_id']:{a:x[a] for a in ['status','prediction','top5','correct','top5_correct']} for x in pred};assert set(scores)==set(meta)
  recognition[k]=dict(source=p,sha256=R.sha(p),predictions=scores);pins[p]=R.sha(p)
 for m in mf['rows']:
  cid=m['clip_id'];b=meta[cid];assert base_meta[cid]['role']=='development_only';rp=V/'records'/(cid+'.json');rec=R.read(rp);assert rec['context_sha256']==R.sha(V/'FROZEN.json');R.check(rec['outputs']);video=Path(rec['source_video']);assert video.is_file() and R.sha(video)==b['source']['sha256']==rec['source_sha256'];d=R.B/'data/clips'/cid
  paths=dict(capture=str(d/'native_capture/candidate.npz'),raw=str(d/'native_capture/raw.npz'),observation=m['source_observation'],channels=str(d/'masked/channels.npz'))
  n=R.load(paths['capture']);raw=R.load(paths['raw']);assign=R.load(d/'native_capture/assignment.npz')['sides'];z=R.load(m['output']);record=R.read(R.B/'data/records'/(cid+'.json'));w=record['calibration']['source_width'];h=record['calibration']['source_height'];box=b['box'];assert box==record['calibration']['box_source_pixels']
  inp=dict(info=dict(source_arrays=paths,canvas=[w,h],original_roi=box),source=b['source'],input_pins={p:R.sha(p) for p in list(paths.values())+[str(d/'native_capture/assignment.npz')]})
  x=n['xy_px'][:,33:].reshape(-1,2,21,2);v=S.H.H.roi(x,n['usable'][:,33:].reshape(-1,2,21),box);np.testing.assert_array_equal(z['xy_px'][:,33:].reshape(x.shape),np.where(v[...,None],x,np.nan));np.testing.assert_array_equal(z['usable'][:,33:].reshape(v.shape),v);np.testing.assert_array_equal(z['frame_id'],n['frame_id']);np.testing.assert_array_equal(z['timestamp_us'],n['timestamp_us'])
  src=dict(clip_id=cid,word_code=m['word_code'],label=m['label'],signer_id=m['signer_id'],input=inp,old_records=str(rp),video=str(video))
  old=S.baseline(src,n,raw,x,v,assign,z['available'][:,:,:15]);state=S.state(old,z['available'][:,:,:15]);before=np.zeros_like(z['available']);before[:,:,:15]=state==5
  # Keep old quality semantics as an untouched field. New image evidence is separate.
  z.update(state_before=state,state_after=state.copy(),quality_before=before,quality_after=before.copy(),requested_before=old['requested'],requested_current=old['requested'].copy(),yolo_conflict=np.zeros_like(state,bool))
  # This unit does not re-interpret old YOLO from its proposals as per-angle truth.
  z['yolo_audit_not_prepared']=np.array(True)
  op=R.U/'baseline'/(cid+'.npz');np.savez_compressed(op,**z);need=z['core'][:,None]&z['available'][:,:,:15].any(-1)&~old['requested'];cache={r['i']:r for r in rec['frames']};targets=[]
  for i in np.flatnonzero(need.any(1)):
   rr=cache.get(int(i),{});hasfull='mp_full' in rr;counts['cached_full' if hasfull else 'new_full']+=1;boxes={};ss=np.flatnonzero(need[i]).tolist()
   for side in ss:
    crop=R.C.crop_box(x[i,side],v[i,side],w,h);boxes[str(side)]=crop;seen=list(rr.get('boxes',{}).values());counts['cached_crop' if crop in seen and 'mp_crop' in rr else 'new_crop']+=1
   targets.append(dict(index=int(i),frame_id=int(n['frame_id'][i]),sides=ss,boxes=boxes))
  rows.append(dict(clip_id=cid,source=src,baseline_input=str(op),original_input=m['output'],recent_record=str(rp),targets=targets));counts.update(clips=1,frames=len(targets),hand_frames=int(need.sum()));pins[str(op)]=R.sha(op);pins[str(rp)]=R.sha(rp);pins[m['output']]=R.sha(m['output']);pins.update(inp['input_pins']);pins[str(video)]=R.sha(video)
 assert len(rows)==300
 for p in [Path(__file__),R.U/'prepare.py',R.U/'recheck.py',R.U/'verify_images.py',R.U/'integrate.py',M/'FROZEN.json',M/'RESULT.json',M/'VERIFIED.json',V/'FROZEN.json',V/'prepare.py',N/'references.py',N/'integrate.py',R.V/'images.py',R.V/'pixel_guard.py',Z/'FEEDBACK_V2_INDEPENDENT.json',Z/'source_feedback_v2.py',Z/'FROZEN.json',R.ROOT/'Tools/models/hand_landmarker.task']:pins[str(p)]=R.sha(p)
 assert vf['pins'][str(R.ROOT/'Tools/models/hand_landmarker.task')]==R.sha(R.ROOT/'Tools/models/hand_landmarker.task')
 timing=R.read(Z/'VERIFIED_IMAGES.json')['latency'];estimate=(counts['new_full']*timing['full']['mean_ms']+counts['new_crop']*timing['crop']['mean_ms'])/1000
 R.write(R.U/'FROZEN.json',dict(at=R.now(),rows=rows,pairs=mr['rows'],pins=pins,protected=mf['protected'],mp_options=zf['mp_options'],counts=dict(counts),estimated_detector_seconds=estimate,scores=recognition,authorization='User autonomous research continuation. Public DEV300 only, no training or selection. Preserve operating C/T3F/E3.',scope='100words,recordedIDs11/12/13,300same publicDEV input videos and600saved directed pairs. Development evaluation,not untouched finaltest. No personal/mentor or extra199/200.',method='Same fixed current-frame original/full/crop hand identity and angle/pixel corroboration from TRAIN. Additional core computable hand requests only,exact existing frame/crop/model caches reused. Original raw coordinates,scales,paths,differences,domains and predictions all unchanged. Existing quality field preserved separately from new currentimage evidence. No anatomy or sign-correctness claims. OldYOLO records retained separately,not a new per-angle veto or accuracy source.',gates='Independent image/source checks,raw input/pair equality,600pair feedback source intervals,no unknown-as-zero,no reference-shortage blame,oneangle/hand absence controls. Report gains and shortfalls,latency offlineonly. Scores never enter observation decisions.',training=0,model_inference=0))
 R.write(R.U/'CURRENT_EXECUTION.json',dict(at=R.now(),stage='frozen_inputs_no_new_detector_yet',active_sessions=[]));print('DEV_FROZEN',dict(counts),'estimated seconds',round(estimate,1))
if __name__=='__main__':
 try:main()
 except BaseException:
  p=R.U/'FREEZE_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
