"""Prepare the complete original TRAIN bank; reuse verified image evidence."""
import prepare as R
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np,time,zipfile,shutil,traceback,sys
N=R.O/'word100_source_separate_multiref_20261026';Z=R.O/'word100_core_image_evidence_20261026'
sys.path.insert(0,str(N))
S=R.mod('bank_source_integration',N/'integrate.py')

def freeze():
 f=R.read(R.V/'FROZEN.json');zp=R.read(Z/'PREPARED.json');R.check(zp['outputs']);assert R.read(Z/'VERIFIED_IMAGES.json')['clips']==300;assert R.read(Z/'INTEGRATION_VERIFIED.json')['pairs']==200
 meta={r['clip_id']:r for r in R.read(R.B/'PREP_CONTEXT.json')['ordered_rows']};old={r['clip_id']:r for r in zp['rows']};assert len(f['rows'])==899 and len(old)==300
 for n in ['baseline','sources','cache_records','outputs','records','starts']:(R.U/n).mkdir()
 pins={};todo=[];catalog=[];counts=Counter();by=defaultdict(list);beg=time.perf_counter()
 for row in f['rows']:
  cid=row['clip_id'];assert meta[cid]['role']=='train_candidate' and row['signer_id'] in range(1,10);by[row['word_code']].append(row['signer_id'])
  if cid in old:
   q=old[cid];p=Path(q['output']);pins[str(p)]=R.sha(p);catalog.append(dict(clip_id=cid,word_code=row['word_code'],label=row['label'],signer_id=row['signer_id'],input=str(p),evidence_complete=True,reused_from=str(Z/'PREPARED.json')));continue
  inp=row['input'];R.check(inp['input_pins']);pins.update(inp['input_pins']);op=Path(inp['info']['source_arrays']['observation']);obs=S.Observation.load(str(op));obs.validate();n=R.load(inp['info']['source_arrays']['capture']);raw=R.load(inp['info']['source_arrays']['raw']);assignment=R.load(Path(inp['info']['source_arrays']['capture']).parent/'assignment.npz')['sides'];x=n['xy_px'][:,33:].reshape(-1,2,21,2);w,h=inp['info']['canvas'];box=inp['info']['original_roi'];valid=S.H.H.roi(x,n['usable'][:,33:].reshape(-1,2,21),box)
  np.testing.assert_array_equal(obs.xy_px[:,33:].reshape(-1,2,21,2),np.where(valid[...,None],x,np.nan));np.testing.assert_array_equal(obs.usable[:,33:].reshape(-1,2,21),valid);np.testing.assert_array_equal(obs.frame_id,n['frame_id'])
  rp=R.I/'records'/(cid+'__masked.json');rr=R.read(rp);R.check(rr['outputs']);pins[str(rp)]=R.sha(rp);src=dict(row,old_records=str(rp));video=Path(rr['source_video']) if rr['source_video'] else R.U/'sources'/(cid+'.mp4')
  if not video.exists():
   source=inp['source'];video=R.U/'sources'/(cid+'.mp4')
   with zipfile.ZipFile(source['resolved_archive'],'r') as archive:
    zi=archive.getinfo(source['member']);assert zi.CRC==source['crc32'] and zi.file_size==source['bytes']
    with archive.open(zi) as a,video.open('xb') as b:shutil.copyfileobj(a,b,1024*1024)
   counts['locally_extracted_videos']+=1
  assert R.sha(video)==inp['source']['sha256'];src['video']=str(video)
  # Metadata-only copy for the frozen runner; cached detector frames remain exactly identical.
  recent=R.U/'cache_records'/(cid+'.json');R.write(recent,dict(rr,source_video=str(video)));pins[str(recent)]=R.sha(recent)
  values,available,_,scale=S.A.channel_view(np.where(valid[...,None],x,np.nan),valid,np.zeros(valid.shape[:2]+(15,),np.uint8),obs.xy_px[:,:33],obs.usable[:,:33],np.zeros(valid.shape[:2],bool))
  b=S.baseline(src,n,raw,x,valid,assignment,available[:,:,:15]);state=S.state(b,available[:,:,:15]);quality=np.zeros_like(available);quality[:,:,:15]=state==5;core=R.load(row['diagnostics'])['core'];features=S.F.sequence_features(obs,trim=False)
  target=R.U/'baseline'/(cid+'.npz');np.savez_compressed(target,xy_px=obs.xy_px,usable=obs.usable,timestamp_us=obs.timestamp_us,frame_id=obs.frame_id,features=features['features'],present=features['present'],core=core,values=values,available=available,quality_eligible=np.zeros_like(available),quality_before=quality,quality_after=quality,state_before=state,state_after=state,requested_before=b['requested'],requested_current=b['requested'],yolo_conflict=np.zeros(state.shape,bool),scale=np.array(scale))
  pins[str(target)]=R.sha(target);pins[row['diagnostics']]=R.sha(row['diagnostics']);cache={q['i']:q for q in rr['frames']};need=core[:,None]&available[:,:,:15].any(-1)&~b['requested'];targets=[]
  for i in np.flatnonzero(need.any(1)):
   cc=cache.get(int(i),{});sides=np.flatnonzero(need[i]).tolist();counts['cached_full' if 'mp_full' in cc else 'new_full']+=1;boxes={}
   for side in sides:
    bb=R.C.crop_box(x[i,side],valid[i,side],w,h);boxes[str(side)]=bb;seen=list(cc.get('boxes',{}).values())+[a['box'] for a in cc.get('calls',[]) if a.get('detector','mp')=='mp' and a['box'] is not None];counts['cached_crop' if bb in seen and 'mp_crop' in cc else 'new_crop']+=1
   targets.append(dict(index=int(i),frame_id=int(n['frame_id'][i]),sides=sides,boxes=boxes))
  todo.append(dict(clip_id=cid,source=src,baseline_input=str(target),recent_record=str(recent),targets=targets));catalog.append(dict(clip_id=cid,word_code=row['word_code'],label=row['label'],signer_id=row['signer_id'],input=str(target),evidence_complete=False));counts.update(clips=1,frames=len(targets),hand_frames=int(need.sum()))
  if len(todo)%50==0:print('BANK_PREPARED',len(todo),599,flush=True)
 assert len(todo)==599 and len(catalog)==899 and len(by)==100 and sorted(map(len,by.values()))==[8]+[9]*99
 for word,ids in by.items():assert len(ids)==len(set(ids))
 extra=dict(counts);extracted=extra.pop('locally_extracted_videos',0)
 for p in [Path(__file__),R.U/'prepare.py',R.U/'recheck.py',R.U/'verify_images.py',R.U/'verify_bank.py',Z/'PREPARED.json',Z/'INTEGRATION_VERIFIED.json',Z/'VERIFIED_IMAGES.json',R.V/'FROZEN.json',R.V/'images.py',R.V/'pixel_guard.py',R.V/'verify_images.py',R.V/'verify_pixel.py',N/'integrate.py',N/'references.py',Path(S.A.__file__),Path(S.F.__file__),R.ROOT/'Tools/models/hand_landmarker.task']:pins[str(p)]=R.sha(p)
 R.check(f['protected']);R.write(R.U/'FROZEN.json',dict(at=R.now(),rows=todo,catalog=catalog,pins=pins,protected=f['protected'],mp_options=f['mp_options'],counts=extra,locally_extracted_videos=extracted,expected_clips=599,reused_complete_clips=300,
  authorization='Continue approved autonomous common research goal; entire public TRAIN899 only. ExistingC/T3F/E3/operating/raw outputs preserved; no user/mentor/extra199/200 or new learning/inference.',
  scope='100words,recordedIDs1-9,899clips. Reuse300verified currentimage inputs; prepare599remaining original inputs and only previously unrequested computable core hand checks. No normal-range fitting or DEV tuning yet.',
  method='Frozen prior fullframe+fixed1.6handcrop MP checks/current-arm identity/original pixel geometry. Raw coords/left-right/normalization/core/features unchanged. Cached calls reused by exact frame/box/task/options. Do not infer sign meaning from model agreement.',
  planned_range='Keep namedreference trajectories and perchannel point/sequence comparisons. Leave target recordedID out of all calibration sources. Minimum5other observed peers for any empirical bound. Separate raw observation from stronger currentimage corroboration. Unknown remains unknown. No pose-envelope averaging, no whole-sign correct verdict.',
  future_gates='Freeze range algorithm before results. Normal speed/rest/crossing/rawdomain preservation plus shortdeletion and visibilityloss controls; do not count unknown as improved. Fullbank coverage first,then common calibrated reference comparison. No rule/class/model choice from DEV300 images.',
  elapsed_seconds=time.perf_counter()-beg,new_training=0,new_classifier_inference=0,operating_changed=False))
 print('TRAIN_BANK_FROZEN',dict(counts),'reuse300',flush=True)

if __name__=='__main__':
 try:freeze()
 except BaseException:
  p=R.U/'PREP_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
