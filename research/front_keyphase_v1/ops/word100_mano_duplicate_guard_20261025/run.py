from pathlib import Path
import json,hashlib,sys,time
import numpy as np
from candidate import inspect_pair
U=Path(__file__).resolve().parent;O=U.parent;OLD=O/'word100_mano_threeway_20261025'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def scale_of(pose,v):
    return float(np.linalg.norm(pose[11]-pose[12])) if v[[11,12]].all() and np.isfinite(pose[[11,12]]).all() else 0.
def main():
 f=read(U/'FROZEN.json');assert sha(U/'candidate.py')==f['candidate_sha256'];assert not (U/'RESULT.json').exists()
 out=U/'outputs';out.mkdir(exist_ok=True);rows=[];requested=[];clips=[]
 start=time.perf_counter()
 for k,m in enumerate(f['manifest']):
  assert sha(m['prepared'])==m['prepared_sha256']
  z=np.load(m['prepared']);p=np.load(OLD/'comparisons'/f"{m['clip_id']}.npz")
  gx=p['H_xy'].copy();gv=p['H_valid'].copy();gi=p['H_inferred'].copy()
  tx=p['HT_xy'].copy();tv=p['HT_valid'].copy();ti=p['HT_inferred'].copy();status=np.zeros(z['trigger'].shape,np.uint8)
  for t,s in np.argwhere(z['requested']):
   t=int(t);s=int(s);raw=np.load(OLD/f'raw_predictions/{k:02d}_{t:04d}_{s}.npz');a=raw['xy_px'];va=np.isfinite(a).all(-1);sc=scale_of(z['pose'][t],z['pose_valid'][t]);pairs=[]
   for side in range(2):
    if not z['full_valid'][t,side].any():continue
    pairs.append({'existing_side':side,**inspect_pair(a,z['full_xy'][t,side],va,z['full_valid'][t,side],sc)})
   rec={'clip_index':k,'clip_id':m['clip_id'],'frame':t,'side':s,'trigger':bool(z['trigger'][t,s]),'pairs':pairs,'conflict':any(q['hold_new'] for q in pairs)}
   requested.append(rec)
   if not rec['trigger']:continue
   rec=rec.copy();rec.update(old_H=bool(p['H_inferred'][t,s].any()),old_HT=bool(p['HT_inferred'][t,s].any()))
   if rec['conflict']:
    # Preserve the original observation and raw candidate. Only new inferred use held.
    if rec['old_H']:
     gx[t,s]=p['B_xy'][t,s];gv[t,s]=p['B_valid'][t,s];gi[t,s]=False
    if rec['old_HT']:
     tx[t,s]=p['B_xy'][t,s];tv[t,s]=p['B_valid'][t,s];ti[t,s]=False
    status[t,s]=1
    for q in pairs:
     if q['hold_new']:status[t,q['existing_side']]=1
   rec.update(new_G=bool(gi[t,s].any()),new_GT=bool(ti[t,s].any()))
   rows.append(rec)
  b=p['B_valid']
  assert np.array_equal(gx[b],p['B_xy'][b]) and np.array_equal(tx[b],p['B_xy'][b])
  dest=out/(m['clip_id']+'.npz');assert not dest.exists()
  np.savez_compressed(dest,G_xy=gx,G_valid=gv,G_inferred=gi,GT_xy=tx,GT_valid=tv,GT_inferred=ti,
    G_evidence=p['B_evidence'],GT_evidence=p['B_evidence'],identity_conflict=status,
    frame_id=p['frame_id'],timestamp_us=p['timestamp_us'],roi_box=p['roi_box'])
  clips.append({'clip':m['clip_id'],'frames':len(b),'baseline_points':int(b.sum()),'old_H':int(p['H_inferred'].any(-1).sum()),'new_G':int(gi.any(-1).sum()),'old_HT':int(p['HT_inferred'].any(-1).sum()),'new_GT':int(ti.any(-1).sum())})
 broad=[];broad_counts=[];total_frames=0
 for m in f['controls']:
  assert sha(m['path'])==m['sha256'];z=np.load(m['path']);total_frames+=len(z['frame_id']);qualified=0;flagged=0
  for t in range(len(z['frame_id'])):
   v=z['usable'][t];a=z['xy_px'][t,33:54];b=z['xy_px'][t,54:75]
   if not v[33:75].all():continue
   scale=scale_of(z['xy_px'][t,:33],v[:33])
   if scale<=1:continue
   q=inspect_pair(a,b,v[33:54],v[54:75],scale);qualified+=1
   if q['hold_new']:
    flagged+=1;broad.append({'clip_id':m['clip_id'],'frame_index':t,'frame_id':int(z['frame_id'][t]),'word_code_for_index_only':m['word_code'],'signer':m['signer'],'pair':q})
  broad_counts.append({'clip':m['clip_id'],'frames':len(z['frame_id']),'qualifying_complete_pairs':qualified,'flagged':flagged})
 result={'status':'REPLAY_COMPLETE_VISUAL_REVIEW_REQUIRED','candidate_sha256':f['candidate_sha256'],'primary':{'clips':7,'frames':812,'missing_slots':len(rows),'requested_saved':len(requested),'old_H':sum(r['old_H'] for r in rows),'new_G':sum(r['new_G'] for r in rows),'old_HT':sum(r['old_HT'] for r in rows),'new_GT':sum(r['new_GT'] for r in rows),'new_H_holds':sum(r['old_H'] and not r['new_G'] for r in rows),'new_HT_holds':sum(r['old_HT'] and not r['new_GT'] for r in rows)},'rows':rows,'clips':clips,
 'broad_proxy':{'clips':len(f['controls']),'words':100,'signers':sorted(set(x['signer'] for x in f['controls'])),'frames':total_frames,'complete_pairs':sum(x['qualifying_complete_pairs'] for x in broad_counts),'flagged':len(broad),'flagged_clips':len(set(x['clip_id'] for x in broad)),'counts':broad_counts,'flags':broad,'interpretation':'Coordinate pair conflict proxy, not new HaMeR detections or image-verified false-positive rate'},
 'elapsed_seconds':time.perf_counter()-start,'training':0,'new_inference':0,'user_data_access':0,'production_change':False}
 write(U/'REQUESTED_DIAGNOSTIC.json',requested);write(U/'RESULT.json',result)
 print(json.dumps({k:result[k] for k in ['primary','elapsed_seconds']},ensure_ascii=False));print('broad',total_frames,result['broad_proxy']['complete_pairs'],'flagged',len(broad),'clips',result['broad_proxy']['flagged_clips'])
if __name__=='__main__':main()

