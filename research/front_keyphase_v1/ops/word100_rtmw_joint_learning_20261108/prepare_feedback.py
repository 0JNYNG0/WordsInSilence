from common import *
import numpy as np
from collections import defaultdict

def coord_mask(x):
 return np.concatenate([np.repeat(x[:,186:237]>0,2,axis=1),np.repeat(x[:,237:279]>0,2,axis=1)],1)
def normalize(x,stats):
 z=x.copy();mask=coord_mask(x)
 z[:,:186]=np.where(mask,(x[:,:186]-np.asarray(stats['mean']))/np.asarray(stats['std']),0)
 return z.astype(np.float32)
def main():
 assert not (U/'FEEDBACK_READY.json').exists()
 source=read(O/'word100_learned_guide_feedback_20261102/INPUT_MANIFEST.json');f=read(U/'FROZEN.json')['feedback'];words=f['words']
 prior=read(O/'word100_learned_guide_feedback_20261102/NORMALIZATION.json');guides=read(O/'word100_learned_guide_feedback_20261102/GUIDE_SUPPORT.json')['guides']
 ids=[g['canonical_stage_id'] for g in guides];inputs={};recordpins={};xs=[]
 for r in source['rows']:
  p=U/'records'/f"{r['clip_id']}.json";rec=read(p);assert rec['status']=='OK';assert sha(rec['channel_path'])==rec['channel_sha256'];recordpins[str(p)]=sha(p)
  a=np.load(rec['channel_path'],allow_pickle=False);inputs[r['clip_id']]={k:a[k].copy() for k in a.files}
  if r['role']=='fit':xs.append(np.where(coord_mask(a['feedback']),a['feedback'][:,:186],np.nan))
 x=np.concatenate(xs);mean=np.nanmean(x,axis=0);std=np.nanstd(x,axis=0);mean=np.nan_to_num(mean);std=np.where(np.isfinite(std)&(std>1e-4),std,1)
 stats=dict(mean=mean.tolist(),std=std.tolist(),guide_vectors=prior['guide_vectors'],guide_words=prior['guide_words'],prototype_lengths=prior['prototype_lengths'],guide_ids=ids,
   fit_only_normalization=True,coordinate_missing='zero after standardization, with masks retained')
 write(U/'FEEDBACK_NORMALIZATION.json',stats);rows=[];overlap_count=0;positive=np.zeros(7,int);negative=np.zeros(7,int)
 for r in source['rows']:
  a=inputs[r['clip_id']];n=len(a['frame_id']);y=np.zeros((n,7),np.float32);ym=np.zeros_like(y);sy=np.full(n,-1,np.int64);sw=np.zeros(n,np.float32);ci=np.full(n,-1,np.int64)
  for l in r['labels']:
   ix=np.flatnonzero(a['frame_id']==l['center_frame']);assert len(ix)==1 and int(a['timestamp_us'][ix[0]])==l['center_timestamp_us']
   y[ix[0]]=l['y'];ym[ix[0]]=l['target_mask']
  if r['role']=='fit':
   overlap=np.zeros(n,int)
   for s in r['stages']:
    lo,hi=s['frames'];sel=(a['frame_id']>=lo)&(a['frame_id']<=hi);overlap[sel]+=1;sy[sel]=ids.index(s['canonical_stage_id']);sw[sel]=1 if r['person']==1 else .2
   sy[overlap>1]=-1;sw[overlap>1]=0;overlap_count+=int((overlap>1).sum())
   mp=read(O/'word100_multipeople_live_feedback_20261101/stage_mappings'/f"{r['clip_id']}.json")
   assert mp['path_axes']==['other_reference_native_index','canonical_native_index']
   bucket=defaultdict(list)
   for qi,ri in mp['path']:bucket[int(mp['reference_frame_id'][qi])].append(ri)
   for i,fid in enumerate(a['frame_id']):
    if int(fid) in bucket:ci[i]=int(round(np.mean(bucket[int(fid)])))
   positive+=((y>.5)&(ym>0)).sum(0);negative+=((y<=.5)&(ym>0)).sum(0)
  path=U/'feedback_inputs'/f"{r['clip_id']}.npz";path.parent.mkdir(exist_ok=True)
  np.savez_compressed(path,x=normalize(a['feedback'],stats),raw_x=a['feedback'],y=y,ym=ym,sy=sy,sw=sw,ci=ci,frame_id=a['frame_id'],timestamp_us=a['timestamp_us'])
  rows.append(dict(clip_id=r['clip_id'],word=words.index(r['word_id']),person=r['person'],role=r['role'],path=str(path),sha256=sha(path)))
 assert np.all(positive+negative>0)
 write(U/'FEEDBACK_READY.json',dict(status='PASS',rows=rows,record_pins=recordpins,fit_positive=positive.tolist(),fit_negative=negative.tolist(),
  overlap_weak_frames_removed=overlap_count,stages=len(guides),unknowns_are_not_negative=True,development_weak_stage_labels_not_used=True,
  frozen_sha256=sha(U/'FROZEN.json'),normalization_sha256=sha(U/'FEEDBACK_NORMALIZATION.json')))
 event('feedback_inputs_ready',clips=len(rows),stages=len(guides))
if __name__=='__main__':main()
