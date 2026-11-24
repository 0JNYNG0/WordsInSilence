"""Prespecified engineering perturbations, never new KSL correctness labels."""
from common import *
import numpy as np,torch,traceback
from feedback_model import PoseGuideNet
from prepare_feedback import normalize
sys.path.insert(0,str(O/'word100_learned_guide_feedback_20261102'))
from alignment import dtw,representative,stage_at

def infer(model,x,word):
 with torch.inference_mode():
  o=model(torch.from_numpy(x[None]),torch.tensor([len(x)]),torch.tensor([word]))
 return o['attributes'][0].sigmoid().numpy(),o['stage'][0].softmax(-1).numpy(),o['latent'][0].numpy()
def gate(x,ts):
 # This is a non-semantic evidence gate, not a detector visibility or correctness guarantee.
 sides=[]
 for s in range(2):
  available=(x[:,237+s*21:258+s*21].sum(1)>0);state=np.full(len(x),'coordinate_estimate',dtype='<U28')
  for i in np.flatnonzero(~available):
   left=np.flatnonzero(available[:i]);right=np.flatnonzero(available[i+1:])+i+1
   short=len(left) and len(right) and ts[right[0]]-ts[left[-1]]<=100000
   state[i]='temporal_estimate_only' if short else 'insufficient_observation'
  sides.append(state)
 return np.stack(sides,1)
def remove_hands(x,lo,hi):
 z=x.copy();z[lo:hi,18:102]=0;z[lo:hi,102:186]=0;z[lo:hi,195:237]=0;z[lo:hi,237:279]=0
 return z
def main():
 assert not (U/'FEEDBACK_CONTROLS.json').exists();assert (U/'FEEDBACK_TRAINING_COMPLETE.json').exists()
 torch.set_num_threads(4);stats=read(U/'FEEDBACK_NORMALIZATION.json');f=read(U/'FROZEN.json')['feedback'];ready=read(U/'FEEDBACK_READY.json')
 old=O/'word100_learned_guide_feedback_20261102';manifest={r['clip_id']:r for r in read(old/'INPUT_MANIFEST.json')['rows']};guides=read(old/'GUIDE_SUPPORT.json')['guides'];ids=stats['guide_ids'];anchors=read(old/'BLIND_STAGE_ANCHORS.json')['rows'];output=[]
 for condition in f['conditions']:
  for seed in f['seeds']:
   d=U/'feedback_runs'/f'{condition}_{seed}';model=PoseGuideNet(stats['guide_vectors'],stats['guide_words'],stats['prototype_lengths']);model.load_state_dict(torch.load(d/'model.pt',map_location='cpu',weights_only=True));model.eval()
   for row in [r for r in ready['rows'] if r['role']=='development']:
    cid=row['clip_id'];z=np.load(row['path']);raw=z['raw_x'];ts=z['timestamp_us'];n=len(ts);meta=manifest[cid];wi=row['word'];pred=np.load(d/'predictions'/f'{cid}.npz');center=meta['labels'][1]['center_frame'];rows=[]
    canonical=model.prototype[wi,:stats['prototype_lengths'][wi]].detach().numpy()
    main_lo=max(0,int(round(meta['intervals'][0][0]*30)));main_hi=min(n,int(round(meta['intervals'][0][1]*30))+1)
    for kind in ['speed_075','speed_125','gap_2_frames','gap_30_frames','main_reversed','main_cut_5_frames']:
     mapping=np.arange(n);x=raw.copy();out_ts=ts.copy();removed=[]
     if kind.startswith('speed_'):
      speed=.75 if kind=='speed_075' else 1.25;mapping=np.rint(np.linspace(0,n-1,round(n/speed))).astype(int);x=raw[mapping].copy();out_ts=np.rint(np.arange(len(mapping))*float(np.median(np.diff(ts)))).astype(np.int64)
     elif kind.startswith('gap_'):
      size=2 if kind=='gap_2_frames' else 30;lo=max(0,min(n-size,center-size//2));x=remove_hands(raw,lo,lo+size);removed=list(range(lo,lo+size))
     elif kind=='main_reversed':mapping[main_lo:main_hi]=mapping[main_lo:main_hi][::-1];x=raw[mapping].copy()
     else:
      lo=max(main_lo,min(main_hi-5,center-2));mapping=np.delete(mapping,np.arange(lo,lo+5));x=raw[mapping].copy();out_ts=ts[:len(mapping)].copy();removed=list(range(lo,lo+5))
     x[:,-1]=np.r_[0,np.diff(out_ts)/1e6];p,st,h=infer(model,normalize(x,stats),wi);al=dtw(h,canonical);states=gate(x,out_ts)
     labels=[];stage_labels=[]
     for l in meta['labels']:
      if kind=='main_cut_5_frames' and l['center_frame'] in removed:continue
      ix=int(np.argmin(np.abs(mapping-l['center_frame'])));known=np.asarray(l['target_mask'],bool);gt=np.asarray(l['y'])>.5
      labels.append(dict(frame_id=l['center_frame'],mapped_index=ix,known=int(known.sum()),correct=int(((p[ix]>=.5)==gt)[known].sum()),probability=p[ix].tolist(),gate=states[ix].tolist()))
     for a in [a for a in anchors if a['clip_id']==cid]:
      if kind=='main_cut_5_frames' and a['frame_id'] in removed:continue
      ix=int(np.argmin(np.abs(mapping-a['frame_id'])));j=representative(al['path'],ix)
      stage_labels.append(dict(frame_id=a['frame_id'],prediction=ids[int(st[ix].argmax())],dtw_prediction=stage_at(j,guides,meta['word_id']),truth=a['stage_id']))
     result=dict(condition=condition,seed=seed,clip_id=cid,kind=kind,frames=len(x),removed_original_frames=removed,center_labels=labels,stage_labels=stage_labels,
       mean_dtw_cost=al['mean_cost'],max_attribute_delta=float(np.max(np.abs(p-pred['probability'][mapping]))),
       insufficient_hand_frames=int((states=='insufficient_observation').any(1).sum()),estimated_only_hand_frames=int((states=='temporal_estimate_only').any(1).sum()),
       semantic_wrong_label=False,all_query_frames_in_alignment=len(np.unique(al['path'][:,0]))==len(x))
     if kind=='gap_30_frames':assert result['insufficient_hand_frames']>=30
     assert result['all_query_frames_in_alignment'];output.append(result)
 write(U/'FEEDBACK_CONTROLS.json',dict(status='ENGINEERING_CONTROLS_COMPLETE',rows=output,
  missing_policy='Never announce normal/correct when long hand observations absent. Short gaps explicitly temporal estimate, not directly observed.',
  limitations=['Synthetic feature-time/occlusion/order controls, not actual camera speed effects or labelled learner mistakes','In-frame RTMW hallucinations are not detected by geometric availability alone','No threshold chosen using these control scores']))
 event('feedback_controls_complete',cases=len(output))
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'CONTROL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
