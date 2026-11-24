import core
P=core.U;core.U=P/'source_corrected'
from core import *
import time,copy
GOLD={14:{0:[97,100],1:[46,49,55,58,64,67,70,73,76,79,82,85,88],2:[52,61]},
      15:{0:[41,44,92,95,98],1:[50,53,62,65],2:[56,59,68,71,74,77,80,83,86]}}
def main():
 torch.set_num_threads(2);manifest=read(U/'CONFIRMATION_INPUTS.json');truth=[]
 for r in manifest['rows']:
  for state,frames in GOLD[r['person']].items():
   for f in frames:truth.append(dict(clip_id=r['clip_id'],frame=f,state=state,source_sha=r['source_sha256']))
 assert not(U/'CONFIRMATION_RESULTS.json').exists();write(U/'CONFIRMATION_TRUTH.json',dict(rows=truth,review='Large native RGB crops before any candidate prediction. Proximity, not physical contact; no guessed labels.'))
 factory=Features();ob=Observer(read(U/'CONFIRMATION_FROZEN.json')['chosen']);out=[];anchor_results=[]
 sys.path.insert(0,str(O/'word100_rtmw_shared_input_20261108'));import adapter
 sys.path.insert(0,str(O/'word100_observation_guard_20261108'));from guard import quarantine
 for r in manifest['rows']:
  raw=dict(np.load(r['raw']));n=len(raw['frame_id']);first=GOLD[r['person']][2][0];far=GOLD[r['person']][1][0]
  for kind in ['original','live_guard','fast_timestamps','slow_timestamps','prefix_one_approach','constant_pose','missing_hands','scale_translation']:
   d={k:v.copy()for k,v in raw.items()}
   if kind=='live_guard':d['xy_px']=quarantine(d['xy_px'],d['score'],d['timestamp_us'],int(d['source_width']),int(d['source_height']))['xy_px']
   if kind=='prefix_one_approach':
    for k in ['xy_px','score','timestamp_us','frame_id']:d[k]=d[k][:first+2]
   if kind=='constant_pose':d['xy_px']=np.repeat(d['xy_px'][far:far+1],n,0);d['score']=np.repeat(d['score'][far:far+1],n,0)
   if kind=='missing_hands':d['xy_px'][:,91:]=np.nan
   if kind=='fast_timestamps':d['timestamp_us']=(d['timestamp_us']*.65).astype(np.int64)
   if kind=='slow_timestamps':d['timestamp_us']=(d['timestamp_us']*1.5).astype(np.int64)
   if kind=='scale_translation':d['xy_px']=d['xy_px']*.75+np.array([90,40],np.float32)
   t=time.perf_counter();ch=adapter.convert(d['xy_px'],d['score'],d['timestamp_us'],d['frame_id'],int(d['source_width']),int(d['source_height']));f,q,coarse=factory(ch);pred=ob(f,q,ch['timestamp_us']);dt=time.perf_counter()-t
   success=len(pred['approaches'])==2 if kind in ['original','live_guard','fast_timestamps','slow_timestamps','scale_translation'] else len(pred['approaches'])<=1
   if kind in ['constant_pose','missing_hands']:success=len(pred['approaches'])==0
   rr=dict(clip_id=r['clip_id'],kind=kind,approaches=pred['approaches'],observed_frames=int((pred['states']>=0).sum()),seconds=dt,diagnostic_pass=success,whole_word_correct=None)
   out.append(rr)
   np.savez_compressed(U/'confirmation'/f'{r["clip_id"]}_{kind}.npz',states=pred['states'],probabilities=pred['probabilities'],ts=d['timestamp_us'],xy=d['xy_px'],coarse=coarse)
   if kind=='original':
    for a in truth:
     if a['clip_id']==r['clip_id']:anchor_results.append(dict(**a,pred=int(pred['states'][a['frame']])))
 write(U/'CONFIRMATION_RESULTS.json',dict(rows=out,anchors=anchor_results,
  summary=dict(anchors=len(anchor_results),correct=sum(a['state']==a['pred']for a in anchor_results),wrong=sum(a['state']!=a['pred']and a['pred']>=0 for a in anchor_results),unknown=sum(a['pred']<0 for a in anchor_results),diagnostic_passes=sum(r['diagnostic_pass']for r in out),diagnostic_total=len(out)),
  globally_unseen=False,scope='Current candidate selection withheld people, previously used in finger work; not fresh project-wide final test. No natural learner errors. Timing excludes extraction/capture/DTW/render.',web_admitted=False))
 print('CONFIRMATION',read(U/'CONFIRMATION_RESULTS.json')['summary'])
if __name__=='__main__':main()
