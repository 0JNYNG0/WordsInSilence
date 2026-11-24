from common import *
import numpy as np,torch
from runtime_candidate import Candidate
def main():
 assert not (U/'MODEL_TIMING.json').exists();assert (U/'TCN_TRAINING_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 selected=[r for r in read(U/'SOURCES.json') if r['feedback'] and r['role']=='development'][:3];rows=[];cold=[]
 for tc in ['fresh','finetune_C']:
  for fc in ['pose_fresh','pose_transfer']:
   t=time.perf_counter();m=Candidate(tc,fc);cold.append(dict(tcn=tc,feedback=fc,seconds=time.perf_counter()-t))
   for r in selected:
    rec=read(U/'records'/f"{r['clip_id']}.json");z=np.load(rec['channel_path']);x=z['full'];ts=z['timestamp_us'];fid=z['frame_id']
    # Public full actual image path, not a reference-derived crop at serving time.
    m.from_channels(x,ts,fid,r['word_id']);dur=[];parts=[];last=None
    for _ in range(20):
     t=time.perf_counter();last=m.from_channels(x,ts,fid,r['word_id']);dur.append(time.perf_counter()-t);parts.append(last['timing'])
    raw=np.load(rec['raw_path']);t=time.perf_counter();full=m.from_raw(raw,r['word_id']);input_and_models=time.perf_counter()-t
    assert full['tcn_top3']==last['tcn_top3'];assert full['path']==last['path']
    p=U/'combined_outputs'/f"{tc}_{fc}_{r['clip_id']}.json";write(p,full)
    rows.append(dict(tcn=tc,feedback=fc,clip_id=r['clip_id'],frames=len(x),repeats=20,warm_total_p50_ms=float(np.median(dur)*1000),warm_total_p95_ms=float(np.percentile(dur,95)*1000),
     parts_p50_ms={k:float(np.median([p[k] for p in parts])*1000) for k in parts[0]},raw_input_and_models_ms=input_and_models*1000,output=str(p),sha256=sha(p)))
 write(U/'MODEL_TIMING.json',dict(status='COMPLETE',rows=rows,cold_model_load=cold,scope='Already extracted coordinates → shared features, TCN, feedback, DTW and JSON-ready result. No camera, recording, detector or GUI time.',
   sample_selection='First three feedback development clips in frozen source list, fixed before scores',threads=4,seeds={'tcn':20261013,'feedback':42},models_on='CPU'))
 event('model_timing_complete',conditions=4,clips=3)
if __name__=='__main__':main()
