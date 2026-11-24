from core import *
import copy,time
def main():
 torch.set_num_threads(2);factory=Features();ob=Observer('context_gru');out=[]
 sys.path.insert(0,str(O/'word100_rtmw_shared_input_20261108'));import adapter
 for r in read(U/'INPUTS.json')['rows']:
  if r['role']!='development':continue
  raw=dict(np.load(r['raw']));orig=dict(np.load(r['channels']));y=np.load(U/'cache'/f'{r["clip_id"]}.npz')['y'];first=int(np.flatnonzero(y==2)[0]);far=int(np.flatnonzero(y==1)[0]);n=len(y)
  for kind in ['original','fast_timestamps','slow_timestamps','prefix_one_approach','constant_pose','missing_hands','scale_translation']:
   d={k:v.copy() for k,v in raw.items()};dd=None
   if kind=='prefix_one_approach':
    for k in ['xy_px','score','timestamp_us','frame_id']:d[k]=d[k][:first+2]
   if kind=='constant_pose':d['xy_px']=np.repeat(d['xy_px'][far:far+1],n,0);d['score']=np.repeat(d['score'][far:far+1],n,0)
   if kind=='missing_hands':d['xy_px'][:,91:]=np.nan
   if kind=='fast_timestamps':d['timestamp_us']=(d['timestamp_us']*.65).astype(np.int64)
   if kind=='slow_timestamps':d['timestamp_us']=(d['timestamp_us']*1.5).astype(np.int64)
   if kind=='scale_translation':
    d['xy_px']=d['xy_px']*.75+np.array([90,40],np.float32)
   t=time.perf_counter();ch=adapter.convert(d['xy_px'],d['score'],d['timestamp_us'],d['frame_id'],int(d['source_width']),int(d['source_height']));f,q,coarse=factory(ch);pred=ob(f,q,ch['timestamp_us']);dt=time.perf_counter()-t
   success=(len(pred['approaches'])>=2 if kind in ['original','fast_timestamps','slow_timestamps','scale_translation'] else len(pred['approaches'])<2)
   if kind in ['missing_hands','constant_pose']:success=len(pred['approaches'])==0
   out.append(dict(clip_id=r['clip_id'],kind=kind,detected_approaches=pred['approaches'],known_frames=int((pred['states']>=0).sum()),seconds=dt,diagnostic_pass=success,full_sign_acceptance=None))
 write(U/'DIAGNOSTICS.json',dict(rows=out,passed=all(r['diagnostic_pass']for r in out),scope='Engineered observability/partial/speed probes, not natural learner error accuracy'))
 print('DIAGNOSTICS',sum(r['diagnostic_pass']for r in out),'/',len(out),flush=True)
if __name__=='__main__':main()
