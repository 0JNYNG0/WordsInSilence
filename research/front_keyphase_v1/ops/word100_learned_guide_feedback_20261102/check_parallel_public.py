"""All frames retained: parallel/serial parity + real pipeline latency."""
import json,numpy as np
from live_adapter import U,O,WORDS,analyze,warmup,write,read
def main():
 assert not (U/'PARALLEL_PUBLIC_VERIFICATION.json').exists()
 cold=warmup();rows=[]
 for w in WORDS:
  source=O/'word100_seated_web_framing_20261102/public_crops'/f'NIA_SL_{w}_REAL12_F.mp4';dest=U/'parallel_public'/w
  if (dest/'RESULT.json').exists():r=read(dest/'RESULT.json')
  else:r=analyze(source,dest,w,progress=lambda e:print(w,e,flush=True))
  old=read(U/'live_public'/w/'RESULT.json');a=np.load(U/'live_public'/w/'analysis.npz');b=np.load(dest/'analysis.npz')
  assert np.array_equal(a['frame_id'],b['frame_id']) and np.array_equal(a['timestamp_us'],b['timestamp_us']) and np.array_equal(a['mask'],b['mask'])
  delta={k:float(np.max(np.abs(a[k]-b[k]))) for k in ['x','rgb','prelinear','probability','stage_probability','latent']}
  assert all(v<.0001 for v in delta.values()),delta
  assert old['path']==r['path'] and [s['status'] for s in old['stages']]==[s['status'] for s in r['stages']]
  rows.append(dict(word_id=w,serial_seconds=old['latency']['total_analysis_seconds'],parallel_seconds=r['latency']['total_analysis_seconds'],delta=delta,exact_path=True,exact_status=True))
  print('verified',rows[-1],flush=True)
 write(U/'PARALLEL_PUBLIC_VERIFICATION.json',dict(status='PASS_ALL_FRAMES_SAME_RESULTS',rows=rows,cold_start=cold,no_precision_resolution_stride_change=True,private_inputs=0))
if __name__=='__main__':main()
