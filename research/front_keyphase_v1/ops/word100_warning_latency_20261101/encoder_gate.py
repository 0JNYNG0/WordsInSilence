"""Once-only paired encoder gate on identical frozen capture/ROI and pixels."""
from benchmark import *
class Recorder:
 def __init__(self,compiled):self.compiled=compiled;self.inputs=[]
 def __call__(self,items):
  a=items[0];self.inputs.append(hashlib.sha256(a.tobytes()).hexdigest())
  return self.compiled(items)
def main():
 assert (U/'BENCHMARK.json').is_file(),'No concurrent GPU benchmark'
 save(U/'ENCODER_GATE_STARTED.json',dict(time=time.time(),candidate_sha256=sha(U/'candidate_adapter.py')))
 base=module('gate_serial',B/'adapter.py');cand=module('gate_candidate',U/'candidate_adapter.py')
 compiled,device=base._compile_signrep();compiled([np.zeros((1,3,16,224,224),np.float32)])
 results=[]
 for ci,row in enumerate(read(U/'FROZEN.json')['public_clips']):
  d=U/'bench'/f'clip{ci+1}_1_serial'/'adapter';meta=read(d/'COMPLETE.json')
  with np.load(d/'analysis.npz') as z:fid=z['frame_id'].copy();ts=z['timestamp_us'].copy()
  outs=[];records=[]
  for condition,m in [('serial',base),('candidate',cand)]:
   rec=Recorder(compiled);t=time.perf_counter();a=m._encode(Path(row['source']),meta['calibration']['box_source_pixels'],fid,ts,rec,None)
   outputs=dict(zip(['rgb','features','starts','centers'],a));out=U/f'gate_{ci+1}_{condition}.npz'
   with out.open('xb') as f:np.savez_compressed(f,**outputs)
   outs.append(out);records.append(dict(condition=condition,seconds=time.perf_counter()-t,input_sha256=rec.inputs))
  equality=compare(*outs)
  passed=all(x['equal'] for x in equality.values()) and records[0]['input_sha256']==records[1]['input_sha256']
  results.append(dict(clip=row['clip_id'],native_frames=len(fid),windows=len(records[0]['input_sha256']),records=records,equality=equality,passed=passed))
  print(row['clip_id'],'PASS' if passed else 'FAIL',flush=True)
 save(U/'ENCODER_GATE.json',dict(status='PASS' if all(r['passed'] for r in results) else 'FAIL',results=results,device=device))
if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/'ENCODER_GATE_FIRST_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
