"""Independent formulas over saved raw observations. Imports no adapter/model."""
from pathlib import Path
import json,hashlib
import numpy as np
U=Path(__file__).resolve().parent
def main():
 prep=json.loads((U/'PREPARED.json').read_text(encoding='utf-8'));rows=[]
 idx=[0,5,6,7,8,9,10,11,12]+list(range(91,133))
 for r in prep['rows']:
  assert hashlib.sha256(Path(r['source']).read_bytes()).hexdigest()==r['source_sha256']
  with np.load(r['source']) as z:xy=z['xy_px'];scores=z['score'];ts=z['timestamp_us'];fid=z['frame_id'];w=int(z['source_width']);h=int(z['source_height'])
  ok=np.isfinite(xy).all(2)&(xy[:,:,0]>=0)&(xy[:,:,0]<w)&(xy[:,:,1]>=0)&(xy[:,:,1]<h)
  shoulder=ok[:,5]&ok[:,6];scale=np.median(np.linalg.norm(xy[shoulder,5]-xy[shoulder,6],axis=1));center=(xy[:,5]+xy[:,6])/2
  body=np.zeros((len(ts),51,2),np.float32);bv=ok[:,idx]&shoulder[:,None]
  rel=(xy[:,idx]-center[:,None])/scale;body[bv]=rel[bv]
  hand=np.zeros((len(ts),42,2),np.float32);hv=np.zeros((len(ts),42),bool)
  for side,start in enumerate([91,112]):
   distances=[np.linalg.norm(xy[t,start+j]-xy[t,start]) for t in range(len(ts)) for j in [5,9,13,17] if ok[t,start] and ok[t,start+j]]
   hs=float(np.median(distances)) if distances else float('nan')
   if np.isfinite(hs) and hs>1e-6:
    for j in range(21):
     valid=ok[:,start]&ok[:,start+j];hv[:,side*21+j]=valid
     hand[valid,side*21+j]=(xy[valid,start+j]-xy[valid,start])/hs
  dt=np.r_[0,np.diff(ts)/1e6].astype(np.float32)[:,None];expected=np.concatenate([body.reshape(len(ts),102),hand.reshape(len(ts),84),bv,hv,dt],1).astype(np.float32)
  with np.load(r['bundle']) as z:
   assert np.array_equal(z['raw_xy_px'],xy,equal_nan=True) and np.array_equal(z['raw_score'],scores,equal_nan=True)
   assert np.array_equal(z['timestamp_us'],ts) and np.array_equal(z['frame_id'],fid)
   assert np.array_equal(z['point_available'],ok) and np.array_equal(z['visibility_evidence'],np.where(ok,-1,0))
   err=float(np.max(np.abs(z['tcn280']-expected)));assert err<1e-5
  rows.append(dict(clip_id=r['clip_id'],frames=len(ts),channels_checked=int(expected.size),max_error=err))
 result=dict(status='PASS',rows=rows,frame_total=sum(x['frames'] for x in rows),scalars_checked=sum(x['channels_checked'] for x in rows),training=0,inference=0,verification='Same Codex separate arithmetic; not visibility truth or model accuracy')
 (U/'VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
