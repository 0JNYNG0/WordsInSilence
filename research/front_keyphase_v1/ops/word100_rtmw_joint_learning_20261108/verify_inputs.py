"""Separate arithmetic: no imports of production converter or model."""
from common import *
import numpy as np,argparse
def expected(xy,ts,w,h):
 ok=np.isfinite(xy).all(2)&(xy[:,:,0]>=0)&(xy[:,:,0]<w)&(xy[:,:,1]>=0)&(xy[:,:,1]<h)
 sh=ok[:,5]&ok[:,6];scale=np.median(np.linalg.norm(xy[sh,5]-xy[sh,6],axis=1));origin=(xy[:,5]+xy[:,6])/2
 ids=[0,5,6,7,8,9,10,11,12]+list(range(91,133));bv=ok[:,ids]&sh[:,None]
 body=np.zeros((len(ts),51,2),np.float32);delta=(xy[:,ids]-origin[:,None])/scale;body[bv]=delta[bv]
 hand=np.zeros((len(ts),42,2),np.float32);hv=np.zeros((len(ts),42),bool)
 for side,start in enumerate([91,112]):
  inds=np.asarray([5,9,13,17])+start;dist=np.linalg.norm(xy[:,inds]-xy[:,start,None],axis=2);valid=ok[:,inds]&ok[:,start,None]
  hs=np.median(dist[valid]) if valid.any() else np.nan
  if np.isfinite(hs) and hs>1e-6:
   v=ok[:,start:start+21]&ok[:,start,None];hv[:,side*21:side*21+21]=v
   local=(xy[:,start:start+21]-xy[:,start,None])/hs;hand[:,side*21:side*21+21][v]=local[v]
 dt=np.r_[0,np.diff(ts)/1e6].astype(np.float32)[:,None]
 return np.concatenate([body.reshape(len(ts),102),hand.reshape(len(ts),84),bv,hv,dt],1).astype(np.float32),ok
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--feedback',action='store_true');args=ap.parse_args()
 name='FEEDBACK_INPUT_VERIFICATION.json' if args.feedback else 'INPUT_VERIFICATION.json';assert not (U/name).exists()
 if not args.feedback:assert (U/'EXTRACTION_COMPLETE.json').exists()
 rows=[r for r in read(U/'SOURCES.json') if r['feedback'] or not args.feedback];reports=[];fails=[];values=0
 fb={r['clip_id']:r for r in read(O/'word100_learned_guide_feedback_20261102/INPUT_MANIFEST.json')['rows']}
 for r in rows:
  rp=U/'records'/f"{r['clip_id']}.json";rec=read(rp);assert rec['source_sha256']==r['source_sha256'];assert sha(rec['raw_path'])==rec['raw_sha256'];assert sha(r['baseline_record'])==r['baseline_record_sha256']
  original=read(r['baseline_record']);assert original['source']['sha256']==r['source_sha256']
  with np.load(rec['raw_path'],allow_pickle=False) as a:
   xy=a['xy_px'];ts=a['timestamp_us'];fid=a['frame_id'];assert len(ts)==r['frames_header'];assert np.array_equal(fid,np.arange(len(fid)));assert np.all(np.diff(ts)>0)
   if rec['status']!='OK':fails.append(r['clip_id']);continue
   assert sha(rec['channel_path'])==rec['channel_sha256']
   with np.load(rec['channel_path'],allow_pickle=False) as z:
    assert np.array_equal(z['frame_id'],fid) and np.array_equal(z['timestamp_us'],ts)
    views={'full':r['height'],'upper_mask':int(original['calibration']['box_source_pixels'][3])}
    if r['feedback']:views['feedback']=fb[r['clip_id']]['box'][3]
    error=0
    for view,height in views.items():
     ex,ok=expected(xy,ts,r['width'],height);err=float(np.max(np.abs(z[view]-ex)));assert err<1e-5,(r['clip_id'],view,err);error=max(error,err);values+=ex.size
     assert z[view].shape==(len(fid),280)
    if r['feedback']:
     for label in fb[r['clip_id']]['labels']:assert int(ts[label['center_frame']])==label['center_timestamp_us']
  reports.append(dict(clip_id=r['clip_id'],frames=len(ts),max_error=error))
 # Failed development inputs stay in denominator. Missing classes/large input failures stop training.
 fit_failed=sum(r['role']=='fit' and r['clip_id'] in fails for r in rows);assert fit_failed<=.05*sum(r['role']=='fit' for r in rows)
 if args.feedback:assert not fails
 else:assert {r['word_id'] for r in rows if r['role']=='fit' and r['clip_id'] not in fails}==set(read(U/'FROZEN.json')['classes'])
 write(U/name,dict(status='PASS',clips=len(rows),rows=reports,input_failures=fails,scalars_checked=values,max_error=max(r['max_error'] for r in reports),
  record_hashes={r['clip_id']:sha(U/'records'/f"{r['clip_id']}.json") for r in rows},
  method='Same Codex separate raw-coordinate arithmetic, not independent human labels or detector accuracy'))
 event('input_verification_pass',scope='feedback' if args.feedback else 'all',clips=len(rows),scalars=values)
if __name__=='__main__':main()
