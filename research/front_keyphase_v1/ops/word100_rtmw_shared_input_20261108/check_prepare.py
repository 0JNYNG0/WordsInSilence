from pathlib import Path
import json,hashlib,re
from collections import Counter
import numpy as np
from adapter import U,convert,load,SCHEMA
O=U.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 checks=[]
 def check(n,v):assert v,n;checks.append(n)
 # Distinct left/right and body markers; no language/word-specific branches.
 xy=np.empty((5,133,2),np.float32);xy[:,:,0]=np.arange(133)*2+100;xy[:,:,1]=np.arange(133)+100
 score=np.ones((5,133),np.float32)*5;ts=np.arange(5,dtype=np.int64)*33333;fid=np.arange(5,dtype=np.int64)
 a=convert(xy,score,ts,fid,1000,1000)
 for dst,src in [(0,0),(5,5),(2,6),(6,7),(3,8),(7,9),(4,10),(12,11),(9,12)]:check('body_map_'+str(src),np.array_equal(a['native67'][:,dst],xy[:,src]))
 check('anatomical_left21',np.array_equal(a['native67'][:,25:46],xy[:,91:112]));check('anatomical_right21',np.array_equal(a['native67'][:,46:],xy[:,112:133]))
 b=convert(xy*2+50,score,ts+100000,fid,2100,2100);check('translation_scale_invariant',np.allclose(a['tcn280'],b['tcn280'],atol=1e-5))
 b=convert(xy,score*100,ts,fid,1000,1000);check('raw_score_not_visibility',np.array_equal(a['visibility_evidence'],b['visibility_evidence']) and not np.any(b['visibility_evidence']==1))
 bad=xy.copy();bad[2,91:112]=np.nan;b=convert(bad,score,ts,fid,1000,1000)
 check('gap_timeline_preserved',np.array_equal(b['timestamp_us'],ts) and len(b['tcn280'])==5)
 check('missing_hand_not_filled',not b['point_available'][2,91:112].any() and np.isnan(b['raw_xy_px'][2,91:112]).all())
 try:convert(xy,score,ts[::-1],fid,1000,1000);raise AssertionError('invalid clock accepted')
 except ValueError:checks.append('nonmonotone_rejected')
 prior=O/'word100_rtmw_browser_capture_20261108';complete=json.loads((prior/'NATIVE_PARALLEL_COMPLETE.json').read_text(encoding='utf-8'));rows=[]
 for r in complete['runs']:
  source=prior/'native_runs'/r['id']/'rtmw/observations.npz';bundle=load(source);out=U/'bundles'/(r['clip_id']+'.npz');out.parent.mkdir(exist_ok=True)
  with np.load(source) as z:
   check(r['label']+' raw exact',np.array_equal(bundle['raw_xy_px'],z['xy_px'],equal_nan=True))
   check(r['label']+' pts exact',np.array_equal(bundle['timestamp_us'],z['timestamp_us']))
  check(r['label']+' channel shape',bundle['tcn280'].shape==(r['frames'],280))
  np.savez_compressed(out,**bundle);rows.append(dict(clip_id=r['clip_id'],source=str(source),source_sha256=sha(source),bundle=str(out),bundle_sha256=sha(out),frames=r['frames']))
 train=json.loads((O/'word100_training_20261013/MANIFEST.json').read_text(encoding='utf-8'))
 feedback=json.loads((O/'word100_learned_guide_feedback_20261102/INPUT_MANIFEST.json').read_text(encoding='utf-8'))
 inventory=dict(tcn_roles=dict(Counter(r['role'] for r in train)),tcn_words=len(set(r['word_code'] for r in train)),tcn_signers_by_role={role:sorted(set(r['signer_id'] for r in train if r['role']==role)) for role in set(r['role'] for r in train)},feedback_roles=dict(Counter(r['role'] for r in feedback['rows'])),feedback_targets=feedback['targets'],feedback_words=sorted(set(r['word_id'] for r in feedback['rows'])),public_manifest_only=True,video_archives_not_extracted=True,old_development_not_new_test=True)
 result=dict(status='INPUT_ADAPTER_VERIFIED_TRAINING_NOT_STARTED',schema=SCHEMA,checks=checks,rows=rows,inventory=inventory,limits='No new detector calls. In-frame point availability is not visibility truth. No learned RTMW classifier or feedback weights yet; full training source/labels/hash audit remains.',training=0,private_data=False,production_changed=False)
 (U/'PREPARED.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(status=result['status'],checks=len(checks),frames=sum(r['frames'] for r in rows),inventory=inventory),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
