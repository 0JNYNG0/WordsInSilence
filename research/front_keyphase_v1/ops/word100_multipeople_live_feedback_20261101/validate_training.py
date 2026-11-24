import json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 d=read(U/'TRAIN_DATASET.json'); f=read(U/'FROZEN.json'); old=read(O/'word100_semantic_feature_cycle_20261101/DATASET.json')
 checks=0
 def ck(v):
  nonlocal checks
  assert v;checks+=1
 ck(len(d['rows'])==288)
 base=[r for r in d['rows'] if r['person']<=3];oldbase=[r for r in old['rows'] if r['role']=='fit']
 ck(len(base)==len(oldbase)==180)
 before={r['sample_id']:r for r in oldbase}
 for r in base:
  b=before[r['sample_id'].removeprefix('old:')]
  for key in ['clip_id','input_sha256','frame_ids','y','target_mask']:ck(r[key]==b[key])
 fit=[r for r in d['rows'] if r['person'] in f['fit_person_ids']]
 dev=[r for r in d['rows'] if r['person'] in f['development_person_ids']]
 ck(len(fit)==252 and len(dev)==36);ck({r['person'] for r in fit}.isdisjoint({r['person'] for r in dev}))
 ck({r['clip_id'] for r in fit}.isdisjoint({r['clip_id'] for r in dev}))
 ck(len({r['sample_id'] for r in d['rows']})==len(d['rows']))
 cache={}
 for r in d['rows']:
  p=r['input'];ck(sha(p)==r['input_sha256'])
  if p not in cache:
   with np.load(p,allow_pickle=False) as z:cache[p]={k:z[k].copy() for k in ['x','mask','rgb','frame_id','timestamp_us']}
  a=cache[p];idx=[int(np.flatnonzero(a['frame_id']==v)[0]) for v in r['frame_ids']]
  ck(len(idx)==13 and idx==list(range(idx[0],idx[0]+13)));ck(a['timestamp_us'][idx].tolist()==r['timestamps_us']);ck(r['center_frame']==r['frame_ids'][6])
  ck(a['x'].shape[1]==69 and a['rgb'].shape[1]==768);ck(np.isfinite(a['x'][a['mask']]).all() and np.isfinite(a['rgb']).all())
  ck(len(r['y'])==len(r['target_mask'])==7 and all(isinstance(m,bool) for m in r['target_mask']))
 ck(all(any(r['target_mask'][j] and r['y'][j]==1 for r in base) for j in range(7)))
 sec=read(U/'LABEL_SECONDARY_REVIEW.json');ck(sec['development_centers_reviewed']==36);ck(sec['label_file_sha256']==sha(U/'data_audit/LABELS.json'))
 out=dict(status='PASS_INPUTS_LABEL_JOIN_AND_PERSON_DISJOINT',checks=checks,fit_people=list(range(1,10)),development_people=[11,12,13],fit_windows=252,development_windows=36,base_replay_windows=180,new_condition_windows=72,second_visual_review=sec,known_dev_cells=sum(sum(r['target_mask']) for r in dev),unknown_dev_cells=sum(7-sum(r['target_mask']) for r in dev),limitations=['Development protocol, historical project exposure declared','Same-Codex visual research labels, not independent KSL educational truth'])
 (U/'DATASET_VALIDATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps({k:v for k,v in out.items() if k!='second_visual_review'},ensure_ascii=False))
if __name__=='__main__':main()
