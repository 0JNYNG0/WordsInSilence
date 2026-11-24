"""Reinterpret saved posteriors; preserve original evaluations and time paths."""
from copy import deepcopy
from collections import Counter
from prepare import U,read,save
def main():
 rows=[]
 for r in read(U/'FEEDBACK_EVALUATION.json')['rows']:
  old=read(r['path']);d=deepcopy(old);changes=[]
  for s in d['stages']:
   for f in s['features']:
    if f['status']=='DIFFERENCE_CANDIDATE' and int(f['observed_mean']>=.5)==int(f['expected']):
     changes.append([s['guide_id'],f['name']]);f['status']='CLASSIFICATION_CONFIDENCE_LOW'
   if s['status']=='FEATURE_DIFFERENCE_CANDIDATE' and not any(f['status']=='DIFFERENCE_CANDIDATE' for f in s['features']):
    s['status']='IMAGE_TIME_ESTIMATE_ONLY';s['message']='확신이 낮은 항목을 동작 오류로 확정하지 않습니다.'
  assert d['path']==old['path'] and d['evidence_gaps']==old['evidence_gaps']
  name='feedback_v2/'+__import__('pathlib').Path(r['path']).name;save(name,d)
  rows.append(dict(condition=r['condition'],seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],changes=changes,
   statuses=dict(Counter(s['status'] for s in d['stages'])),difference_candidates=sum(s['status']=='FEATURE_DIFFERENCE_CANDIDATE' for s in d['stages']),path=str(U/name)))
 save('FEEDBACK_V2_AUDIT.json',dict(status='PASS_SAME_CLASS_CONFIDENCE_SEPARATED',rows=rows,new_inference=0,
  retraining=0,changed_class_predictions=0,changed_dtw_paths=0,scope='Development bug correction after v1 review; not fresh evaluation',
  short_omission_limit_unchanged=True,notes=['Posterior confidence is not a physical angle/distance.','Original model and v1 outputs preserved.']))
 print('v2 cases',len(rows),'changed candidates',sum(len(r['changes']) for r in rows))
if __name__=='__main__':main()
