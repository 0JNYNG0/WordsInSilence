from pathlib import Path
import json,hashlib,collections
import numpy as np
from core import values,measurement
U=Path(__file__).resolve().parent;O=U.parent.parent;S=O/'word100_goal_state_learning_20261114'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
data=read(U/'INPUTS.json');ev=read(U/'EVALUATION.json');train=read(U/'TRAINING_COMPLETE.json');arr=np.load(S/'source_arrays.npz');checks=[]
def ck(n,b):checks.append(dict(name=n,passed=bool(b)))
for r in train['rows']:
 ck('checkpoint_hash',sha(Path(r['model']))==r['sha256'])
 ck('FIT_only',all(data['rows'][i]['split']=='fit'and 2<=data['rows'][i]['person']<=7 for i in r['fit_indices']))
 ck('weights_changed',r['weights_changed'])
for mode in ['body_wrist','stable_palm']:
 for axis in [0,1]:
  for split in ['fit','exposed_probe']:
   rr=[r for r in ev['rows']if r['mode']==mode and r['axis']==axis and r['split']==split]
   s=next(s for s in ev['summary']if (s['mode'],s['axis'],s['split'])==(mode,axis,split))
   ck('summary_denominator',s['n']==len(rr)and s['correct']+s['unknown']+s['wrong']==len(rr))
for cid in dict.fromkeys(r['clip_id']for r in data['rows']):
 x=arr[cid+'_x'];p=x[:,:102].reshape(-1,51,2);base,ok=values(x)
 # Common translation and camera scale do not change these body relations.
 for scale,shift in [(1.7,[.3,-.6]),(.6,[-.2,.4])]:
  xx=x.copy();xx[:,:102]=(p*scale+np.array(shift)).reshape(-1,102)
  q,v=values(xx);ck('body_scale_translation_invariant',np.array_equal(ok,v)and np.allclose(base[ok],q[ok],atol=1e-5))
 for axis in [0,1]:
  ts=arr[cid+'_timestamp_us'];d,available,_=measurement(x,ts,[0,len(x)-1],axis)
  rev,rv,_=measurement(x[::-1],ts[0]+ts[-1]-ts[::-1],[0,len(x)-1],axis)
  ck('reverse_signed_change',available==rv and abs(d+rev)<1e-5)
  xx=x.copy();xx[:,186:279]=0;_,v,_=measurement(xx,ts,[0,len(x)-1],axis);ck('unobserved_not_motion',not v)
# Review is explicitly AFTER the fixed scores. Preserve them, do not relabel
# the probe, remove a failure from its denominator, or retrain to the probe.
review=dict(status='PHYSICAL_TARGET_SCOPE_LIMITATION_FOUND_NOT_MODEL_ADOPTION',
 image_review=str(S/'review_record08.jpg'),image_sha256=sha(S/'review_record08.jpg'),
 observed_ko='기록08 f60→66→72에서는 손가락이 닫히고 팔꿈치가 들리며 손이 어깨 가까이에 온다. 손목/손바닥 중심이 크게 아래로 이동해야 한다는 정답은 이 사진만으로 뒷받침되지 않는다. f74→84는 팔꿈치 위치가 바뀌며 팔뚝이 더 세워진다.',
 invalid_inference='Family2 close_chest source identity does not itself label positive palm-y displacement. Generic guide action and one coordinate axis are different targets.',
 affected_probe_rows=[r['index']for r in data['rows']if r['split']=='exposed_probe'and r['axis']==0 and not r['freeze']],
 original_labels_and_scores_preserved=True,posthoc_accuracy_recalculation=False,new_probe_truth_created=False,
 decision='Do not adopt either calibration as an action-completion or guide-identity model; keep original frozen web/classifiers. Forearm change itself remains a limited physical observation, not a full action.',
 next_target='Directly review hand closing and hand/body region separately; allow simultaneous traits. Require trace evidence for which source action produced each state. Do not infer mandatory down-displacement from the family name.',
 source_variation_not_declared_error=True,independent_expert=False)
write('TARGET_REVIEW.json',review)
write('VERIFIED.json',dict(implementation_checks_pass=all(c['passed']for c in checks),checks=checks,failures=[c for c in checks if not c['passed']],
 training_runs=6,model_adopted=False,goal_model_qualified=False,
 distinction='Implementation/input separation verified. Target validity and held-out semantic capability NOT passed.',
 same_Codex_separate_review=True))
print(json.dumps(dict(checks=len(checks),implementation_pass=all(c['passed']for c in checks),target_validity_pass=False,model_adopted=False)))
