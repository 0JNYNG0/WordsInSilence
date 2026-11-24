from pathlib import Path
import json,hashlib,collections,sys
import numpy as np
from change_core import pair_features,completion_evidence,CLASSES
from reconcile import reconcile,attach
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_terminal_relation_targets_20261114';S=O/'word100_goal_state_learning_20261114';G=O/'word100_web_completion_graph_20261111'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(U/'FROZEN.json');meta=read(U/'PAIR_LABELS.json');evaluation=read(U/'EVALUATION.json');checks=[]
def check(n,b):checks.append(dict(name=n,passed=bool(b)))
for p,h in {**cfg['pins'],**meta['pins']}.items():check('frozen_input',sha(Path(p))==h)
for p,h in read(G/'PROTECTIONS.json').items():check('protected_model_web_original',sha(Path(p))==h)
for r in read(U/'TRAINING_COMPLETE.json')['rows']:
 check('checkpoint',sha(Path(r['path']))==r['sha256']);check('changed_weights',r['weights_changed'])
 check('fit_only',all(meta['rows'][i]['record_id']in cfg['fit_records']for i in r['fit_indices']))
 check('visible_fit_only',all(meta['rows'][i]['available']for i in r['fit_indices']))
probs=[np.load(U/'models'/f'{s}_probabilities.npy')for s in cfg['seeds']]
for r in evaluation['rows']:
 pp=np.stack([p[r['index']]for p in probs]);vote=pp.argmax(1);known=r['available']and len(set(vote.tolist()))==1 and pp.max(1).min()>=.8
 pred=CLASSES[int(vote[0])]if known else None
 check('saved_predictions',np.allclose(pp,r['probabilities'])and pred==r['prediction'])
 check('correct_arithmetic',r['correct']==(pred==CLASSES[r['target']]))
 check('not_whole_correctness',r['whole_action_complete']is None)
for s in evaluation['summary']:
 rr=[r for r in evaluation['rows']if r['split']==s['split']and r['variant']==s['scope']and CLASSES[r['target']]==s['target']]
 check('group_denominator',len(rr)==s['n']);check('group_correct',sum(r['correct']for r in rr)==s['correct'])
 check('group_partition',s['correct']+s['unknown']+s['wrong']==s['n'])
old=read(P/'EVALUATION.json')['dense'];joined=[]
for r in evaluation['rows']:
 cid=r['clip_id']
 if cid not in old:continue
 a,b=r['input_indices'];dd=old[cid][r['hand']]['shape'];states=dd['states'][a:b+1];available=dd['available'][a:b+1]
 if r['reverse']:states=states[::-1];available=available[::-1]
 if r['freeze']:states=[states[0]]*len(states);available=[available[0]]*len(available)
 obs=reconcile(states,r['prediction'],available,'four_closed',association_ok=False)
 obs_confirmed=reconcile(states,r['prediction'],available,'four_closed',association_ok=True)
 joined.append(dict(index=r['index'],clip_id=cid,hand=r['hand'],source_frames=r['source_frames'],evidence=obs,confirmed_interval_diagnostic=obs_confirmed))
 check('raw_prediction_preserved',obs['raw_change_prediction']==r['prediction'])
 check('no_correction_promotion',obs['educational_error']is None and obs['whole_action_complete']is None and not obs['display_enabled'])
 check('association_required',obs['status']=='action_association_unconfirmed')
for src in read(O/'word100_goal_binding_integration_20261114/JOINT_SAVED_COMPARISON.json')['rows']:
 candidate=attach(src,dict(status='not_evaluated_on_this_lineage',display_enabled=False));candidate.pop('research_handshape_change');check('existing_246_output_contract',candidate==src)
# A held bent pose is an unchanged pose, independent of its absolute curl.
arr=np.load(S/'source_arrays.npz');cid='NIA_SL_WORD0739_REAL08_F';raw=arr[cid+'_x'];xx=raw[62:67]
f,ok=pair_features(np.repeat(xx[:1],5,axis=0),'R');check('held_pose_zero_change',np.max(np.abs(f[:20]))==0)
f,ok=pair_features(xx,'R');rev,rok=pair_features(xx[::-1],'R');check('time_reverse_sign',np.allclose(f[:20],-rev[:20])and np.allclose(f[20:],rev[20:]))
# Do not drop a hidden internal repetition when the first and last posture match.
test=reconcile(['four_open','four_closed','four_open'],'shape_maintained',[True]*3,'four_closed',True)
check('internal_changes_retained',test['component_status']=='multiple_category_changes_retained'and len(test['observed_category_transitions'])==2)
test=reconcile(['four_closed']*4,'opening',[True]*4,'four_closed',True)
check('projection_conflict_not_error',test['component_status']=='change_without_confirmed_category_change'and test['educational_error']is None)
test=reconcile([None]*4,'closing',[False]*4,'four_closed',True);check('missing_not_wrong',test['component_status']=='observation_uncertain')
for r in read(U/'AUDIT_INPUTS.json')['rows']:check('pixel_mapping',r['mapping_max_px_error']<.01)
probe=[r for r in evaluation['rows']if r['split']=='exposed_probe'];errors=[r for r in probe if not r['correct']]
check('admission_not_fabricated',len(errors)==2)
assert all(c['passed']for c in checks)
with(U/'RECONCILIATION.json').open('x',encoding='utf-8')as f:json.dump(dict(rows=joined,raw_outputs_unchanged=True,automatic_educational_correction=False,web_changed=False),f,ensure_ascii=False,indent=2)
out=dict(passed=True,total=len(checks),checks=checks,model_admission_passed=False,web_changed=False,
 probe_conditions=len(probe),probe_correct=sum(r['correct']for r in probe),errors=errors,
 source_video_count=len({r['clip_id']for r in meta['rows']}),
 verification='Same Codex separate numerical/contract verification, not independent human KSL validation',
 original_endpoint_hash_unchanged=sha(P/'DENSE_PROBABILITIES.npz')==evaluation['endpoint_predictions_sha256'])
with(U/'VERIFIED.json').open('x',encoding='utf-8')as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({k:out[k]for k in ['passed','total','probe_conditions','probe_correct','source_video_count','original_endpoint_hash_unchanged']},ensure_ascii=False))
