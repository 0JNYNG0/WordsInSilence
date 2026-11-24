from pathlib import Path
import json,hashlib,collections
import numpy as np
T=Path(__file__).resolve().parent;U=T.parent
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(T/'FROZEN.json');data=read(T/'EVALUATION.json');labels=read(T/'LABELS.json')['rows'];checks=[]
def check(name,value):checks.append(dict(name=name,passed=bool(value)))
for p,h in cfg['pins'].items():check('frozen:'+Path(p).name,sha(Path(p))==h)
for r in read(T/'TRAINING_COMPLETE.json')['rows']:
 check('model_sha',sha(Path(r['path']))==r['sha256']);check('weights_changed',r['weights_changed'])
 check('fit_only',all(labels[i]['record_id']in cfg['fit_records']for i in r['fit_indices']))
 check('known_visible_only',all(labels[i]['label']is not None and labels[i]['available']for i in r['fit_indices']))
pp=np.load(T/'PROBABILITIES.npz')
for r,l in zip(data['rows'],labels):
 p=pp[r['clip_id']+'_'+r['hand']][:,l['array_index']];votes=p.argmax(1)
 known=l['available']and len(set(votes.tolist()))==1 and np.min(p.max(1))>=.8
 pred=cfg['classes'][votes[0]]if known else None;check('saved_probability_recount',pred==r['new'])
for s in data['summary']:
 rr=[r for r in data['rows']if(r['split'],r['origin'])==(s['split'],s['origin'])and r['label']is not None]
 check('denominator',len(rr)==s['n']);check('correct_count',sum(r['new']==r['label']for r in rr)==s['new_correct'])
 check('partition',s['new_correct']+s['new_wrong']+s['new_unknown']==s['n'])
for r in data['prefixes']:
 check('actual_prefix_not_completed',not r['result']['new']['feature_sequence_observed'])
 check('no_educational_error',r['result']['new']['educational_error']is None)
for r in data['traces']:
 check('trace_no_whole_completion',r['result']['whole_action_complete']is None)
known=[r for r in data['rows']if r['split']=='exposed_probe'and r['origin']=='original_visible_state'and r['label']]
regress=[r for r in known if r['old']==r['label']and r['new']!=r['label']]
old_good=sum(r['old']==r['label']for r in known);new_good=sum(r['new']==r['label']for r in known)
new_transition=[r for r in data['rows']if r['split']=='exposed_probe'and r['origin']=='RGB_transition_review'and r['label']=='transition']
short=[r for r in data['traces']if r['split']=='exposed_probe'and r['variant']=='short_goal_prefix']
check('regression_recorded',len(regress)==old_good-new_good)
check('no_failed_model_adoption',data['existing_web_replaced']is False)
out=dict(passed=all(c['passed']for c in checks),total=len(checks),checks=checks,
 training_valid=True,admission_passed=False,model_adopted=False,
 reason='Three-class fine-tuning identifies reviewed transitional fingers but regresses previously correct endpoint observations. Keep frozen baseline; do not merge it into web.',
 original_known_probe=dict(n=len(known),old_correct=old_good,new_correct=new_good,regressions=regress),
 intermediate_probe=dict(n=len(new_transition),correct=sum(r['new']==r['label']for r in new_transition)),
 short_goal_probe=dict(n=len(short),preserved=sum(r['result']['feature_sequence_observed']for r in short)),
 actual_prefixes=dict(n=len(data['prefixes']),false_shape_goal=sum(r['result']['new']['feature_sequence_observed']for r in data['prefixes'])),
 reviewer='Same Codex separate numerical and input verification; not independent human annotation',
 next='Freeze endpoint recognition and learn transition/change separately with cross-word FIT examples; test intermediate-state false alarms on normal handshape holds. Do not lower confidence or require a longer hold to pass.',
 no_new_holdout=True,no_learner_error_accuracy=True)
with(T/'VERIFIED.json').open('x',encoding='utf-8')as f:json.dump(out,f,ensure_ascii=False,indent=2)
assert out['passed'];print(json.dumps({k:v for k,v in out.items()if k not in ['checks','original_known_probe']},ensure_ascii=False))
