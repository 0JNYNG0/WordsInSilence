"""Separate verification entry point; no learning and no neural re-inference."""
from pathlib import Path
import json,hashlib,collections,copy,ast
import numpy as np
from core import features,assess,attach,CLASSES
from component_feedback import describe_component
U=Path(__file__).resolve().parent;O=U.parent;S=O/'word100_goal_state_learning_20261114';P=O/'word100_goal_binding_integration_20261114';G=O/'word100_web_completion_graph_20261111'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def check(name,condition):checks.append(dict(name=name,passed=bool(condition)))
lab=read(U/'LABELS.json');cfg=read(U/'FROZEN.json');data=read(U/'EVALUATION.json');dec=read(U/'DECOMPOSITION.json')
for p,h in {**cfg['training_pins'],**lab['source_pins']}.items():check('frozen_input:'+Path(p).name,sha(Path(p))==h)
for p,h in read(G/'PROTECTIONS.json').items():check('protected:'+Path(p).name,sha(Path(p))==h)
for row in read(U/'TRAINING_COMPLETE.json')['rows']:
 check('weights_changed',row['weights_changed']);check('checkpoint_sha',sha(Path(row['model']))==row['sha256'])
 check('fit_records_only',all(lab['rows'][i]['record_id']in cfg['fit_records']for i in row['fit_indices']))
 check('known_visible_only',all(lab['rows'][i]['target']>=0 and lab['rows'][i]['available']for i in row['fit_indices']))
check('disjoint_records',not set(cfg['fit_records'])&set(cfg['exposed_probe_records']))
check('probe_not_called_holdout',read(U/'DECOMPOSITION_FROZEN.json')['not_new_holdout'])
check('training_three_only',len(read(U/'TRAINING_COMPLETE.json')['rows'])==3)
check('failure_preserved',(U/'FIRST_SERIALIZATION_FAILURE.partial.json').exists())
check('inference_not_repeated',data['serialization_recovered_from_saved_predictions']and not data['neural_inference_repeated'])
probs=np.load(U/'DENSE_PROBABILITIES.npz')
for r,s in zip(lab['rows'],data['state_rows']):
 pp=probs[r['clip_id']+'_'+r['hand']+'_'+r['domain']][:,r['array_index']]
 votes=np.argmax(pp,axis=1);known=r['available']and len(set(votes.tolist()))==1 and np.min(np.max(pp,axis=1))>=.8
 expected=CLASSES[r['domain']][votes[0]]if known else None
 check('saved_consensus',s['predicted']==expected)
for s in data['state_summary']:
 rr=[r for r in data['state_rows']if r['split']==s['split']and r['domain']==s['domain']and r['label']is not None]
 check('state_denominator',len(rr)==s['total'])
 check('state_partition',s['total']==s['supported_correct']+s['wrong']+s['unknown'])
 check('state_arithmetic',sum(r['predicted']==r['label']for r in rr)==s['supported_correct'])
for s in dec['summary']:
 rr=[r for r in dec['rows']if(r['split'],r['variant'])==(s['split'],s['variant'])]
 check('decomposition_count',len(rr)==s['n'])
 check('decomposition_statuses',dict(collections.Counter(r['result']['parts'][0]['evidence']['status']for r in rr))==s['statuses'])
for r in dec['rows']:
 check('partial_observation_not_whole_success',r['result']['whole_action_complete']is None and r['result']['educational_error']is None)
 check('region_remains_unresolved',r['result']['additional_requirements']['hand_body_relation']=='unqualified_region_model')
 check('no_display',r['result']['display_enabled']is False)
probe=[r for r in dec['rows']if r['split']=='exposed_probe']
for r in probe:
 status=r['result']['parts'][0]['evidence']['status'];variant=r['variant']
 if variant in ['held_start','goal_only','reverse','missing_tail','missing_all','middle_long_gap']:
  check('no_false_full_shape_transition:'+variant,status!='required_feature_sequence_observed')
 if variant in ['normal','short_goal_prefix','speed075','speed125']:
  check('preserve_known_shape_change:'+variant,status=='required_feature_sequence_observed')

# Geometry invariance checks on public cached data, no private inputs.
arr=np.load(S/'source_arrays.npz');sample=np.concatenate([arr[c+'_x'][::10]for c in sorted({r['clip_id']for r in lab['rows']})])
def transform(x,scale=1.,shift=(0.,0.),angle=0.,mirror=False):
 out=x.copy();p=out[:,:102].reshape(-1,51,2)
 rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
 p[:]=(p@rot.T)*scale+shift
 if mirror:
  p[:,:,0]*=-1;order=np.arange(51);order[[1,2,3,4,5,6]]=[2,1,4,3,6,5]
  order[9:30]=np.arange(30,51);order[30:51]=np.arange(9,30)
  p[:]=p[:,order].copy();out[:,186:237]=out[:,186:237][:,order].copy()
 return out
for hand in ['R','L']:
 for domain in CLASSES:
  f,v=features(sample,domain,hand)
  for name,xx,hh in [('translation',transform(sample,shift=(2.,-3.)),hand),('scale',transform(sample,scale=1.7),hand),('rotation',transform(sample,angle=.08),hand),('mirror',transform(sample,mirror=True),'L'if hand=='R'else'R')]:
   g,w=features(xx,domain,hh);check('validity_'+name,np.array_equal(v,w));check('geometry_'+name,np.allclose(f[v],g[v],atol=3e-4,rtol=3e-4))
  blank=sample.copy();blank[:,186:237]=0;_,bv=features(blank,domain,hand);check('missing_not_measurement',not bv.any())

# Contract examples use symbolic states, not invented model performance data.
req={'start':{'shape':'four_open'},'goal':{'shape':'four_closed'}}
cases=[(['four_open']*5,[True]*5,'still_at_start_state'),
       (['four_closed']*5,[True]*5,'goal_posture_without_start'),
       (['four_open',None,'four_closed'],[True]*3,'required_feature_sequence_observed'),
       (['four_open',None,None],[True,False,False],'observation_uncertain'),
       (['four_closed','four_closed','four_open'],[True]*3,'reverse_feature_order')]
for states,valid,want in cases:
 ts=np.arange(len(states))*33333
 r=assess({'shape':states},{'shape':valid},ts,req);check('symbolic:'+want,r['status']==want)
 for speed in [.75,1.25,2.]:
  r=assess({'shape':states},{'shape':valid},np.rint(ts/speed).astype(int),req);check('no_minimum_duration',r['status']==want)
 for conflict in ['competing_action_assignment','action_identity_unconfirmed','posture_snapshot_only']:
  r=assess({'shape':states},{'shape':valid},ts,req,conflict);check('assignment_guard',r['status']=='action_association_uncertain')

# A new diagnostic is additive. Existing boundary/shape/direction/normal-order
# results remain exact; this is a serialization/interface regression, NOT
# an unrun full integration experiment or new 246-example accuracy claim.
existing=read(P/'JOINT_SAVED_COMPARISON.json')['rows']
for original in existing:
 new=attach(original,{'status':'unvalidated_for_this_lineage','whole_action_complete':None,'display_enabled':False})
 new.pop('terminal_trait_diagnostics');check('preserve_existing_comparison',new==original)
failures=[r for r in checks if not r['passed']]
out=dict(passed=not failures,total=len(checks),checks=checks,failures=failures,
 component_research_scope='shape change observation only; combined body-goal model not qualified',
 model_admitted_for_web=False,whole_graph_complete=False,
 source_videos=32,fit_records=6,exposed_probe_records=2,
 verification='Same Codex separate script, not independent human or KSL-expert annotation',
 pins={str(U/n):sha(U/n)for n in ['LABELS.json','FROZEN.json','TRAINING_COMPLETE.json','EVALUATION.json','DECOMPOSITION.json','component_feedback.py']})
with(U/'VERIFIED.json').open('x',encoding='utf-8')as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(passed=not failures,total=len(checks),failures=failures),ensure_ascii=False));assert not failures
