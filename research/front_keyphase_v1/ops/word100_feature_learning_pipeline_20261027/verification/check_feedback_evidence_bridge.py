"""Finite synthetic API contracts, not accuracy or KSL validation."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import math
from datetime import datetime, timezone

U=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bridge_under_test',U/'feedback_evidence_bridge.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
mapping=json.loads((U/'research/DEEPHAND_PHYSICAL_PARTIAL_MAPPING.json').read_text(encoding='utf-8'))
source={r['class_name']:r for r in mapping['rows']}
names=list(source)
checks=[]

def check(name, condition):
    checks.append({'name':name,'passed':bool(condition)})
    assert condition,name

def p(**classes):
    return [classes.get(n,0.) for n in names]

def project(values, **kwargs):
    return b.project_class_probabilities(values,names,hand_present=kwargs.get('present',True),
                                         identity_resolved=kwargs.get('identity',True))

def obs(state=1,measurement=None):
    return b.observed_hand({f:{'state':state,'measurement':measurement} for f in b.FEATURES},
                          hand_present=True,identity_resolved=True,evidence_id='synthetic-contract-only')

def frames(hands):
    return [{'time_s':i*.04,'hands':{'R':h} if h is not None else {}} for i,h in enumerate(hands)]

def ref(fid, fs, path, required=True):
    return {'reference_id':fid,'frames':fs,'path':path,
            'required':{h:{f:[required if h=='R' else False]*len(fs) for f in b.FEATURES} for h in b.HANDS}}

def get(result, feature='index_only', refindex=0, hand='R'):
    return result['references'][refindex]['hands'][hand][feature]

def rejects(fn):
    try:fn()
    except (ValueError,TypeError):return True
    return False

for name in names:
    out=project(p(**{name:1.}))
    for f in b.FEATURES:
        s=source[name][f];d=out['features'][f]
        check('source_onehot_'+name+'_'+f,d['state']==s and d['kind']=='model_estimate'
              and d['p_pos']==int(s==1) and d['p_neg']==int(s==0) and d['p_unknown']==int(s is None))

reversed_names=names[::-1];one=p(**{'2':1})
permuted=b.project_class_probabilities(one[::-1],reversed_names,hand_present=True,identity_resolved=True)
check('explicit_class_order_not_numeric_name_heuristic',permuted['features']==project(one)['features']
      and project(p(**{'1':1}))['features']['four_closed']['state']==1)
boundary=project(p(**{'2':.8,'5':.2}));below=project(p(**{'2':.799999,'5':.200001}))
check('fixed_threshold_exact_boundary',boundary['features']['index_only']['state']==1
      and below['features']['index_only']['state'] is None)
unkclass=next(n for n in names if source[n]['index_only'] is None)
unknown=project(p(**{unkclass:.6,'2':.4}))
check('unknown_source_mass_not_renormalized',unknown['features']['index_only']['state'] is None
      and math.isclose(unknown['features']['index_only']['p_pos'],.4)
      and math.isclose(unknown['features']['index_only']['p_unknown'],.6))
for reason, kwargs in [('absent',{'present':False}),('identity',{'identity':False})]:
    hidden=project(one,**kwargs)
    check('visibility_gate_'+reason,all(c['state'] is None for c in hidden['features'].values())
          and hidden['raw_class_probabilities']==one
          and hidden['features']['index_only']['p_pos']==1.)

q=frames([obs(1,{'value':37.,'unit':'degree'})]);r=frames([obs(0,{'value':31.,'unit':'degree'})])
result=b.compare_references(q,[ref('numeric',r,[[0,0]])]);pair=get(result)['cells'][0]['pairs'][0]
check('numeric_difference_and_classification_difference_separate',pair['observed_value_abs_difference']==6
      and pair['classification_state_difference']==1 and pair['model_positive_mass_abs_difference'] is None
      and get(result)['complete_observed_only_count']==1 and get(result)['complete_observed_value_comparison_count']==1)
model=b.compare_references(frames([project(one)]),[ref('model',frames([project(one)]),[[0,0]])])
check('model_agreement_never_becomes_observation',get(model)['complete_observed_only_count']==0
      and get(model)['complete_involving_estimate_count']==1
      and get(model)['cells'][0]['pairs'][0]['observed_value_abs_difference'] is None)
mismatch=b.compare_references(q,[ref('units',frames([obs(1,{'value':31.,'unit':'pixel'})]),[[0,0]])])
check('incompatible_units_not_subtracted',get(mismatch)['complete_observed_value_comparison_count']==0
      and get(mismatch)['complete_comparison_count']==1)

q=frames([obs(),obs()]);r=frames([obs(),obs()])
duplicated=b.compare_references(q,[ref('repeat_pair',r,[[0,0],[0,0],[1,1]])])
check('exact_path_duplicate_does_not_inflate_required_denominator',get(duplicated)['required_sample_count']==2
      and get(duplicated)['complete_comparison_count']==2
      and duplicated['references'][0]['duplicate_pairs_ignored_for_count']==1)
q=frames([obs(),None,obs()]);r=frames([obs(),obs()])
partial=b.compare_references(q,[ref('partial',r,[[0,0],[1,0],[2,1]])])
g=get(partial)
check('multiple_query_pairs_keep_partial_missing',g['required_sample_count']==2 and g['complete_comparison_count']==1
      and g['partial_comparison_count']==1 and g['cells'][0]['known_pair_count']==1
      and g['cells'][0]['aligned_pair_count']==2 and g['cells'][0]['state_relation_scope'].startswith('known_pairs'))
check('missing_time_points_explicit',g['insufficient_sample_runs'][0]['reference_indices']==[0]
      and g['insufficient_sample_runs'][0]['sample_times_s']==[0.]
      and g['insufficient_sample_runs'][0]['continuous_interval_claim'] is False)
empty=b.compare_references([],[ref('empty_query',frames([obs(),obs(),obs()]),[])])
check('entire_query_absent_keeps_required_axis',get(empty)['required_sample_count']==3
      and get(empty)['insufficient_count']==3 and get(empty)['complete_comparison_count']==0
      and len(get(empty)['insufficient_sample_runs'][0]['reference_indices'])==3)
missingref=b.compare_references(frames([obs(),obs()]),[ref('missing_reference',frames([None,obs()]),[[0,0],[1,1]])])
check('reference_missing_also_keeps_denominator',get(missingref)['required_sample_count']==2
      and get(missingref)['insufficient_count']==1)
rest=b.compare_references(frames([None,None]),[ref('normal_rest',frames([None,None]),[[0,0],[1,1]],False)])
check('reference_defined_rest_not_called_error',get(rest)['required_sample_count']==0
      and get(rest)['insufficient_count']==0 and get(rest)['complete_comparison_fraction'] is None)
uncovered=b.compare_references(frames([obs()]),[ref('unmapped_end',frames([obs(),obs()]),[[0,0]])])
check('unmapped_reference_time_not_deleted',get(uncovered)['required_sample_count']==2
      and get(uncovered)['complete_comparison_count']==1)
many=b.compare_references(frames([obs()]),[ref('agree',frames([obs(1)]),[[0,0]]),ref('disagree',frames([obs(0)]),[[0,0]])])
check('separate_reference_results_no_best_splice',len(many['references'])==2
      and get(many,refindex=0)['cells'][0]['state_relation']=='same'
      and get(many,refindex=1)['cells'][0]['state_relation']=='different'
      and many['reference_splicing_or_best_selection'] is False)
cross=[{'time_s':0.,'hands':{'L':obs(1),'R':obs(0)}}, {'time_s':.04,'hands':{'L':obs(1),'R':obs(0)}}]
crossref=ref('cross',copy.deepcopy(cross),[[0,0],[1,1]])
for f in b.FEATURES:crossref['required']['L'][f]=[True,True]
crossresult=b.compare_references(cross,[crossref])
check('externally_assigned_sides_never_swapped',all(get(crossresult,hand=h)['complete_observed_only_count']==2
      and all(c['state_relation']=='same' for c in get(crossresult,hand=h)['cells']) for h in b.HANDS))
twozero=b.observed_hand({'index_only':{'state':0},'open_four':{'state':0},'four_closed':{'state':None}},
                       hand_present=True,identity_resolved=True,evidence_id='synthetic')
check('two_absent_features_do_not_imply_closed_hand',twozero['features']['four_closed']['state'] is None)

cases={
 'probability_wrong_length':lambda:project([0.]*44),
 'probability_is_logit':lambda:project([2.]*45),
 'probability_nan':lambda:project([float('nan')]+[0.]*44),
 'probability_negative':lambda:project([-1.]+[2.]+[0.]*43),
 'probability_zero_mass':lambda:project([0.]*45),
 'class_duplicate':lambda:b.project_class_probabilities(one,[names[0]]*45,hand_present=True,identity_resolved=True),
 'path_reversal':lambda:b.compare_references(frames([obs(),obs()]),[ref('bad',frames([obs(),obs()]),[[1,0],[0,1]])]),
 'path_out_of_range':lambda:b.compare_references(frames([obs()]),[ref('bad',frames([obs()]),[[1,0]])]),
 'path_jump':lambda:b.compare_references(frames([obs(),obs(),obs()]),[ref('bad',frames([obs(),obs(),obs()]),[[0,0],[2,2]])]),
 'path_float':lambda:b.compare_references(frames([obs()]),[ref('bad',frames([obs()]),[[0.,0]])]),
 'frame_duplicate_time':lambda:b.compare_references([{'time_s':0.,'hands':{}},{'time_s':0.,'hands':{}}],[ref('bad',frames([obs()]),[])]),
 'identity_non_boolean':lambda:project(one,identity=1),
 'feature_non_binary':lambda:b.observed_hand({'index_only':{'state':2}},hand_present=True,identity_resolved=True,evidence_id='test'),
 'empty_reference':lambda:b.compare_references([],[ref('bad',[],[])]),
 'duplicate_reference_id':lambda:b.compare_references([],[ref('same',frames([obs()]),[]),ref('same',frames([obs()]),[])]),
}
badreq=ref('badreq',frames([obs()]),[]);badreq['required']['R']['index_only']=[]
cases['required_length']=lambda:b.compare_references([], [badreq])
for name,fn in cases.items():check('reject_'+name,rejects(fn))

q=frames([project(one)]); rr=[ref('preserve',frames([project(one)]),[[0,0],[0,0]])]
before=json.dumps([q,rr],sort_keys=True);after=b.compare_references(q,rr)
check('all_input_bytes_semantically_preserved',json.dumps([q,rr],sort_keys=True)==before
      and after['raw_query_frames']==q and after['references'][0]['raw_reference_frames']==rr[0]['frames']
      and after['references'][0]['path_original']==rr[0]['path'])
def walk(x):
    if isinstance(x,dict):
        for k,v in x.items():
            if k=='educational_verdict':assert v is None
            if k=='educational_feedback_enabled':assert v is False
            walk(v)
    elif isinstance(x,list):
        for v in x:walk(v)
walk(after);check('no_educational_verdict',True)
result={'status':'PASS','at':datetime.now(timezone.utc).isoformat(),'checks':checks,'count':len(checks),
        'scope':'Synthetic API and source mapping arithmetic contracts; not independent model performance, real crossing preservation, or KSL pedagogical accuracy',
        'production_changed':False,'user_mentor_access':False,'new_model_inference':False,
        'bridge_sha256':hashlib.sha256((U/'feedback_evidence_bridge.py').read_bytes()).hexdigest(),
        'protocol_sha256':hashlib.sha256((U/'BRIDGE_PROTOCOL.json').read_bytes()).hexdigest(),
        'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
pout=U/'verification/BRIDGE_CONTRACT_CHECKS.json'
assert not pout.exists(),'Preserve previous contract audit'
pout.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','checks':len(checks),'real_world_accuracy_claim':False}))
