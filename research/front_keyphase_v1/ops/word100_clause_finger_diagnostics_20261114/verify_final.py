from pathlib import Path
import sys,json,hashlib,collections,copy
import numpy as np,torch
U=Path(__file__).resolve().parent;O=U.parent
sys.path[:0]=[str(O/'word100_guide_requirement_learning_20261111'),str(O/'word100_joint_completion_block_20261114')]
import requirement_core as rc
from stable_geometry import geometry
from motion_core import reverse_geometry
from direction_adapter import merge_state
from presentation_evidence import prepare
from feedback_draft import draft
from clause_feedback import compose
torch.set_num_threads(2)
read=rc.read;write=rc.write;checks=[]
def check(name,value,detail=None):
    checks.append(dict(name=name,passed=bool(value),detail=detail))
    if not value:raise AssertionError(name)
pins=dict(read(U/'PROTECTED.json'))
graph_pins=read(O/'word100_web_completion_graph_20261111/PROTECTIONS.json')
pins.update(graph_pins)
motion_initial=read(U/'MOTION_FROZEN.json')
pins[motion_initial['init']]=motion_initial['init_sha256']
for name,field in [('FINGER_FROZEN.json','source_pins'),('MOTION_INPUTS.json','source_pins'),('INTEGRATION_FROZEN.json','pins'),('REPAIR_FROZEN_V2.json','pins')]:
    pins.update(read(U/name)[field])
changed=[p for p,h in pins.items()if rc.sha(p)!=h]
check('protected_source_model_and_contract_hashes',not changed,dict(paths=len(pins),changed=changed,protected_count=len(read(U/'PROTECTED.json'))))

rows=read(U/'FINGER_INPUTS.json')['rows'];z=np.load(U/'finger_arrays.npz');truth=z['targets'];mask=(truth>=0)&z['availability'];base=z['baseline']
fi=np.array([i for i,r in enumerate(rows)if r['role']=='fit']);di=np.array([i for i,r in enumerate(rows)if r['role']=='development'])
check('public_only_identifiers',all(r['clip_id'].startswith('NIA_SL_')for r in rows))
check('finger_person_and_clip_split',not({rows[i]['person']for i in fi}&{rows[i]['person']for i in di})and not({rows[i]['clip_id']for i in fi}&{rows[i]['clip_id']for i in di}))
check('no_digit_auxiliary_development',not any(rows[i]['auxiliary']for i in di))
metrics=[];model_count=0
for suffix,folder in [('', 'finger_models'),('_V2','finger_models_v2')]:
    for run in read(U/f'FINGER_RESULTS{suffix}.json')['runs']:
        seed=run['seed'];pred=np.load(U/folder/f'{seed}_predictions.npz')['predictions'];pr=pred[di];y=truth[di];m=mask[di]
        actual=dict(known=int(m.sum()),old_correct=int(((base[di]==y)&m).sum()),new_correct=int(((pr==y)&m).sum()),new_wrong=int(((pr>=0)&(pr!=y)&m).sum()),old_correct_regressions=int(((base[di]==y)&(pr!=y)&m).sum()))
        check(f'finger_recount{suffix}_{seed}',all(run['development'][k]==v for k,v in actual.items())and rc.sha(run['model'])==run['sha256'])
        check(f'thumb_disabled{suffix}_{seed}',np.all(pred[:,0]==-1))
        if suffix:metrics.append(dict(seed=seed,**actual))
        model_count+=1
for suffix in ['', '_V2']:
    for run in read(U/f'MOTION_TRAINED{suffix}.json')['rows']:
        check(f'motion_checkpoint{suffix}_{run["seed"]}',rc.sha(run['model'])==run['sha256']and run['reversal_equivariance_max_error']<1e-5);model_count+=1

coverage={}
for role,ix in [('fit',fi),('development',di)]:
    valid=(truth[ix,3]>=0)&(truth[ix,4]>=0)
    coverage[role]=dict(anchors=len(ix),clips=len({rows[i]['clip_id']for i in ix}),words=len({rows[i]['word']for i in ix}),known_observable_finger_cells=int(mask[ix].sum()),ring_little_labeled_together=int(valid.sum()),ring_little_different_state_labels=int((valid&(truth[ix,3]!=truth[ix,4])).sum()),patterns=[dict(states=list(k),anchors=v)for k,v in collections.Counter(tuple(int(t)for t in truth[i])for i in ix).items()])
write(U/'FINGER_PATTERN_COVERAGE.json',coverage)

result=read(U/'INTEGRATED_RESULTS.json');normal=errors=0;counts=collections.Counter();changed_cards=[];cases=[]
for r in result['rows']:
    check('source_'+r['dataset']+'_'+r['clip_id']+'_'+r['kind'],rc.sha(r['source'])==r['source_sha256'])
    d=read(r['result']);b=d['before'];a=d['after'];word=r['clip_id'].split('_')[2]
    check('same_candidate_order_'+str(len(cases)),b['original_candidates']==a['original_candidates']and b['original_orders']==a['original_orders'])
    rendered=prepare(a,rc.FAMILIES[word],draft)
    for bc,ac,pc in zip(b['joint_cards'],a['joint_cards'],rendered['joint_cards']):
        check('shape_preserved_'+str(len(cases))+'_'+ac['card'],[e for e in bc['observations']if e['task']!='direction']==[e for e in ac['observations']if e['task']!='direction'])
        if r['normal']:
            check('normal_warning_'+str(len(cases))+'_'+ac['card'],not(ac['state']in ['attributable_feature_difference','evidence_conflict']and bc['state']not in ['attributable_feature_difference','evidence_conflict']))
        for be,ae in zip([e for e in bc['observations']if e['task']=='direction'],[e for e in ac['observations']if e['task']=='direction']):
            if be['state']=='opposite_direction' or (r['normal']and be['state']=='supported'):check('old_evidence_kept_'+str(len(cases))+'_'+ac['card'],be['state']==ae['state'])
            if be['state']!=ae['state']:
                check('all_three_learned_witnesses_'+str(len(cases)),len(ae['supplementary_motion'])==3 and all(e['state']==ae['state']for e in ae['supplementary_motion']))
                changed_cards.append(dict(clip_id=r['clip_id'],kind=r['kind'],card=ac['card'],from_state=be['state'],to_state=ae['state'],span=ae['span']))
        if 'missing'in r['kind'] and r['kind'].startswith('all_'):
            check('missing_never_error_'+str(len(cases))+'_'+ac['card'],ac['state']!='attributable_feature_difference')
        if pc['state']in ['evidence_conflict','action_link_uncertain','component_link_uncertain']:
            check('uncertain_not_displayed_as_error_'+str(len(cases))+'_'+pc['card'],all(m['kind']=='uncertain'for m in pc['presentation']['messages']))
    # Exactly the resolved evidence, not stale old drafts, is exported for review.
    dest=U/'presentation'/Path(r['result']).name;write(dest,dict(source=r['result'],source_sha256=rc.sha(r['result']),result=rendered))
    normal+=r['normal'];errors+=bool(r['new_warnings'])and r['normal'];counts[r['dataset']]+=1;cases.append(r)
check('all_cases_present',dict(counts)=={'development82':82,'public8_40conditions':40},dict(counts))
check('normal_recount',normal==64 and errors==0)
check('simultaneous_shape_and_direction_gain',result['summary']['combined_before']==2 and result['summary']['combined_after']==3 and result['summary']['combined_n']==3)
combo=read(U/'presentation/development82__NIA_SL_WORD1133_REAL13_F__closed_hand_and_reversed.json')['result']['joint_cards'][0]['presentation']['messages']
check('simultaneous_feedback_from_actual_evidence',len([c for c in combo if c['kind']=='observed_difference'])==2)

# Public coordinate perturbations test the representation, not model accuracy.
sample=next(r for r in cases if r['clip_id']=='NIA_SL_WORD1133_REAL13_F'and r['kind']=='full_normal');arr=np.load(sample['source']);x=arr['x'];ts=arr['ts'];a,b=54,71
g,ok,_=geometry(rc.prior,x,ts,a,b,2)
check('observed_test_interval',ok)
shift=x.copy();shift[:,:102]=(shift[:,:102].reshape(-1,51,2)*1.7+np.array([.31,-.24])).reshape(-1,102)
gs,oks,_=geometry(rc.prior,shift,ts,a,b,2)
check('body_translation_scale_invariance',oks and np.allclose(g,gs,atol=2e-5),float(abs(g-gs).max()))
gr,okr,_=geometry(rc.prior,x,ts,a,b,2,reverse=True)
check('reverse_geometry_consistency',okr and np.allclose(gr,reverse_geometry(torch.tensor(g[None]))[0].numpy(),atol=2e-5))
xx=x.copy();xx[:,18:102]=xx[:,18:102]*.7+.17
gh,okh,_=geometry(rc.prior,xx,ts,a,b,2)
check('finger_change_does_not_change_body_motion',okh and np.allclose(g,gh))
for ax in [0,1,3,4]:
    ga,oa,_=geometry(rc.prior,x,ts,a,b,ax);gb,ob,_=rc.prior.geometry(x,ts,a,b,ax)
    check('other_relation_axis_unchanged_'+str(ax),oa==ob and np.array_equal(ga,gb))
check('weak_new_does_not_remove_normal',merge_state('supported',['movement_not_established']*3)=='supported')
check('strong_contradiction_is_unknown',merge_state('supported',['opposite_direction']*3)=='evidence_conflict')
check('mixed_new_is_unknown',merge_state('uncertain',['supported','opposite_direction','uncertain'])=='evidence_conflict')
check('two_votes_insufficient',merge_state('uncertain',['opposite_direction','opposite_direction','uncertain'])=='uncertain')
actions=read(U/'CLAUSE_LINKS.json')['actions'];action=next(a for a in actions if a['action_id']=='brush_arm')
check('actual_phrase_offsets',all(a['display_instruction'][slice(*c['char_span'])]==c['text']for a in actions for c in a['clauses']))
for state in ['evidence_conflict','association_unknown','observation_unknown','uncertain']:
    q=compose(action,[dict(hand='R',finger='index',span=[2,4],state=state)],[dict(span=[2,4],state=state)],True)
    check('clause_unknown_'+state,not q['cues']and len(q['unknown'])==2)
conflict=copy.deepcopy(read(cases[0]['result'])['after']);conflict['joint_cards'][0]['state']='evidence_conflict';conflict['joint_cards'][0]['differences']=[dict(task='direction',span=[2,4])]
view=prepare(conflict,rc.FAMILIES[cases[0]['clip_id'].split('_')[2]],draft)
check('conflict_never_emits_old_correction',view['joint_cards'][0]['presentation']['tone']=='uncertain'and all(m['kind']=='uncertain'for m in view['joint_cards'][0]['presentation']['messages']))
audit=dict(status='PASS_LIMITED_RESEARCH',implementer='Codex',verifier='Same Codex, separate saved-output recount and invariant audit; no independent human expert review',checks=checks,protected_paths=len(set(graph_pins)|set(read(U/'PROTECTED.json'))),total_hashes=len(pins),model_training_runs=model_count,source_cases=dict(counts),normal_cases=normal,new_normal_warnings=errors,changed_cards=changed_cards,finger_metrics=metrics,finger_coverage=coverage,
    decisions=dict(direction_supplement='PASS isolated regression; web not changed',finger='HOLD: known-cell regressions and limited independent finger patterns',whole_action_correctness='Not established',sentence_understanding='Seven manually structured action descriptions; automatic observed intervals; no unrestricted language parser',actual_camera='Deferred by user',C_T3F_common100='Unchanged'),new_finger_feedback_enabled=False)
write(U/'VERIFIED.json',audit)
print(json.dumps({k:audit[k]for k in ['status','protected_paths','total_hashes','model_training_runs','source_cases','normal_cases','new_normal_warnings','decisions']},ensure_ascii=False))
