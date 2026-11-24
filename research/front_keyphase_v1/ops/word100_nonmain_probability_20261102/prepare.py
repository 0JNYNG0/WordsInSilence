from contract import *

def main():
    assert not (U/'FROZEN.json').exists();tests=behavior()
    v=read(T/'VERIFIED.json');rows=[]
    for tag in ['old_frozen','new_frozen','old_adapt','new_adapt']:
        rows.append(dict(tag=tag,report=str(OUT/'evaluations'/f'{tag}.json'),arrays=str(OUT/'predictions'/tag),content_key='raw_action_probability',baseline=v['conditions'][tag]))
    f=read(O/'word100_learned_flow_role_20261102/VERIFIED.json')['candidate']
    rows.append(dict(tag='prior_flow',report=str(FLOW/'evaluation.json'),arrays=str(FLOW/'predictions'),content_key='content_probability',
        baseline=dict(raw_correct=f['raw_correct'],used_correct=f['main_eligible_correct'],normal_order=f['normal_order'],short_reverse=f['short_reverse'],false_order=f['false_order'],reverse_false_pass=f['short_reverse_false_supported'])))
    pins={str(p):sha(p) for p in [U/'contract.py',U/'prepare.py',U/'compare.py',U/'verify.py',T/'VERIFIED.json']}
    for r in rows:
        pins[r['report']]=sha(r['report']);rep=read(r['report'])
        for s in rep['sequences']:
            p=Path(r['arrays'])/(s['key']+'.npz');pins[str(p)]=sha(p)
    exclusive(U/'FROZEN.json',dict(at=clock(),reason='Same nonmain total can be split between preparation and rest, making max(subtype) an incorrect test of nonmain certainty.',
        change='Use sum(role preparation/return, role rest)>=existing0.8 for event eligibility; preserve strong rest-main conflict and raw observations. No word IDs, new threshold or trained weights.',
        models=rows,pins=pins,behavior=tests,primary='Common calculation rule, all five fixed models reported; no winner selection or automatic web adoption.',
        comparison_scope='NEW interpretation candidate on saved probabilities. Old model inference and old scores unchanged. 5x88 new calculation comparisons only.',
        gates=['raw_time_observations_exact','no_additional_known_main_exclusions_each_model','no_order_regression_each_model','no_short_reverse_regression_each_model',
            'normal_false_order0_each_model','reverse_false_pass0_each_model','missing_held_all','normal_order_gain_somewhere'],
        overall_scope='Even a calculation gate pass does not adopt a previously failed neural candidate. Full model/feedback/latency gates remain.',whole_goal_complete=False))
    print('Frozen categorical nonmain-union correction;five saved models,440 new interpretation cases;no neural runs')
if __name__=='__main__':main()
