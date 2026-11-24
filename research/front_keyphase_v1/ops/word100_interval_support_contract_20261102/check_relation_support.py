"""Focused regressions for relation support. No models or public footage."""
from pathlib import Path
import json
import hashlib
import numpy as np
from relation_support import relation_support, inclusive_indices

U = Path(__file__).resolve().parent


def main():
    out=U/'STRUCTURAL_CHECKS.json'
    if out.exists():
        assert out.read_bytes() == (U/'before_compact'/out.name).read_bytes(), 'Only preserved initial result may be replaced'
    cases=[]
    def case(name,p,a,check):
        r=relation_support(p,a);check(r);cases.append(name)
        for c in r['candidates']:
            assert all(a[i] for i in inclusive_indices(c['context_envelope_range']))
            assert c['selected_end'] is None and c['process_probability'] is None
            assert set(inclusive_indices(c['former_first_only_range'])).issubset(inclusive_indices(c['context_envelope_range']))
        assert r['scored_candidates']==0 and not r['exact_event_boundaries']
    def eq(value,expected):assert value==expected,(value,expected)
    case('late_arrival_support_survives',[[.9,0],[.9,.85],[0,.9],[0,.95]], [1]*4,
         lambda r:eq(list(inclusive_indices(r['candidates'][0]['end_support_range'])),[1,2,3]))
    case('single_frame_relations_and_short_action_survive',[[.9,0],[0,.9]],[1,1],
         lambda r:eq(list(inclusive_indices(r['candidates'][0]['context_envelope_range'])),[0,1]))
    case('overlapping_relations_are_not_forced_exclusive',[[.9,.9],[.9,.9]],[1,1],
         lambda r:eq(list(inclusive_indices(r['candidates'][0]['end_support_range'])),[1]))
    case('missing_image_breaks_connection',[[.9,0],[.9,.9],[0,.9]],[1,0,1],
         lambda r:eq(len(r['candidates']),0))
    case('normal_hold_is_not_a_wrong_action',[[.9,0],[0,.9],[0,.9],[0,.9]],[1]*4,
         lambda r:eq(r['candidates'][0]['learner_error'],None))
    case('repeat_runs_remain_distinct',[[.9,0],[0,.9],[0,0],[.9,0],[0,.9]], [1]*5,
         lambda r:eq(len(r['contexts'][0]['arrival_relation_run_ranges']),2))
    p=np.zeros((20,2));p[::2,0]=.9;p[1::2,1]=.9
    case('no_chronological_cap_silently_discards_later_repetitions',p,[1]*20,
         lambda r:eq(len(r['candidates']),55))
    case('no_supported_relation_is_not_a_wrong_sign',[[0,0],[0,0]], [1,1],
         lambda r:eq(r['candidates'],[]))
    case('all_missing_is_no_evidence',[[.9,.9],[.9,.9]],[0,0],
         lambda r:eq(r['candidates'],[]))
    p=np.zeros((2000,2));p[::2,0]=.9;p[1::2,1]=.9
    r=relation_support(p,[1]*len(p))
    assert r['status']=='UNRESOLVED_BUDGET_EXCEEDED'
    assert r['candidate_pair_count']==500500 and r['candidates']==[]
    assert not r['candidate_generation_complete'] and not r['candidates_silently_truncated']
    assert len(r['contexts'][0]['setup_relation_run_ranges'])==1000
    assert len(r['contexts'][0]['arrival_relation_run_ranges'])==1000
    cases.append('large_pair_count_is_known_before_any_pair_materialization')
    p=np.zeros((44,2));p[::2,0]=.9;p[1::2,1]=.9
    assert len(relation_support(p,[1]*44,max_pairs=253)['candidates'])==253
    assert relation_support(p,[1]*44,max_pairs=252)['status']=='UNRESOLVED_BUDGET_EXCEEDED'
    cases.append('explicit_budget_boundary_all_or_none')
    for p,a in [([[float('nan'),0]],[1]),([[2,0]],[1]),([[0,0]],[1,1]),
                ([[0,0]],[float('nan')]),([[0,0]],['False']),([[0,0]],[2])]:
        try:relation_support(p,a)
        except ValueError:pass
        else:raise AssertionError('Invalid input accepted')
    for budget in (0,-1,1.5,True,None):
        try:relation_support([[.9,0],[0,.9]],[1,1],max_pairs=budget)
        except ValueError:pass
        else:raise AssertionError('Invalid pair budget accepted')
    result=dict(status='PASS_STRUCTURAL_CONTRACT_NOT_MODEL_ACCURACY',cases=cases,
        valid_cases=len(cases),invalid_cases=11,model_inference=0,training=0,
        compact_ranges=True,default_pair_budget=256,budget_is_semantic_selection=False,
        public_accuracy_claim=False,current_web_changed=False,
        sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),U/'relation_support.py',U/'SCOPE.json']})
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(result['status'],len(cases)+11)


if __name__=='__main__':main()
