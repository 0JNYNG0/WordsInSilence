"""Focused regressions for relation support. No models or public footage."""
from pathlib import Path
import json
import hashlib
import numpy as np
from relation_support import relation_support

U = Path(__file__).resolve().parent


def main():
    out=U/'STRUCTURAL_CHECKS.json'
    assert not out.exists()
    cases=[]
    def case(name,p,a,check):
        r=relation_support(p,a);check(r);cases.append(name)
        for c in r['candidates']:
            assert all(a[i] for i in c['context_envelope_indices'])
            assert c['selected_end'] is None and c['process_probability'] is None
            assert set(c['former_first_only_indices']).issubset(c['context_envelope_indices'])
        assert r['scored_candidates']==0 and not r['exact_event_boundaries']
    def eq(value,expected):assert value==expected,(value,expected)
    case('late_arrival_support_survives',[[.9,0],[.9,.85],[0,.9],[0,.95]], [1]*4,
         lambda r:eq(r['candidates'][0]['end_support_indices'],[1,2,3]))
    case('single_frame_relations_and_short_action_survive',[[.9,0],[0,.9]],[1,1],
         lambda r:eq(r['candidates'][0]['context_envelope_indices'],[0,1]))
    case('overlapping_relations_are_not_forced_exclusive',[[.9,.9],[.9,.9]],[1,1],
         lambda r:eq(r['candidates'][0]['end_support_indices'],[1]))
    case('missing_image_breaks_connection',[[.9,0],[.9,.9],[0,.9]],[1,0,1],
         lambda r:eq(len(r['candidates']),0))
    case('normal_hold_is_not_a_wrong_action',[[.9,0],[0,.9],[0,.9],[0,.9]],[1]*4,
         lambda r:eq(r['candidates'][0]['learner_error'],None))
    case('repeat_runs_remain_distinct',[[.9,0],[0,.9],[0,0],[.9,0],[0,.9]], [1]*5,
         lambda r:eq(len(r['contexts'][0]['arrival_relation_runs']),2))
    p=np.zeros((20,2));p[::2,0]=.9;p[1::2,1]=.9
    case('no_chronological_cap_silently_discards_later_repetitions',p,[1]*20,
         lambda r:eq(len(r['candidates']),55))
    case('no_supported_relation_is_not_a_wrong_sign',[[0,0],[0,0]], [1,1],
         lambda r:eq(r['candidates'],[]))
    case('all_missing_is_no_evidence',[[.9,.9],[.9,.9]],[0,0],
         lambda r:eq(r['candidates'],[]))
    for p,a in [([[float('nan'),0]],[1]),([[2,0]],[1]),([[0,0]],[1,1])]:
        try:relation_support(p,a)
        except ValueError:pass
        else:raise AssertionError('Invalid input accepted')
    result=dict(status='PASS_STRUCTURAL_CONTRACT_NOT_MODEL_ACCURACY',cases=cases,
        valid_cases=len(cases),invalid_cases=3,model_inference=0,training=0,
        public_accuracy_claim=False,current_web_changed=False,
        sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),U/'relation_support.py',U/'SCOPE.json']})
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(result['status'],len(cases)+3)


if __name__=='__main__':main()
