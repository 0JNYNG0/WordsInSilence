"""Saved-output diagnosis. Counterfactual counts are NOT adopted scores."""
from episode_model import *
from collections import Counter
NORMAL=['native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest']
SUPPORTED='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'

def main():
    assert not (D/'DIAGNOSIS.json').exists();r=read(D/'evaluations/episode_42.json');rows=[]
    for q in r['sequences']:
        if q['kind'] not in NORMAL:continue
        edges=q['precedences'];all_edges=bool(edges) and all(e['status']=='SUPPORTED_ORDER_OBSERVED' for e in edges)
        rows.append(dict(key=q['key'],word=q['word'],kind=q['kind'],status=q['status'],edge_order_supported=all_edges,
          foreign_frames=q['unexpected_action_frames'],core_unconfirmed=q['previous_reviewed_core_unconfirmed'],action_errors=q['evaluation_action_errors'],
          recurrent_family_runs=[dict(group=s['group'],runs=s['runs'],isolated=s['isolated_candidate_frames']) for s in q['states'] if len(s['runs'])>1 or s['isolated_candidate_frames']]))
    normal=[p for p in r['points'] if p['kind'] in NORMAL];rest=[p for p in normal if p['y']==8]
    cm=Counter((p['y'],p['pred']) for p in normal if p['y']!=p['pred'])
    readout=read(D/'evaluations/readout_42.json')
    native_wrist=[p for p in r['points'] if p['kind']=='native' and p['y'] in [3,4] and p['y']!=p['pred']]
    out=dict(at=clock(),primary='episode_42',normal_action=r['normal_action_metrics'],readout_normal_action=readout['normal_action_metrics'],
      rest_correct=sum(p['y']==p['pred'] for p in rest),rest_known=len(rest),confusions=[dict(true=a,pred=b,count=n) for (a,b),n in cm.items()],normal_sequences=rows,
      diagnostic_edge_support_without_foreign_veto=sum(x['edge_order_supported'] for x in rows),
      diagnostic_scope='This number explains which gate blocks a result; it is not an alternative score, acceptance rule or threshold tuning. The frozen0/21result stands.',
      arm_travel_endpoint_disagreements=native_wrist,
      findings=[
       'Normal action witnesses585/615 are recognized, but this is a sparse event-content task, not word recognition or complete feedback correctness.',
       'The new9way softmax has weak rest generalization34/107; rest often becomes upright-forearm family2. Removing target-masked old state input also removed useful conditioned evidence; this factor was not isolated by the readout/episode comparison.',
       'Preparation and return are largely unlabelled for the new event task. A family can reappear during return; all-occurrence precedence then becomes uncertain. Do not erase those frames or choose the longest run using DEV outcomes.',
       'Arm traversal versus wrist-endpoint witnesses have timing disagreements. A hold is not mandatory and a coordinate-like endpoint class cannot certify motion completion.',
       'The added recurrent head does not beat the same-input readout:585vs587/615 and0vs4normal order support. More recurrence alone is not the established remedy.',
       'Goal-blind physical observation and target-conditioned guide interpretation are distinct. The former must stay fixed; requiring the latter to ignore the target is stronger than the project requirement. Investigate it as a preregistered factor, not a DEV-selected shortcut.'
      ],next='Before another full training wave, audit FIT support for preparation/return and conditional guide interpretation. Preserve physical observations, compare matched ablations on the same sparse action labels, and retain reverse/cut/gap tests. Do not simply relax the foreign-content veto or mask return frames.',
      adopted=False,web_changed=False,whole_goal_complete=False)
    exclusive(D/'DIAGNOSIS.json',out);print('Stored causal limits and saved-output diagnosis; no re-inference or revised score')
if __name__=='__main__':main()
