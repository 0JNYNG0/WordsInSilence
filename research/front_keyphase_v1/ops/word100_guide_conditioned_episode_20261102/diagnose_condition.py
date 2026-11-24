"""Frozen-output causal description; never revise scores or choose a seed."""
from condition_model import *
from collections import Counter
NORMAL=['native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest']

def main():
    assert not (E/'DIAGNOSIS.json').exists();verified=read(E/'VERIFIED.json');plan=read(E/'FROZEN.json');r=read(E/'evaluations'/f"{plan['primary']}.json")
    confusion=Counter((p['y'],p['pred']) for p in r['points'] if p['kind'] in NORMAL and p['y']!=p['pred'])
    rows=[]
    for q in r['sequences']:
        if q['kind'] not in NORMAL:continue
        rows.append(dict(key=q['key'],word=q['word'],kind=q['kind'],status=q['status'],edges=q['precedences'],
          unexpected=q['unexpected_action_frames'],core_unconfirmed=q['previous_reviewed_core_unconfirmed'],action_errors=q['evaluation_action_errors'],
          approach_return=q['approach_return_observed_frames'],rest=q['rest_observed_frames'],
          runs=[dict(group=s['group'],runs=s['runs'],isolated=s['isolated_candidate_frames']) for s in q['states']]))
    old=read(D/'VERIFIED.json');primary=verified['conditions'][plan['primary']]
    result=dict(at=clock(),primary=plan['primary'],same_task_conditions=verified['conditions'],paired_factor_differences=verified['paired_factor_differences'],
      historical_D_primary=dict(normal_action=old['normal_action'],normal_order=len(old['normal_order_after']),local_reverse=len(old['local_reverse_detected'])),
      comparison_scope='Four matched new conditions isolate the two interventions at fixed seed42. D is historical, different output vocabulary, not the matched control. Repeated DEV and same-source speed variants are not independent examples.',
      confusion=[dict(truth=a,prediction=b,count=n) for (a,b),n in sorted(confusion.items())],normal_sequences=rows,
      context_as_action_errors=[p for p in r['points'] if p['kind'] in NORMAL and p['y']<8 and p['pred']==9],
      lessons_for_interpretation=[
       'Target-conditioned posterior is a learned hint, not independent proof of the requested action. Preserve goal-blind RGB/geometry evidence and old raw outputs.',
       'Additional preparation/return labels leave96ambiguous candidate times unknown; they do not establish that every movement outside sparse main witnesses is irrelevant.',
       'A sparse event classifier and a reliable full sequence verifier are distinct outcomes. Normal order failures and altered-action detection must be reported even if frame accuracy increases.',
       'Do not erase repeated late detections, choose the longest run, lower confidence thresholds or choose a different primary from DEV results.',
       'Before any runtime adoption, passing development must be followed by fixed stability tests, target-mismatch challenges, complete latency measurement and a separate demo. Already used H14/15 cannot be reused to select changes.'
      ],adopted=False,web_changed=False,whole_goal_complete=False)
    exclusive(E/'DIAGNOSIS.json',result);print('Saved frozen-output diagnosis without reinference')
if __name__=='__main__':main()
