"""Saved-result cause separation; no new model inference or score tuning."""
from role_model import *
from evaluate_roles import NORMAL
from collections import Counter
def main():
    r=read(A/'evaluations/temporal_42.json');before=read(T/'evaluations/current_42.json');b={s['key']:s for s in before['sequences']}
    wrong=[p for p in r['points'] if p['kind'] in NORMAL and p['y']!=p['pred']]
    confusion=[[sum(p['y']==i and p['pred']==j for p in r['points'] if p['kind'] in NORMAL) for j in range(3)] for i in range(3)]
    harmless_role_swap=[p for p in wrong if {p['y'],p['pred']}=={0,2}]
    knowncore=[p for p in r['points'] if p['kind'] in NORMAL and p['y']==1]
    high_conf_core_lost=[p for p in knowncore if p['pred']!=1 and max(p['probability'])>=.8]
    failures=[];changed=[]
    for s in r['sequences']:
        if s['kind'] not in NORMAL+['actual_reverse_main']:continue
        if s['status']!=b[s['key']]['status']:changed.append(dict(key=s['key'],before=b[s['key']]['status'],after=s['status']))
        if s['status'] not in ['IMPORTANT_STATE_OR_ORDER_UNCONFIRMED','ACTION_ROLE_OR_CONFIGURATIONS_UNCONFIRMED']:continue
        path=A/'predictions/temporal_42'/f"{s['key']}.npz";a=np.load(path);role=a['role_probability'];cur=a['current_probability']
        item=dict(key=s['key'],kind=s['kind'],status=s['status'],missing_groups=s['missing_groups'],edges=s['precedences'],
          groups=[dict(group=x['group'],runs=x['runs'],short=x['isolated_candidate_frames']) for x in s['states']],
          core_witnesses=s['role_core_witnesses'],unknown_role_on_confident_state=[])
        for i in np.flatnonzero((cur[:,:9].max(1)>=.8)&(role.max(1)<.8)):
            item['unknown_role_on_confident_state'].append(dict(frame=int(i),state=int(cur[i].argmax()),state_probability=float(cur[i].max()),role=role[i].tolist()))
        failures.append(item)
    out=dict(at=clock(),source_sha256=sha(A/'evaluations/temporal_42.json'),normal_confusion_true_rows_pred_columns=confusion,
      normal_errors=len(wrong),transit_rest_only_swaps=len(harmless_role_swap),high_confidence_core_relabelled_noncore=high_conf_core_lost,
      changes=changed,unresolved=failures,
      interpretation=['Rest-vs-transit mistakes are distinct from removing presented action; neither should hide the native timeline.','A correct sparse role label does not establish dense source semantics.','Current state broad categories and passing role cases are preserved.','Inspect ambiguous approach-to-main boundaries and opposite-order evidence jointly; do not lower role threshold or pick seed44 from DEV.'],
      no_new_inference=True,no_changes_to_frozen_results=True,whole_goal_complete=False)
    exclusive(A/'CAUSE_DIAGNOSIS.json',out)
    print('Confusion',confusion,'errors',len(wrong),'transit/rest swaps',len(harmless_role_swap),'highconf core lost',len(high_conf_core_lost))
    print('Changes',changed)
    for f in failures:
        if f['kind'] in ['native','actual_reverse_main']:print(f['key'],f['groups'],f['unknown_role_on_confident_state'][:4])
if __name__=='__main__':main()
