"""Generic action evidence grouping; no target direction/order chooses windows.

The display may split movement and arrival into separate cards. Their guide-
declared phase groups can belong to one action. Preserve them and their gaps.
"""
from pathlib import Path
import copy,json
import numpy as np
from binding import GoalObserver,CONTRACTS
O=Path(__file__).resolve().parent.parent
direction=json.loads((O/'word100_positive_direction_20261109/GUIDE_CONTRACTS.json').read_text('utf-8-sig'))['contracts']
class ContextGoalObserver(GoalObserver):
 def attach(self,result,x,ts,fid,contracts=CONTRACTS,phase=None):
    if phase is None:phase=np.asarray(result.get('phase_probability',[]))
    if phase.ndim!=2 or len(phase)!=len(x):raise ValueError('Model phase evidence required')
    local=copy.deepcopy(result)
    for card in local.get('cards',[]):
        if card['id']not in contracts:continue
        # The same groups are already declared in the unchanged directional
        # observer: movement and optional arrival are not separate actions.
        groups=direction[card['id']]['phase_groups']if card['id']in direction else [g for gg in card['groups']for g in gg]
        active=phase[:,np.asarray(groups)+1].sum(1)>=.65
        ids=np.flatnonzero(active);cuts=np.flatnonzero((np.diff(ids)>1)|(np.diff(np.asarray(ts)[ids])>120000))+1
        runs=[z for z in np.split(ids,cuts)if len(z)]
        spans=[];context=[]
        for z in runs:
            a,b=int(z[0]),int(z[-1]);start=a
            # One adjacent observed boundary sample, not a search for a desired
            # state. No completion/direction label is consulted here.
            if a>0 and 0<ts[a]-ts[a-1]<=120000:start=a-1
            spans.append([start,b]);context.append(dict(model_span=[a,b],context_span=[start,b],
                included_phase_groups=groups,expected_order_used=False,goal_state_used_for_selection=False))
        card['phase_evidence']=[dict(available_runs=spans)]
        card['query_indices']=None
        card['_temporary_goal_context']=context
    attached=super().attach(local,x,ts,fid,contracts)
    out=copy.deepcopy(result);extra=attached['goal_components']
    byid={c['id']:c for c in local['cards']}
    for c in extra:
        c['context_contract']=byid[c['card']].get('_temporary_goal_context',[])
        c['arrival_hold_required']=False
    out['goal_components']=extra
    return out
