"""Read-only localized goal evidence. Localization is never goal conditioned."""
from pathlib import Path
import sys,copy
import numpy as np
import torch
U=Path(__file__).resolve().parent;S=U.parent/'word100_goal_state_learning_20261114'
sys.path.insert(0,str(S))
from state_model import StateHeads,features,consensus,transition_evidence
from goal_contract import summarize_component
CONTRACTS={'brush_arm':dict(domain='arm_region',path=['upper_arm_side','wrist_side'],hand='R'),
 'spread_hands':dict(domain='separation',path=['hands_near','hands_apart'],hand='R')}
class GoalObserver:
 def __init__(self):
    self.models=[]
    for seed in [42,43,44]:
        m=StateHeads('mlp');m.load_state_dict(torch.load(S/'models'/f'mlp_{seed}.pt',map_location='cpu',weights_only=True));m.eval();self.models.append(m)
 def states(self,x,contract):
    f,ok=features(x,contract['domain'],contract.get('hand','R'))
    with torch.inference_mode():pp=[torch.softmax(m(torch.from_numpy(f),contract['domain']),-1).numpy()for m in self.models]
    return consensus(pp,ok,contract['domain'],.8),ok
 def attach(self,result,x,ts,fid,contracts=CONTRACTS):
    out=copy.deepcopy(result);extra=[]
    if len(x)!=len(ts)or len(x)!=len(fid)or not np.all(np.diff(ts)>0):raise ValueError('Native monotone timing required')
    for card in out.get('cards',[]):
        contract=contracts.get(card['id'])
        if contract is None:continue
        # Keep each observed available island. Do not bridge unrelated motions.
        spans=[s for e in card.get('phase_evidence',[])for s in e.get('available_runs',[])]
        if not spans and card.get('query_indices')is not None:spans=[card['query_indices']]
        spans=sorted({tuple(map(int,s))for s in spans});events=[]
        for a,b in spans:
            if not 0<=a<=b<len(x):raise ValueError('Invalid automatic observation span')
            states,available=self.states(x[a:b+1],contract)
            ev=transition_evidence(states,contract['path'],available)
            hints=summarize_component(ev,available,states,contract['domain'])
            events.append(dict(span=[a,b],seconds=[float(ts[a]/1e6),float(ts[b]/1e6)],source_frames=[int(fid[a]),int(fid[b])],
                states=states,available=available.tolist(),evidence=ev,hint=hints,
                witness_indices=[a+i for i in ev['witness_indices']],association='existing_model_estimate'))
        extra.append(dict(card=card['id'],contract=contract,events=events,status='no_localized_observation'if not events else'observed_component_events',
          semantic_completion=None,educational_error=None,display_enabled=False))
    out['goal_components']=extra
    return out
