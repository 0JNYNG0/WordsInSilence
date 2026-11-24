import json,numpy as np
from pathlib import Path
from identity_v3 import conflict_mask,choose
U=Path(__file__).resolve().parent
rows=[{'clip_index':0,'frame':0,'side':1,'pairs':[{'existing_side':1,'hold_new':True}]},
      {'clip_index':0,'frame':1,'side':1,'pairs':[{'existing_side':0,'hold_new':True}]},
      {'clip_index':0,'frame':2,'side':1,'pairs':[{'existing_side':0,'hold_new':False}]}]
m=conflict_mask(rows,0,(3,2))
tests={'same_side_confirmation_not_quarantined':not m[0].any(),'cross_side_conflict_quarantines_both':m[1].all(),'separate_hands_not_quarantined':not m[2].any(),
'unknown_not_forced':choose([], [{'supported_side':-1},{'supported_side':-1}],1,0)['chosen']==-1,
'two_supported_candidates_not_arbitrarily_selected':choose([], [{'supported_side':1},{'supported_side':1}],1,0)['chosen']==-1,
'unique_supported_candidate':choose([], [{'supported_side':-1},{'supported_side':1}],1,0)['chosen']==1,
'same_side_quality_tie_not_used_for_replacement':choose([], [{'supported_side':-1},{'supported_side':1}],1,1)['chosen']==-1}
assert all(tests.values()),tests
with (U/'CONTRACT_TESTS.json').open('x',encoding='utf-8') as q:json.dump({'status':'PASS','tests':{k:bool(v) for k,v in tests.items()},'scope':'Policy invariants, not recognition accuracy'},q,indent=2)
print(len(tests),'contract checks PASS')
