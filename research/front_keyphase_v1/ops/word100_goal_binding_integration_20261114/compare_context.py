from pathlib import Path
import json,hashlib,collections
import numpy as np
import torch
from context_binding import ContextGoalObserver
U=Path(__file__).resolve().parent;O=U.parent;D=O/'word100_positive_direction_20261109';R=O/'word100_candidate_role_learning_20261111'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,j):
 with (U/n).open('x',encoding='utf-8')as f:json.dump(j,f,ensure_ascii=False,indent=2)
write('CONTEXT_FROZEN.json',dict(reason='Initial integration omitted guide-declared arrival support; the movement/arrival UI cards are not independent actions.',
    code_sha256=sha(U/'context_binding.py'),initial_sha256=sha(U/'INITIAL_COMPARISON.json'),
    source_groups='Existing direction GUIDE_CONTRACTS groups; otherwise current card groups',phase_threshold=.65,max_gap_us=120000,
    context='One previous native sample only, not search for expected start state; no expected order or goal score for selection',
    models_unchanged=True,threshold_tuning=False,old_cards_immutable=True))
torch.set_num_threads(2);model=ContextGoalObserver();rows=[]
for p in sorted((D/'final_decisions').glob('*.json')):
    z=np.load(D/'predictions'/(p.stem+'.npz'));old=read(p)['after'];out=model.attach(old,z['x'],z['timestamp_us'],z['frame_id'],phase=z['phase'])
    extra=out.pop('goal_components');assert out==old
    rows.append(dict(source='web',case=p.stem,normal=any(k in p.stem for k in ['_normal','_speed','_translation_scale']),components=extra))
# This comparison uses current-web phase tensors. Original role arrays have a
# different family schema and cannot silently be used as web phases.
summary=[]
for normal in [True,False]:
 for card in ['brush_arm','spread_hands']:
    rr=[e for r in rows if r['normal']==normal for c in r['components']if c['card']==card for e in c['events']]
    summary.append(dict(normal=normal,card=card,states=dict(collections.Counter(e['evidence']['status']for e in rr))))
write('CONTEXT_COMPARISON.json',dict(rows=rows,summary=summary,old_cards_preserved=True,semantic_completion=None))
print(json.dumps(summary,ensure_ascii=False))
