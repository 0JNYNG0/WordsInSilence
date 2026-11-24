from pathlib import Path
import json,hashlib,collections
import numpy as np
import torch
from binding import GoalObserver
U=Path(__file__).resolve().parent;O=U.parent;D=O/'word100_positive_direction_20261109';R=O/'word100_candidate_role_learning_20261111'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,j):
 with (U/name).open('x',encoding='utf-8')as f:json.dump(j,f,ensure_ascii=False,indent=2)
torch.set_num_threads(2);observer=GoalObserver();rows=[];pins={}
assert not(U/'INITIAL_COMPARISON.json').exists()
fixed=read(O/'word100_goal_state_learning_20261114/TRAINING_FROZEN.json')
for p,h in fixed['protected'].items():assert sha(Path(p))==h
write('INITIAL_FROZEN.json',dict(candidate_sha256=sha(U/'binding.py'),plan_sha256=sha(U/'PLAN.json'),
 confidence=.8,models={str(p):sha(p)for p in (O/'word100_goal_state_learning_20261114/models').glob('mlp_*.pt')},
 data='Existing exposed current-web93 + original G2 role234; two selectors reported separately. No new detector, encoder or role inference.'))
for p in sorted((D/'final_decisions').glob('*.json')):
    z=np.load(D/'predictions'/(p.stem+'.npz'));old=read(p)['after']
    new=observer.attach(old,z['x'],z['timestamp_us'],z['frame_id']);extra=new.pop('goal_components');assert new==old
    rows.append(dict(source='web',case=p.stem,normal=any(k in p.stem for k in ['_normal','_speed','_translation_scale']),
      preserved=True,components=extra))
    pins[str(p)]=sha(p);pins[str(D/'predictions'/(p.stem+'.npz'))]=sha(D/'predictions'/(p.stem+'.npz'))
source={(r['clip_id'],r['kind']):r for r in read(R/'EVALUATION_INPUTS.json')['rows']};cache={}
for r in read(R/'safety_reconcile/FULLPATH_COMPLETE.json')['rows']:
    s=source[r['clip_id'],r['kind']];p=Path(s['path']);assert sha(p)==s['sha256'];pins[str(p)]=s['sha256']
    if str(p)not in cache:cache[str(p)]=dict(np.load(p))
    z=cache[str(p)];old=read(Path(r['result']));assert sha(Path(r['result']))==r['sha256'];pins[r['result']]=r['sha256']
    fid=z.get('fid',z.get('frame_id',np.arange(len(z['x']))))
    # If source-frame IDs are not cached, do not fabricate an original identity.
    identities_known='fid'in z or 'frame_id'in z
    new=observer.attach(old,z['x'],z['ts'],fid);extra=new.pop('goal_components');assert new==old
    if not identities_known:
        for c in extra:
            for e in c['events']:e['source_frames']=None
    rows.append(dict(source='original_role',case=s['key'],seed=r['seed'],kind=r['kind'],
        normal=r['kind']in ['full_normal','speed075','speed125'],preserved=True,components=extra,source_frame_identity_known=identities_known))
summary=[]
for source_id in ['web','original_role']:
 for normal in [True,False]:
  for card in ['brush_arm','spread_hands']:
    rr=[e for r in rows if r['source']==source_id and r['normal']==normal for c in r['components']if c['card']==card for e in c['events']]
    summary.append(dict(source=source_id,normal=normal,card=card,states=dict(collections.Counter(e['evidence']['status']for e in rr))))
write('INITIAL_COMPARISON.json',dict(rows=rows,summary=summary,pins=pins,old_cards_preserved=True,whole_sign_correctness=False))
print(json.dumps(summary,ensure_ascii=False))
