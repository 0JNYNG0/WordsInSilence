from pathlib import Path
import sys,json,importlib.util,collections
import numpy as np,torch
U=Path(__file__).resolve().parent;O=U.parent
sys.path.insert(0,str(O/'word100_guide_requirement_learning_20261111'))
import requirement_core as rc
read=rc.read;write=rc.write
torch.set_num_threads(2)
net=rc.frozen_observer();ready={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows']}
motion={r['clip_id']:r for r in read(O/'word100_motion_components_20261109/READY.json')['rows']}
old=read(O/'word100_joint_completion_block_20261114/JOINT_FINAL_V2.json')['rows']
pm={(r['clip_id'],r['kind']):r for r in read(O/'word100_web_completion_graph_20261111/typed_motion_recovery_20261113/null_candidate_learning/evaluation/COMPARISON.json')['predictions'] if r['seed']==42}
edits={r['clip_id']:r for r in read(O/'word100_observation_guard_20261108/INPUT_PLAN.json')['rows']}
out=[]
for r in old:
 if r['seed']!=42 or r['kind'] not in ['full_normal','right_closed_hand_main','main_reversed','closed_hand_and_reversed']:continue
 cid=r['clip_id'];pairs=[p for p in motion.get(cid,{}).get('pairs',[]) if p['target']!=1]
 if not pairs:continue
 inp=pm[cid,r['kind']];assert rc.sha(inp['source'])==inp['source_sha256'];z=np.load(inp['source']);x=z['x'];ts=z['ts'];h=z['z'][:,392:456]
 # Recompute the deployed observer: cached role-model latents are not the same model.
 with torch.no_grad():hh=net.shared(torch.tensor(rc.prior.shared.normalize(x)[None]),torch.tensor([len(x)]),torch.tensor([ready[cid]['word']]))['latent']
 source=ready[cid];fid=np.load(source['path'])['frame_id'];mapping=np.arange(len(fid))
 if r['kind'] in ['main_reversed','closed_hand_and_reversed']:
  crop=np.load(edits[cid].get('cache')or O/'word100_observation_guard_20261108/crops'/f'{cid}.npz');ct=crop['timestamp_us'];lo,hi=edits[cid]['intervals'][0];ii=np.flatnonzero((ct>=lo*1e6)&(ct<=hi*1e6));mapping[ii]=mapping[ii[::-1]]
 sf=fid[mapping]
 for c in r['joint_cards']:
  ds=[v for v in c['observations'] if v['task']=='direction' and 'span'in v]
  for d in ds:
   pp=[p for p in pairs if p['axis']==d.get('axis')]
   for p in pp:
    ids=np.flatnonzero((sf>=p['frame_a'])&(sf<=p['frame_b']));
    if len(ids)<4:continue
    a,b=int(ids.min()),int(ids.max());target=2-p['target'] if r['kind']in ['main_reversed','closed_hand_and_reversed']else p['target']
    gg,ok,info=rc.prior.geometry(x,ts,a,b,p['axis'],p.get('moving','R'),p.get('anchor','L'))
    with torch.no_grad():prob=net.movement(hh,torch.tensor(gg[None]),torch.tensor([[0,a,b]])).softmax(-1)[0].numpy()
    ag,aok,ai=rc.prior.geometry(x,ts,*d['span'],p['axis'],p.get('moving','R'),p.get('anchor','L'))
    out.append(dict(clip_id=cid,kind=r['kind'],card=c['card'],axis=p['axis'],truth_interval=[a,b],source_interval=[p['frame_a'],p['frame_b']],target=target,
      automatic_span=d['span'],automatic_state=d['state'],automatic_probabilities=d['probabilities'],automatic_delta=ai.get('signed_change'),
      known_span_state=rc.direction_verdict(prob,target,ok),known_span_probabilities=prob.tolist(),known_span_delta=info.get('signed_change'),
      truth_scope='Previously reviewed observed movement interval, not whole-sign correctness; edited channels are synthetic controls',source=inp['source']))
write(U/'INTERVAL_DIAGNOSIS.json',dict(rows=out,review='Known intervals used only for diagnosis, never runtime selection',public_only=True))
print(json.dumps(out,ensure_ascii=False))
