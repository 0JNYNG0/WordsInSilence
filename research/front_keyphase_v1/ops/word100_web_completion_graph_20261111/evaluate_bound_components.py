"""Audit shapes/direction on the actual selected intervals, without TCN/DTW replay.

Uses the frozen common100 observer's saved latent and unchanged attribute layer.
No learned word recognition, pose observation, or old experiment is rerun.
This is an isolated new combination; it cannot promote itself to the web.
"""
from graph import *
from prepare_linker import load
from collections import Counter
import numpy as np,torch
O=U.parent;G=O/'word100_guide_requirement_learning_20261111';R=O/'word100_candidate_role_learning_20261111'


def main():
 assert not (U/'BOUND_COMPONENTS_COMPLETE.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rc=load('_bound_requirements',G/'requirement_core.py');role=load('_bound_role',R/'role_core.py')
 pe=load('_bound_shapes',O/'word100_positive_direction_20261109/positive_evidence.py')
 base=rc.frozen_observer();motion={}
 for seed in [42,43,44]:
  n=role.mc.MotionNet();n.load_state_dict(torch.load(role.M/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'),strict=False);n.eval();motion[seed]=n
 sources={(s['clip_id'],s['kind']):s for s in read(R/'EVALUATION_INPUTS.json')['rows']}
 saved={(s['seed'],s['clip_id'],s['kind']):s for s in read(R/'safety_reconcile/FULLPATH_COMPLETE.json')['rows']}
 write(U/'BOUND_COMPONENTS_FROZEN.json',dict(
  selector='INTERVAL_PRESERVATION_RESULTS.json',selector_sha256=hashlib.sha256((U/'INTERVAL_PRESERVATION_RESULTS.json').read_bytes()).hexdigest(),
  shape='Frozen common100 coupled42 attributes(saved h); four-finger/index features 0..4 only.',
  movement='Existing coupled42 plus already trained seed-specific short-motion rescue; threshold unchanged .65, geometry sign agreement.',
  invariants=['No target word ID branch','No absolute pixel thresholds','No duration as semantic error','No missing or held endpoint promoted to motion completion','No gap between supported intervals painted as known','No overall correctness percentage'],
  default_seed=42,new_training=False,new_pose_or_TCN_DTW_inference=False,web_changed=False,
  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 out=[];cached={};directions={}
 for r in read(U/'INTERVAL_PRESERVATION_RESULTS.json'):
  cid,kind,seed=r['clip_id'],r['kind'],r['seed'];key=(cid,kind);src=sources[key]
  if key not in cached:
   z={k:v for k,v in np.load(src['path']).items()}
   with torch.inference_mode():z['attrs']=base.shared.base.attributes(torch.tensor(z['fh'])).sigmoid().numpy()
   cached[key]=z
  z=cached[key];before=read(saved[seed,cid,kind]['result']);word=cid.split('_')[2];cards=[]
  for c in r['cards']:
   spec=next(s for s in rc.FAMILIES[word] if s['id']==c['id']);desc=next(s for s in rc.CARDS[word] if s['id']==c['id'])
   shape=[];movement=[]
   spans=c['evidence_intervals']
   for span in spans:
    assert all(int(j)<5 for j in desc['features']), 'Do not reinterpret relation heads as hand heights.'
    shape.extend(pe.shape_evidence(z['attrs'],z['x'],z['ts'],span,desc['features'],desc.get('shape_scope')=='late'))
   if spec['axis'] is not None:
    for a,b in spans:
     k=(cid,kind,seed,a,b,spec['axis'])
     if k not in directions:
      d=role.jc.integration.direction(base,z['x'],z['ts'],z['fh'],a,b,spec['axis'])
      nd=role.mc.direction(motion[seed],z['x'],z['ts'],a,b,spec['axis'])
      directions[k]=role.mc.fused(d,nd)
     movement.append(directions[k])
   signs={d['prediction']-1 for d in movement if d['eligible'] and d['confidence']>=.65 and d['prediction']!=1}
   wanted=spec['sign'];ms='not_required' if spec['axis'] is None else 'unknown'
   if spec['axis'] is not None and wanted is not None:
    ms='supported' if signs=={wanted} else 'opposite_direction' if signs=={-wanted} else 'mixed' if len(signs)>1 else 'movement_not_established'
   elif spec['axis'] is not None:
    ms='opposite_directions_observed' if signs=={-1,1} else 'insufficient_alternation_evidence'
   old=next(a for a in before['cards'] if a['id']==c['id'])
   shapes_old=[(s['feature'],s['state']) for s in old.get('shape_evidence',[])]
   shapes_new=[(s['feature'],s['state']) for s in shape]
   cards.append(dict(id=c['id'],span=c['span'],evidence_intervals=spans,shape=shape,
       movement=movement,movement_state=ms,expected_direction=wanted,
       before_shape_states=shapes_old,after_shape_states=shapes_new,
       before_span=old.get('query_indices'),same_window=c['span']==old.get('query_indices'),
       observed_correspondence=c['span'] is not None,
       action_completed=None,whole_sign_correct=None,
       note='A matching held posture does not prove the required movement occurred.'))
  out.append(dict(seed=seed,clip_id=cid,kind=kind,cards=cards,order=r['pairs']))
 write(U/'BOUND_COMPONENTS_RESULTS.json',out)
 summary={str(s):{k:dict(Counter(c['movement_state'] for r in out if r['seed']==s and r['kind']==k for c in r['cards'])) for k in sorted({r['kind'] for r in out})} for s in [42,43,44]}
 shape_changes=[dict(seed=r['seed'],clip_id=r['clip_id'],card=c['id'],before=c['before_shape_states'],after=c['after_shape_states'],same_window=c['same_window']) for r in out if r['kind']=='full_normal' for c in r['cards'] if c['before_shape_states']!=c['after_shape_states']]
 write(U/'BOUND_COMPONENTS_COMPLETE.json',dict(summary=summary,normal_shape_state_changes=shape_changes,web_changed=False,semantic_accuracy_claim=False));print(summary);print('normal_shape_changes',len(shape_changes))

if __name__=='__main__':main()
