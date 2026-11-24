"""One new observation pass on saved automatic candidates, without new selection."""
from pathlib import Path
import json,hashlib,sys,importlib.util,collections,time
import numpy as np,torch
from sequence_core import features,decide
from evidence_path import summarize
U=Path(__file__).resolve().parent;O=U.parent;I=O/'word100_goal_binding_integration_20261114';T=O/'word100_terminal_relation_targets_20261114';G=O/'word100_web_completion_graph_20261111';D=U/'state_path'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
new=module('shape_path_auto',D/'core.py');old=module('terminal_auto',T/'core.py')
rows=read(I/'FULL_GRAPH_COMPARISON.json')['rows'];association={(r['seed'],r['clip_id'],r['kind']):r for r in read(I/'ASSOCIATION_SCOPE_COMPARISON.json')['rows']}
components={(r['seed'],r['clip_id'],r['kind']):r for r in read(I/'VISIBILITY_COMPARISON.json')['rows']}
pm={(r['clip_id'],r['kind']):r for r in read(G/'typed_motion_recovery_20261113/null_candidate_learning/evaluation/COMPARISON.json')['predictions']if r['seed']==42}
specs=read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words']
assert len(rows)==246 and len(pm)==82
oldpaths=[Path(r['model'])for r in read(T/'TRAINING_COMPLETE.json')['rows']]
write('AUTO_INTEGRATION_FROZEN.json',dict(
 input='82 existing exposed public edited scenarios; all 246 saved automatic candidate results retained',
 policy='Attach a separate shape state path to existing full candidate spans, never search for a desired state or replace original shape/direction/order evidence',
 ownership='Explicit independently supplied association state required. A single role occurrence is not proof of whole action completion.',
 unchanged=['candidate spans','order outputs','qualified shape and direction','model selection','thresholds','whole completion remains unknown'],
 preserve=['normal/speed per seed','all correction evidence','all missing masks','conflicts and unconfirmed action identity'],
 expected_order_not_used_for_candidate_selection=True,final8_unopened=True,
 pins={str(p):sha(p)for p in [Path(__file__),U/'evidence_path.py',U/'sequence_core.py',D/'core.py',T/'core.py',I/'FULL_GRAPH_COMPARISON.json',I/'ASSOCIATION_SCOPE_COMPARISON.json',I/'VISIBILITY_COMPARISON.json']+oldpaths+[D/'models'/f'{s}.pt'for s in [42,43,44]]},
 source_pins={r['source']:r['source_sha256']for r in pm.values()},
 caveat='No direct new correctness labels for record11-13; diagnostic transfer is not an accuracy test. Existing endpoint head sees these inputs first time; prior completed outputs are reused.'))
torch.set_num_threads(2);nets=[];oldnets=[]
for s,p in zip([42,43,44],oldpaths):
 m=new.StatePath();m.load_state_dict(torch.load(D/'models'/f'{s}.pt',weights_only=True,map_location='cpu'));m.eval();nets.append(m)
 m=old.TraitHeads();m.load_state_dict(torch.load(p,weights_only=True,map_location='cpu'));m.eval();oldnets.append(m)
dense={};saved={};times=[]
for sid,p in pm.items():
 assert sha(Path(p['source']))==p['source_sha256'];z=np.load(p['source']);x=z['x'];ts=z['ts'];n=len(x);assert x.shape[1]==280
 dense[sid]={};st=time.perf_counter()
 for hand in ['L','R']:
  ff,ok=features(x,hand);contexts=np.zeros((n,7,59),np.float32)
  for i in range(n):
   for j,k in enumerate(range(i-3,i+4)):
    if 0<=k<n:contexts[i,j]=ff[k]
  f0,ok0=old.features(x,'shape',hand)
  with torch.inference_mode():
   pos=np.stack([m.position(torch.tensor(contexts)).numpy()for m in nets])
   pp=np.stack([m(torch.tensor(f0),'shape').softmax(-1).numpy()for m in oldnets])
  states=old.consensus(pp,ok0,'shape');dense[sid][hand]=dict(positions=pos,available=ok&ok0,states=states,ts=ts)
  key=sid[0]+'__'+sid[1]+'__'+hand
  saved[key+'__positions']=pos;saved[key+'__old_probabilities']=pp;saved[key+'__available']=ok&ok0
 times.append(dict(clip_id=sid[0],kind=sid[1],seconds=time.perf_counter()-st,frames=n))
np.savez_compressed(U/'AUTO_DENSE_OUTPUTS.npz',**saved)
out=[];diagnostics=[]
for r in rows:
 key=(r['seed'],r['clip_id'],r['kind']);sid=key[1:];spec={f['id']:f for f in specs[r['clip_id'].split('_')[2]]}
 ac={c['card']:c for c in association[key]['association']['cards']};extra=[]
 for c in r['cards']:
  # Nonmovement role-only observations are kept; no artificial continuous span.
  chosen=c.get('chosen');span=chosen.get('span')if chosen else None
  if span is None:
   extra.append(dict(card=c['card'],status='no_automatic_movement_span',role_observations_retained=True));continue
  a,b=map(int,span);perhand=[]
  for hand in spec[c['card']]['hands']:
   d=dense[sid][hand];pos=d['positions'][:,a:b+1];ok=d['available'][a:b+1];ts=d['ts'][a:b+1]
   assert len(pos[0])==b-a+1
   probs=[]
   for k,m in enumerate(nets):
    delta=float(pos[k,-1]-pos[k,0]);margin=float(torch.sigmoid(m.margin_raw).detach());scale=float(torch.nn.functional.softplus(m.scale_raw).detach())
    probs.append(torch.tensor([-delta,margin,delta]).mul(scale).softmax(-1).numpy())
   change=decide(np.stack(probs),bool(ok[0]and ok[-1]and ok.mean()>=.8))
   assoc=ac[c['card']]['state']=='role_observations_available'
   ev=summarize(change,pos,d['states'][a:b+1],ok,ts,association=assoc)
   # Endpoint equality is a net change statement, not proof of no internal motion.
   ev['net_change_only']=True;ev['whole_interval_shape_maintenance']=None
   ev['automatic_span']=span;ev['hand']=hand;ev['association_state']=ac[c['card']]['state']
   ev['probabilities']=np.stack(probs).tolist();ev['whole_action_complete']=None
   perhand.append(ev)
  extra.append(dict(card=c['card'],status='observation_attached',hands=perhand))
  diagnostics.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['card'],span=span,
      association=ac[c['card']]['state'],states=[h['status']for h in perhand],changes=[h['raw_change']for h in perhand]))
 out.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],original_candidates=r['cards'],original_orders=r['orders'],
   original_qualified_components=components[key]['components'],original_association=association[key]['association'],shape_path=extra,
   whole_action_complete=None,educational_error=None,web_enabled=False))
write('AUTO_INTEGRATION.json',dict(rows=out,diagnostics=diagnostics,source_scenarios=len(pm),source_videos=len({r['clip_id']for r in rows}),
 dense_source_hand_passes=len(pm)*2,old_result_reinference=0,timing=times,
 inference_scope='New state path and old endpoint on new previously exposed auto-candidate inputs. Not end-to-end speed, not new held-out accuracy.',whole_G2_passed=False))
print(json.dumps(dict(rows=len(out),scenarios=len(pm),normal=[r for r in diagnostics if r['kind']=='full_normal'and r['seed']==42]),ensure_ascii=False))
