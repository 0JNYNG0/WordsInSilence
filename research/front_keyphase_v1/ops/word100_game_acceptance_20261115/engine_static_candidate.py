"""Frozen current G2/G3 on NEW public inputs. No witness labels in selection."""
from pathlib import Path
import sys
O=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(O/'word100_public_g4_20261114'))
from prepare import read,sha,write,load
from conflict_guard import qualify
import sys,time,itertools,copy
import numpy as np,torch
G=O/'word100_web_completion_graph_20261111';W=G/'candidate_description_supervision';H=G/'typed_motion_recovery_20261113';I=O/'word100_goal_binding_integration_20261114';J=O/'word100_joint_completion_block_20261114';B=O/'word100_action_evidence_block_20261114';R=O/'word100_candidate_role_learning_20261111'
sys.path[:0]=[str(W),str(I),str(J),str(B)]
from local_rank_core import LocalRankHead,F,AX
from source_boundary_core import BodyWindow,body_context,pool_windows
from transfer_local_timefix_core import normalized_pose,encode_windows
from core_correspondence import attach_role_core,core_order
from association_scope import scope_associations
from component_visibility import guard_component
from assemble_evidence import assemble
from joint_core import integrate
from feedback_draft import draft
from traits import Traits,face_geometry,old
rc=load('g4req',O/'word100_guide_requirement_learning_20261111/requirement_core.py')
role=load('g4role',R/'role_core.py');binder=load('g4bind',G/'binding.py')
def netload(cls,p):
 st=torch.load(p,weights_only=True,map_location='cpu');m=cls(st['mean'].numpy(),st['std'].numpy());m.load_state_dict(st);m.eval();return m
def latent(base,pose):
 with torch.inference_mode():
  x=torch.tensor(pose[None]);h=torch.nn.functional.gelu(base.input(x)).transpose(1,2)
  for layer in base.temporal:h=layer(h)
  h,_=base.context(h.transpose(1,2));return h[0].numpy()
def initialize():
 global _supplement
 sys.path.insert(0,str(O/'word100_finger_repair_20261115'))
 from runtime_extension import Extension
 assert read(O/'word100_finger_repair_20261115/CONNECT_GATE.json')['passed']
 _supplement=Extension(rc)
 net=rc.frozen_observer();base=net.shared.base;rn=role.frozen_model();traits=Traits();heads={};caps={(r['word'],r['card'],r['feature'],r['expected']):r['qualified']for r in read(rc.U/'GUIDE_CAPABILITY.json')['rows']}
 for s in [42,43,44]:
  rr=torch.load(R/'runs'/str(s)/'model.pt',weights_only=True,map_location='cpu');rolehead=torch.nn.Sequential(torch.nn.Linear(684,96),torch.nn.GELU(),torch.nn.Linear(96,11));rolehead.load_state_dict({k[5:]:v for k,v in rr.items()if k.startswith('role.')});rolehead.eval()
  heads[s]=(netload(BodyWindow,W/'source_boundary_candidate_v3/runs/body_window'/str(s)/'model.pt'),netload(LocalRankHead,W/'local_rank_learning/runs/local_full'/str(s)/'model.pt'),netload(LocalRankHead,H/'full_bag_rate_rank_learning/runs'/str(s)/'model.pt'),rolehead)
 return net,base,rn,traits,heads,caps
def infer(models,ch,raw,word):
 net,base,rn,traits,heads,caps=models
 tick=time.perf_counter();stop=len(ch['frame_id']);rate=1;cid='live';kind='live'
 if word not in rc.FAMILIES:raise ValueError('Unsupported feedback word')
 if stop<4:raise ValueError('At least four observed frames required')
 if stop>480:raise ValueError('Recording exceeds the 15-second capture envelope')
 x=ch['tcn280'][:stop].copy();ts=np.round(ch['timestamp_us'][:stop]/rate).astype(np.int64);fid=ch['frame_id'][:stop];x[:,279]=np.r_[0,np.diff(ts)/1e6];pose=normalized_pose(x);fh=latent(base,pose);ob=role.features(rn,x,ts,fh);z=ob['z'];qr=z[:,292:392].reshape(-1,10,10)
 specs=[s for s in rc.FAMILIES[word]if s['required']];move=[s for s in specs if s['families'][0]in F];cx=body_context(qr,ts);T=len(x);ss=[]
 for spec in move:
  v=qr[:,AX[spec['families'][0]]*2,8]>.5;cs=np.r_[0,v.cumsum()];ss.append(np.array([(a,b)for a in range(T-3)for b in range(a+3,T)if v[a]and v[b]and cs[b+1]-cs[a]>=.8*(b-a+1)],np.int32).reshape(-1,2))
 union=sorted({tuple(v)for sp in ss for v in sp});sp=np.array(union,np.int32).reshape(-1,2);lu={v:i for i,v in enumerate(union)};bf=pool_windows(cx,sp)
 ff=np.concatenate([encode_windows(base,pose,qr.reshape(-1,100),sp[a:a+128])for a in range(0,len(sp),128)])if len(sp)else np.empty((0,1160),np.float32)
 face,faceok=face_geometry(raw['xy_px'][:stop],ch['point_available'][:stop]);dense={h:traits.infer(x,face,faceok,h)for h in ['L','R']};av=rc.prior.shared.available(x)
 with torch.inference_mode():shape=net.shared.base.attributes(torch.tensor(fh)).sigmoid().numpy()
 variants=[]
 for seed,(bw,rank,ratehead,rh)in heads.items():
  with torch.inference_mode():
   presence=bw(torch.tensor(bf)).numpy();rankscore=rank(torch.tensor(ff)).numpy();ratescore=ratehead(torch.tensor(ff)).numpy();logits=rh(torch.tensor(z));logits[:,10]=-1e4;prob=logits.softmax(-1).numpy()
  visible=x[:,186:237]>.5;right=visible[:,[30]+list(range(35,51))].all(1);left=visible[:,[9]+list(range(14,30))].all(1);occ=[]
  for f in sorted({f for s in specs for f in s['families']}):occ+=binder.bind_occurrences(prob,right if f in [1,2,3]else right&left,ts,[f],.65)
  for condition in ['current_g3']:
   cards=[]
   for spec,spans in zip(move,ss):
    family=spec['families'][0];j=F.index(family);ids=np.array([lu[tuple(v)]for v in spans],int);score=(ratescore if condition=='current_g3'and family in [2,3]else rankscore)[ids,j];values=np.where(presence[ids,j,0]>=0,score,-np.inf);chosen=None
    if len(values)and np.isfinite(values).any():
     best=int(values.argmax());a,b=map(int,spans[best]);chosen=dict(span=[a,b],score=float(values[best]),semantic_completion=None)
    cards.append(dict(card=spec['id'],family=family,chosen=chosen))
   cards=attach_role_core(cards,occ)
   for spec in specs:
    if spec['id']in[c['card']for c in cards]:continue
    match=[dict(family=e['family'],span=[e['a'],e['b']],original_role_span=[e['a'],e['b']])for e in occ if e['family']in spec['families']]
    cards.append(dict(card=spec['id'],family=spec['families'][0],chosen=None,role_confirmation=dict(matching=match,other_observed_roles=[],state='matching_role_observed'if match else'no_role_confirmation')))
   lookup={c['card']:c for c in cards};cards=[lookup[s['id']] for s in specs]
   allowed={tuple([a['id'],b['id']]) for a,b in zip(specs,specs[1:]) if a['families']!=b['families']}
   order=[p for p in core_order(cards) if tuple(p['cards']) in allowed];association=scope_associations(cards,{s['id']:s['axis']is not None for s in specs});am={c['card']:c['state']for c in association['cards']};components=[];paths=[];extras=[]
   for c in cards:
    spec=next(s for s in specs if s['id']==c['card']);guide=next(v for v in rc.CARDS[word]if v['id']==c['card']);span=c['chosen']['span']if c['chosen']else None;es=[];ph=[]
    # A stationary role has no movement-window candidate. Reuse its independently
    # detected native span to observe shape, rather than silently dropping it.
    # Never merge gaps or search for the required shape inside a larger region.
    if span is None and spec['axis'] is None:
     matches=c.get('role_confirmation',{}).get('matching',[])
     unique=sorted({tuple(e['span']) for e in matches if e['family'] in spec['families']})
     if len(unique)==1:span=list(unique[0])
    if span:
     a,b=span;aa=a+(b-a)//2 if guide.get('shape_scope')=='late'else a
     for j,want in guide.get('features',{}).items():
      j=int(j);v=av[aa:b+1,j];pval=float(shape[aa:b+1,j][v].mean())if v.any()else None;pval=pval if want or pval is None else 1-pval;qualified=caps.get((word,c['card'],j,want),False)
      es.append(dict(task='shape',feature=j,expected=want,span=[aa,b],probability=pval,state=rc.decide(pval,bool(v.mean()>=.8),qualified)if pval is not None else'observation_unknown'))
     if spec['axis']is not None:
      if spec['sign']is None:es.append(dict(task='direction',state='sequence_required'))
      else:
       geom,ok,info=rc.prior.geometry(x,ts,a,b,spec['axis'],spec['hands'][0],'L'if spec['hands'][0]=='R'else'R')
       with torch.inference_mode():dp=net.movement(torch.tensor(fh[None]),torch.tensor(geom[None]),torch.tensor([[0,a,b]])).softmax(-1)[0].numpy()
       es.append(dict(task='direction',span=span,probabilities=dp.tolist(),state=rc.direction_verdict(dp,spec['sign']+1,ok)))
     joint=None
     for h in spec['hands']:
      v=traits.path(dense[h],ts,fid,a,b,am[c['card']]=='role_observations_available');ev=v['shape_path']['raw'];ev.update(hand=h,automatic_span=span);ph.append(ev)
      if c['card']=='close_chest'and h=='R':joint=v['joint']
     extras.append(dict(card=c['card'],span=span,joint=joint,source_frames=fid[a:b+1].tolist(),timestamps_us=ts[a:b+1].tolist()))
    components.append(guard_component(dict(card=c['card'],span=span,evidence=es),x,spec['hands']))
    paths.append(dict(card=c['card'],status='observation_attached'if span else'no_automatic_movement_span',hands=ph))
   saved=dict(seed=seed,clip_id=cid,kind=kind,original_candidates=cards,original_orders=order,original_qualified_components=components,original_association=association,shape_path=paths)
   merged=integrate(assemble(saved),extras)
   for c in merged['joint_cards']:c['drafts']=draft(c,next(s['instruction']for s in specs if s['id']==c['card']))
   variants.append(dict(condition=condition,**merged))
 result=qualify(variants)
 result=_supplement.apply(result,x,ts,word)
 result['runtime']=dict(version='G6-finger-20261115',frames=stop,candidate_windows=len(sp),seconds=time.perf_counter()-tick,
                        trained=False,reference_video_reads=0,rgb_model_calls=0)
 return result
