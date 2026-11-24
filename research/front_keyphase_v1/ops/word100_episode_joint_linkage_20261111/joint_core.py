from pathlib import Path
import sys, json, time, hashlib, importlib.util, copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_action_episode_supervision_20261110'
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ep=load('_joint_episode',P/'episode_core.py');sys.modules['episode_core']=ep
integration=load('_joint_previous_integration',P/'integration.py')
bc,lc,le,old,prior,B,L=ep.bc,ep.lc,ep.le,ep.old,ep.prior,ep.B,ep.L
read,write,sha=ep.read,ep.write,ep.sha
def event(kind,**kw):
 d=dict(time=time.time(),kind=kind,**kw)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
 print(json.dumps(d,ensure_ascii=False),flush=True)

class JointNet(ep.EpisodeNet):
 def __init__(self):
  super().__init__();self.load_state_dict(torch.load(P/'runs/42/model.pt',weights_only=True,map_location='cpu'))
  self.quality=nn.Sequential(nn.Linear(304,64),nn.GELU(),nn.Linear(64,4))
  self.pair=nn.Sequential(nn.Linear(609,96),nn.GELU(),nn.Linear(96,3))
 def segment(self,h,q,fh,specs):
  z=(q-self.qm)/self.qs;z=torch.cat([z[...,:4]*q[...,8,None],z[...,4:8]*q[...,9,None],q[...,8:]],-1);out=[]
  for i,a,b,k in specs:
   a,b=int(a),int(b);hh=h[i,a:b+1];zz=z[i,a:b+1,k];ff=fh[i,a:b+1]
   out.append(torch.cat([hh.mean(0),hh[0],hh[-1],zz.mean(0),zz[0],zz[-1],zz[-1]-zz[0],ff.mean(0),F.one_hot(torch.tensor(k//2),5).float(),F.one_hot(torch.tensor(k%2),2).float(),torch.tensor([(b-a+1)/100.])]))
  return torch.stack(out) if out else h.new_zeros((0,304))

def example_specs(rows,reverse=False):
 ss=[];yy=[];pairs=[];py=[]
 for i,r in enumerate(rows):
  n=len(r['frame_id']);byaxis={}
  for a in r.get('annotations',[]):
   k=a['axis']*2+(a['moving']=='L');evs=[]
   for e in a['events']:
    lo,hi=e['onset'][0],e['offset'][1]
    if reverse:lo,hi=n-1-hi,n-1-lo
    evs.append((lo,hi));mid=(lo+hi)//2
    ss.append((i,lo,hi,k));yy.append(2)
    # Derived fragments of an observed action, not invented biological phase labels.
    idx=len(ss);ss.extend([(i,lo,mid,k),(i,mid,hi,k)]);yy.extend([1,1]);pairs.append((idx,idx+1,0.));py.append(0)
   evs.sort();byaxis[k]=evs
   for (a0,b0),(a1,b1) in zip(evs,evs[1:]):
    ss.append((i,a0,b1,k));yy.append(3)
    idx=len(ss);ss.extend([(i,a0,b0,k),(i,a1,b1,k)]);yy.extend([2,2]);pairs.append((idx,idx+1,(a1-b0)/100.));py.append(1)
   for lo,hi in a['holds']+a['background']:
    if reverse:lo,hi=n-1-hi,n-1-lo
    ss.append((i,lo,hi,k));yy.append(0)
  flat=sorted((a,b,k) for k,evs in byaxis.items() for a,b in evs)
  for (a,b,k),(c,d,l) in zip(flat,flat[1:]):
   if k!=l and b<=c:
    idx=len(ss);ss.extend([(i,a,b,k),(i,c,d,l)]);yy.extend([2,2]);pairs.append((idx,idx+1,(c-b)/100.));py.append(2)
 return ss,yy,pairs,py

def candidate_spans(bp,q,legacy):
 out=[];seen=set()
 def add(a,b,k,source,complete=False):
  key=(int(a),int(b),int(k))
  if b-a<3 or key in seen:return
  seen.add(key);valid=q[a:b+1,k,-2:].max(-1)>.5
  out.append(dict(a=int(a),b=int(b),axis=k//2,moving='L' if k%2 else 'R',source=source,complete=complete,coverage=float(valid.mean())))
 for e in legacy:add(e['a'],e['b'],e['axis']*2+(e['moving']=='L'),'retained',e.get('complete',False))
 for k in range(10):
  valid=q[:,k,-2:].max(-1)>.5;active=(bp[:,k,0]>=.5)&valid
  starts=bc.groups((bp[:,k,1]>=.5)&valid);ends=bc.groups((bp[:,k,2]>=.5)&valid)
  for a,aa in starts:
   for b,bb in ends:
    if b<aa:continue
    lo=a+int(bp[a:aa+1,k,1].argmax());hi=b+int(bp[b:bb+1,k,2].argmax())
    if hi-lo<3 or active[lo:hi+1].mean()<.6 or valid[lo:hi+1].mean()<.8:continue
    add(lo,hi,k,'boundary_pair',True)
  parts=bc.groups(active)
  for a,b in parts:add(a,b,k,'active_fragment')
  for (a,b),(c,d) in zip(parts,parts[1:]):
   if valid[a:d+1].mean()>=.8:add(a,d,k,'fragment_union')
 return out

def infer(net,x,ts,fh,legacy):
 g=old.old.geometry(x,ts);q=old.relation_features(x)
 with torch.inference_mode():
  tq=torch.tensor(q[None]);tf=torch.tensor(fh[None]);rp,bp,_,h=net(torch.tensor(g[None]),tq,tf,torch.tensor([len(x)]));r=rp.softmax(-1)[0].numpy();b=bp.sigmoid()[0].numpy()
  cand=candidate_spans(b,q,legacy);spec=[(0,e['a'],e['b'],e['axis']*2+(e['moving']=='L')) for e in cand];feat=net.segment(h,tq,tf,spec);qual=net.quality(feat).softmax(-1).numpy() if len(spec) else []
  for e,p in zip(cand,qual):
   e.update(quality=p.tolist(),quality_class=int(p.argmax()),quality_confidence=float(p.max()));pp=r[e['a']:e['b']+1].mean(0);e.update(family=int(pp.argmax()),family_confidence=float(pp.max()))
  # Pair decisions supplement interval quality. Runtime receives no expected word,
  # desired direction, edit point or reference-frame ground truth.
  ps=[];pi=[]
  for i,e in enumerate(cand):
   for j,f in enumerate(cand):
    if i==j or e['axis']!=f['axis'] or e['moving']!=f['moving'] or e['b']>f['a']:continue
    if f['a']-e['b']>max(e['b']-e['a']+1,f['b']-f['a']+1):continue
    ps.append(torch.cat([feat[i],feat[j],torch.tensor([(f['a']-e['b'])/100.])])) ;pi.append((i,j))
  pp=net.pair(torch.stack(ps)).softmax(-1).numpy() if ps else [];links=[dict(first=i,second=j,probabilities=p.tolist(),label=int(p.argmax()),confidence=float(p.max())) for (i,j),p in zip(pi,pp)]
 accepted=[]
 for e in cand:
  k=e['axis']*2+(e['moving']=='L');a,bb=e['a'],e['b'];ends=b[a:min(a+4,bb+1),k,1].max()>=.5 and b[max(a,bb-3):bb+1,k,2].max()>=.5
  if e['quality_class']==2 and e['quality_confidence']>=.65 and e['coverage']>=.8 and ends:accepted.append(dict(e,complete=True))
 # Keep multiple hypotheses auditable. Non-overlap decoding never reuses one
 # event as two repetitions; unaccepted old candidates remain in the ledger.
 selected=[]
 for e in sorted(accepted,key=lambda v:(-v['quality'][2],v['b']-v['a'])):
  if any(e['axis']==f['axis'] and e['moving']==f['moving'] and max(e['a'],f['a'])<min(e['b'],f['b']) for f in selected):continue
  # A confidently different repetition within a union blocks joining it.
  blocked=any(l['label']!=0 and l['confidence']>=.65 and cand[l['first']]['a']>=e['a'] and cand[l['second']]['b']<=e['b'] for l in links)
  if not blocked:selected.append(e)
 selected.sort(key=lambda e:(e['a'],e['b'],e['axis']))
 return dict(rp=r,bp=b,q=q,candidates=cand,selected=selected,links=links)

def legacy_role(before):
 p=np.asarray(before['guide_link_probabilities']);out=np.zeros((len(p),8),np.float32)
 for k in range(p.shape[1]):out[:,ep.MAP[k]]+=p[:,k]
 return out

def supplement(before,obs,x,ts,fh,base,contracts,mode):
 out=copy.deepcopy(before);rp=legacy_role(before) if mode=='interval_only' else obs['rp'];ss=integration.role_spans(rp,obs['q'],ts)
 # Preserve baseline evidence; new interval effects do not silently replace it.
 events=before['phase_events']+before['phase_partials']
 if mode!='role_only':events=events+obs['selected']
 binds=[];cache={}
 for c,spec in zip(out['cards'],contracts):
  matches=[s for s in ss if s['family'] in spec['families']];cc=integration.conflict(matches,before['phase_probability']);bound=[];seen=set()
  if spec['axis'] is not None:
   for e in events:
    a,b=e['a'],e['b'];key=(a,b,spec['axis'])
    if e['axis']!=spec['axis'] or e['moving']!='R' or b-a<3 or key in seen:continue
    overlap=len({i for s in matches for i in range(max(a,s['a']),min(b,s['b'])+1)})
    if overlap/(b-a+1)<.25:continue
    seen.add(key)
    if key not in cache:cache[key]=integration.direction(base,x,ts,fh,a,b,spec['axis'])
    bound.append(dict(cache[key],complete=e.get('complete',False),source=e.get('source','retained')))
  signs={e['prediction']-1 for e in bound if e['eligible'] and e['confidence']>=.65 and e['prediction']!=1};expected=spec['sign'];state='not_specified' if expected is None else 'unknown'
  if expected is not None and len(signs)==1:state='supported' if signs=={expected} else 'contradicted'
  elif expected is not None and len(signs)>1:state='mixed'
  blocked=any(v['conflict'] for v in cc);bind=dict(card=c['id'],spans=matches,movements=bound,state=state,conflict=blocked,conflict_checks=cc);binds.append(bind);c['joint_binding']=bind
  if state=='contradicted' and not blocked and not c.get('observation_quarantine_overlap'):integration.add_warning(c,'joint_direction','표시된 움직임이 참고와 반대 방향으로 보여요. 이동 방향을 확인해 보세요.')
 order=[];required=[i for i,s in enumerate(contracts) if s['required'] and not s.get('parent')]
 for ia,ib in zip(required,required[1:]):
  a,b=binds[ia],binds[ib];state=lc.order_state(a['spans'],b['spans']);blocked=a['conflict'] or b['conflict'];order.append(dict(pair=[a['card'],b['card']],state=state,conflict=blocked))
  if state=='contradicted' and not blocked and not any(out['cards'][i].get('observation_quarantine_overlap') for i in [ia,ib]):integration.add_warning(out['cards'][ib],'joint_order','표시된 행동의 순서가 참고와 반대로 보여요. 먼저 할 행동부터 이어 보세요.')
 # No complete trajectory => completion unconfirmed. Do not label it an actual
 # omission or erase existing attempts simply because the guide expects one.
 out.update(joint_mode=mode,joint_bindings=binds,joint_order_checks=order,joint_completion=[dict(card=b['card'],complete_observed=any(e['complete'] and e['eligible'] for e in b['movements']),unconfirmed_is_not_omission=True) for b in binds])
 return out
