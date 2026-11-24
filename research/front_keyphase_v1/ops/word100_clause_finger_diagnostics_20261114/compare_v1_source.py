from pathlib import Path
import sys,json,time,collections
import numpy as np,torch
from motion_core import SymmetricMotion,reverse_geometry
from finger_core import FingerHead,local_sequence,windows,FINGERS
U=Path(__file__).resolve().parent;O=U.parent
sys.path.insert(0,str(O/'word100_guide_requirement_learning_20261111'))
import requirement_core as rc
torch.set_num_threads(2);old=rc.frozen_observer();contract=rc.read(U/'CLAUSE_CONTRACT.json');cm={(r['word'],r['action_id']):r for r in contract['actions']}
models={};fmodels={}
for seed in [42,43,44]:
 m=SymmetricMotion(old);m.load_state_dict(torch.load(U/'motion_models'/f'{seed}.pt',weights_only=True));m.eval();models[seed]=m
 ck=torch.load(U/'finger_models'/f'{seed}.pt',weights_only=True);fm=FingerHead();fm.load_state_dict(ck['state_dict']);fm.eval();fmodels[seed]=(fm,np.array(ck['mean'],np.float32),np.array(ck['std'],np.float32))
baseline=rc.read(O/'word100_joint_completion_block_20261114/JOINT_FINAL_V2.json')['rows'];pm={(r['clip_id'],r['kind']):r for r in rc.read(O/'word100_web_completion_graph_20261111/typed_motion_recovery_20261113/null_candidate_learning/evaluation/COMPARISON.json')['predictions']if r['seed']==42};cache={};out=[];times=[]
for row in baseline:
 sid=row['clip_id'],row['kind'];seed=row['seed'];word=row['clip_id'].split('_')[2]
 if sid not in cache:
  r=pm[sid];assert rc.sha(r['source'])==r['source_sha256'];z=np.load(r['source']);x=z['x'];ts=z['ts']
  t=time.perf_counter()
  with torch.no_grad():h=old.shared(torch.tensor(rc.prior.shared.normalize(x)[None]),torch.tensor([len(x)]),torch.tensor([0]))['latent'][0].numpy()
  hands={}
  for side in ['L','R']:
   f,k=local_sequence(x,side);w=windows(np.c_[f,h],np.arange(len(x)));pred={}
   for sd,(fm,mu,sigma)in fmodels.items():
    with torch.no_grad():pred[sd]=fm(torch.tensor((w-mu)/sigma)).sigmoid().numpy()
   hands[side]=(k,pred)
  cache[sid]=(x,ts,h,hands);times.append(dict(clip_id=sid[0],kind=sid[1],seconds=time.perf_counter()-t,scope='common latent plus all3 finger candidates both hands; excludes pose extraction, camera and UI'))
 x,ts,h,hands=cache[sid];cards=[]
 for c in row['joint_cards']:
  spec=cm[word,c['card']];oldshape=[e for e in c['observations']if e['task']=='shape'];ds=[e for e in c['observations']if e['task']=='direction'];newd=[];newf=[]
  for e in ds:
   if 'span'not in e:continue
   a,b=e['span'];fs=next(f for f in rc.FAMILIES[word]if f['id']==c['card']);g,ok,info=rc.prior.geometry(x,ts,a,b,e['axis'],fs['hands'][0],'L'if fs['hands'][0]=='R'else'R')
   with torch.no_grad():p=models[seed](torch.tensor(g[None])).softmax(-1)[0].numpy()
   newd.append(dict(span=e['span'],axis=e['axis'],desired=e['desired'],before=e['state'],after=rc.direction_verdict(p,e['desired'],ok),probabilities=p.tolist(),signed_change=info.get('signed_change'),available=bool(ok)))
  for e in oldshape:
   if 'span'not in e:continue
   a,b=e['span']
   for req in spec['finger_requirements']:
    if req['source_feature']!=e['feature']:continue
    side=req['hand'];j=FINGERS.index(req['finger']);known,pr=hands[side];v=known[a:b+1,j];valid=bool(v.mean()>=.8);p=float(pr[seed][a:b+1,j][v].mean())if v.any()else None;expected=int(req['expected']=='extended')
    state='observation_unknown'if not valid or p is None else'uncertain'if .15<p<.85 else'supported'if int(p>=.85)==expected else'difference'
    newf.append(dict(hand=side,finger=req['finger'],expected=req['expected'],span=[a,b],probability=p,state=state,coarse_before=e['state'],association=next(q['state']for q in row['original_association']['cards']if q['card']==c['card'])))
  cards.append(dict(card=c['card'],before_state=c['state'],direction=newd,fingers=newf,order_evidence=c.get('order_evidence',[]),original_candidate=next(q for q in row['original_candidates']if q['card']==c['card'])))
 out.append(dict(seed=seed,clip_id=sid[0],kind=sid[1],cards=cards,source=pm[sid]['source'],source_sha256=pm[sid]['source_sha256']))
normal={'full_normal','speed075','speed125','dynamic_complete_blocks'}
summary=[]
for seed in [42,43,44]:
 rr=[r for r in out if r['seed']==seed];newwarnings=[];lost=[];errgains=[];errlost=[]
 for r in rr:
  for c in r['cards']:
   for d in c['direction']:
    item=dict(clip_id=r['clip_id'],kind=r['kind'],card=c['card'],**d)
    if r['kind']in normal:
     if d['after']=='opposite_direction'and d['before']!='opposite_direction':newwarnings.append(item)
     if d['before']=='supported'and d['after']!='supported':lost.append(item)
    elif r['kind']in ['main_reversed','closed_hand_and_reversed','reversed_core_auto_order']:
     if d['before']!='opposite_direction'and d['after']=='opposite_direction':errgains.append(item)
     if d['before']=='opposite_direction'and d['after']!='opposite_direction':errlost.append(item)
 combo=[r for r in rr if r['kind']=='closed_hand_and_reversed'];both=sum(any(f['state']=='difference'for c in r['cards']for f in c['fingers'])and any(d['after']=='opposite_direction'for c in r['cards']for d in c['direction'])for r in combo)
 s=dict(seed=seed,scenarios=len(rr),new_normal_direction_warnings=newwarnings,normal_direction_losses=lost,reverse_gains=errgains,reverse_losses=errlost,combined_both_detected=int(both),combined_sources=len(combo));summary.append(s)
 print(json.dumps(dict(seed=seed,normal_new_warning=len(newwarnings),normal_lost=len(lost),error_gained=len(errgains),error_lost=len(errlost),both=int(both)),ensure_ascii=False),flush=True)
rc.write(U/'AUTOMATIC_COMPARISON.json',dict(rows=out,summary=summary,timings=times,automatic_candidates_unchanged=True,independent_order_evidence_unchanged=True,unchanged_sources=len(pm),new_real_errors=False))
# Component oracle comparison, all known public DEV intervals, not selected successes.
rr=rc.public_rows();review=[]
for r in rr:
 if r['role']!='development':continue
 with torch.no_grad():h=old.shared(torch.tensor(r['x_full'][None]),torch.tensor([len(r['x_full'])]),torch.tensor([r['word']]))['latent']
 for pi,pair in enumerate(r['pairs']):
  for rev in [False,True]:
   a,b=pair['a'],pair['b'];g,ok,inf=rc.prior.geometry(r['raw_x'],r['timestamp_us'],a,b,pair['axis'],pair.get('moving','R'),pair.get('anchor','L'),rev);tar=2-pair['target']if rev else pair['target'];gg=torch.tensor(g[None])
   with torch.no_grad():op=old.movement(h,gg,torch.tensor([[0,b,a]if rev else[0,a,b]])).softmax(-1)[0].numpy();pp={s:m(gg).softmax(-1)[0].numpy()for s,m in models.items()}
   review.append(dict(clip_id=r['clip_id'],axis=pair['axis'],pair=pi,reverse=rev,target=tar,available=bool(ok),baseline_pred=int(op.argmax()),baseline_state=rc.direction_verdict(op,tar,ok),candidate={str(s):dict(pred=int(p.argmax()),state=rc.direction_verdict(p,tar,ok))for s,p in pp.items()}))
rc.write(U/'KNOWN_INTERVAL_COMPARISON.json',dict(rows=review,baseline_mode='For historical supplied-interval reverse diagnostic, reverse observed geometry and endpoints; auto-input comparison is reported separately.'))
