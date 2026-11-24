from pathlib import Path
import sys,json,time,hashlib,importlib.util
import numpy as np,torch
U=Path(__file__).resolve().parent;O=U.parent
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item(),allow_nan=False)+'\n','utf-8')
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
for p,h in read(U/'STATIC_LINK_FROZEN.json')['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
sys.path.insert(0,str(U));sys.path.insert(0,str(O/'word100_live_g6_20261114'))
base=load('game_acceptance_base',O/'word100_live_g6_20261114/engine.py');new=load('game_acceptance_candidate',U/'engine_static_candidate.py')
torch.set_num_threads(2);models=base.initialize();new._supplement=base._supplement
labels=read(O/'word100_public_g4_20261114/LABELS_FROZEN.json')['actions'];rows=[]
for r in read(O/'word100_public_g4_20261114/EXTRACTED.json')['rows']:
 cid=r['clip_id'];word=cid.split('_')[2];cz=np.load(r['channel_path']);rz=np.load(r['raw_path']);N=len(cz['tcn280']);cases=['normal','continuous_full_reverse','all_hands_missing']
 if word=='WORD0739':cases+=['prefix_before_goal']
 for kind in cases:
  n=next(a['partial_end']for a in labels if a['clip_id']==cid and a['card']=='close_chest')+1 if kind=='prefix_before_goal'else N
  ids=np.arange(n)[::-1].copy()if kind=='continuous_full_reverse'else np.arange(n)
  ch={k:v[ids].copy()if v.ndim and len(v)==N else v.copy()for k,v in cz.items()};raw={k:v[ids].copy()if v.ndim and len(v)==N else v.copy()for k,v in rz.items()}
  ch['timestamp_us']=cz['timestamp_us'][:n];ch['frame_id']=np.arange(n);raw['timestamp_us']=ch['timestamp_us'];raw['frame_id']=ch['frame_id']
  if kind=='all_hands_missing':ch['tcn280'][:,18:186]=0;ch['tcn280'][:,195:279]=0;ch['point_available'][:,9:]=False
  before=base.infer(models,ch,raw,word);after=new.infer(models,ch,raw,word)
  assert before['original_candidates']==after['original_candidates'];assert before['original_orders']==after['original_orders']
  changes=[];warnings=[]
  for b,a in zip(before['joint_cards'],after['joint_cards']):
   if b['state']!=a['state']or b['observations']!=a['observations']:changes.append(dict(card=a['card'],before=b['state'],after=a['state'],observations=a['observations']))
   if kind=='normal'and a['state']=='attributable_feature_difference'and b['state']!=a['state']:warnings.append(a['card'])
   if b['state']=='attributable_feature_difference':assert a['state']==b['state']
  path=U/'static_replay'/f'{cid}__{kind}.json';path.parent.mkdir(exist_ok=True);write(path,dict(before=before,after=after))
  rows.append(dict(clip_id=cid,kind=kind,changes=changes,new_normal_warnings=warnings,output=str(path),preserved_candidates_and_order=True))
 print(cid,'checked',flush=True)
write(U/'STATIC_LINK_RESULTS.json',dict(cases=len(rows),changed_cases=sum(bool(r['changes'])for r in rows),new_normal_warnings=sum(len(r['new_normal_warnings'])for r in rows),rows=rows,scope='Paired fixed-model public replay. Stationary observed features only, not full lesson completion.'))
print('static-link results',len(rows),'cases')
