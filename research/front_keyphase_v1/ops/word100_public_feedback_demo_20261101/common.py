from pathlib import Path
import hashlib,json
import numpy as np
U=Path(__file__).resolve().parent; O=U.parent
S=O/'word100_semantic_feature_cycle_20261101'
B=O/'word100_stage_feedback_cycle_20261031'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
 with (U/n).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def pins():
 for p,h in read(U/'FROZEN.json')['pins'].items():
  assert sha(p)==h,p
def support(mask,names,target):
 if target.endswith('higher'):
  cols=[names.index(s+'_palm_center_'+d) for s in ['L','R'] for d in ['x','y']]
  return mask[:,cols].all(1)
 s=target[0];w=[names.index(s+'_hand_wrist_'+d) for d in ['x','y']]
 a=[i for i,n in enumerate(names) if n.startswith(s+'_') and '_angle' in n and 'elbow' not in n]
 assert len(a)==14
 return mask[:,w].all(1)&(mask[:,a].sum(1)>=10)
def normalize(a,stats):
 x=np.where(a['mask'],(a['x']-np.asarray(stats['pose_mean']))/np.asarray(stats['pose_std']),0)
 x=np.concatenate([x,a['mask'].astype(np.float32)],axis=-1).astype(np.float32)
 z=((a['rgb']-np.asarray(stats['rgb_mean']))/np.asarray(stats['rgb_std'])).astype(np.float32)
 return x,z
