"""Direct cached SignRep video evidence, added without replacing prior heads."""
from pathlib import Path
import sys,numpy as np,torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
R=Path(__file__).resolve().parent;O=R.parent;Q=O/'word100_learned_motion_path_20261102';N=O/'word100_learned_motion_invariance_20261102';M=O/'word100_learned_motion_order_20261102';P=O/'word100_learned_guide_feedback_20261102'
sys.path.insert(0,str(N));sys.path.insert(0,str(Q));sys.path.insert(0,str(M))
from path_model import RelativeBranch,relative_features,PathNet,records_and_sequences,batch,predict,read

def rgb_input(a,stats):
 rgb=np.asarray(a['rgb']);available=np.asarray(a.get('image_available',np.ones(len(rgb),bool)),bool)&np.isfinite(rgb).all(1)
 values=np.where(available[:,None],(rgb-np.asarray(stats['mean']))/np.asarray(stats['std']),0)
 return np.concatenate([values,available[:,None]],axis=1).astype('float32')

class RGBBranch(RelativeBranch):
 def __init__(self):
  super().__init__();self.rgb=nn.Linear(768,32,bias=False);nn.init.zeros_(self.rgb.weight)
 def forward(self,x,lengths,video):
  h=self.embed[1](self.embed[0](x)+self.relative(relative_features(x,lengths))+self.rgb(video))
  _,last=self.temporal(pack_padded_sequence(h,lengths.cpu(),batch_first=True,enforce_sorted=False))
  return self.out(torch.cat([last[0],last[1]],dim=-1))

class RGBMotionNet(nn.Module):
 def __init__(self,condition):
  super().__init__();assert condition in ['no_rgb_replay','rgb_residual'];self.condition=condition
  self.direction=RGBBranch();self.continuity=RGBBranch()
  keep=read(N/'TRAINING_CONTRACT.json')['motion_value_indices'];v=torch.zeros(146);v[keep]=1;v[np.array(keep)+73]=1;self.register_buffer('valid',v)
  if condition=='no_rgb_replay':
   self.direction.rgb.requires_grad_(False);self.continuity.rgb.requires_grad_(False)
 def initialize(self,seed):
  s=torch.load(Q/'runs'/f'separated_relative_{seed}/model.pt',map_location='cpu',weights_only=True);result=self.load_state_dict(s,strict=False)
  assert set(result.missing_keys)=={'direction.rgb.weight','continuity.rgb.weight'} and not result.unexpected_keys
 def forward(self,x,lengths):
  assert x.shape[-1]==915
  geometry=x[:,:,:146]*self.valid
  temporal=torch.arange(x.shape[1],device=x.device)[None,:]<lengths[:,None]
  available=(x[:,:,914]>.5)&temporal
  video=torch.where(available[:,:,None],x[:,:,146:914],0)
  d=self.direction(geometry,lengths,video);c=self.continuity(geometry,lengths,video)
  return torch.cat([d[:,:5],c[:,5:6]],dim=-1)

def training_data():
 seq,y,mask,records,groups=records_and_sequences();stats=read(R/'RGB_NORMALIZATION.json')
 sources={r['key']:r for r in read(R/'INPUT_READY.json')['rows']};cache={};out=[]
 for r,x in zip(records,seq):
  key=r['source_key'];assert sources[key]['person'] in range(1,10)
  if key not in cache:cache[key]=rgb_input(dict(np.load(sources[key]['input'])),stats)
  video=cache[key][r['frame_ids']]
  if r['synthetic_feature_indices'] is not None:video=video[r['synthetic_feature_indices']]
  assert len(video)==len(x);out.append(np.concatenate([x,video],axis=1))
 return out,y,mask,records,groups

