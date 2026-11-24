from pathlib import Path
import json,numpy as np,torch
from torch import nn
V=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
class RelationNet(nn.Module):
 def __init__(self,use_rgb=True):
  super().__init__();self.use_rgb=use_rgb
  self.pose=nn.Linear(138,32);self.rgb=nn.Linear(768,32)
  self.temporal=nn.Sequential(nn.Conv1d(64,64,3,padding=1),nn.GELU(),nn.Conv1d(64,32,3,padding=2,dilation=2),nn.GELU())
  self.out=nn.Linear(32,3)
 def forward(self,p,z):
  a=torch.nn.functional.gelu(self.pose(p));b=torch.nn.functional.gelu(self.rgb(z)) if self.use_rgb else torch.zeros_like(a)
  h=self.temporal(torch.cat([a,b],-1).transpose(1,2)).transpose(1,2)
  return self.out(h[:,h.shape[1]//2])
def normalized(a,stats):
 x=np.asarray(a['x']);m=np.asarray(a['mask'],bool);z=np.asarray(a['rgb'])
 p=np.where(m,(x-np.array(stats['pm']))/np.array(stats['ps']),0)
 z=(z-np.array(stats['zm']))/np.array(stats['zs'])
 return np.concatenate([p,m],1).astype('float32'),z.astype('float32')
def dense(model,a,stats):
 p,z=normalized(a,stats);n=len(p);result=np.full((n,3),np.nan,np.float32)
 ids=np.asarray(a['frame_id']);centers=[i for i in range(6,n-6) if np.array_equal(ids[i-6:i+7],np.arange(ids[i]-6,ids[i]+7))]
 with torch.inference_mode():
  for b in range(0,len(centers),64):
   ix=np.array(centers[b:b+64])[:,None]+np.arange(-6,7)
   result[np.array(centers[b:b+64])]=model(torch.from_numpy(p[ix]),torch.from_numpy(z[ix])).sigmoid().numpy()
 return result
