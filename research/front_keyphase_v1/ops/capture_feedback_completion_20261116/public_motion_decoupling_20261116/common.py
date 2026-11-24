from pathlib import Path
import sys,json,hashlib,time,datetime
E=Path(__file__).resolve().parent;G=E.parent;O=G.parent;D=G/'public_shape_endpoint_learning_20261116';H=G/'public_shape_arrival_repair_20261116';V=G/'completion_v1';B=O/'basic_demo_feedback_20261115'
sys.path.insert(0,str(D))
from form_v5 import QualifiedFormLesson
from form_candidate import np,torch,adapter
from core import PhaseNet,states
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sources():return read(H/'PUBLIC_INPUTS.json')['rows']+read(H/'public_camera/INPUTS.json')['rows']+read(H/'public_views/INPUTS.json')['rows']+read(H/'confirmation_views/INPUTS.json')['rows']+read(D/'confirmation/INPUTS.json')['rows']+read(D/'final_confirmation/INPUTS.json')['rows']
def channels(raw):return adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
def geometry(f):
 # Original frozen Features layout: 17 body-relative points, availability,
 # detailed finger features, frozen temporal embedding. Only these eight
 # shoulder/elbow/wrist/root positions are used for posture classification.
 ids=[1,2,3,4,5,6,7,12];loc=f[:,:34].reshape(-1,17,2)[:,ids];av=f[:,34:51][:,ids]>.5
 pairs=np.concatenate([loc[:,6]-loc[:,7],loc[:,6]-loc[:,2],loc[:,7]-loc[:,3],loc[:,4]-loc[:,5]],axis=1)
 distance=np.linalg.norm(loc[:,6]-loc[:,7],axis=1,keepdims=True)
 x=np.c_[loc.reshape(len(f),-1),av,pairs,distance].astype(np.float32)
 good=av.all(1)&np.isfinite(x).all(1)
 return np.nan_to_num(x),good
class GeometryRelation:
 def __init__(self):
  self.models=[]
  for seed in[42,43,44]:
   c=torch.load(E/'relation'/f'{seed}.pt',map_location='cpu',weights_only=False);m=PhaseNet(len(c['mean']),'frame_mlp');m.load_state_dict(c['state']);m.eval();self.models.append((m,c['mean'],c['std']))
 def __call__(self,f,quality,ts):
  x,good=geometry(f);ss=[];pp=[]
  with torch.inference_mode():
   for m,mu,sd in self.models:
    p=m(torch.tensor(((x-mu)/sd)[None])).softmax(-1)[0].numpy();pp.append(p);ss.append(states(p.copy(),good))
  ss=np.array(ss);return np.where((ss==ss[0]).all(0),ss[0],-1),np.array(pp)
class MotionLesson(QualifiedFormLesson):
 def __init__(self):super().__init__();self.relation=GeometryRelation()
