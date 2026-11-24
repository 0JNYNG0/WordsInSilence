from prepare import U,O,read,sha,write,load,Path
import sys,time
import numpy as np,torch
T=O/'word100_terminal_relation_targets_20261114';B=O/'word100_action_evidence_block_20261114';J=O/'word100_joint_completion_block_20261114'
sys.path[:0]=[str(J),str(B)]
from region_extent import RegionExtent,features as region_features
from body_core import face_geometry
from sequence_core import features as sequence_features,decide
from evidence_path import summarize
from assemble_evidence import qualify_path
from joint_core import joint_sequence
old=load('g4oldtraits',T/'core.py');new=load('g4statepath',B/'state_path/core.py')
class Traits:
 def __init__(self):
  self.old=[];self.reg=[];self.seq=[]
  for s,r in zip([42,43,44],read(T/'TRAINING_COMPLETE.json')['rows']):
   a=old.TraitHeads();a.load_state_dict(torch.load(r['model'],weights_only=True,map_location='cpu'));a.eval();self.old.append(a)
   a=RegionExtent();a.load_state_dict(torch.load(J/'extent_augmented'/f'{s}.pt',weights_only=True,map_location='cpu'));a.eval();self.reg.append(a)
   a=new.StatePath();a.load_state_dict(torch.load(B/'state_path/models'/f'{s}.pt',weights_only=True,map_location='cpu'));a.eval();self.seq.append(a)
 def infer(self,x,face,faceok,h):
  out={}
  with torch.inference_mode():
   for d in ['shape','region']:
    f,v=old.features(x,d,h);p=np.stack([m(torch.tensor(f),d).softmax(-1).numpy()for m in self.old]);out[d+'_old']=dict(states=old.consensus(p,v,d),prob=p,available=v)
   f,v=region_features(x,face,faceok,h);p=np.stack([m(torch.tensor(f)).softmax(-1).numpy()for m in self.reg]);out['region_new']=dict(states=old.consensus(p,v,'region'),prob=p,available=v)
   f,v=sequence_features(x,h);ctx=np.zeros((len(x),7,59),np.float32)
   for i in range(len(x)):
    for j,k in enumerate(range(i-3,i+4)):
     if 0<=k<len(x):ctx[i,j]=f[k]
   out['positions']=np.stack([m.position(torch.tensor(ctx)).numpy()for m in self.seq]);out['available']=v&out['shape_old']['available']
  return out
 def path(self,d,ts,fid,a,b,association=True,mask=None):
  pos=d['positions'][:,a:b+1];v=d['available'][a:b+1].copy();states=d['shape_old']['states'][a:b+1]
  if mask is not None:v&=mask
  pp=[]
  for i,m in enumerate(self.seq):
   delta=float(pos[i,-1]-pos[i,0]);margin=float(torch.sigmoid(m.margin_raw).detach());scale=float(torch.nn.functional.softplus(m.scale_raw).detach());pp.append(torch.tensor([-delta,margin,delta]).mul(scale).softmax(-1).numpy())
  change=decide(np.stack(pp),bool(v[0]and v[-1]and v.mean()>=.8));ev=summarize(change,pos,states,v,ts[a:b+1],association)
  q=qualify_path(ev);sh=dict(ev,content_status=q['content_status'])
  reg=d['region_new'];av=v&reg['available'][a:b+1]&np.array([s is not None for s in states])
  joint=joint_sequence(sh,reg['states'][a:b+1],av.tolist(),ts[a:b+1].tolist(),fid[a:b+1].tolist(),association)
  req=dict(start=dict(shape='four_open',region='face_front'),goal=dict(shape='four_closed',region='upper_body'))
  base=old.assess(dict(shape=states,region=d['region_old']['states'][a:b+1]),dict(shape=v,region=v&d['region_old']['available'][a:b+1]),ts[a:b+1],req)
  return dict(shape_path=q,joint=joint,baseline=base)
def main():
 torch.set_num_threads(2);assert not(U/'TRAIT_RESULTS_V2.json').exists();engine=Traits();out=[];dense={};timing=[]
 cached=np.load(U/'TRAIT_DENSE.npz')if(U/'TRAIT_DENSE.npz').exists()else None
 labels=read(U/'LABELS_FROZEN.json');extract=read(U/'EXTRACTED.json')
 write(U/'TRAIT_RECOVERY_FROZEN.json',dict(pins={str(p):sha(p)for p in [Path(__file__),U/'LABELS_FROZEN.json',U/'EXTRACTED.json']},models_fixed=True,confidence='.8 and all3 agree, unchanged',evaluation_only=True,recovery='Initial model path and import errors occurred before inference. NumPy bool JSON serialization failed after dense8-source save; reuse all saved normal outputs, recompute only2 unsaved reverse controls. Partial JSON preserved.'))
 for r in extract['rows']:
  cid=r['clip_id'];z=np.load(r['channel_path']);raw=np.load(r['raw_path']);x=z['tcn280'];ts=z['timestamp_us'];fid=z['frame_id'];face,ok=face_geometry(raw['xy_px'],z['point_available']);st=time.perf_counter();dd={}
  for h in ['L','R']:
   if cached is None:d=engine.infer(x,face,ok,h)
   else:
    d={};prefix=cid+'__'+h+'__'
    for key in ['shape_old','region_old','region_new']:
     p=cached[prefix+key+'__prob'];v=cached[prefix+key+'__available'];d[key]=dict(prob=p,available=v,states=old.consensus(p,v,key.split('_')[0]))
    d['positions']=cached[prefix+'positions'];d['available']=cached[prefix+'available']
   dd[h]=d
   for key in ['shape_old','region_old','region_new']:
    dense[cid+'__'+h+'__'+key+'__prob']=d[key]['prob'];dense[cid+'__'+h+'__'+key+'__available']=d[key]['available']
   dense[cid+'__'+h+'__positions']=d['positions'];dense[cid+'__'+h+'__available']=d['available']
  points=[]
  for l in labels['rows']:
   if l['clip_id']!=cid:continue
   f=l['frame'];d=dd[l['hand']]
   points.append(dict(**l,predictions={k:d[k]['states'][f]for k in ['shape_old','region_old','region_new']}))
  controls=[]
  for act in labels['actions']:
   if act['clip_id']!=cid or act['card']!='close_chest':continue
   a,b=act['span']
   for kind,aa,bb,rate in [('normal',a,b,1),('speed075',a,b,.75),('speed125',a,b,1.25),('prefix',a,act['partial_end'],1),('goal_only',b,b,1)]:
    result=engine.path(dd['R'],np.round(ts/rate).astype(np.int64),fid,aa,bb);controls.append(dict(kind=kind,span=[aa,bb],**result))
   controls.append(dict(kind='all_missing',span=[a,b],**engine.path(dd['R'],ts,fid,a,b,mask=np.zeros(b-a+1,bool))))
   # Reversed source trajectory re-inferred once; time still increases.
   ids=np.arange(a,b+1)[::-1];rx=x[ids].copy();rx[:,279]=np.r_[0,np.diff(ts[a:b+1])/1e6];rd=engine.infer(rx,face[ids],ok[ids],'R')
   controls.append(dict(kind='reverse',span=[a,b],**engine.path(rd,ts[a:b+1]-ts[a],fid[ids],0,len(ids)-1)))
  out.append(dict(clip_id=cid,points=points,controls=controls));timing.append(dict(clip_id=cid,seconds=time.perf_counter()-st))
 if cached is None:np.savez_compressed(U/'TRAIT_DENSE.npz',**dense)
 write(U/'TRAIT_RESULTS_V2.json',dict(rows=out,timing=timing,scope='Supplied witness intervals; NOT automatic localization; eight public sources; actual normal RGB + diagnostic edits',timing_scope='Recovery assembly from normal cached outputs plus2 unsaved reverse inferences; not end-to-end latency',no_training=True))
 print('trait evaluation saved',len(out),'sources',flush=True)
if __name__=='__main__':main()
