from pathlib import Path
import sys,json,hashlib,collections
import numpy as np,torch
from finger_core import local_sequence,windows,baseline_fingers,FINGERS
U=Path(__file__).resolve().parent;O=U.parent
sys.path.insert(0,str(O/'word100_guide_requirement_learning_20261111'))
import requirement_core as rc
read=rc.read;write=rc.write
assert not(U/'FINGER_FROZEN.json').exists()
torch.set_num_threads(2);net=rc.frozen_observer();T=O/'word100_handshape_transfer_20261113';A=O/'demo_handshape_learning_20261113';RT=O/'word100_rtmw_joint_learning_20261108'
# Direct visual review, before new candidate predictions. -1 means not resolved.
fit={i:[-1,1,0,0,0]for i in list(range(9))+[10,11,14,16,17,18,19,20]}
fit.update({9:[-1]*5,12:[-1]*5,13:[-1]*5,15:[-1,1,-1,-1,-1],21:[-1,-1,0,0,0],22:[-1,-1,0,0,0],23:[-1,-1,0,0,0],24:[-1,0,0,0,0],25:[-1,0,0,0,0],26:[-1,0,0,0,0]})
dev={i:[-1,1,0,0,0]for i in[1,3,4,5,6,7,14,17,21]}
dev.update({0:[-1,-1,0,0,0],2:[-1,-1,0,0,0],8:[-1]*5,9:[-1]*5,10:[-1]*5,11:[-1,0,0,0,0],12:[-1]*5,13:[-1,1,-1,-1,-1],15:[-1,-1,0,0,0],16:[-1,-1,0,0,0],18:[-1,-1,0,0,0],19:[-1,-1,0,0,0],20:[-1]*5,22:[-1,0,0,0,0]})
review=[];direct={}
for name,labels in [('RGB_REVIEW_INPUTS.json',fit),('dev_RGB_REVIEW_INPUTS.json',dev)]:
 for r in read(U/name)['rows']:
  y=labels[r['index']];assert rc.sha(r['image'])==r['image_sha256'];key=r['clip_id'],r['frame'],r['side'];direct[key]=y
  review.append({**r,'finger_states':y,'reviewer':'Codex source RGB crop visual review, not independent expert','definition':'1 visibly extended; 0 visibly flexed (not necessarily touching palm); -1 hidden/ambiguous; no thumb imputation'})
write(U/'FINGER_RGB_LABELS.json',dict(rows=review,review_before_training=True,fit_frames=len(fit),development_frames=len(dev)))
manifest=read(T/'INPUT_MANIFEST.json');rows=[];seq=[];knowns=[];baselines=[];pins={};cache={}
def get(cid,aux=False):
 if cid in cache:return cache[cid]
 if aux:
  r=next(r for r in read(A/'INPUTS.json')['rows']if r['clip_id']==cid);p=Path(r['channels']);assert rc.sha(p)==r['channels_sha256'];z=np.load(p);x=z['tcn280'];fid=z['frame_id']
 else:
  r=read(RT/'records'/f'{cid}.json');p=Path(r['channel_path']);assert rc.sha(p)==r['channel_sha256'];z=np.load(p);x=z['feedback']if'feedback'in z.files else z['full'];fid=z['frame_id']
 pins[str(p)]=rc.sha(p)
 with torch.inference_mode():out=net.shared(torch.tensor(rc.prior.shared.normalize(x)[None]),torch.tensor([len(x)]),torch.tensor([0]));h=out['latent'][0].numpy();pr=out['attributes'][0].sigmoid().numpy()
 cache[cid]=(x,fid,h,pr);return cache[cid]
def add(r,y,origins,aux=False):
 x,fid,h,pr=get(r['clip_id'],aux);i=int(np.flatnonzero(fid==r['frame'])[0]);side=r.get('side','R');f,k=local_sequence(x,side);seq.append(windows(np.c_[f,h],[i])[0]);knowns.append(k[i]);baselines.append(baseline_fingers(pr,side)[i]);rows.append(dict(clip_id=r['clip_id'],word=r['word'],person=r.get('person',r.get('record_id')),role=r['role'],frame=r['frame'],side=side,targets=y,origins=origins,auxiliary=aux))
for r in manifest['ordinary']:
 y=[-1]*5;orig=[];coarse=r['y']
 if coarse[1]==1:y[1:]=[1]*4;orig.append('positive_four_extended_RGB_label')
 if coarse[2]==1:
  assert all(v in [-1,0]for v in y[1:]);y[1:]=[0]*4;orig.append('positive_four_flexed_RGB_label')
 if coarse[0]==1:
  state=[1,0,0,0];assert all(a==-1 or a==b for a,b in zip(y[1:],state));y[1:]=state;orig.append('positive_index_only_RGB_label')
 key=r['clip_id'],r['frame'],r['side']
 if key in direct:
  for j,v in enumerate(direct[key]):
   if v>=0:assert y[j]in[-1,v],(key,j,y,direct[key]);y[j]=v
  orig.append('current_individual_finger_RGB_review')
 if any(v>=0 for v in y):add(r,y,orig)
for r in manifest['auxiliary']:add(r,r['finger_states'],['previous_direct_digit_finger_RGB_review'],True)
arr=np.array(seq);av=np.array(knowns);truth=np.array([r['targets']for r in rows]);fi=np.array([r['role']=='fit'for r in rows]);fitpersons={r['person']for r in rows if r['role']=='fit'};devpersons={r['person']for r in rows if r['role']!='fit'};assert fitpersons.isdisjoint(devpersons),(fitpersons,devpersons)
mu=arr[fi].reshape(-1,127).mean(0);sd=arr[fi].reshape(-1,127).std(0).clip(.05)
np.savez_compressed(U/'finger_arrays.npz',x=arr,availability=av,targets=truth,baseline=np.array(baselines),mean=mu,std=sd)
write(U/'FINGER_INPUTS.json',dict(rows=rows,fit_persons=sorted(fitpersons),development_persons=sorted(devpersons),definition='Visible extension/flexion, not named handshape or exact contact',unknown_not_negative=True))
cfg=read(U/'PLAN.json')['learning'];write(U/'FINGER_FROZEN.json',dict(**cfg,source_pins=pins,labels_sha256=rc.sha(U/'FINGER_RGB_LABELS.json'),inputs_sha256=rc.sha(U/'finger_arrays.npz'),common100_sha256=rc.sha(rc.B/'runs/coupled_42/model.pt'),training='Only new temporalConv+GRU per-finger readout; pretrained common100 frozen and reused; no new word recognition weights',test='All historical ordinary DEV known fingers; exact input masks; no new human-error accuracy claim',promotion='No regression in any seed on old confident correct cells; no increased false contradiction; per-finger unknown never correction'))
coverage={}
for split in ['fit','development']:
 ix=np.array([r['role']==split for r in rows]);coverage[split]=dict(anchors=int(ix.sum()),clips=len({r['clip_id']for r in rows if r['role']==split}),words=len({r['word']for r in rows if r['role']==split}),known=int((truth[ix]>=0).sum()),unobservable_known=int(((truth[ix]>=0)&~av[ix]).sum()),by_finger={f:dict(extended=int((truth[ix,j]==1).sum()),flexed=int((truth[ix,j]==0).sum()))for j,f in enumerate(FINGERS)})
write(U/'FINGER_COVERAGE.json',coverage);print(json.dumps(coverage))
