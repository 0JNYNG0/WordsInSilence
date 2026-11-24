"""Public-only bounded trial: shared anatomical shape and interval motion observers.

No target word or guide enters either observer. The old heads remain immutable.
Intervals are the prespecified source token span divided uniformly, NOT learned
semantic stages. Full frame outputs/unknown masks are preserved in all files.
"""
from pathlib import Path
import sys,json,hashlib,time,traceback,importlib.util
import numpy as np
import torch
from torch import nn
U=Path(__file__).resolve().parent; OLD=U.parent/'word100_stage_feedback_cycle_20261031'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
 p=U/n;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
spec=importlib.util.spec_from_file_location('frozen_learning',OLD/'learning.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
ALL=read(OLD/'LEARNING_INPUTS.json')['all_targets'];TARGETS=read(OLD/'LEARNING_INPUTS.json')['targets']
SIDES=['R','L'];SHAPES=['index_only','open_four'];DIRS=['up','down','left','right']
MOTION_NAMES=['hand_wrist_x','hand_wrist_y','pose_wrist_x','pose_wrist_y','elbow_x','elbow_y','palm_center_x','palm_center_y','elbow_angle','wrist_velocity_x','wrist_velocity_y']
def get_rows(fresh=False):
 return read(U/'fresh/INPUTS.json')['rows'] if fresh else [r for r in read(OLD/'LEARNING_INPUTS.json')['rows'] if r['role'] in ['fit','development']]
def feature_ids(r,side,kind):
 names=r['feature_names']
 return [i for i,n in enumerate(names) if n.startswith(side+'_') and n.endswith('_angle') and n!=side+'_elbow_angle'] if kind=='shape' else [names.index(side+'_'+n) for n in MOTION_NAMES]
def intervals(alias,fresh=False):
 vis=read((U/'fresh' if fresh else OLD)/'VISUAL_INDEX.json')['rows'];v=next(r for r in vis if r['alias']==alias)
 return [(v['frames'][a]['frame_id'],v['frames'][b]['frame_id']) for a,b in [(0,3),(3,6),(6,9),(9,12)]]
def labels(r,fresh=False):
 p=U/'fresh/targets'/f"{r['alias']}.npz" if fresh else Path(r['targets_file'])
 return dict(np.load(p,allow_pickle=False))
def assemble(rows,stats=None,fresh=False):
 """All original interval frames, no interpolation or frame dropping."""
 arrays=[dict(np.load(r['input'],allow_pickle=False)) for r in rows];ys=[labels(r,fresh) for r in rows]
 if stats is None:
  vals=[];masks=[]
  for r,a in zip(rows,arrays):
   if r['role']!='fit':continue
   for s in SIDES:
    ix=feature_ids(r,s,'motion');vals.append(a['x'][:,ix]);masks.append(a['mask'][:,ix])
  x=np.concatenate(vals);m=np.concatenate(masks)
  mean=np.array([x[m[:,j],j].mean() if m[:,j].any() else 0 for j in range(x.shape[-1])]);sd=np.array([max(x[m[:,j],j].std(),.001) if m[:,j].any() else 1 for j in range(x.shape[-1])])
  stats={'motion_mean':mean.tolist(),'motion_std':sd.tolist(),'shape':'projected angles / pi, no learned normalization','fit_only':True}
 T=max(len(a['x']) for a in arrays);n=len(rows)*2
 sx=np.zeros((n,T,28),np.float32);sy=np.zeros((n,T,2),np.float32);sm=np.zeros((n,T,2),bool);sv=np.zeros((n,T),np.float32)
 mx=[];my=[];mm=[];meta=[]
 for ri,(r,a,y) in enumerate(zip(rows,arrays,ys)):
  t=len(a['x']);ints=intervals(r['alias'],fresh)
  for si,s in enumerate(SIDES):
   j=ri*2+si;ix=feature_ids(r,s,'shape');assert len(ix)==14
   sx[j,:t]=np.concatenate([np.where(a['mask'][:,ix],a['x'][:,ix],0),a['mask'][:,ix]],1);sv[j,:t]=1
   iy=[ALL.index(s+'_'+f) for f in SHAPES];sy[j,:t]=y['y'][:,iy];sm[j,:t]=y['target_mask'][:,iy]
   ix=feature_ids(r,s,'motion');iy=[ALL.index(s+'_motion_'+d) for d in DIRS]
   for low,high in ints:
    ids=np.flatnonzero((a['frame_id']>=low)&(a['frame_id']<=high));assert len(ids)>0
    ts=a['timestamp_us'][ids];idx=ids
    mask=a['mask'][idx][:,ix].copy();mask[0,-2:]=False
    xx=(a['x'][idx][:,ix]-stats['motion_mean'])/np.array(stats['motion_std']);xx=np.where(mask,xx,0)
    # Add displacement from first *observed* point only where BOTH points exist.
    # Keep absolute torso-relative positions too; no hidden positions filled.
    anchor=a['x'][ids[0],ix[:2]];ag=a['mask'][ids[0],ix[:2]]
    dm=mask[:,:2]&ag;dx=np.where(dm,a['x'][idx][:,ix[:2]]-anchor,0)
    mx.append(np.concatenate([xx,mask,dx,dm],1).astype('float32'))
    # Reuse the exact frozen label midpoint convention, including tie rounding.
    # Integer-microsecond midpoint would choose the other frame for some ties.
    center_seconds=(float(ts[0]/1e6)+float(ts[-1]/1e6))/2
    mid=int(np.abs(a['timestamp_us']/1e6-center_seconds).argmin());my.append(y['y'][mid,iy]);mm.append(y['target_mask'][mid,iy]);meta.append(dict(alias=r['alias'],side=s,frame_start=int(low),frame_end=int(high),center_index=mid,role=r['role']))
  centers={q['center_index'] for q in meta if q['alias']==r['alias']}
  motion_columns=[ALL.index(s+'_motion_'+di) for s in SIDES for di in DIRS]
  assert set(np.flatnonzero(y['target_mask'][:,motion_columns].any(1))).issubset(centers),r['alias']
 lengths=np.array([len(q) for q in mx],np.int64);padded=np.zeros((len(mx),max(lengths),26),np.float32)
 for i,q in enumerate(mx):padded[i,:len(q)]=q
 return dict(shape=[torch.from_numpy(z) for z in [sx,sy,sm,sv]],motion=[torch.from_numpy(z) for z in [padded,np.array(my),np.array(mm),lengths]],meta=meta,arrays=arrays,labels=ys),stats
class Shape(nn.Module):
 def __init__(self):
  super().__init__();self.fc=nn.Linear(28,32);self.c=nn.Conv1d(32,32,5,padding=2);self.out=nn.Linear(32,2)
 def forward(self,x,v):
  h=nn.functional.gelu(self.fc(x))*v[:,:,None];h=nn.functional.gelu(self.c(h.transpose(1,2))).transpose(1,2)*v[:,:,None];return self.out(h)
class Motion(nn.Module):
 def __init__(self):
  super().__init__();self.gru=nn.GRU(26,24,batch_first=True);self.out=nn.Linear(24,4)
 def forward(self,x,lengths):
  packed=nn.utils.rnn.pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
  _,state=self.gru(packed);return self.out(state[0])
def weights(y,m):
 cnt=torch.stack([((y==v)&m).sum(tuple(range(y.ndim-1))) for v in [0,1]]).float();assert (cnt>0).all();return cnt.sum(0)[None]/(2*cnt),cnt
def freeze():
 rows=get_rows();assert len(rows)==48 and sum(r['role']=='fit' for r in rows)==36
 torch.set_num_threads(4)
 data,stats=assemble(rows);sx,sy,sm,sv=data['shape'];mx,my,mm,ml=data['motion'];fit=[i*2+s for i,r in enumerate(rows) if r['role']=='fit' for s in [0,1]];mi=[i for i,r in enumerate(data['meta']) if r['role']=='fit']
 sw,sc=weights(sy[fit],sm[fit]);mw,mc=weights(my[mi],mm[mi]);checks={}
 torch.manual_seed(0);sh=Shape();mo=Motion();p=sh(sx,sv);loss=old.loss_fn(p,sy,sm,sw);yy=sy.clone();yy[~sm]=float('nan');assert torch.equal(loss,old.loss_fn(p,yy,sm,sw));checks['unknown_shape_targets_do_not_change_loss']=True
 l=old.loss_fn(mo(mx,ml)[:,None,:],my[:,None,:],mm[:,None,:],mw);y2=my.clone();y2[~mm]=float('nan');assert torch.equal(l,old.loss_fn(mo(mx,ml)[:,None,:],y2[:,None,:],mm[:,None,:],mw));checks['unknown_motion_targets_do_not_change_loss']=True
 # Hand permutation invariant by construction: no side ID, word or guide input.
 assert torch.equal(sh(sx[:1],sv[:1]),sh(sx[:1].clone(),sv[:1].clone()));checks['same_hand_evidence_same_shared_weights']=True
 for i in [0,17,71,95]:
  t=int(sv[i].sum());assert torch.allclose(sh(sx[i:i+1,:t],sv[i:i+1,:t]),p[i:i+1,:t],atol=2e-6)
 checks['shape_padding_invariant']=True
 for i in [0,19,100,300]:assert torch.allclose(mo(mx[i:i+1,:ml[i]],ml[i:i+1]),mo(mx,ml)[i:i+1],atol=2e-6)
 checks['motion_padding_invariant']=True
 assert mx.shape[-1]==26 and torch.isfinite(mx).all() and torch.isfinite(sx).all();checks['finite_masked_inputs_all_original_interval_frames']=True
 assert all(int(ml[i])==q['frame_end']-q['frame_start']+1 for i,q in enumerate(data['meta']))
 pins={str(p):sha(p) for p in [Path(__file__),OLD/'learning.py',OLD/'LEARNING_INPUTS.json',OLD/'RGB_RUBRIC.json',U/'fresh/SELECTION.json']}
 for r in rows:
  assert sha(r['input'])==r['input_sha256'] and sha(r['targets_file'])==r['targets_sha256'];pins[r['input']]=r['input_sha256'];pins[r['targets_file']]=r['targets_sha256']
 protected=read(OLD/'PROTOCOL.json')['protected']
 for p,h in protected.items():assert sha(p)==h
 save('QUALITY_PROTOCOL.json',dict(status='FROZEN_BEFORE_TRAINING_AND_FRESH_LABEL_SCORES',seeds=[42,43,44],epochs=160,selected_epoch=160,optimizer='AdamW lr=.001 weight_decay=.01 clip1',threshold=.5,targets=TARGETS,shape_targets=SHAPES,shape='Shared bilateral 14 projected finger angles+14 masks, 5-frame conv, two binary physical attributes. Thumb is an input although targets describe four non-thumb fingers. No synthetic extra people; same labeled source cells.',motion='Shared bilateral full prespecified interval, ALL original frames, 11 features+masks+2 displacement+mask, packed GRU26->24->4. First velocity masked: no outside-interval frame. Mixed/unknown excluded, no missing interpolation. Padding ignored.',scope='Anatomical feature observer only. Uniform intervals from known source token span, NOT autonomous stage segmentation, linguistic correctness or a word recognizer.',comparison=['frozen_previous_pose','frozen_previous_rgb','shared_shape_interval_motion'],unchanged_outputs=['R_face_overlap','hands_overlap'],unchanged_source='frozen_previous_pose for same seed',primary='Pooled class-balanced accuracy per target then macro across ten targets; fresh_known and fresh_expression separately. 3 seed mean/range. No outcome-based selection.',training_fit=[r['alias'] for r in rows if r['role']=='fit'],development='12 fixed ID4 clips; no search or early stopping',fresh='32 fixed ID7/8 public TRAIN clips, new to observer, previously TCN TRAIN. Old held32 not accessed.',pins=pins,protected=protected))
 save('QUALITY_READY.json',dict(status='PASS_INPUT_CONTRACT_ONLY',checks=checks,shape_counts=sc.tolist(),motion_counts=mc.tolist(),human_truth=False));save('NORMALIZATION.json',stats)
def train():
 proto=read(U/'QUALITY_PROTOCOL.json')
 for p,h in proto['pins'].items():assert sha(p)==h,p
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);rows=get_rows();d,_=assemble(rows,read(U/'NORMALIZATION.json'));sx,sy,sm,sv=d['shape'];mx,my,mm,ml=d['motion'];fi=[i*2+s for i,r in enumerate(rows) if r['role']=='fit' for s in [0,1]];mi=[i for i,r in enumerate(d['meta']) if r['role']=='fit'];sw,sc=weights(sy[fi],sm[fi]);mw,mc=weights(my[mi],mm[mi]);runs=[]
 for seed in proto['seeds']:
  out=U/'runs'/str(seed);out.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed);sh=Shape();mo=Motion();opt=torch.optim.AdamW(list(sh.parameters())+list(mo.parameters()),lr=.001,weight_decay=.01);trace=[];start=time.perf_counter()
  with (out/'EXPOSURES.jsonl').open('x',encoding='utf8') as f:
   for epoch in range(160):
    opt.zero_grad();ls=old.loss_fn(sh(sx[fi],sv[fi]),sy[fi],sm[fi],sw);lm=old.loss_fn(mo(mx[mi],ml[mi])[:,None,:],my[mi,None,:],mm[mi,None,:],mw);loss=ls+lm;assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(list(sh.parameters())+list(mo.parameters()),1);opt.step();trace.append([float(ls),float(lm)])
    f.write(json.dumps(dict(epoch=epoch+1,clip_ids=[r['clip_id'] for r in rows if r['role']=='fit'],shape_loss=float(ls),motion_loss=float(lm)))+'\n')
  elapsed=time.perf_counter()-start;torch.save({'shape':sh.state_dict(),'motion':mo.state_dict()},out/'heads.pt');run=dict(seed=seed,epochs=160,selected_epoch=160,seconds=elapsed,parameters=sum(p.numel() for h in [sh,mo] for p in h.parameters()),shape_parameters=sum(p.numel() for p in sh.parameters()),motion_parameters=sum(p.numel() for p in mo.parameters()),trace=trace,checkpoint_sha256=sha(out/'heads.pt'),exposure_sha256=sha(out/'EXPOSURES.jsonl'));save(f'runs/{seed}/RUN.json',run);runs.append({k:run[k] for k in ['seed','epochs','seconds','parameters']});print(json.dumps(runs[-1]),flush=True)
 save('TRAINING.json',dict(status='COMPLETE',runs=runs,backbone_updates=0,old_model_updates=0))
def evaluate(fresh=False):
 torch.set_num_threads(4);rows=get_rows(fresh);rows=[dict(r,targets_file=str(U/'fresh/targets'/f"{r['alias']}.npz")) for r in rows] if fresh else rows
 d,_=assemble(rows,read(U/'NORMALIZATION.json'),fresh);sx,sy,sm,sv=d['shape'];mx,my,mm,ml=d['motion'];info={'rows':rows,'target_indices':[ALL.index(f) for f in TARGETS]};batch,_,aa=old.assemble(info,read(OLD/'NORMALIZATION.json'));x,z,y,m,v=batch;proj=torch.from_numpy(old.projection());outrows=[];predpins={};stage='fresh' if fresh else 'development'
 proto=read(U/'QUALITY_PROTOCOL.json');pins=dict(proto['pins']);pins[str(U/'NORMALIZATION.json')]=sha(U/'NORMALIZATION.json');pins[str(OLD/'NORMALIZATION.json')]=sha(OLD/'NORMALIZATION.json')
 for p,h in proto['pins'].items():assert sha(p)==h
 for r in rows:
  assert sha(r['input'])==r['input_sha256'];pins[r['input']]=r['input_sha256'];pins[r['targets_file']]=sha(r['targets_file'])
 for seed in [42,43,44]:
  cp=U/'runs'/str(seed)/'heads.pt';assert sha(cp)==read(cp.parent/'RUN.json')['checkpoint_sha256'];pins[str(cp)]=sha(cp)
  for folder in ['pose_capacity_matched','pose_plus_signrep']:
   cp=OLD/'runs'/f'{folder}_{seed}'/'head.pt';assert sha(cp)==read(cp.parent/'RUN.json')['checkpoint_sha256'];pins[str(cp)]=sha(cp)
 save(stage.upper()+'_EVALUATION_FREEZE.json',dict(status='FROZEN_BEFORE_FIRST_INFERENCE',pins=pins,output_mask='candidate_availability means model output exists, NOT actual observation confidence; measurement masks must gate any feedback',no_threshold_selection=True))
 y=y.numpy();m=m.numpy();comparisons={}
 for seed in [42,43,44]:
  for cond,folder in [('frozen_previous_pose','pose_capacity_matched'),('frozen_previous_rgb','pose_plus_signrep')]:
   h=old.Head(len(TARGETS));cp=OLD/'runs'/f'{folder}_{seed}'/'head.pt';h.load_state_dict(torch.load(cp,weights_only=True));h.eval()
   with torch.no_grad():p=h(x,x@proj if folder=='pose_capacity_matched' else z,v).sigmoid().numpy()
   comparisons[(cond,seed)]=p
  sh=Shape();mo=Motion();cp=U/'runs'/str(seed)/'heads.pt';ck=torch.load(cp,weights_only=True);sh.load_state_dict(ck['shape']);mo.load_state_dict(ck['motion']);sh.eval();mo.eval()
  with torch.no_grad():sp=sh(sx,sv).sigmoid().numpy();mp=mo(mx,ml).sigmoid().numpy()
  p=comparisons[('frozen_previous_pose',seed)].copy()
  # Only supervised/evaluated interval centers get a motion estimate. Other
  # times are explicitly unavailable, not synthesized as a continuous claim.
  availability=np.zeros_like(p,dtype=bool)
  for ri,r in enumerate(rows):
   n=len(d['arrays'][ri]['x'])
   for si,s in enumerate(SIDES):
    for jj,f in enumerate(SHAPES):
     if s+'_'+f in TARGETS:
      j=TARGETS.index(s+'_'+f);p[ri,:n,j]=sp[ri*2+si,:n,jj];availability[ri,:n,j]=True
  for j,f in enumerate(TARGETS):
   if '_motion_' in f:p[:,:,j]=np.nan
   if f in ['R_face_overlap','hands_overlap']:availability[:,:,j]=v.numpy().astype(bool)
  for ii,meta in enumerate(d['meta']):
   ri=next(i for i,r in enumerate(rows) if r['alias']==meta['alias'])
   for k,di in enumerate(DIRS):
    f=meta['side']+'_motion_'+di
    if f in TARGETS:p[ri,meta['center_index'],TARGETS.index(f)]=mp[ii,k];availability[ri,meta['center_index'],TARGETS.index(f)]=True
  assert np.isfinite(p[m]).all(), 'Every known target requires a genuine estimate'
  comparisons[('shared_shape_interval_motion',seed)]=p
  npfile=U/'predictions'/f'{stage}_candidate_availability_{seed}.npy';npfile.parent.mkdir(exist_ok=True);assert not npfile.exists();np.save(npfile,availability,allow_pickle=False);predpins[str(npfile)]=sha(npfile)
 for (cond,seed),p in comparisons.items():
  npfile=U/'predictions'/f'{stage}_{cond}_{seed}.npy';assert not npfile.exists();np.save(npfile,p,allow_pickle=False);predpins[str(npfile)]=sha(npfile)
  for role in sorted({r['role'] for r in rows}):
   ids=[i for i,r in enumerate(rows) if r['role']==role];val=old.metric(p[ids].reshape(-1,len(TARGETS)),y[ids].reshape(-1,len(TARGETS)),m[ids].reshape(-1,len(TARGETS)));outrows.append(dict(condition=cond,seed=seed,role=role,clips=len(ids),**val))
 summary=[]
 for cond in sorted({r['condition'] for r in outrows}):
  for role in sorted({r['role'] for r in rows}):
   rr=[r for r in outrows if r['condition']==cond and r['role']==role];vv=[r['macro_balanced_accuracy'] for r in rr];summary.append(dict(condition=cond,role=role,mean=float(np.mean(vv)),range=[min(vv),max(vv)],target_means=[float(np.mean([r['targets'][j]['balanced_accuracy'] for r in rr])) if rr[0]['targets'][j]['known'] else None for j in range(len(TARGETS))]))
 save(stage.upper()+'_SCORES.json',dict(rows=outrows,summary=summary,targets=TARGETS,input_aliases=[r['alias'] for r in rows],prediction_pins=predpins,note='Provisional same-Codex RGB physical labels, not independently expert-verified educational correctness. Fresh inference once.'))
 print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':
 try:
  if sys.argv[1]=='fresh':evaluate(True)
  elif sys.argv[1]=='development':evaluate(False)
  else:globals()[sys.argv[1]]()
 except BaseException:
  p=U/(sys.argv[1].upper()+'_FIRST_FAILURE.txt')
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
