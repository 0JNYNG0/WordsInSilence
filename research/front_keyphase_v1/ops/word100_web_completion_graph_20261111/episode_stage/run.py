"""One fixed sparse episode-stage continuation; frozen upstream and no runtime writes."""
from pathlib import Path
import sys,json,hashlib,time,collections,traceback
import numpy as np,torch
from torch import nn
W=Path(__file__).resolve().parent;U=W.parent;O=U.parent
sys.path.insert(0,str(U))
from graph import read,write,stamp
FAMILIES=[2,3,4,5,7]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class StageNet(nn.Module):
 def __init__(self,seed,supported):
  super().__init__();self.hidden=nn.Linear(684,96);self.output=nn.Linear(96,15);self.register_buffer('supported',torch.tensor(supported,dtype=torch.bool))
  ck=torch.load(O/'word100_candidate_role_learning_20261111/runs'/str(seed)/'model.pt',map_location='cpu',weights_only=True)
  with torch.no_grad():
   self.hidden.weight.copy_(ck['role.0.weight']);self.hidden.bias.copy_(ck['role.0.bias'])
   for j,family in enumerate(FAMILIES):
    for cls,old in enumerate([0,family,family]):
     self.output.weight[3*j+cls].copy_(ck['role.2.weight'][old]);self.output.bias[3*j+cls].copy_(ck['role.2.bias'][old])
 def forward(self,x):
  p=self.output(torch.nn.functional.gelu(self.hidden(x))).reshape(-1,5,3)
  return p.masked_fill(~self.supported[None],-10000.)

def freeze():
 assert not (W/'FROZEN.json').exists();meta=read(W/'INPUTS_DENSE_COMPLETE.json');src=read(U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json');smap={r['clip_id']:r for r in src['rows']};seen=set();counts=np.zeros((5,3),int);probe=np.zeros_like(counts);audit=[]
 for p,h in meta['pins'].items():assert sha(p)==h
 for r in meta['rows']:
  assert sha(r['path'])==r['sha256'] and sha(r['cache'])==r['cache_sha256'];d=np.load(r['path']);z=np.load(r['cache'])['z'];sr=smap[r['clip_id']];direct=set(f for s in sr['sheets'] for f in s['frames'])
  assert set(d['frame_id'].tolist())<=direct and np.array_equal(d['x'],z[d['frame_id']]);assert np.isfinite(d['x']).all()
  expected=np.full_like(d['y'],-1);loc={int(f):j for j,f in enumerate(d['frame_id'])}
  for t in r['labels']:
   f=t['frame'];j=FAMILIES.index(t['family']);expected[loc[f],j]=t['label'];g=next(g for g in sr['groups'] if g['family']==t['family'])
   if t['label']==1:assert g['onset'][1]<f<g['offset'][0]
   if t['label']==2:assert f>g['offset'][1] and f in sr['hold_sample_frames']
   if t['basis']=='different_reviewed_action_in_same_source':assert any(gg['family']!=g['family'] and gg['onset'][1]<f<gg['offset'][0] for gg in sr['groups'])
   (counts if r['split']=='train' else probe)[j,t['label']]+=1
  assert np.array_equal(expected,d['y']);audit.append(dict(clip_id=r['clip_id'],view=r['view'],targets=len(r['labels']),arrays_and_direct_provenance=True))
 train={r['source_sha256'] for r in meta['rows'] if r['split']=='train'};test={r['source_sha256'] for r in meta['rows'] if r['split']=='probe'};assert len(train)==16 and len(test)==8 and not train&test
 assert np.all(counts[:,:2]>0) and np.all(probe[:,:2]>0),('missing movement/rest class',counts,probe)
 supported=counts>0;assert all(probe[j,k]==0 or supported[j,k] for j in range(5) for k in range(3))
 write(W/'PRETRAIN_VERIFICATION.json',dict(at=stamp(),status='PASS_ROOT_SECOND_SOURCE_ARRAY_AUDIT',checks=audit,train_counts=counts.tolist(),probe_counts=probe.tolist(),supported=supported.tolist(),independent_human=False,training_ready=True,limits=meta['limits']))
 paths=[Path(__file__),W/'prepare_dense_inputs.py',W/'INPUTS_DENSE_COMPLETE.json',W/'PRETRAIN_VERIFICATION.json',U/'context_role_training/RESULTS.json',O/'word100_candidate_role_learning_20261111/EVALUATION_INPUTS.json',U/'dynamic_blocks/INPUTS_COMPLETE.json',U/'reviews/sequence_target/MOTION_EXTENTS.json']+[O/'word100_candidate_role_learning_20261111/runs'/str(s)/'model.pt' for s in [42,43,44]]
 write(W/'FROZEN.json',dict(at=stamp(),seeds=[42,43,44],epochs=120,batch=64,lr=.0003,weight_decay=.01,gradient_clip=2,supported=supported.tolist(),train_counts=counts.tolist(),loss='Masked class-balanced per-family cross entropy; weights from train only. Unsupported family/stage combinations masked, not invented.',initialization='Copy matching-seed existing role hidden layer and family/background weights. Only isolated stage head/hidden fine-tuned. All common100 encoders, TCN, shape, direction and runtime unchanged.',input='Existing 684 before/current/after body-normalized learned features. No word ID, frame ID, desired order or absolute source time.',selection='last epoch only; no hyperparameter selection from probe/DEV',evaluation='First direct sampled-point stage confusion on continuation-excluded IDs08/09, then one saved prediction pass on82 existing exposed public scenarios. No new camera/pose/encoder. Stage output alone is not complete feedback or G2 PASS.',proposal_contract='For dynamic family only, retain all contiguous observed motion predictions >=0.5; no longest/highest singleton, no invented filling of gaps. Hold predictions separately retained, never upgraded to movement. Compare motion witness recovery, opposite-action contamination, preserved normal evidence, crossing/missing and speed. No required repetition count or semantic pass.',g2_adoption=False,pins={str(p):sha(p) for p in paths}))
 print('Verified data and froze stage comparison. Train/probe counts:',counts.tolist(),probe.tolist())

def metrics(y,p):
 pred=p.argmax(-1);cm=np.zeros((5,3,3),int)
 for j in range(5):
  for a,b in zip(y[:,j],pred[:,j]):
   if a>=0:cm[j,a,b]+=1
 return dict(confusion=cm.tolist(),correct=int(sum(np.trace(c) for c in cm)),total=int(cm.sum()),recall=[[None if cm[j,k].sum()==0 else float(cm[j,k,k]/cm[j,k].sum()) for k in range(3)] for j in range(5)])

def train():
 cfg=read(W/'FROZEN.json');assert not (W/'TRAINING_COMPLETE.json').exists()
 for p,h in cfg['pins'].items():assert sha(p)==h
 torch.set_num_threads(4);torch.set_num_interop_threads(1);rows=read(W/'INPUTS_DENSE_COMPLETE.json')['rows'];data={split:[] for split in ['train','probe']}
 for r in rows:
  assert sha(r['path'])==r['sha256'];d=np.load(r['path']);data[r['split']].append((r,d['x'],d['y']))
 x=torch.tensor(np.concatenate([d[1] for d in data['train']]));y=torch.tensor(np.concatenate([d[2] for d in data['train']]));px=torch.tensor(np.concatenate([d[1] for d in data['probe']]));py=np.concatenate([d[2] for d in data['probe']]);counts=np.array(cfg['train_counts']);weights=np.where(counts>0,counts.sum(1,keepdims=True)/(np.maximum(counts,1)*np.maximum((counts>0).sum(1,keepdims=True),1)),0);weights=torch.tensor(weights,dtype=torch.float32);done=[]
 for seed in cfg['seeds']:
  dst=W/'runs'/str(seed);assert not (dst/'TRAINING.json').exists();torch.manual_seed(seed);net=StageNet(seed,cfg['supported']);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);start=time.perf_counter();hist=[];steps=0
  for epoch in range(cfg['epochs']):
   order=np.random.default_rng(seed*1000+epoch).permutation(len(x));losses=[]
   for a in range(0,len(order),cfg['batch']):
    ii=order[a:a+cfg['batch']];logit=net(x[ii]);target=y[ii];mask=target>=0;safe=target.clamp_min(0);raw=torch.nn.functional.cross_entropy(logit.reshape(-1,3),safe.reshape(-1),reduction='none').reshape(-1,5);cw=weights[torch.arange(5)[None],safe];loss=(raw*cw*mask).sum()/(cw*mask).sum();assert torch.isfinite(loss)
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip']);opt.step();steps+=1;losses.append(float(loss.detach()))
   hist.append(float(np.mean(losses)))
  dst.mkdir(parents=True,exist_ok=True);torch.save(net.state_dict(),dst/'model.pt');net.eval()
  with torch.inference_mode():p=net(px).softmax(-1).numpy()
  np.savez_compressed(dst/'PROBE.npz',y=py,p=p);result=metrics(py,p);rec=dict(seed=seed,path=str(dst/'model.pt'),sha256=sha(dst/'model.pt'),seconds=time.perf_counter()-start,steps=steps,history=hist,probe=result,scope='Continuation-excluded stage probe only; encoders previously saw people. Not fresh or sign correctness.')
  write(dst/'TRAINING.json',rec);done.append(rec);print(seed,result['correct'],'/',result['total'],round(rec['seconds'],3),'seconds',flush=True)
 write(W/'TRAINING_COMPLETE.json',dict(at=stamp(),runs=done,web_changed=False,private_used=False))

def evaluate():
 cfg=read(W/'FROZEN.json');assert not (W/'EVALUATION_COMPLETE.json').exists()
 for p,h in cfg['pins'].items():assert sha(p)==h
 sources=read(O/'word100_candidate_role_learning_20261111/EVALUATION_INPUTS.json')['rows']+[dict(r,kind='dynamic_complete_blocks') for r in read(U/'dynamic_blocks/INPUTS_COMPLETE.json')['rows']];assert len(sources)==82
 torch.set_num_threads(4);torch.set_num_interop_threads(1);records=[]
 for run in read(W/'TRAINING_COMPLETE.json')['runs']:
  assert sha(run['path'])==run['sha256'];net=StageNet(run['seed'],cfg['supported']);net.load_state_dict(torch.load(run['path'],weights_only=True,map_location='cpu'));net.eval();start=time.perf_counter()
  for r in sources:
   assert sha(r['path'])==r['sha256'];z=np.load(r['path']);dest=W/'predictions'/str(run['seed'])/(r['clip_id']+'_'+r['kind']+'.npz');assert not dest.exists()
   with torch.inference_mode():p=net(torch.tensor(z['z'])).softmax(-1).numpy()
   dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,p=p);records.append(dict(seed=run['seed'],clip_id=r['clip_id'],kind=r['kind'],source=r['path'],source_sha256=r['sha256'],path=str(dest),sha256=sha(dest)))
  print('Saved82 stage predictions',run['seed'],round(time.perf_counter()-start,3),'s; cached features only',flush=True)
 write(W/'EVALUATION_COMPLETE.json',dict(at=stamp(),rows=records,web_changed=False,private_used=False,source_encoder_replayed=False,reserved_final_unread=True,scope='Saved exposed public DEV diagnostics, not adopted'))

if __name__=='__main__':
 try:{'freeze':freeze,'train':train,'evaluate':evaluate}[sys.argv[1]]()
 except BaseException:
  (W/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
