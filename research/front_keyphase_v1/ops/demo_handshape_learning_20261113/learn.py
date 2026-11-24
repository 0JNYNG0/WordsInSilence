"""Isolated predeclared supervised probes; never replaces production models."""
import sys,json,time,hashlib
from pathlib import Path
import numpy as np,torch
from torch import nn
U=Path(__file__).resolve().parent;O=U.parent;ROOT=U.parents[3]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'research/sign_model_pilot_v1/src'))
from pilot.model_masked_v1 import MaskedSignTCN
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def invariants():
 inp=read(U/'INPUTS.json');a=read(U/'ADMISSION.json');assert len(inp['rows'])+len(inp['failures'])==60
 frozen=read(U/'FROZEN.json');assert all(sha(p)==h for p,h in frozen['protection'].items())
 for r in inp['rows']:
  assert r['person'] in list(range(1,10))+[11,12,13]
  assert sha(r['raw'])==r['raw_sha256'] and sha(r['channels'])==r['channels_sha256'] and sha(r['video'])==r['source_sha256']
  z=np.load(r['channels']);assert len(z['frame_id'])==r['frames'] and np.all(np.diff(z['timestamp_us'])>0) and np.isfinite(z['tcn280']).all()
 assert len(a['rows'])==105 and len({r['clip_id'] for r in a['rows'] if r['role']=='development'})==9
 assert {r['person'] for r in a['rows'] if r['role']=='fit'}.isdisjoint({r['person'] for r in a['rows'] if r['role']=='development'})
 write(U/'PRELEARN_VERIFY.json',dict(status='PASS',source_clips=len(inp['rows']),planned=60,failures=inp['failures'],admission_sha256=sha(U/'ADMISSION.json'),contract_sha256=sha(U/'LEARNING_CONTRACT.json'),protect_originals=True))
 return inp,a
def shape_features(z,f):
 xy=z['raw_xy_px'][f,112:133].copy();mask=z['point_available'][f,112:133].copy()
 # Remove image position, scale and in-plane palm direction. Direction must be
 # assessed separately; numeral six must not be recognized just by pointing left.
 axis=xy[9]-xy[0];length=np.linalg.norm(axis)
 ok=bool(mask[0] and mask[9] and mask.sum()>=16 and length>1)
 if not ok:return np.zeros(63,np.float32),False
 ey=axis/length;ex=np.array([ey[1],-ey[0]],np.float32)
 scale=np.median(np.linalg.norm(xy[[5,9,13,17]]-xy[0],axis=1));scale=max(float(scale),1.)
 local=np.stack([(xy-xy[0])@ex,(xy-xy[0])@ey],1)/scale;local[~mask]=0
 return np.r_[local.flatten(),mask.astype(np.float32)].astype(np.float32),True
class ShapeNet(nn.Module):
 def __init__(self):
  super().__init__();self.shared=nn.Sequential(nn.Linear(63,32),nn.ReLU());self.cls=nn.Linear(32,3);self.fingers=nn.Linear(32,5)
 def forward(self,x):
  h=self.shared(x);return self.cls(h),self.fingers(h)
def shape_probe(inp,a):
 assert not (U/'SHAPE_RESULTS.json').exists(),'Do not repeat completed comparison'
 byid={r['clip_id']:r for r in inp['rows']};obs=[];xx=[];usable=[]
 for row in a['rows']:
  z=np.load(byid[row['clip_id']]['channels']);x,ok=shape_features(z,row['frame']);xx.append(x);usable.append(ok);obs.append(row)
 x=np.array(xx);usable=np.array(usable);train=np.array([r['role']=='fit' for r in obs]);test=~train
 mu=x[train&usable].mean(0);sd=x[train&usable].std(0).clip(.05);xt=torch.tensor((x-mu)/sd)
 yy=torch.tensor([r['shape_class'] for r in obs]);ft=torch.tensor([r['finger_states'] for r in obs],dtype=torch.float32);fm=(ft>=0).float();ft=ft.clamp(0)
 ii=np.where(train&usable)[0];jj=np.where(test)[0];proto=np.array([[0,1,1,0,0],[0,1,1,1,1],[1,1,0,0,0]],np.float32);runs=[]
 for seed in [42,43,44]:
  for condition in ['class_only','class_plus_fingers']:
   torch.manual_seed(seed);m=ShapeNet();opt=torch.optim.Adam(m.parameters(),lr=.001);start=time.perf_counter()
   for step in range(600):
    opt.zero_grad();c,f=m(xt[ii]);loss=nn.functional.cross_entropy(c,yy[ii])
    if condition=='class_plus_fingers':loss=loss+(nn.functional.binary_cross_entropy_with_logits(f,ft[ii],reduction='none')*fm[ii]).sum()/fm[ii].sum()
    loss.backward();opt.step()
   train_seconds=time.perf_counter()-start;m.eval()
   with torch.inference_mode():
    c,f=m(xt);prob=c.softmax(-1).numpy();fp=f.sigmoid().numpy() if condition=='class_plus_fingers' else prob@proto
   preds=[]
   for i in jj:
    r=obs[i];p=prob[i];preds.append(dict(clip_id=r['clip_id'],person=r['person'],frame=r['frame'],truth=r['shape_class'],prob=p.tolist(),top1=int(p.argmax()),finger_truth=r['finger_states'],finger_prob=fp[i].tolist(),input_available=bool(usable[i]),confident=bool(usable[i] and p.max()>=.85)))
   videos=[]
   for cid in sorted({r['clip_id'] for r in preds}):
    rr=[r for r in preds if r['clip_id']==cid];p=np.mean([r['prob'] for r in rr],0);videos.append(dict(clip_id=cid,truth=rr[0]['truth'],top1=int(p.argmax()),prob=p.tolist(),correct=bool(p.argmax()==rr[0]['truth']),confident=bool(all(r['input_available'] for r in rr) and p.max()>=.85)))
   known=[(t,p) for r in preds for t,p in zip(r['finger_truth'],r['finger_prob']) if t>=0]
   t0=time.perf_counter()
   with torch.inference_mode():m(xt[jj])
   ms=(time.perf_counter()-t0)*1000
   path=U/f'shape_{condition}_{seed}.pt';torch.save(dict(state_dict=m.state_dict(),mean=mu.tolist(),std=sd.tolist(),schema='palm_normalized_xy_mask63',condition=condition,seed=seed),path)
   run=dict(seed=seed,condition=condition,train_seconds=train_seconds,parameter_count=sum(p.numel() for p in m.parameters()),batch27_inference_ms=ms,checkpoint_sha256=sha(path),videos=videos,anchors=preds,video_correct=sum(r['correct'] for r in videos),video_total=len(videos),confident_video_correct=sum(r['correct'] and r['confident'] for r in videos),confident_video_total=sum(r['confident'] for r in videos),finger_correct=sum((p>=.5)==bool(t) for t,p in known),finger_total=len(known))
   runs.append(run);print('SHAPE',condition,seed,run['video_correct'],'/',run['video_total'],flush=True)
 write(U/'SHAPE_RESULTS.json',dict(runs=runs,fit_videos=26,fit_anchors=int(train.sum()),dev_videos=9,dev_anchors=int(test.sum()),input_unavailable=int((~usable).sum()),scope='Same 3 numeric shapes, held recorded IDs; not cross-word transfer or learner feedback accuracy'))
def classifier(inp,a):
 assert not (U/'TCN_RESULTS.json').exists(),'Do not repeat completed comparison'
 j=O/'word100_rtmw_joint_learning_20261108';base=j/'tcn_runs/finetune_C_20261013/model.pt';ck=torch.load(base,map_location='cpu',weights_only=True)
 m=MaskedSignTCN(**ck['model_config']);m.load_state_dict(ck['state_dict']);m.eval()
 for p in m.parameters():p.requires_grad_(False)
 new=['WORD2633','WORD2635','WORD2637','WORD0099'];src=read(j/'SOURCES.json');rows=[]
 for r in src:
  rec=read(j/'records'/f'{r["clip_id"]}.json');assert rec['status']=='OK';assert sha(rec['channel_path'])==rec['channel_sha256']
  rows.append(dict(id=r['clip_id'],role=r['role'],person=r['person'],y=r['label_index'],path=rec['channel_path'],key='full',new=False))
 for r in inp['rows']:
  if r['clip_id'] in a['classifier_admitted']:rows.append(dict(id=r['clip_id'],role=r['role'],person=r['person'],y=100+new.index(r['word']),path=r['channels'],key='tcn280',new=True))
 assert sum(r['role']=='fit' and r['new'] for r in rows)==34
 features=[];original=[];captured=[];hook=m.head.register_forward_pre_hook(lambda mod,args:captured.append(args[0].detach().clone()))
 for lo in range(0,len(rows),32):
  rr=rows[lo:lo+32];arrays=[np.load(r['path'])[r['key']] for r in rr];T=max(map(len,arrays));x=np.zeros((len(rr),T,280),np.float32);mask=np.zeros((len(rr),T),np.float32)
  for i,v in enumerate(arrays):x[i,:len(v)]=v;mask[i,:len(v)]=1
  with torch.inference_mode():lg=m(torch.tensor(x),torch.tensor(mask))
  original.extend(lg.numpy());features.extend(captured.pop().numpy())
 hook.remove();x=torch.tensor(np.array(features));old=torch.tensor(np.array(original));y=torch.tensor([r['y'] for r in rows]);fit=np.array([i for i,r in enumerate(rows) if r['role']=='fit']);dev=np.array([i for i,r in enumerate(rows) if r['role']=='development'])
 # Check the frozen old logits against their saved baseline rather than rescoring
 # to choose a different seed/candidate. Small batch floating-point tolerance only.
 saved={r['clip_id']:r for r in read(j/'tcn_runs/finetune_C_20261013/PREDICTIONS_full.json')};delta=max(float(np.abs(np.array(saved[r['id']]['logits'])-old[i].numpy()).max()) for i,r in enumerate(rows) if r['role']=='development' and not r['new']);assert delta<.001,delta
 torch.manual_seed(42);head=nn.Linear(x.shape[1],4);opt=torch.optim.Adam(head.parameters(),lr=.003);start=time.perf_counter()
 for epoch in range(80):
  order=np.random.default_rng(42+epoch).permutation(fit)
  for lo in range(0,len(order),64):
   ix=order[lo:lo+64];opt.zero_grad();lg=torch.cat([old[ix],head(x[ix])],1);loss=nn.functional.cross_entropy(lg,y[ix]);loss.backward();opt.step()
 train_seconds=time.perf_counter()-start
 with torch.inference_mode():extra=head(x).numpy()
 pred=[]
 for i in dev:
  r=rows[i];logits=np.r_[old[i].numpy(),extra[i]];order=np.argsort(-logits);bo=np.argsort(-old[i].numpy());pred.append(dict(clip_id=r['id'],person=r['person'],new=r['new'],truth=r['y'],top1=int(order[0]),top3=order[:3].tolist(),baseline_top1=int(bo[0]),baseline_top3=bo[:3].tolist(),logits=logits.tolist()))
 torch.save(dict(base_checkpoint=str(base),base_sha256=sha(base),new_classes=new,state_dict=head.state_dict(),schema='frozen_pooled_width32_appended4',seed=42,epochs=80),U/'appended_classifier.pt')
 def summary(newflag):
  pp=[r for r in pred if r['new']==newflag];return dict(total=len(pp),top1=sum(r['top1']==r['truth'] for r in pp),top3=sum(r['truth'] in r['top3'] for r in pp),baseline_top1=sum(r['baseline_top1']==r['truth'] for r in pp),baseline_top3=sum(r['truth'] in r['baseline_top3'] for r in pp))
 write(U/'TCN_RESULTS.json',dict(status='ISOLATED_FIXED_CANDIDATE',train_seconds=train_seconds,fit_clips=len(fit),new_fit_clips=34,old100=summary(False),added4=summary(True),baseline_equivalence_max_abs=delta,predictions=pred,old_weights_unchanged=True,production_replaced=False,scope='Full-frame public development; no web/new-user camera test. New labels absent from old classifier, old new-class scores not a fair accuracy comparison.'))
 print('CLASSIFIER',summary(False),summary(True),flush=True)
def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);inp,a=invariants();shape_probe(inp,a);classifier(inp,a)
if __name__=='__main__':main()
