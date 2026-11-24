"""One prespecified six-run comparison, frozen RGB backbone, matched trainable heads."""
from pathlib import Path
import sys,os,json,hashlib,time,traceback
import numpy as np
import torch
from torch import nn
U=Path(__file__).resolve().parent;OPS=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
 p=U/n;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def projection():return np.random.default_rng(20261031).normal(0,1/np.sqrt(138),(138,768)).astype('float32')
class Head(nn.Module):
 def __init__(self,k):
  super().__init__();self.pose=nn.Linear(138,32);self.aux=nn.Linear(768,32);self.c1=nn.Conv1d(64,64,3,padding=1);self.c2=nn.Conv1d(64,32,3,padding=1);self.out=nn.Linear(32,k)
 def forward(self,x,z,v):
  h=torch.cat([nn.functional.gelu(self.pose(x)),nn.functional.gelu(self.aux(z))],-1)*v[:,:,None]
  h=nn.functional.gelu(self.c1(h.transpose(1,2)))*v[:,None,:];h=nn.functional.gelu(self.c2(h))*v[:,None,:]
  return self.out(h.transpose(1,2))
def loss_fn(logits,y,m,w):
 safe=torch.where(m,y,torch.zeros_like(y));loss=nn.functional.binary_cross_entropy_with_logits(logits,safe,reduction='none')
 loss=torch.where(m,loss*torch.where(safe>.5,w[1],w[0]),torch.zeros_like(loss));counts=m.sum(1);known=counts>0
 return ((loss.sum(1)/counts.clamp_min(1))*known).sum()/known.sum().clamp_min(1)
def assemble(info,stats=None):
 rows=info['rows'];ids=info['target_indices'];arrays=[]
 for r in rows:
  a=dict(np.load(r['input'],allow_pickle=False));labels=dict(np.load(r['targets_file'],allow_pickle=False));a.update(y=labels['y'][:,ids],target_mask=labels['target_mask'][:,ids]);arrays.append(a)
 if stats is None:
  fit=[a for r,a in zip(rows,arrays) if r['role']=='fit'];xf=np.concatenate([a['x'] for a in fit]);mf=np.concatenate([a['mask'] for a in fit]);zf=np.concatenate([a['rgb'] for a in fit])
  pm=np.array([xf[mf[:,j],j].mean() if mf[:,j].any() else 0 for j in range(69)]);ps=np.array([max(xf[mf[:,j],j].std(),.001) if mf[:,j].any() else 1 for j in range(69)])
  stats=dict(pose_mean=pm.tolist(),pose_std=ps.tolist(),rgb_mean=zf.mean(0).tolist(),rgb_std=np.maximum(zf.std(0),.001).tolist())
 n=len(rows);T=max(len(a['x']) for a in arrays);k=len(ids)
 x=np.zeros((n,T,138),np.float32);z=np.zeros((n,T,768),np.float32);y=np.zeros((n,T,k),np.float32);m=np.zeros((n,T,k),bool);v=np.zeros((n,T),np.float32)
 for i,a in enumerate(arrays):
  t=len(a['x']);p=np.where(a['mask'],(a['x']-stats['pose_mean'])/np.array(stats['pose_std']),0);x[i,:t]=np.concatenate([p,a['mask']],1);z[i,:t]=(a['rgb']-stats['rgb_mean'])/np.array(stats['rgb_std']);y[i,:t]=a['y'];m[i,:t]=a['target_mask'];v[i,:t]=1
 return [torch.from_numpy(b) for b in [x,z,y,m,v]],stats,arrays
def freeze():
 info=read(U/'LEARNING_INPUTS.json');assert read(U/'LABEL_AUDIT.json')['known_disagreement_rate']<=.05
 assert read(U/'review/INPUTS_INDEPENDENT.json')['status']=='PASS_NUMERIC_ONLY'
 assert read(U/'review/TARGETS_V2_INDEPENDENT.json')['status']=='PASS_TARGET_MASKS'
 assert read(U/'review/ROI_VISUAL_GATE.json')['allow_bounded_feature_study'] is True
 for p,h in info['source_pins'].items():assert sha(p)==h
 for r in info['rows']:
  assert sha(r['input'])==r['input_sha256'] and sha(r['targets_file'])==r['targets_sha256']
 fit=[r for r in info['rows'] if r['role']=='fit'];assert len(fit)==36
 protected=read(OPS/'word100_face_body_stage_20261027/FROZEN.json')['protected']
 for p,h in protected.items():assert sha(p)==h,p
 proto=dict(status='FROZEN_BEFORE_TRAINING_AND_SCORES',conditions=['pose_capacity_matched','pose_plus_signrep'],seeds=[42,43,44],epochs=160,selected_epoch=160,selection='fixed final epoch; no eval-based checkpoint/model/threshold choice',targets=info['targets'],fit=[r['alias'] for r in fit],fit_people=[1,2,3],development_people=[4],held_people=[5,6],matched_parameters=True,baseline_aux='Fixed seeded dense linear expansion of normalized pose+mask138 into768, no added information. All aux weights active; independent information dimensions still inherently differ.',architecture='same pose138->32,aux768->32,concat64,Conv1d64->64 k3 GELU,Conv1d64->32 k3 GELU,Linear32->K',optimizer=dict(name='AdamW',lr=.001,weight_decay=.01,gradient_clip=1,full_fit_batch=True),loss='Masked binary CE; target/class-balanced, equal per-clip/per-target. Uncertain cells excluded, not whole clips.',backbone='official frozen SignRep, existing Intel GPU features; zero backbone updates',prediction_threshold=.5,feedback_seed=42,feature_feedback_confidence=.8,feedback_probability_difference=.6,stage_alignment_weight=.25,band_ratio=.2,dtw_budget=96,calibration='Only fit-person reference pairs and ID4 development, before locked evaluation; descriptive deviation thresholds not linguistic tolerance',protected=protected,pins={str(p):sha(p) for p in [Path(__file__),U/'build_labels.py',U/'LEARNING_INPUTS.json',U/'LABEL_AUDIT.json',U/'RGB_RUBRIC.json',U/'feedback.py']})
 save('PROTOCOL.json',proto)
def prechecks(batch):
 x,z,y,m,v=batch;torch.manual_seed(0);h=Head(y.shape[-1]);w=torch.ones((2,y.shape[-1]));p=h(x,z,v)
 a=loss_fn(p,y,m,w);ga=torch.autograd.grad(a,tuple(h.parameters()),retain_graph=True)
 altered=y.clone();altered[~m]=float('nan');b=loss_fn(p,altered,m,w);gb=torch.autograd.grad(b,tuple(h.parameters()))
 assert torch.equal(a,b) and all(torch.equal(s,t) for s,t in zip(ga,gb))
 with torch.no_grad():
  whole=h(x,z,v)
  for i in [0,15,35,79]:
   t=int(v[i].sum());single=h(x[i:i+1,:t],z[i:i+1,:t],v[i:i+1,:t]);assert torch.allclose(whole[i,:t],single[0],atol=2e-6,rtol=2e-5)
 return dict(unknown_values_loss_and_gradient_invariant=True,padding_invariant=True)
def train():
 proto=read(U/'PROTOCOL.json');info=read(U/'LEARNING_INPUTS.json')
 for p,h in proto['pins'].items():assert sha(p)==h,p
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 batch,stats,arrays=assemble(info);x,z,y,m,v=batch;fit=[i for i,r in enumerate(info['rows']) if r['role']=='fit']
 counts=torch.stack([((y[fit]==c)&m[fit]).sum((0,1)) for c in [0,1]]).float();assert (counts>0).all();w=counts.sum(0)[None]/(2*counts)
 save('TRAIN_READY.json',dict(status='PASS_PROVISIONAL_PHYSICAL_LABELS_ONLY',checks=prechecks(batch),fit=len(fit),targets=info['targets'],independent_human_truth=False))
 save('NORMALIZATION.json',stats);pr=torch.from_numpy(projection());baseaux=x@pr
 runs=[]
 for condition in proto['conditions']:
  for seed in proto['seeds']:
   name=f'{condition}_{seed}';d=U/'runs'/name;d.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed);head=Head(len(info['targets']));opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
   aux=baseaux if condition=='pose_capacity_matched' else z;start=time.perf_counter();trace=[]
   ids=[info['rows'][i]['clip_id'] for i in fit]
   with (d/'EXPOSURES.jsonl').open('x',encoding='utf8') as ef:
    for ep in range(proto['epochs']):
     head.train();opt.zero_grad();loss=loss_fn(head(x[fit],aux[fit],v[fit]),y[fit],m[fit],w);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1);opt.step();trace.append(float(loss))
     ef.write(json.dumps(dict(epoch=ep+1,clip_ids=ids,loss=float(loss)))+'\n')
   seconds=time.perf_counter()-start;head.eval()
   with torch.no_grad():p=head(x,aux,v).sigmoid().numpy()
   np.save(d/'probabilities.npy',p,allow_pickle=False);torch.save(head.state_dict(),d/'head.pt')
   result=dict(condition=condition,seed=seed,epochs=160,selected_epoch=160,seconds=seconds,parameters=sum(p.numel() for p in head.parameters()),loss_trace=trace,class_counts=counts.tolist(),class_weights=w.tolist(),fit_aliases=[info['rows'][i]['alias'] for i in fit],checkpoint_sha256=sha(d/'head.pt'),prediction_sha256=sha(d/'probabilities.npy'),exposure_sha256=sha(d/'EXPOSURES.jsonl'),backbone_updates=0)
   save(f'runs/{name}/RUN.json',result);runs.append({k:result[k] for k in ['condition','seed','seconds','parameters']});print(json.dumps(runs[-1]),flush=True)
 assert len({r['parameters'] for r in runs})==1
 save('TRAINING.json',dict(status='COMPLETE',runs=runs,locked_test_scores_not_read=True,backbone_updates=0))
def metric(prob,y,m):
 pred=prob>=.5;targets=[]
 for j in range(y.shape[-1]):
  valid=m[:,j];n=int(valid.sum());actual=y[valid,j];p=pred[valid,j];tp=int(((actual==1)&p).sum());tn=int(((actual==0)&~p).sum());fp=int(((actual==0)&p).sum());fn=int(((actual==1)&~p).sum());rec=[]
  if tp+fn:rec.append(tp/(tp+fn))
  if tn+fp:rec.append(tn/(tn+fp))
  targets.append(dict(known=n,tp=tp,tn=tn,fp=fp,fn=fn,balanced_accuracy=float(np.mean(rec)) if rec else None,accuracy=float((tp+tn)/n) if n else None))
 vals=[t['balanced_accuracy'] for t in targets if t['balanced_accuracy'] is not None]
 return dict(macro_balanced_accuracy=float(np.mean(vals)) if vals else None,targets=targets,known=int(m.sum()))
def score():
 info=read(U/'LEARNING_INPUTS.json');batch,_,arrays=assemble(info,read(U/'NORMALIZATION.json'));_,_,y,m,v=batch;y=y.numpy();m=m.numpy();rows=info['rows'];out=[];clips=[]
 for run in read(U/'TRAINING.json')['runs']:
  p=np.load(U/'runs'/f"{run['condition']}_{run['seed']}"/'probabilities.npy',allow_pickle=False)
  for group in ['fit','development','held_person','held_expression_person']:
   ix=[i for i,r in enumerate(rows) if r['role']==group];vals=metric(p[ix].reshape(-1,len(info['targets'])),y[ix].reshape(-1,len(info['targets'])),m[ix].reshape(-1,len(info['targets'])))
   cms=[]
   for i in ix:
    c=metric(p[i],y[i],m[i]);cms.append(c['macro_balanced_accuracy']);clips.append(dict(condition=run['condition'],seed=run['seed'],alias=rows[i]['alias'],role=group,**c))
   out.append(dict(condition=run['condition'],seed=run['seed'],group=group,clips=len(ix),clip_macro_mean=float(np.mean([c for c in cms if c is not None])),**vals))
 summary=[]
 for cond in ['pose_capacity_matched','pose_plus_signrep']:
  for group in ['fit','development','held_person','held_expression_person']:
   rs=[r for r in out if r['condition']==cond and r['group']==group];vals=[r['macro_balanced_accuracy'] for r in rs]
   summary.append(dict(condition=cond,group=group,mean=float(np.mean(vals)),range=[min(vals),max(vals)],clip_macro_mean=float(np.mean([r['clip_macro_mean'] for r in rs])),per_seed=vals))
 save('SCORES.json',dict(rows=out,summary=summary,targets=info['targets'],note='Physical provisional RGB labels; not word accuracy or educational error rate. All rows retained; correlated frames not independent videos.'))
 save('CLIP_SCORES.json',dict(rows=clips))
 print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  f=U/f'{sys.argv[1].upper()}_FIRST_FAILURE.txt'
  if not f.exists():f.write_text(traceback.format_exc(),encoding='utf8')
  raise
