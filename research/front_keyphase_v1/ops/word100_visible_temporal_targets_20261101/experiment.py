from pathlib import Path
import json,hashlib,time,sys,traceback,os
import numpy as np
import torch
from head import Head,masked_loss
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 Path(p).parent.mkdir(exist_ok=True,parents=True)
 with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def projection():return np.random.default_rng(20261101).normal(0,1/np.sqrt(138),(138,768)).astype('float32')

def assemble(info,stats=None):
 rows=info['rows'];ids=info['selected_indices'];raw=[];cache={}
 for r in rows:
  if r['alias'] not in cache:
   assert sha(r['input'])==r['input_sha256'];cache[r['alias']]=dict(np.load(r['input'],allow_pickle=False))
  a=cache[r['alias']];lookup={int(f):i for i,f in enumerate(a['frame_id'])};ix=[lookup[f] for f in r['frame_ids']]
  assert len(ix)==13 and ix==list(range(ix[0],ix[0]+13)) and r['center_frame']==r['frame_ids'][6]
  assert a['timestamp_us'][ix].tolist()==r['timestamps_us']
  raw.append(dict(x=a['x'][ix],mask=a['mask'][ix]&np.isfinite(a['x'][ix]),rgb=a['rgb'][ix]))
 if stats is None:
  ix=[i for i,r in enumerate(rows) if r['role']=='fit'];px=np.concatenate([raw[i]['x'] for i in ix]);pm=np.concatenate([raw[i]['mask'] for i in ix]);rz=np.concatenate([raw[i]['rgb'] for i in ix])
  mean=np.array([px[pm[:,j],j].mean() if pm[:,j].any() else 0 for j in range(69)])
  std=np.array([max(float(px[pm[:,j],j].std()),.001) if pm[:,j].any() else 1 for j in range(69)])
  stats=dict(pose_mean=mean.tolist(),pose_std=std.tolist(),rgb_mean=rz.mean(0).tolist(),rgb_std=np.maximum(rz.std(0),.001).tolist(),scope='fit13-frame windows only; repeated overlapping training-frame occurrences retained')
 x=[];z=[]
 for a in raw:
  v=np.where(a['mask'],(a['x']-stats['pose_mean'])/np.array(stats['pose_std']),0)
  x.append(np.concatenate([v,a['mask']],axis=-1));z.append((a['rgb']-stats['rgb_mean'])/np.array(stats['rgb_std']))
 y=np.asarray([r['y'] for r in rows],np.float32)[:,ids];m=np.asarray([r['target_mask'] for r in rows],bool)[:,ids]
 xx=np.asarray(x,np.float32);zz=np.asarray(z,np.float32);assert np.isfinite(xx).all() and np.isfinite(zz).all()
 return [torch.from_numpy(a) for a in [xx,zz,y,m]],stats

def checks():
 torch.manual_seed(9);x=torch.randn(5,13,138);z=torch.randn(5,13,768);h=Head(4)
 y=torch.randint(0,2,(5,4)).float();m=torch.rand(5,4)>.3;w=torch.ones(2,4);ew=torch.ones(5,4)
 p=h(x,z);a=masked_loss(p,y,m,w,ew);ga=torch.autograd.grad(a,tuple(h.parameters()),retain_graph=True)
 changed=y.clone();changed[~m]=float('nan');b=masked_loss(p,changed,m,w,ew);gb=torch.autograd.grad(b,tuple(h.parameters()))
 assert torch.equal(a,b) and all(torch.equal(s,t) for s,t in zip(ga,gb))
 h.eval()
 with torch.no_grad():
  batch=h(x,z);one=torch.cat([h(x[i:i+1],z[i:i+1]) for i in range(5)])
  assert torch.allclose(batch,one,atol=2e-6)
  assert not torch.equal(h(x.flip(1),z.flip(1)),batch)
 return dict(unknown_label_gradient_invariant=True,batch_independent=True,temporal_order_not_pooled_away=True,word_text_inputs=False)

def freeze():
 audit=read(U/'LABEL_AUDIT.json');assert audit['status']=='READY_PROVISIONAL_RGB_FEATURE_STUDY'
 assert read(U/'review/INPUT_LABEL_VERIFICATION.json')['status'].startswith('PASS')
 assert read(U/'review/TRAIN_CODE_REVIEW.json')['status'].startswith('PASS')
 info=read(U/'DATASET.json')
 for p,h in info['pins'].items():assert sha(p)==h
 assert {r['person'] for r in info['rows'] if r['role']=='fit'}=={1,2,3}
 checks_result=checks();sel=read(U/'SELECTION.json')
 protected=sel['protected'];assert all(sha(p)==h for p,h in protected.items())
 pins={str(U/n):sha(U/n) for n in ['AUTHORIZATION.json','RUBRIC.json','PLAN_KO.txt','head.py','experiment.py','build_targets.py','compare_saved.py','diagnostics.py','make_report.py','report_template.html','DATASET.json','LABEL_AUDIT.json','review/INPUT_LABEL_VERIFICATION.json','review/TRAIN_CODE_REVIEW.json']}
 write(U/'TRAIN_FROZEN.json',dict(status='FROZEN_BEFORE_NEW_SCORES',conditions=['pose_only','pose_plus_signrep'],seeds=[42,43,44],epochs=160,lr=.001,weight_decay=.01,threshold=.5,selection='fixed final160, no evaluation tuning or candidate replacement',targets=info['selected_targets'],fit_windows=[r['sample_id'] for r in info['rows'] if r['role']=='fit'],loss='masked binary; inverse class weights and equal per clip-target mass',backbone='frozen existing SignRep, no backbone updates',architecture='13-frame small head, separate ordered thirds and centre',controls='same trainable parameter count; pose-only aux is fixed seeded pose projection',checks=checks_result,protected=protected,pins=pins,source_pins=info['pins'],evaluation_scope='source-drawn centre/window supervision, not automatic full-video segmentation or educational correctness'))

def train():
 p=read(U/'TRAIN_FROZEN.json');info=read(U/'DATASET.json')
 assert not (U/'TRAINING.json').exists()
 assert not (U/'TRAIN_RUNNING.json').exists(), 'Inspect existing execution; never duplicate training'
 for path,h in {**p['pins'],**p['source_pins']}.items():assert sha(path)==h
 write(U/'TRAIN_RUNNING.json',dict(pid=os.getpid(),started=time.time(),stage='six_fixed_runs'))
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 (x,z,y,m),stats=assemble(info);write(U/'NORMALIZATION.json',stats)
 fit=[i for i,r in enumerate(info['rows']) if r['role']=='fit'];counts=torch.stack([((y[fit]==c)&m[fit]).sum(0) for c in [0,1]]).float();assert (counts>0).all()
 w=counts.sum(0)[None]/(2*counts);ew=torch.zeros_like(y)
 for clip in {info['rows'][i]['clip_id'] for i in fit}:
  ix=[i for i in fit if info['rows'][i]['clip_id']==clip];cnt=m[ix].sum(0).clamp_min(1);ew[ix]=1/cnt
 auxbase=x@torch.from_numpy(projection());results=[]
 for condition in p['conditions']:
  for seed in p['seeds']:
   dest=U/'runs'/f'{condition}_{seed}';dest.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed);h=Head(len(info['selected_targets']));opt=torch.optim.AdamW(h.parameters(),lr=p['lr'],weight_decay=p['weight_decay']);aux=auxbase if condition=='pose_only' else z
   started=time.perf_counter();losses=[]
   with (dest/'EXPOSURES.jsonl').open('x',encoding='utf8') as log:
    for epoch in range(p['epochs']):
     h.train();opt.zero_grad();loss=masked_loss(h(x[fit],aux[fit]),y[fit],m[fit],w,ew[fit]);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),1);opt.step();losses.append(float(loss.detach()));log.write(json.dumps(dict(epoch=epoch+1,sample_ids=p['fit_windows'],loss=losses[-1]))+'\n')
   training_seconds=time.perf_counter()-started;h.eval()
   with torch.no_grad():pred=h(x,aux).sigmoid().numpy()
   torch.save(h.state_dict(),dest/'head.pt');np.save(dest/'probabilities.npy',pred,allow_pickle=False)
   result=dict(condition=condition,seed=seed,epochs=p['epochs'],selected_epoch=p['epochs'],training_seconds=training_seconds,parameters=sum(v.numel() for v in h.parameters()),losses=losses,class_counts=counts.tolist(),class_weights=w.tolist(),checkpoint_sha256=sha(dest/'head.pt'),probabilities_sha256=sha(dest/'probabilities.npy'),exposures_sha256=sha(dest/'EXPOSURES.jsonl'))
   write(dest/'COMPLETE.json',result);results.append({k:result[k] for k in ['condition','seed','training_seconds','parameters']});print(json.dumps(results[-1]),flush=True)
 assert len({r['parameters'] for r in results})==1
 write(U/'TRAINING.json',dict(status='COMPLETE_SIX_RUNS',runs=results,backbone_updates=0))
 (U/'TRAIN_RUNNING.json').unlink()

def metrics(pred,y,m,targets):
 out=[]
 for j,name in enumerate(targets):
  pp=pred[:,j]>=.5;known=m[:,j];gt=y[:,j]
  tp=int((known&(gt==1)&pp).sum());tn=int((known&(gt==0)&~pp).sum());fp=int((known&(gt==0)&pp).sum());fn=int((known&(gt==1)&~pp).sum());n=tp+tn+fp+fn
  ba=(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None
  out.append(dict(target=name,known=n,tp=tp,tn=tn,fp=fp,fn=fn,accuracy=(tp+tn)/n if n else None,balanced_accuracy=ba))
 valid=[r['balanced_accuracy'] for r in out if r['balanced_accuracy'] is not None];n=sum(r['known'] for r in out)
 return dict(targets=out,macro_balanced_accuracy=float(np.mean(valid)) if valid else None,balanced_supported_targets=len(valid),known_cells=n,known_cell_accuracy=sum(r['tp']+r['tn'] for r in out)/n if n else None)

def score():
 info=read(U/'DATASET.json');p=read(U/'TRAIN_FROZEN.json');(x,z,y,m),_=assemble(info,read(U/'NORMALIZATION.json'));y=y.numpy();m=m.numpy();scores=[]
 for rr in read(U/'TRAINING.json')['runs']:
  dest=U/'runs'/f"{rr['condition']}_{rr['seed']}";d=read(dest/'COMPLETE.json');assert sha(dest/'probabilities.npy')==d['probabilities_sha256']
  prob=np.load(dest/'probabilities.npy',allow_pickle=False)
  for role in sorted({r['role'] for r in info['rows']}):
   ix=[i for i,r in enumerate(info['rows']) if r['role']==role]
   scores.append(dict(condition=rr['condition'],seed=rr['seed'],role=role,windows=len(ix),**metrics(prob[ix],y[ix],m[ix],info['selected_targets'])))
 summary=[]
 for condition in p['conditions']:
  for role in sorted({r['role'] for r in info['rows']}):
   rows=[s for s in scores if s['condition']==condition and s['role']==role];v=[r['macro_balanced_accuracy'] for r in rows if r['macro_balanced_accuracy'] is not None]
   summary.append(dict(condition=condition,role=role,mean=float(np.mean(v)) if v else None,range=[min(v),max(v)] if v else None,known_cells=rows[0]['known_cells'],balanced_supported_targets=rows[0]['balanced_supported_targets']))
 write(U/'SCORES.json',dict(status='ONE_FROZEN_COMPARISON_COMPLETE',rows=scores,summary=summary,targets=info['selected_targets'],not_sign_word_or_educational_accuracy=True));print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  path=U/f'{sys.argv[1].upper()}_FIRST_FAILURE.txt'
  if not path.exists():path.write_text(traceback.format_exc(),encoding='utf8')
  raise
