"""Isolated boundary-augmentation candidate. Frozen C; original T3-F untouched.

Only its 9,600-parameter temporal auxiliary matrix is fine-tuned in a new file.
Frozen feature caches accelerate the exact fixed-backbone head computation.
"""
from pathlib import Path
import sys,os,json,time,hashlib,copy,traceback
import numpy as np
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path('C:/MyUnityProjects/WordsInSilence');OPS=ROOT/'research/front_keyphase_v1/ops'
U=OPS/'word100_boundary_aux_20261026';P=OPS/'word100_boundary_evidence_20261026'
sys.path.insert(0,str(P))
import boundary_probe as b
import torch
from frozen_temporal import FrozenCTemporal
from training_core import MODEL,padded_arrays,schedule,prediction

SEEDS=(20261013,20261014,20261015);VIEWS=('masked','sensor');TEMP=('x_whole','x_context')
KINDS=('clean','start_spike','end_spike','both_spikes')
LR=0.0001;EPOCHS=40;BATCH=32
def write(name,x): b.write(U/name,x)
def statehash(s):
 h=hashlib.sha256()
 for k,v in sorted(s.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def event(**kw):
 x=dict(at=time.time(),**kw)
 (U/'STATUS.json').write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 print(json.dumps(x,ensure_ascii=False),flush=True)
def checks():
 c=b.read(U/'FROZEN.json')
 for p,h in c['pins'].items():assert b.sha(p)==h,p
 return c
def freeze():
 assert b.read(P/'VERIFIED.json')['status']=='PASS'
 assert all((P/f'{s}_SCALE_FIXED_SCORES.json').exists() for s in ('C13','T3F13'))
 paths=[Path(__file__),P/'boundary_probe.py',P/'VERIFIED.json',P/'FROZEN.json',b.B/'training_CONTEXT.json',
   b.T/'frozen_temporal.py',b.T/'temporal.py',b.B/'training_core.py',
   ROOT/'research/sign_model_pilot_v1/src/pilot/data.py',ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py']
 for seed in SEEDS:paths.extend([b.T/f'T3F_{seed}_best.pt',b.T/f'T3F_{seed}_RESULT.json'])
 write('FROZEN.json',dict(at=time.time(),name='T3F boundary augmentation auxiliary',seeds=SEEDS,lr=LR,epochs=EPOCHS,batch=BATCH,
  training_data='899 public TRAIN; existing C spatial/whole/context schedule retained; each selected training clip contributes clean and one augmented view equally.',
  augmentation='One second of copied body-only waiting each end, with a copied edge hand point at start, end or both. No original motion/frame/mask deletion. Synthetic diagnostic nuisance; not natural-user prevalence.',
  trainable='Only 100x96 phase_weight matrix; initialize from respective original T3F; C backbone/BN/head frozen.',
  loss='0.5 clean CE + 0.5 augmented CE, AdamW lr0.0001 weight_decay0.001 clip_norm1',
  selection='Mean unmodified DEV CE across sensor/masked only, epoch0 included; strict decrease1e-6; no stress score selection; fixed40 epochs.',
  limits='This uses repeatedly exposed DEV, no fresh199/known200 or user/mentor accesses. No word-specific rules, default deployment, detector changes or claimed anatomy recovery.',
  acceptance=['mean original DEV top1 and top5 not lower for either view',
              'mean synthetic top1 improves over original T3F for either view and does not decline for the other',
              'all C tensors and original observation arrays preserved',
              'report per-seed, per-ID regressions; do not call synthetic gain real webcam improvement'],
  pins={str(p):b.sha(p) for p in paths}))
 event(stage='FROZEN')

def load_source_rows():
 ctx=b.read(b.B/'training_CONTEXT.json');tr=[];dv=[]
 for r in ctx['ordered_rows']:(tr if r['role']=='train_candidate' else dv).append(r)
 assert len(tr)==899 and len(dv)==300
 assert {r['signer_id'] for r in tr}==set(range(1,10)) and {r['signer_id'] for r in dv}=={11,12,13}
 return ctx,tr,dv

def row_inputs(r,training):
 rec=b.read(b.B/'data/records'/f"{r['clip_id']}.json");out=[];keys=[]
 for vi,v in enumerate(VIEWS):
  q=rec['views'][v];obs=b.arrays(q['observation']);saved=b.arrays(q['channel_file'])
  assert b.sha(q['channel_file'])==r['views'][v]['sha256']
  native,present=b.to_native_like(obs['xy_px'],obs['usable'])
  for ti,t in enumerate(TEMP if training else TEMP[:1]):
   ids=saved['whole_frame_id' if t=='x_whole' else 'context_frame_id']
   select=np.flatnonzero(np.isin(obs['frame_id'],ids))
   assert len(select)==len(ids) and np.array_equal(obs['frame_id'][select],ids)
   assert np.array_equal(select,np.arange(select[0],select[-1]+1))
   xy=native[select];pr=present[select];ts=obs['timestamp_us'][select]
   clean=b.build_channels(xy,pr,ts)['x'];assert np.array_equal(clean,saved[t])
   # Context can contain absent edge frames. Original valid frame IDs and
   # internal missing slots must still be preserved by the augmentation.
   # Every chosen native context intersects existing whole hand span, and
   # endpoints need at least one hand point for exact spike construction.
   # If annotation padding creates no hand at an endpoint, copy the nearest
   # observed edge hand point without deleting any original context frames.
   for ki,k in enumerate(KINDS):
    if k=='clean':x=clean
    else:x=augment_preserve_context(xy,pr,ts,k)
    out.append(x);keys.append((vi,ti,ki))
 return out,keys

def augment_preserve_context(xy,pr,ts,kind):
 step=int(np.median(np.diff(ts)));assert step>0;n=max(1,round(1e6/step))
 l=np.repeat(xy[:1],n,axis=0);r=np.repeat(xy[-1:],n,axis=0)
 lm=np.repeat(pr[:1],n,axis=0);rm=np.repeat(pr[-1:],n,axis=0)
 l[:,25:67]=0;r[:,25:67]=0;lm[:,25:67]=False;rm[:,25:67]=False
 observed=np.flatnonzero(pr[:,25:67].any(1));assert len(observed)>0
 for active,a,mask,source,idx in [(kind in ('start_spike','both_spikes'),l,lm,observed[0],0),
                                 (kind in ('end_spike','both_spikes'),r,rm,observed[-1],-1)]:
  if active:
   js=np.flatnonzero(pr[source,25:67])+25;j=next((j for j in (25,46) if pr[source,j]),int(js[0]))
   a[idx,j]=xy[source,j];mask[idx,j]=True
 xx=np.concatenate([l,xy,r]);pm=np.concatenate([lm,pr,rm]);tt=np.concatenate([ts[0]-step*np.arange(n,0,-1),ts,ts[-1]+step*np.arange(1,n+1)])
 # One-sided augmentation retains all original context; only the *added*
 # opposite empty waiting is removed. This matches whole-span stress exactly.
 start=0 if kind in ('start_spike','both_spikes') else n
 stop=len(tt) if kind in ('end_spike','both_spikes') else n+len(ts)
 assert np.array_equal(xx[n:n+len(xy)],xy) and np.array_equal(pm[n:n+len(pr)],pr)
 assert np.array_equal(tt[n:n+len(ts)],ts)
 return b.build_channels(xx[start:stop],pm[start:stop],tt[start:stop])['x']

def prepare_cache(seed,tr,dv):
 cp=U/f'features_{seed}.npz';meta=U/f'features_{seed}.json'
 if cp.exists():
  m=b.read(meta);assert b.sha(cp)==m['cache_sha256'] and m['protocol_sha256']==b.sha(U/'FROZEN.json')
  z=b.arrays(cp);return z,b.read(b.T/f'T3F_{seed}_best.pt') if False else None
 ck=torch.load(b.T/f'T3F_{seed}_best.pt',map_location='cpu',weights_only=True)
 model=FrozenCTemporal(**MODEL);model.load_state_dict(ck['state_dict']);model.eval();model.assert_frozen()
 initial=statehash(model.state_dict());cache={};input_checks=0;probe_error=0.
 with torch.inference_mode():
  for split,rows,training in [('train',tr,True),('dev',dv,False)]:
   store=np.zeros((len(rows),2,2 if training else 1,4,196),np.float32)
   for ri,row in enumerate(rows):
    xs,keys=row_inputs(row,training);x,p=padded_arrays(xs)
    xx=torch.from_numpy(x);pp=torch.from_numpy(p);base,d=model.base_and_contrast(xx,pp)
    fd=torch.cat([base,d],1).numpy();assert fd.shape==(len(keys),196) and np.isfinite(fd).all()
    for (vi,ti,ki),feature in zip(keys,fd):store[ri,vi,ti,ki]=feature
    # Fixed TRAIN-only exact model/cache connection, no baseline rescore.
    if training and ri==0:
     direct=model(xx,pp);calc=base+torch.nn.functional.linear(d,model.phase_weight)
     probe_error=max(probe_error,float((direct-calc).abs().max()));assert torch.equal(direct,calc)
    input_checks+=len(xs)
    if (ri+1)%200==0:event(stage='CACHE',seed=seed,split=split,completed=ri+1,total=len(rows))
   cache[split]=store
 assert initial==statehash(model.state_dict());checks()
 np.savez_compressed(cp,**cache)
 write(meta.name,dict(seed=seed,protocol_sha256=b.sha(U/'FROZEN.json'),cache_sha256=b.sha(cp),
  input_variants=input_checks,train_ids=[r['clip_id'] for r in tr],dev_ids=[r['clip_id'] for r in dv],
  frozen_state_before=initial,frozen_state_after=statehash(model.state_dict()),cache_model_probe_error=probe_error,
  no_deleted_original_frames=True,normal_channels_exact=True))
 return cache,None

def logits(feat,w):return feat[...,:100]+feat[...,100:]@w.T
def ce(zz,y):return float(torch.nn.functional.cross_entropy(torch.from_numpy(zz.reshape(-1,100)),torch.from_numpy(np.broadcast_to(y[:,None],zz.shape[:-1]).copy().reshape(-1))))
def eval_clean(dev,w,y):
 z=np.stack([logits(dev[:,vi,0,0],w) for vi in range(2)],1)
 return ce(z,y),[int((z[:,vi].argmax(1)==y).sum()) for vi in range(2)]
def predict_features(dev,w,rows):
 return {v:{kind:[prediction(r,z) for r,z in zip(rows,logits(dev[:,vi,0,ki],w))]
             for ki,kind in enumerate(KINDS)} for vi,v in enumerate(VIEWS)}

def train(seed,cache,tr,dv,ctx):
 target=U/f'BA_{seed}_RESULT.json'
 if target.exists():assert b.read(target)['protocol_sha256']==b.sha(U/'FROZEN.json');return
 ck=torch.load(b.T/f'T3F_{seed}_best.pt',map_location='cpu',weights_only=True)
 original={k:v.clone() for k,v in ck['state_dict'].items()}
 init=original['phase_weight'].numpy().copy();w=torch.nn.Parameter(torch.from_numpy(init.copy()))
 opt=torch.optim.AdamW([w],lr=LR,weight_decay=.001)
 train_y=np.array([r['label_index'] for r in tr],np.int64);dev_y=np.array([r['label_index'] for r in dv],np.int64)
 initial_ce,initial_hits=eval_clean(cache['dev'],init,dev_y)
 best=dict(epoch=0,ce=initial_ce,weight=init.copy());history=[];batch_count=0;start=time.monotonic()
 trace=U/f'BA_{seed}_TRACE.jsonl';assert not trace.exists()
 for epoch in range(1,EPOCHS+1):
  order=np.random.default_rng(seed+epoch).permutation(len(tr));total_loss=0.
  for step,j in enumerate(range(0,len(tr),BATCH),1):
   ii=order[j:j+BATCH];features=[];labels=[];trace_rows=[]
   for i in ii:
    row=tr[i];v,t=schedule(row['clip_id'],epoch,'C');vi=VIEWS.index(v);ti=TEMP.index(t)
    ki=1+int(hashlib.sha256(f'{seed}:{epoch}:{row["clip_id"]}'.encode()).hexdigest(),16)%3
    features.extend([cache['train'][i,vi,ti,0],cache['train'][i,vi,ti,ki]])
    labels.extend([int(train_y[i])]*2);trace_rows.append([row['clip_id'],v,t,KINDS[ki]])
   feat=torch.from_numpy(np.stack(features));yy=torch.tensor(labels,dtype=torch.long)
   opt.zero_grad(set_to_none=True);zz=feat[:,:100]+torch.nn.functional.linear(feat[:,100:],w)
   loss=torch.nn.functional.cross_entropy(zz,yy);assert torch.isfinite(loss);loss.backward()
   torch.nn.utils.clip_grad_norm_([w],1.);opt.step();total_loss+=float(loss.item())*len(ii);batch_count+=1
   with trace.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(epoch=epoch,batch=step,clips=trace_rows),ensure_ascii=False)+'\n')
  ww=w.detach().numpy().copy();val,hits=eval_clean(cache['dev'],ww,dev_y)
  if val<best['ce']-1e-6:best=dict(epoch=epoch,ce=val,weight=ww.copy())
  h=dict(epoch=epoch,train_ce=total_loss/899,clean_dev_ce=val,clean_hits=hits,selected_epoch=best['epoch'])
  history.append(h)
  if epoch%5==0:event(stage='AUX_TRAIN',seed=seed,**h)
 selected=copy.deepcopy(ck);selected['state_dict']['phase_weight']=torch.from_numpy(best['weight'].copy())
 selected.update(arm='BOUNDARY_AUX',epoch=best['epoch'],protocol_sha256=b.sha(U/'FROZEN.json'),original_t3f_sha256=b.sha(b.T/f'T3F_{seed}_best.pt'))
 assert all(torch.equal(v,selected['state_dict'][k]) for k,v in original.items() if k!='phase_weight')
 checkpoint=U/f'BA_{seed}_best.pt';assert not checkpoint.exists();torch.save(selected,checkpoint)
 # Baseline original score summaries retain authoritative saved logits. The
 # baseline synthetic scores have not been evaluated for seeds14/15 previously.
 after=predict_features(cache['dev'],best['weight'],dv)
 baseline_stress=predict_features(cache['dev'],init,dv)
 original_saved=b.read(b.T/f'T3F_{seed}_RESULT.json')['predictions']
 for v in VIEWS:
  baseline_stress[v]['clean']=original_saved[v]
  if seed==20261013:
   old=b.read(P/'T3F13_STRESS_SCORES.json')['scores'][v]
   for kind in KINDS[1:]:baseline_stress[v][kind]=old[kind]
 write(f'BA_{seed}_history.json',history)
 write(target.name,dict(status='COMPLETE_DEVELOPMENT_ONLY',seed=seed,protocol_sha256=b.sha(U/'FROZEN.json'),
   selected_epoch=best['epoch'],selected_ce=best['ce'],initial_ce=initial_ce,initial_cache_clean_hits=initial_hits,
   completed_epochs=EPOCHS,batches=batch_count,original_clip_exposures=899*EPOCHS,
   training_examples=899*EPOCHS*2,checkpoint_sha256=b.sha(checkpoint),trace_sha256=b.sha(trace),
   predictions=after,baseline=baseline_stress,frozen_C_tensors_equal=True,head_changed=not np.array_equal(init,best['weight']),
   training_seconds=time.monotonic()-start,cache_sha256=b.sha(U/f'features_{seed}.npz'),
   user_data_used=False,production_changed=False))
 event(stage='SEED_DONE',seed=seed,selected_epoch=best['epoch'])

def main():
 U.mkdir(parents=True,exist_ok=True)
 if '--freeze' in sys.argv:freeze();return
 checks();assert not (U/'RUNNING.json').exists()
 write('RUNNING.json',dict(pid=os.getpid(),at=time.time()));start=time.monotonic()
 try:
  torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
  ctx,tr,dv=load_source_rows()
  for seed in SEEDS:
   cache,_=prepare_cache(seed,tr,dv);train(seed,cache,tr,dv,ctx)
  checks();write('DONE.json',dict(status='COMPLETE_PENDING_VERIFICATION',seconds=time.monotonic()-start))
  (U/'RUNNING.json').unlink()
 except BaseException:
  if not (U/'FIRST_ERROR.json').exists():write('FIRST_ERROR.json',dict(at=time.time(),traceback=traceback.format_exc()))
  raise
if __name__=='__main__':main()
