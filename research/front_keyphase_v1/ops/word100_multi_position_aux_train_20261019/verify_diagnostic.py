"""Independent NumPy replay of trained GRU/readout using previously saved frozen TCN features."""
from common import *
import torch
def reference(features,w,head,dt):
 state=np.zeros(16);sequence=[]
 for frame in features:
  i=w['weight_ih_l0']@frame+w['bias_ih_l0'];h=w['weight_hh_l0']@state+w['bias_hh_l0']
  r=1/(1+np.exp(-i[:16]-h[:16]));z=1/(1+np.exp(-i[16:32]-h[16:32]));n=np.tanh(i[32:]+r*h[32:])
  state=(1-z)*n+z*state;sequence.append(state.copy())
 sequence=np.array(sequence);elapsed=np.cumsum(dt.astype('float64'));u=(elapsed-elapsed[0])/(elapsed[-1]-elapsed[0]) if elapsed[-1]>elapsed[0] else np.full(len(dt),.5)
 weights=np.stack([np.maximum(1-2*u,0),1-np.abs(2*u-1),np.maximum(2*u-1,0)],axis=1)
 parts=[]
 for col in weights.T:parts.append((sequence*col[:,None]).sum(0)/col.sum() if col.sum()>0 else sequence.mean(0))
 return head@np.concatenate(parts)
assert not (U/'DIAGNOSTIC_INDEPENDENT.json').exists()
r=read(U/'DIAGNOSTIC.json');old=read(D/'COMPLETE.json');assert sha(D/'NUMERIC_SAMPLES.npz')==old['numeric_samples_sha256']
assert sha(U/'DIAGNOSTIC_SCORES.npz')==r['scores_sha256'] and sha(U/'DIAGNOSTIC_PLAN.json')==r['plan_sha256']
with np.load(D/'NUMERIC_SAMPLES.npz',allow_pickle=False) as z:features={k:z[k].astype('float64') for k in z.files if k.startswith('h')}
with np.load(U/'DIAGNOSTIC_SCORES.npz',allow_pickle=False) as z:scores=z['aux'].astype('float64');ids=z['ids'];ys=z['labels']
ctx=read(U/'TRAIN_CONTEXT.json');train=sorted((x for x in ctx['rows'] if x['role']=='train_candidate'),key=lambda q:q['clip_id'])
sampleidx=read(D/'SCOPE.json')['sample_indices'];maxerr=0.;count=0
for si,seed in enumerate(SEEDS):
 ck=torch.load(U/f'{ARM}_{seed}_best.pt',map_location='cpu',weights_only=True)
 w={k.removeprefix('memory.'):v.numpy().astype('float64') for k,v in ck['state_dict'].items() if k.startswith('memory.')};head=ck['state_dict']['residual.weight'].numpy().astype('float64')
 for vi,v in enumerate(VIEWS):
  for ri in sampleidx:
   assert ids[ri]==train[ri]['clip_id'];h=features[f'h{si}_{vi}_{ri}'];n=len(h)
   with np.load(train[ri]['views'][v]['channel_file'],allow_pickle=False) as z:dt=z['x_whole'][:,279].astype('float64')
   assert len(dt)==n
   q,t=divmod(n,3);x=q+(t>0);y=2*q+(t>0)+(t>1);first=list(range(x));middle=list(range(x,y));last=list(range(y,n))
   for ci,order in enumerate([list(range(n)),middle+first+last,last+middle+first,last]):
    clock=dt[order].copy() if ci==3 else dt.copy();clock[0]=0
    pred=reference(h[order],w,head,clock);err=float(np.max(np.abs(pred-scores[si,vi,ri,ci])));assert err<2e-5,err
    maxerr=max(err,maxerr);count+=1
  for ci,cond in enumerate(read(U/'DIAGNOSTIC_PLAN.json')['conditions']):
   row=next(x for x in r['rows'] if x['seed']==seed and x['view']==v and x['condition']==cond)
   a=scores[si,vi,:,ci];o=scores[si,vi,:,0]
   assert row['same_aux_class']==int((a.argmax(-1)==o.argmax(-1)).sum()) and row['aux_label_hits']==int((a.argmax(-1)==ys).sum())
   ac=a-a.mean(-1,keepdims=True);oc=o-o.mean(-1,keepdims=True)
   relative=np.sqrt(((ac-oc)**2).mean(-1))/np.sqrt((oc**2).mean(-1))
   assert abs(row['relative_centered_rms_median_pct']-100*np.median(relative))<1e-10
check(ctx['pins'])
write(U/'DIAGNOSTIC_INDEPENDENT.json',{'status':'PASS','trained_GRU_numpy_checks':count,'max_abs_logit_error':maxerr,'saved_diagnostic_rows_recounted':24,
 'reused_frozen_features':str(D/'NUMERIC_SAMPLES.npz'),'reused_features_sha256':sha(D/'NUMERIC_SAMPLES.npz'),
 'diagnostic_sha256':sha(U/'DIAGNOSTIC.json'),'verifier_sha256':sha(__file__),'additional_model_inference':0,'training_steps':0,'user_data_read':0},new=True)
print('PASS: trained NumPy GRU/readout240 checks and diagnostic metric recount, max error',maxerr)
