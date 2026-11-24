from core import *
import time

def main():
 torch.set_num_threads(2);rows=read(U/'INPUTS.json')['rows'];truth=read(U/'SOURCE_TRUTH.json')['rows'];cache={}
 (U/'cache').mkdir(exist_ok=True);factory=None
 for r in rows:
  cp=U/'cache'/f'{r["clip_id"]}.npz'
  if not cp.exists():
   if factory is None:factory=Features()
   ch=dict(np.load(r['channels']));f,q,coarse=factory(ch)
   y=np.full(len(f),-1,np.int64)
   for a in truth:
    if a['clip_id']==r['clip_id']:y[a['frame']]=a['state']
   np.savez_compressed(cp,f=f,q=q,coarse=coarse,y=y,ts=ch['timestamp_us'])
  cache[r['clip_id']]=dict(np.load(cp))
 fit=[cache[r['clip_id']] for r in rows if r['role']=='fit'];allf=np.concatenate([r['f'] for r in fit]);mu=allf.mean(0);sd=allf.std(0).clip(.05)
 weight=np.bincount(np.concatenate([r['y'][r['y']>=0] for r in fit]),minlength=3);weight=torch.tensor(weight.sum()/weight,dtype=torch.float32);weight/=weight.mean()
 config=read(U/'TRAIN_FROZEN.json');logs=[]
 for kind in config['architectures']:
  for seed in config['seeds']:
   dest=U/'runs'/f'{kind}_{seed}.pt'
   if dest.exists():raise RuntimeError('Completed training artifact exists; do not rerun')
   torch.manual_seed(seed);rng=np.random.default_rng(seed);model=PhaseNet(len(mu),kind);opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay']);start=time.perf_counter()
   for step in range(config['steps']):
    opt.zero_grad();loss=0
    for idx in rng.choice(len(fit),3,replace=False):
     r=fit[idx];x=(r['f']-mu)/sd;x=x+rng.normal(0,.015,x.shape).astype(np.float32)
     logits=model(torch.tensor(x[None]))[0];mask=(r['y']>=0)&r['q']
     loss+=nn.functional.cross_entropy(logits[mask],torch.tensor(r['y'][mask]),weight=weight)/3
    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
   model.eval();torch.save(dict(state=model.state_dict(),mean=mu,std=sd,kind=kind,seed=seed,steps=config['steps'],fit_people=list(range(1,10)),source_truth_sha=sha(U/'SOURCE_TRUTH.json')),dest)
   logs.append(dict(kind=kind,seed=seed,seconds=time.perf_counter()-start,loss=float(loss),sha256=sha(dest)))
   write(U/'TRAINED.json',{'runs':logs,'existing100_weights_changed':False});print('TRAINED',kind,seed,round(logs[-1]['seconds'],2),flush=True)
 comparisons=[]
 for kind in config['architectures']:
  obs=Observer(kind)
  for r in rows:
   z=cache[r['clip_id']];p=obs(z['f'],z['q'],z['ts']);ids=np.flatnonzero(z['y']>=0)
   out=dict(clip_id=r['clip_id'],split=r['role'],person=r['person'],kind=kind,anchors=len(ids),correct=int((p['states'][ids]==z['y'][ids]).sum()),wrong=int(((p['states'][ids]>=0)&(p['states'][ids]!=z['y'][ids])).sum()),unknown=int((p['states'][ids]<0).sum()),approaches=p['approaches'],per_anchor=[dict(frame=int(i),gold=int(z['y'][i]),pred=int(p['states'][i]))for i in ids])
   comparisons.append(out);np.savez_compressed(U/'results'/f'{r["clip_id"]}_{kind}.npz',**{k:v for k,v in p.items()if k!='approaches'})
 write(U/'DEVELOPMENT.json',dict(rows=comparisons,training_runs=6,scope='Sparse relation state discrimination; not full feedback accuracy',preexisting='Existing4-only action binder has no gratitude clauses; not assigned fabricated 0% score'))
 print('DEVELOPMENT COMPLETE')
if __name__=='__main__':main()
