"""FIT-grounded corrective experiment: an antisymmetric guide/event pairing.

This cannot say both A-before-B and B-before-A match the same observed endpoints.
It does not decide episode completeness, number of repetitions, or hidden contact.
"""
from requirement_core import *
import time,os,traceback
class OrderedBinding(nn.Module):
 def __init__(self):
  super().__init__();self.observe=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,24))
  self.guide=nn.Sequential(nn.Linear(28,32),nn.GELU(),nn.Linear(32,24))
 def forward(self,ob,req):
  ha=self.observe(ob[:,:64]);hb=self.observe(ob[:,128:192])
  qa=self.guide(req[:,20:48]);qb=self.guide(req[:,48:76])
  return ((ha-hb)*(qa-qb)).sum(-1)/(24**.5)

def main():
 assert not (U/'ORDER_BINDING_COMPLETE.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 cfg=read(U/'FROZEN.json');z=np.load(U/'train.npz');manifest=read(U/'TRAIN_MANIFEST.json')
 ix=np.array([i for i,m in enumerate(manifest) if m['task']=='order' and m['eligible']],int)
 impossible=((z['base'][ix]>2)&(z['y'][ix]==0))|((z['base'][ix]<-2)&(z['y'][ix]==1))
 # Bound analysis uses FIT only: +/-2 residual cannot cross zero for these wrong priors.
 fit_audit=dict(order_training_queries=len(ix),wrong_prior_unfixable_under_residual_bound=int(impossible.sum()),source='FIT train.npz, algebraic maximum correction bound2; no evaluation target used',fix='Independent antisymmetric ordered binding, not another residual of the old confident score',feature_scope='Cached frozen100-word Conv+GRU endpoints and structured guide descriptors',new_seeds=cfg['seeds'],epochs=40,lr=.001,thresholds=[.35,.65],selection='No seed selection; same historical DEV is exploratory, not fresh confirmation')
 assert impossible.sum()>0
 write(U/'ORDER_FIT_CAUSE.json',fit_audit);write(U/'ORDER_BINDING_FROZEN.json',dict(fit_audit,code_sha256=sha(__file__),data_sha256=sha(U/'train.npz')))
 ob=torch.tensor(z['ob'][ix]);req=torch.tensor(z['req'][ix]);y=torch.tensor(z['y'][ix]);training=[];scores=[]
 gold=[r for r in read(U/'SCORES.json') if r['seed']==42 and r['task']=='order']
 sources=read(R/'EVALUATION_INPUTS.json')['rows'];devcache={q['clip_id']:q for q in sources if q['kind']=='full_normal'}
 for seed in cfg['seeds']:
  torch.manual_seed(seed);rng=np.random.default_rng(seed);n=OrderedBinding();opt=torch.optim.AdamW(n.parameters(),lr=.001,weight_decay=.01);losses=[];start=time.perf_counter()
  write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING_ORDER_BINDING_FIT_CORRECTION',pid=os.getpid(),seed=seed))
  for epoch in range(40):
   ls=[]
   for ii in np.array_split(rng.permutation(len(ix)),max(1,int(np.ceil(len(ix)/64)))):
    loss=nn.functional.binary_cross_entropy_with_logits(n(ob[ii],req[ii]),y[ii]);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(n.parameters(),2);opt.step();ls.append(float(loss.detach()))
   losses.append(float(np.mean(ls)))
  p=U/'order_runs'/str(seed)/'model.pt';p.parent.mkdir(parents=True,exist_ok=True);torch.save(n.state_dict(),p);n.eval()
  training.append(dict(seed=seed,path=str(p),sha256=sha(p),seconds=time.perf_counter()-start,loss=losses,parameters=sum(v.numel() for v in n.parameters())))
  with torch.inference_mode():
   for r in gold:
    c=np.load(devcache[r['clip_id']]['path']);a,b=r['a'],r['b'];h=c['fh'];oi=torch.tensor(observation(h,a,b)[None]);g=r['guides']
    for swapped in [False,True]:
     gg=g[::-1] if swapped else g;qi=torch.tensor(requirement('order',guides=gg)[None]);log=n(oi,qi);pr=float(log.sigmoid()[0]);scores.append(dict(seed=seed,clip_id=r['clip_id'],pair=r['pair'],swapped=swapped,target=int(not swapped),probability=pr,state=decide(pr),guide_ids=gg,a=a,b=b,not_filmed_learner_error=True))
   # Same endpoint, swapped guide, and missing observation: pure contract tests.
   a=ob[:1];q=req[:1];qs=q.clone();qs[:,20:48]=q[:,48:76];qs[:,48:76]=q[:,20:48]
   anti=float((n(a,q)+n(a,qs)).abs().max());same=a.clone();same[:,128:192]=same[:,:64];identical=float(n(same,q).abs().max())
   training[-1].update(antisymmetry_error=anti,identical_endpoint_logit=identical)
  print(seed,losses[-1],flush=True)
 write(U/'ORDER_BINDING_TRAINING.json',training);write(U/'ORDER_BINDING_SCORES.json',scores)
 write(U/'ORDER_BINDING_COMPLETE.json',dict(training_queries=len(ix),fit_prior_unfixable=int(impossible.sum()),summary={str(seed):{str(sw):dict(n=sum(r['seed']==seed and r['swapped']==sw for r in scores),correct=sum((r['probability']>=.5)==bool(r['target']) for r in scores if r['seed']==seed and r['swapped']==sw),withheld=sum(r['state']=='uncertain' for r in scores if r['seed']==seed and r['swapped']==sw)) for sw in [False,True]} for seed in cfg['seeds']},not_fullpath_deployed=True))
 print(read(U/'ORDER_BINDING_COMPLETE.json'))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'ORDER_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
