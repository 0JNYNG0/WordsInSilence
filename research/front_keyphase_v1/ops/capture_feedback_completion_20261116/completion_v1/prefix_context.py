"""Recover unknown states near an actual recording end using public prefix training."""
from train_terminal_relation import *

class PrefixRelation(RelationObserver):
 def __init__(self):
  super().__init__();self.extra=[]
  for seed in[42,43,44]:
   ck=torch.load(V/'prefix_context'/f'{seed}.pt',map_location='cpu',weights_only=False);m=PhaseNet(len(ck['mean']),'frame_mlp');m.load_state_dict(ck['state']);m.eval();self.extra.append((m,ck['mean'],ck['std']))
 def __call__(self,f,q,ts):
  old,_=super().__call__(f,q,ts);pp=[]
  with torch.inference_mode():
   for m,mu,sd in self.extra:pp.append(states(m(torch.tensor(((f-mu)/sd)[None])).softmax(-1)[0].numpy(),q))
  pp=np.stack(pp);p=np.where((pp==pp[0]).all(0),pp[0],-1)
  scope=np.arange(len(f))>=max(0,len(f)-7)
  merged=np.where((old<0)&scope,p,old)
  return merged,approaches(merged,ts)

def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);out=V/'prefix_context';out.mkdir(exist_ok=True);assert not(out/'FROZEN.json').exists()
 truth=read(B/'source_corrected/SOURCE_TRUTH.json')['rows'];records=read(B/'INPUTS.json')['rows'];ff=Features();xx=[];yy=[];provenance=[]
 for r in records:
  if r['role']!='fit':continue
  ch=dict(np.load(r['channels']));n=len(ch['frame_id']);known=[a for a in truth if a['clip_id']==r['clip_id']];f,q,_=ff(ch)
  for a in known:
   if q[a['frame']]:xx.append(f[a['frame']]);yy.append(a['state']);provenance.append(dict(clip=r['clip_id'],frame=a['frame'],end=n-1,state=a['state']))
  for cls in[0,1,2]:
   aa=[a for a in known if a['state']==cls]
   if not aa:continue
   for a in {a['frame']:a for a in[aa[0],aa[-1]]}.values():
    for offset in[0,3,6]:
     end=min(n-1,a['frame']+offset);prefix={k:(v[:end+1]if v.ndim and len(v)==n else v)for k,v in ch.items()};f,q,_=ff(prefix)
     if q[a['frame']]:xx.append(f[a['frame']]);yy.append(cls);provenance.append(dict(clip=r['clip_id'],frame=a['frame'],end=end,state=cls))
 xraw=np.array(xx,np.float32);target=np.array(yy,np.int64);write(out/'INPUTS.json',dict(rows=provenance))
 write(out/'FROZEN.json',dict(steps=400,lr=.0002,seeds=[42,43,44],confidence=.65,fit_people=list(range(1,10)),
  source_sha=sha(B/'source_corrected/SOURCE_TRUTH.json'),inputs_sha=sha(out/'INPUTS.json'),
  recovery='Only old unknown states in last7 actually recorded frames. All previously decided states retained. No future pose fill.',
  diagnostic='Initial endpoint-only continuation did not recover pose just before cut; supervised anchor with0/3/6real future frames covers this omitted input condition.',
  no_word_identity=True,no_dev_training=True,acceptance='No new definite DEV state errors, preserve normal2, observefirst1, constant/missing0. Then fresh source confirmation.'))
 w=torch.tensor(len(target)/(3*np.bincount(target,minlength=3)),dtype=torch.float32);ty=torch.tensor(target);runs=[]
 for seed in[42,43,44]:
  ck=torch.load(B/'source_corrected/runs'/f'frame_mlp_{seed}.pt',map_location='cpu',weights_only=False);torch.manual_seed(seed);net=PhaseNet(len(ck['mean']),'frame_mlp');net.load_state_dict(ck['state']);x=torch.tensor((xraw-ck['mean'])/ck['std']);opt=torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=.0001);t=time.perf_counter()
  for _ in range(400):
   loss=F.cross_entropy(net(x[None])[0],ty,weight=w);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
  net.eval();p=out/f'{seed}.pt';torch.save(dict(state=net.state_dict(),mean=ck['mean'],std=ck['std']),p);runs.append(dict(seed=seed,seconds=time.perf_counter()-t,sha256=sha(p)))
 write(out/'TRAINED.json',dict(runs=runs))
 # Reuse exactly the prior comparison without editing that frozen result.
 import evaluate_parts as e
 e.TerminalRelation=PrefixRelation
 original_write=e.write
 def scoped_write(path,value):
  original_write(out/'COMPARISON.json' if Path(path).name=='COMPARISON.json' else path,value)
 e.write=scoped_write;e.relation_check()
if __name__=='__main__':main()
