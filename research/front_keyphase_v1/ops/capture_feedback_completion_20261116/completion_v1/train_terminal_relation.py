"""Teach observed terminal poses without treating every clip end as background.

Only public FIT RGB-reviewed anchors are used, including background, separated,
and near endpoints. No synthetic arrival is labelled from DTW or model output.
"""
from train_shape import *
from core import Features,PhaseNet,states,approaches
from basic_runtime import RelationObserver

class TerminalRelation(RelationObserver):
 def __init__(self):
  self.models=[]
  for seed in[42,43,44]:
   ck=torch.load(V/'terminal_relation'/f'{seed}.pt',map_location='cpu',weights_only=False);m=PhaseNet(len(ck['mean']),'frame_mlp');m.load_state_dict(ck['state']);m.eval();self.models.append((m,ck['mean'],ck['std']))

def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);out=V/'terminal_relation';out.mkdir(exist_ok=True)
 assert not(out/'FROZEN.json').exists()
 truth=read(B/'source_corrected/SOURCE_TRUTH.json')['rows'];records=read(B/'INPUTS.json')['rows'];ff=Features();xx=[];yy=[];provenance=[]
 for r in records:
  if r['role']!='fit':continue
  ch=dict(np.load(r['channels']));n=len(ch['frame_id']);known=[a for a in truth if a['clip_id']==r['clip_id']]
  f,q,_=ff(ch)
  for a in known:
   if q[a['frame']]:xx.append(f[a['frame']]);yy.append(a['state']);provenance.append(dict(clip_id=r['clip_id'],frame=a['frame'],state=a['state'],view='full'))
  # At most first and last reviewed endpoint in each class, fixed before DEV.
  for cls in[0,1,2]:
   aa=[a for a in known if a['state']==cls]
   if not aa:continue
   for a in {a['frame']:a for a in[aa[0],aa[-1]]}.values():
    end=a['frame'];prefix={k:(v[:end+1]if v.ndim and len(v)==n else v)for k,v in ch.items()}
    f,q,_=ff(prefix)
    if q[-1]:xx.append(f[-1]);yy.append(cls);provenance.append(dict(clip_id=r['clip_id'],frame=end,state=cls,view='actual_prefix'))
 xraw=np.array(xx,np.float32);target=np.array(yy,np.int64);write(out/'INPUTS.json',dict(rows=provenance,truth_sha=sha(B/'source_corrected/SOURCE_TRUTH.json')))
 write(out/'FROZEN.json',dict(steps=300,lr=.0002,seeds=[42,43,44],confidence=.65,
   parent={str(B/'source_corrected/runs'/f'frame_mlp_{s}.pt'):sha(B/'source_corrected/runs'/f'frame_mlp_{s}.pt')for s in[42,43,44]},
   fit_people=list(range(1,10)),input_sha=sha(out/'INPUTS.json'),existing100_changed=False,
   acceptance='Preserve whole normal2 and recover witnessed first-arrival1 without constant/missing arrivals. No extra definite DEV state errors. This is not complete KSL feedback admission.',
   diagnosis='Frozen BiGRU context changes at terminal prefixes; observed endpoint targets were absent in this context. No future padding to invent evidence.'))
 weight=np.bincount(target,minlength=3);w=torch.tensor(len(target)/(3*weight),dtype=torch.float32);ty=torch.tensor(target);runs=[]
 for seed in[42,43,44]:
  ck=torch.load(B/'source_corrected/runs'/f'frame_mlp_{seed}.pt',map_location='cpu',weights_only=False);torch.manual_seed(seed);net=PhaseNet(len(ck['mean']),'frame_mlp');net.load_state_dict(ck['state']);x=torch.tensor((xraw-ck['mean'])/ck['std']);opt=torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=.0001);start=time.perf_counter()
  for _ in range(300):
   loss=F.cross_entropy(net(x[None])[0],ty,weight=w);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
  net.eval();path=out/f'{seed}.pt';torch.save(dict(state=net.state_dict(),mean=ck['mean'],std=ck['std']),path);runs.append(dict(seed=seed,seconds=time.perf_counter()-start,sha256=sha(path)))
 write(out/'TRAINED.json',dict(runs=runs,rows=len(yy),fit_only=True));print(json.dumps(runs))
if __name__=='__main__':main()
