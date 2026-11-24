from common import *
import torch,time,collections
from torch.nn import functional as F
sys.path.insert(0,str(P));from finger_core import FingerHead
from finger_model import SharedFinger
cfg=read(U/'TRAIN_FROZEN.json')
for p,h in cfg['pins'].items():assert sha(p)==h
assert not(U/'TRAINED.json').exists()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
data=np.load(U/'INPUTS.npz');rows=read(U/'INPUTS.json')['rows'];y=data['y'];mask=(y>=0)&data['availability'];fi=np.array([i for i,r in enumerate(rows)if r['role']=='fit']);di=np.array([i for i,r in enumerate(rows)if r['role']=='development']);ty=torch.tensor(y.clip(0),dtype=torch.float32)
counts=collections.Counter((rows[i]['clip_id'],f,int(y[i,f]))for i in fi for f in range(4)if mask[i,f]);classes=collections.Counter((f,t)for c,f,t in counts);w=np.zeros(y.shape,np.float32)
for i in fi:
 for f in range(4):
  if mask[i,f]:w[i,f]=1/(counts[rows[i]['clip_id'],f,int(y[i,f])]*classes[f,int(y[i,f])])
weights=torch.tensor(w[fi]);weights/=weights.sum();runs=[]
def metrics(pred,ids):
    yy=y[ids];mm=mask[ids];pp=pred[ids];old=data['baseline'][ids]
    return dict(known=int(mm.sum()),answered=int(((pp>=0)&mm).sum()),correct=int(((pp==yy)&mm).sum()),wrong=int(((pp>=0)&(pp!=yy)&mm).sum()),baseline_correct=int(((old==yy)&mm).sum()),old_correct_to_wrong=int(((old==yy)&(pp>=0)&(pp!=yy)&mm).sum()))
for kind in cfg['candidates']:
 if kind=='expanded_whole_hand':xx=(data['x_old']-data['old_mean'])/data['old_std'];xx[:,:,63:]=0
 else:xx=(data['x']-data['mean'])/data['std']
 x=torch.tensor(xx,dtype=torch.float32);outdir=U/kind;outdir.mkdir(exist_ok=True)
 for seed in cfg['seeds']:
  torch.manual_seed(seed);model=FingerHead()if kind=='expanded_whole_hand'else SharedFinger();opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);t=time.perf_counter()
  for step in range(cfg['steps']):
   out=model(x[fi]);out=out[:,1:]if kind=='expanded_whole_hand'else out;loss=(F.binary_cross_entropy_with_logits(out,ty[fi],reduction='none')*weights).sum();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
  model.eval()
  with torch.inference_mode():prob=model(x).sigmoid().numpy();prob=prob[:,1:]if kind=='expanded_whole_hand'else prob
  pred=np.where(prob>=cfg['confidence'],1,np.where(prob<=1-cfg['confidence'],0,-1));pred[~data['availability']]=-1
  p=outdir/f'{seed}.pt';torch.save(dict(state_dict=model.state_dict(),kind=kind,mean=data['mean']if kind=='shared_finger_chain'else data['old_mean'],std=data['std']if kind=='shared_finger_chain'else data['old_std']),p);np.savez_compressed(outdir/f'{seed}_predictions.npz',probabilities=prob,predictions=pred)
  olddi=np.array([i for i in di if not rows[i]['new']]);newdi=np.array([i for i in di if rows[i]['new']]);pattern=np.array([i for i in di if y[i,2]>=0 and y[i,3]>=0 and y[i,2]!=y[i,3]])
  r=dict(kind=kind,seed=seed,model=str(p),sha256=sha(p),seconds=time.perf_counter()-t,old_development=metrics(pred,olddi),added_development=metrics(pred,newdi),distinct_ring_little=metrics(pred,pattern),fit_only_optimizer=True);runs.append(r);print(json.dumps(r),flush=True)
 write(U/'TRAIN_PROGRESS.json',dict(runs=runs))
write(U/'TRAINED.json',dict(runs=runs,confidence=cfg['confidence'],no_dev_selection=True,common100_changed=False,C_T3F_changed=False))
