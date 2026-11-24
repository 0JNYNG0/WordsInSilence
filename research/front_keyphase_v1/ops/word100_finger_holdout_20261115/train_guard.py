from holdout_common import *
import torch,time,collections
from torch.nn import functional as F
sys.path.insert(0,str(D));from finger_core import FingerHead
cfg=read(U/'TRAIN_FROZEN.json');assert not(U/'TRAINED.json').exists()
for p,h in cfg['pins'].items():assert sha(p)==h
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);data=np.load(U/'TRAIN_INPUTS.npz');rows=read(U/'TRAIN_INPUTS.json')['rows'];fi=np.array([i for i,r in enumerate(rows)if r['role']=='fit']);assert all(rows[i]['person']in range(1,10)for i in fi);y=data['y'];mask=(y>=0)&data['availability'];counts=collections.Counter((rows[i]['clip_id'],f,int(y[i,f]))for i in fi for f in range(4)if mask[i,f]);classes=collections.Counter((f,t)for c,f,t in counts);w=np.zeros(y.shape,np.float32)
for i in fi:
 for f in range(4):
  if mask[i,f]:w[i,f]=1/(counts[rows[i]['clip_id'],f,int(y[i,f])]*classes[f,int(y[i,f])])
weights=torch.tensor(w[fi]);weights/=weights.sum();ty=torch.tensor(y.clip(0),dtype=torch.float32);outdir=U/'guard_models';outdir.mkdir(exist_ok=True);runs=[]
for seed in cfg['seeds']:
 torch.manual_seed(seed);ck=torch.load(P/'expanded_whole_hand'/f'{seed}.pt',map_location='cpu',weights_only=False);m=FingerHead();m.load_state_dict(ck['state_dict']);mu=np.array(ck['mean']);sd=np.array(ck['std']);xx=(data['x']-mu)/sd;xx[:,:,63:]=0;x=torch.tensor(xx,dtype=torch.float32);opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);start=time.perf_counter()
 for step in range(cfg['steps']):
  p=m(x[fi])[:,1:];loss=(F.binary_cross_entropy_with_logits(p,ty[fi],reduction='none')*weights).sum();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
 m.eval()
 with torch.inference_mode():prob=m(x)[:,1:].sigmoid().numpy()
 path=outdir/f'{seed}.pt';torch.save(dict(state_dict=m.state_dict(),mean=mu,std=sd),path);np.savez_compressed(outdir/f'{seed}_predictions.npz',probabilities=prob);runs.append(dict(seed=seed,seconds=time.perf_counter()-start,model=str(path),sha256=sha(path),final_loss=float(loss.detach())));print(runs[-1],flush=True)
write(U/'TRAINED.json',dict(runs=runs,trained_fit_anchors=len(fi),additional_anchors=read(U/'TRAIN_INPUTS.json')['new_anchors'],common100_unchanged=True,C_T3F_unchanged=True,user_recordings_used=False))
