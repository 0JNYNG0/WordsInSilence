from pathlib import Path
import json,time,hashlib,collections
import numpy as np,torch
from torch.nn import functional as F
from finger_core import FingerHead,FINGERS
U=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=read(U/'FINGER_FROZEN.json');assert sha(U/'finger_arrays.npz')==cfg['inputs_sha256'];assert not(U/'FINGER_RESULTS.json').exists()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
data=np.load(U/'finger_arrays.npz');rows=read(U/'FINGER_INPUTS.json')['rows'];x=torch.tensor((data['x']-data['mean'])/data['std']);truth=data['targets'];mask=(truth>=0)&data['availability'];y=torch.tensor(truth.clip(0),dtype=torch.float32);fi=np.array([i for i,r in enumerate(rows)if r['role']=='fit']);di=np.array([i for i,r in enumerate(rows)if r['role']=='development']);base=data['baseline']
count=collections.Counter((rows[i]['clip_id'],j,int(truth[i,j]))for i in fi for j in range(5)if mask[i,j]);classes=collections.Counter((j,t)for c,j,t in count);weights=np.zeros_like(truth,dtype=np.float32)
for i in fi:
 for j in range(5):
  if mask[i,j]:weights[i,j]=1/(count[rows[i]['clip_id'],j,int(truth[i,j])]*classes[j,int(truth[i,j])])
w=torch.tensor(weights[fi]);w/=w.sum()
def summary(pred,ix):
 m=mask[ix];old=base[ix];new=pred[ix];return dict(known=int(m.sum()),old_answered=int(((old>=0)&m).sum()),old_correct=int(((old==truth[ix])&m).sum()),new_answered=int(((new>=0)&m).sum()),new_correct=int(((new==truth[ix])&m).sum()),new_wrong=int(((new>=0)&(new!=truth[ix])&m).sum()),old_correct_regressions=int(((old==truth[ix])&(new!=truth[ix])&m).sum()),wrong_vs_correct=int(((old==truth[ix])&(new>=0)&(new!=truth[ix])&m).sum()))
runs=[];dest=U/'finger_models';dest.mkdir(exist_ok=True)
for seed in cfg['seeds']:
 torch.manual_seed(seed);m=FingerHead();opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);start=time.perf_counter();initial={k:v.clone()for k,v in m.state_dict().items()}
 for step in range(cfg['finger_steps']):
  out=m(x[fi]);loss=(F.binary_cross_entropy_with_logits(out,y[fi],reduction='none')*w).sum();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
 m.eval()
 with torch.inference_mode():pr=m(x).sigmoid().numpy()
 pred=np.where(pr>=cfg['confidence'],1,np.where(pr<=1-cfg['confidence'],0,-1));pred[~data['availability']]=-1
 # Individual thumb output is not licensed for ordinary words: no DEV thumb truth.
 pred[:,0]=-1
 p=dest/f'{seed}.pt';torch.save(dict(state_dict=m.state_dict(),mean=data['mean'].tolist(),std=data['std'].tolist(),supported_fingers=FINGERS[1:]),p);np.savez_compressed(dest/f'{seed}_predictions.npz',probabilities=pr,predictions=pred)
 s=summary(pred,di);run=dict(seed=seed,seconds=time.perf_counter()-start,model=str(p),sha256=sha(p),weights_changed=any(not torch.equal(initial[k],v)for k,v in m.state_dict().items()),development=s,by_finger={f:dict(known=int(mask[di,j].sum()),correct=int(((pred[di,j]==truth[di,j])&mask[di,j]).sum()),wrong=int(((pred[di,j]>=0)&(pred[di,j]!=truth[di,j])&mask[di,j]).sum()))for j,f in enumerate(FINGERS)},new_review=summary(pred,[i for i in di if'current_individual_finger_RGB_review'in rows[i]['origins']]),eligible_for_integration=s['old_correct_regressions']==0 and s['new_wrong']==0)
 runs.append(run);print(json.dumps(run),flush=True)
write(U/'FINGER_RESULTS.json',dict(runs=runs,training_steps=cfg['finger_steps'],baseline='Frozen deployed attribute head, conservatively converted to individual-finger evidence at .85; point test, not end-to-end web accuracy',common100_changed=False,web_changed=False,thumb_disabled=True,unseen_person_not_unseen_test=True))
