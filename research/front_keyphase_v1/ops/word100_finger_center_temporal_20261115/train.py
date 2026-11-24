from model import *
import time,random
cfg=read(U/'TRAIN_CONFIG.json');assert not(U/'TRAINED.json').exists()
for p,h in cfg['pins'].items():assert sha(p)==h
torch.set_num_threads(2);data=np.load(U/'TRAIN_INPUTS.npz');runs=[]
for kind in ['state','phase']:
 for seed in cfg['seeds']:
  torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);t=time.perf_counter();ck=torch.load(P/'shared_finger_chain'/f'{seed}.pt',map_location='cpu',weights_only=False);mu=np.array(ck['mean'],np.float32);sd=np.array(ck['std'],np.float32)
  model=SharedFinger()if kind=='state'else ChangeHead()
  if kind=='state':model.load_state_dict(ck['state_dict'])
  X=data[kind+'_x'];X=X[:,3:4]if kind=='state'else X;X=torch.tensor((X-mu)/sd,dtype=torch.float32);Y=torch.tensor(data[kind+'_y'].clip(0),dtype=torch.float32);mask=torch.tensor(data[kind+'_mask'],dtype=torch.float32)
  optim=torch.optim.AdamW(model.parameters(),lr=cfg[kind+'_lr'],weight_decay=cfg['weight_decay']);logs=[]
  if kind=='state':old=torch.tensor(~data['is_new']);new=~old
  else:
   pos=(Y*mask).sum();neg=((1-Y)*mask).sum();weights=torch.where(Y>.5,mask.sum()/(2*pos.clamp(min=1)),mask.sum()/(2*neg.clamp(min=1)))
  for step in range(cfg[kind+'_steps']):
   model.train();optim.zero_grad();logits=model(X);losses=torch.nn.functional.binary_cross_entropy_with_logits(logits,Y,reduction='none')
   if kind=='state':loss=(losses[old]*mask[old]).sum()/mask[old].sum()+cfg['new_weight']*(losses[new]*mask[new]).sum()/mask[new].sum()
   else:loss=(losses*mask*weights).sum()/mask.sum()
   assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.0);optim.step()
   if step%100==0 or step==cfg[kind+'_steps']-1:logs.append(dict(step=step+1,loss=float(loss.detach())))
  model.eval()
  with torch.inference_mode():probs=model(X).sigmoid().numpy()
  path=U/'models'/kind/f'{seed}.pt';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();torch.save(dict(state_dict=model.state_dict(),mean=mu,std=sd,kind=kind,seed=seed,config_sha256=sha(U/'TRAIN_CONFIG.json')),path)
  run=dict(kind=kind,seed=seed,path=str(path),sha256=sha(path),steps=cfg[kind+'_steps'],logs=logs,seconds=time.perf_counter()-t,fit_binary_accuracy=float(((probs>.5)==data[kind+'_y'])[data[kind+'_mask']].mean()),note='FIT accuracy is training diagnostic, not generalization evidence');runs.append(run);write(U/'TRAIN_PROGRESS.json',dict(runs=runs));print(kind,seed,run['logs'][-1],round(run['seconds'],2),flush=True)
write(U/'TRAINED.json',dict(runs=runs,pins=cfg['pins'],device='CPU two threads; small branches only, no RTMW retraining'))
