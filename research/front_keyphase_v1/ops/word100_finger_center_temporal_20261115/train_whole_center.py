from model import *
from finger_runtime import FingerHead,local_sequence
import time,random
assert not(U/'WHOLE_CENTER_TRAINED.json').exists()
cfg=read(U/'TRAIN_CONFIG.json');data=np.load(U/'TRAIN_INPUTS.npz');meta=read(U/'TRAIN_INPUTS.json')['state_rows'];extras={r['clip_id']:r for r in read(O/'word100_feature_expansion_20261113/INPUTS_NEW.json')['rows']+read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']};cache={};xx=[]
for r in meta:
 assert r['role']=='fit'and r['person']<=9
 path=r.get('channel_path')or(extras[r['clip_id']]['channels']if r['clip_id']in extras else read(J/'records'/f'{r["clip_id"]}.json')['channel_path']);hand=r['side'];key=path,hand
 if key not in cache:
  x,f=load_channels(path);fl,known=local_sequence(x,hand);cache[key]=(np.c_[fl,np.zeros((len(fl),64),np.float32)],f)
 fl,f=cache[key];j=int(np.flatnonzero(f==r['frame'])[0]);xx.append(fl[j])
xx=np.array(xx,np.float32)[:,None];np.savez_compressed(U/'WHOLE_CENTER_INPUT.npz',x=xx)
cfg2=dict(base_config_sha256=sha(U/'TRAIN_CONFIG.json'),model='Pretrained whole-hand FingerHead copied, center-frame ONLY, same FIT targets and masks as center-chain. All latent64 inputs zero.',reason='Phase head detects change but is not proof center state is correct. V1 newly committed errors in previously unknown finger chains; require two different center representations (whole hand and per-finger) to agree before temporal fusion.',seeds=[42,43,44],steps=450,lr=.0002,new_weight=.35,threshold=.9,phase_threshold=.85,candidates=['dual_center_only','dual_change_override','dual_change_fill'],priority=['dual_change_override','dual_change_fill'],no_old_threshold_change=True,no_new_GT_or_dev_training=True,confirmation_unopened=not(U/'confirmation_RESULTS.json').exists(),pins={str(p):sha(p)for p in [U/'TRAIN_INPUTS.npz',U/'WHOLE_CENTER_INPUT.npz',U/'runtime.py']+[P/'expanded_whole_hand'/f'{s}.pt'for s in [42,43,44]]})
assert cfg2['confirmation_unopened'];write(U/'WHOLE_CENTER_CONFIG.json',cfg2);torch.set_num_threads(2);runs=[]
for seed in cfg2['seeds']:
 torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);t=time.perf_counter();ck=torch.load(P/'expanded_whole_hand'/f'{seed}.pt',map_location='cpu',weights_only=False);mu=np.array(ck['mean']);sd=np.array(ck['std']);m=FingerHead();m.load_state_dict(ck['state_dict']);v=(xx-mu)/sd;v[:,:,63:]=0;X=torch.tensor(v,dtype=torch.float32);Y=torch.tensor(data['state_y'].clip(0),dtype=torch.float32);mask=torch.tensor(data['state_mask'],dtype=torch.float32);old=torch.tensor(~data['is_new']);new=~old;opt=torch.optim.AdamW(m.parameters(),lr=cfg2['lr'],weight_decay=.0001);logs=[]
 for step in range(cfg2['steps']):
  m.train();opt.zero_grad();p=m(X)[:,1:];l=torch.nn.functional.binary_cross_entropy_with_logits(p,Y,reduction='none');loss=(l[old]*mask[old]).sum()/mask[old].sum()+cfg2['new_weight']*(l[new]*mask[new]).sum()/mask[new].sum();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2.0);opt.step()
  if step%100==0 or step==449:logs.append(dict(step=step+1,loss=float(loss.detach())))
 path=U/'models'/'whole_center'/f'{seed}.pt';path.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(state_dict=m.state_dict(),mean=mu,std=sd,seed=seed,config_sha256=sha(U/'WHOLE_CENTER_CONFIG.json')),path);runs.append(dict(path=str(path),sha256=sha(path),seed=seed,seconds=time.perf_counter()-t,logs=logs));print('whole_center',seed,logs[-1],flush=True)
write(U/'WHOLE_CENTER_TRAINED.json',dict(runs=runs))
