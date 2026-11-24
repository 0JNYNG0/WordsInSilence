from common import *
from motion import *
import importlib.util,traceback
sys.path.insert(0,str(J))
from feedback_model import PoseGuideNet

def makebase():
 s=read(V/'FEEDBACK_NORMALIZATION.json');m=PoseGuideNet(s['guide_vectors'],s['guide_words'],s['prototype_lengths']);m.load_state_dict(torch.load(V/'feedback_runs/pose_transfer_42/model.pt',weights_only=True,map_location='cpu'));return m.eval()
def physical(z):
 s=read(V/'FEEDBACK_NORMALIZATION.json');x=z.copy();x[:,:186]=x[:,:186]*np.array(s['std'])+np.array(s['mean']);return x

def build_fit():
 rows=[r for r in read(U/'READY.json')['rows'] if r['role']=='fit'];base=makebase();xx=[];yy=[];ll=[];meta=[]
 with torch.inference_mode():
  for r in rows:
   z=np.load(r['path'])
   for view in ['full','upper']:
    inp=z['x_'+view];raw=physical(inp);h=base(torch.from_numpy(inp[None]),torch.tensor([len(inp)]),torch.tensor([r['word']]))['latent'][0].numpy()
    for p in r['pairs']:
     for rev in [False,True]:
      f,valid,info=features(raw,z['timestamp_us'],p['a'],p['b'],KINDS[p['axis']],reverse=rev)
      if not valid:continue
      a,b=(p['b'],p['a']) if rev else (p['a'],p['b']);latent=np.r_[h[a],h[b],h[b]-h[a]]
      xx.append(f);ll.append(latent);yy.append(2-p['target'] if rev else p['target']);meta.append(dict(clip_id=r['clip_id'],person=r['person'],view=view,reversed=rev,a=p['a'],b=p['b'],axis=p['axis']))
 return np.asarray(xx,np.float32),np.asarray(ll,np.float32),np.array(yy),meta

def main():
 assert not (U/'TRAINING_COMPLETE.json').exists();cfg=read(U/'FROZEN.json');assert sha(U/'READY.json')==cfg['data_sha256']
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 x,latent,y,meta=build_fit();assert {m['person'] for m in meta}==set(range(1,10));np.savez_compressed(U/'FIT_FEATURES.npz',x=x,latent=latent,y=y)
 write(U/'FIT_MANIFEST.json',meta);allruns=[]
 for condition in cfg['models']:
  f=np.concatenate([x,latent],1) if condition=='motion_plus_frozen_latent' else x;mean=f.mean(0);std=f.std(0);std=np.maximum(std,.05);data=torch.tensor((f-mean)/std);targets=torch.tensor(y)
  for seed in cfg['seeds']:
   folder=U/'runs'/f'{condition}_{seed}';folder.mkdir(parents=True,exist_ok=True)
   if (folder/'COMPLETE.json').exists():allruns.append(read(folder/'COMPLETE.json'));continue
   torch.manual_seed(seed);model=MotionNet(f.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);history=[];start=time.perf_counter();classw=torch.tensor(len(y)/(3*np.bincount(y,minlength=3)),dtype=torch.float32)
   for epoch in range(cfg['epochs']):
    model.train();opt.zero_grad();loss=nn.functional.cross_entropy(model(data),targets,weight=classw);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
    if (epoch+1)%60==0:history.append(dict(epoch=epoch+1,loss=float(loss.detach())))
   torch.save(dict(model=model.state_dict(),mean=torch.tensor(mean),std=torch.tensor(std)),folder/'model.pt')
   result=dict(condition=condition,seed=seed,seconds=time.perf_counter()-start,epochs=cfg['epochs'],parameters=sum(p.numel() for p in model.parameters()),fit_samples=len(y),history=history,checkpoint_sha256=sha(folder/'model.pt'),frozen_sha256=sha(U/'FROZEN.json'),private=False,dev_used=False)
   write(folder/'COMPLETE.json',result);allruns.append(result);event('motion_trained',condition=condition,seed=seed,seconds=result['seconds'])
 write(U/'TRAINING_COMPLETE.json',dict(status='COMPLETE_PENDING_EVAL',runs=allruns,code_pins={str(U/f):sha(U/f) for f in ['motion.py','train_motion.py']},frozen_sha256=sha(U/'FROZEN.json')))
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
