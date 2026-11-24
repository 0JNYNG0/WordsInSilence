from graph import *
from binding import *
import time,os,traceback
def main():
 assert not (U/'LINKER_TRAIN_COMPLETE.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1);cfg=read(U/'LINKER_FROZEN.json');data=np.load(U/'linker_fit.npz');x=torch.tensor(data['x']);y=torch.tensor(data['y']);runs=[]
 assert hashlib.sha256((U/'linker_fit.npz').read_bytes()).hexdigest()==cfg['train_hash']
 write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING_COMMON_BODY_CONTEXT_LINKER',pid=os.getpid(),camera=False,private_access=False))
 for seed in cfg['seeds']:
  torch.manual_seed(seed);net=BodyContextLinker(np.array(cfg['mean'],np.float32),np.array(cfg['std'],np.float32));opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);loss=nn.CrossEntropyLoss(weight=torch.tensor(cfg['weights']));log=[];t=time.perf_counter()
  for epoch in range(cfg['epochs']):
   idx=torch.randperm(len(x));total=0
   for ii in idx.split(cfg['batch']):
    opt.zero_grad();v=loss(net(x[ii]),y[ii]);v.backward();opt.step();total+=float(v.detach())*len(ii)
   log.append(total/len(x))
  net.eval()
  with torch.inference_mode():pred=net(x).argmax(1).numpy()
  p=U/'linker_runs'/str(seed);p.mkdir(parents=True,exist_ok=True);torch.save(net.state_dict(),p/'model.pt')
  item=dict(seed=seed,seconds=time.perf_counter()-t,fit_correct=int((pred==data['y']).sum()),fit_n=len(x),loss=log,checkpoint=str(p/'model.pt'),sha256=hashlib.sha256((p/'model.pt').read_bytes()).hexdigest());write(p/'training.json',item);runs.append(item);print(seed,item['fit_correct'],item['fit_n'],round(item['seconds'],2),flush=True)
 write(U/'LINKER_TRAIN_COMPLETE.json',dict(runs=runs,backbone_updated=False,private_access=False,completed=True));write(U/'CURRENT_EXECUTION.json',dict(status='LINKER_TRAINED_NEXT_FROZEN_EVALUATION',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
