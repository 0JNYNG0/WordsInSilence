from graph import *
from prepare_linker import load
import numpy as np,torch,time,traceback
O=U.parent;R=O/'word100_candidate_role_learning_20261111'
def main():
 assert not (U/'BLOCK_TRAIN_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1);role=load('_train_block_role',R/'role_core.py');cfg=read(U/'BLOCK_FROZEN.json');sources=read(U/'BLOCK_INPUT_COMPLETE.json')['rows'];items=[]
 for s in sources:
  assert role.sha(s['path'])==s['sha256'];z=np.load(s['path']);items.append(dict(z=torch.tensor(z['z']),y=torch.tensor(z['y']).long(),w=torch.tensor(z['w']),normal=s['mode']=='normal'))
 results=[]
 for seed in cfg['seeds']:
  torch.manual_seed(seed);n=role.RoleAdapter(seed);n.load_state_dict(torch.load(R/'runs'/str(seed)/'model.pt',map_location='cpu',weights_only=True));n.eval()
  with torch.no_grad():teachers=[n(it['z']).softmax(-1).detach() if it['normal'] else None for it in items]
  opt=torch.optim.AdamW(n.role.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);t=time.perf_counter();log=[];n.train()
  for epoch in range(cfg['epochs']):
   total=0
   for i in torch.randperm(len(items)).tolist():
    it=items[i];valid=it['w']>0
    if not valid.any():continue
    out=n(it['z']);loss=torch.nn.functional.cross_entropy(out[valid],it['y'][valid])
    if it['normal']:loss=loss+cfg['teacher_kl_weight']*torch.nn.functional.kl_div(out.log_softmax(-1),teachers[i],reduction='batchmean')
    opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())
   log.append(total/len(items))
  p=U/'block_runs'/str(seed);p.mkdir(parents=True,exist_ok=True);torch.save(n.state_dict(),p/'model.pt');r=dict(seed=seed,seconds=time.perf_counter()-t,loss=log,path=str(p/'model.pt'),sha256=role.sha(p/'model.pt'));write(p/'TRAINING.json',r);results.append(r);print(seed,r['seconds'],log[-1],flush=True)
 write(U/'BLOCK_TRAIN_COMPLETE.json',dict(runs=results,private_used=False,existing_backbone_changed=False,original_models_unchanged=True))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'BLOCK_TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
