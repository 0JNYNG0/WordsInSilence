from graph import *
from prepare_linker import load
import numpy as np,torch,time,traceback
O=U.parent;R=O/'word100_candidate_role_learning_20261111';W=U/'occurrence_supervision'


def main():
 assert not (W/'TRAINING_COMPLETE.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 role=load('_role_corrected_training',R/'role_core.py');cfg=read(W/'FROZEN.json');rows=read(W/'INPUTS_COMPLETE.json')['rows'];results=[]
 for condition in cfg['conditions']:
  items=[]
  for s in rows:
   if s['condition']!=condition:continue
   assert role.sha(s['path'])==s['sha256'];z=np.load(s['path'])
   items.append(dict(z=torch.tensor(z['z']),y=torch.tensor(z['y']).long(),w=torch.tensor(z['w'])))
  for seed in cfg['seeds']:
   dst=W/'runs'/condition/str(seed);done=dst/'TRAINING.json'
   if done.exists():
    r=read(done);assert role.sha(r['path'])==r['sha256'];results.append(r);continue
   torch.manual_seed(seed);n=role.RoleAdapter(seed)
   initial=torch.load(R/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu');n.load_state_dict(initial);n.eval()
   with torch.no_grad():teachers=[n(it['z']).softmax(-1).detach() for it in items]
   opt=torch.optim.AdamW(n.role.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);start=time.perf_counter();log=[];n.train()
   for epoch in range(cfg['epochs']):
    loss_sum=0.
    for i in torch.randperm(len(items)).tolist():
     it=items[i];valid=it['w']>0
     if not valid.any():continue
     p=n(it['z']);loss=torch.nn.functional.cross_entropy(p[valid],it['y'][valid])
     if (~valid).any():loss=loss+cfg['teacher_kl_weight']*torch.nn.functional.kl_div(p[~valid].log_softmax(-1),teachers[i][~valid],reduction='batchmean')
     opt.zero_grad();loss.backward();opt.step();loss_sum+=float(loss.detach())
    log.append(loss_sum/len(items))
   final=n.state_dict();unchanged=[k for k in initial if not k.startswith('role.')]
   assert all(torch.equal(initial[k],final[k]) for k in unchanged)
   dst.mkdir(parents=True,exist_ok=True);torch.save(final,dst/'model.pt')
   r=dict(condition=condition,seed=seed,seconds=time.perf_counter()-start,loss=log,path=str(dst/'model.pt'),sha256=role.sha(dst/'model.pt'),only_role_changed=True)
   write(done,r);results.append(r);print(condition,seed,r['seconds'],flush=True)
 write(W/'TRAINING_COMPLETE.json',dict(runs=results,private_used=False,common100_backbone_changed=False,web_changed=False))

if __name__=='__main__':
 try:main()
 except BaseException:
  (W/f'FIRST_TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
