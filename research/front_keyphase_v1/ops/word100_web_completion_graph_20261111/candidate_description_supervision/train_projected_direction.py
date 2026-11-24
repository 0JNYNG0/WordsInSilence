from projected_direction_core import *
import time,traceback
def main():
 assert not(V/'TRAINING_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 frozen=read(V/'FROZEN.json');cfg=frozen['training'];meta=read(V/'INPUTS_COMPLETE.json');assert sha(meta['path'])==meta['sha256'];assert sha(meta['source'])==meta['source_sha256']
 for p,h in {**frozen['pins'],**meta['pins']}.items():assert sha(p)==h,p
 rows=meta['rows'];z=np.load(meta['path']);ctx=torch.tensor(z['context']);g=torch.tensor(z['g']);y=torch.tensor(z['y'],dtype=torch.long);tr=[i for i,r in enumerate(rows)if r['split']=='train'];probe=[i for i,r in enumerate(rows)if r['split']=='probe']
 assert not({rows[i]['source_sha256']for i in tr}&{rows[i]['source_sha256']for i in probe});assert {rows[i]['person']for i in tr}<={1,2,3,4,5,6,7}
 assert all(rows[i]['axis']in[1,2,3,4]for i in tr);assert all(np.isfinite(z[k]).all()for k in z.files)
 pools={}
 for origin in ['original_motion_replay','new_reviewed']:
  ii=[i for i in tr if(rows[i]['origin']=='original_motion_replay')==(origin=='original_motion_replay')]
  pools[origin]={k:[i for i in ii if(rows[i]['axis'],rows[i]['target'])==k]for k in sorted({(rows[i]['axis'],rows[i]['target'])for i in ii})}
 rc=load('_proj_req',O/'word100_guide_requirement_learning_20261111/requirement_core.py');observer=rc.frozen_observer();runs=[]
 write(V/'PRETRAIN_VERIFICATION.json',dict(at=stamp(),source_split=True,finite=True,geometry_norm_fit_only=True,common100_frozen=True,pins_checked=len({**frozen['pins'],**meta['pins']}),pins={str(p):sha(p)for p in[Path(__file__),W/'projected_direction_core.py',V/'FROZEN.json',V/'INPUTS_COMPLETE.json']}))
 for seed in cfg['seeds']:
  torch.manual_seed(seed);rng=np.random.default_rng(seed);head=DirectionHead(observer);head.gm.copy_(torch.tensor(meta['norm']['mean']));head.gs.copy_(torch.tensor(meta['norm']['std']));opt=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);start=time.perf_counter();log=[]
  for step in range(cfg['steps']):
   ix=[]
   for origin,groups in pools.items():
    keys=list(groups)
    for _ in range(128):key=keys[int(rng.integers(len(keys)))];ix.append(int(rng.choice(groups[key])))
   ix=torch.tensor(ix);out=head(ctx[ix],g[ix]);loss=torch.nn.functional.cross_entropy(out,y[ix]);assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),2);opt.step()
   if step%200==0 or step==cfg['steps']-1:log.append(dict(step=step+1,loss=float(loss.detach())));print(seed,step+1,float(loss.detach()),flush=True)
  head.eval();dest=V/'runs'/str(seed);dest.mkdir(parents=True,exist_ok=True);assert not(dest/'model.pt').exists();torch.save(head.state_dict(),dest/'model.pt')
  with torch.inference_mode():pred=torch.cat([head(ctx[i:i+512],g[i:i+512])for i in range(0,len(rows),512)]).numpy()
  np.savez_compressed(dest/'predictions.npz',logits=pred);runs.append(dict(seed=seed,model=str(dest/'model.pt'),model_sha256=sha(dest/'model.pt'),predictions=str(dest/'predictions.npz'),predictions_sha256=sha(dest/'predictions.npz'),seconds=time.perf_counter()-start,log=log));write(V/'TRAINING_PROGRESS.json',dict(at=stamp(),completed=runs))
 write(V/'TRAINING_COMPLETE.json',dict(at=stamp(),rows=runs,common100_changed=False,web_changed=False,private_access=False,final8=False));print('Projected direction head training complete')
if __name__=='__main__':
 try:main()
 except BaseException:(V/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
