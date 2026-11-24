from common import *
import numpy as np,torch,traceback,random
from pilot.model_masked_v1 import MaskedSignTCN

def batch(rows,view='full',epoch=None):
 xx=[]
 for r in rows:
  k=('upper_mask' if (epoch+r['parity'])%2 else 'full') if epoch is not None else view
  xx.append(r[k])
 n=len(xx);T=max(map(len,xx));x=np.zeros((n,T,280),np.float32);p=np.zeros((n,T),np.float32)
 for i,a in enumerate(xx):x[i,:len(a)]=a;p[i,:len(a)]=1
 return torch.from_numpy(x),torch.from_numpy(p),torch.tensor([r['y'] for r in rows],dtype=torch.long)
def dataset():
 check=read(U/'INPUT_VERIFICATION.json');assert check['status']=='PASS';rows=[];bad=[]
 for r in read(U/'SOURCES.json'):
  rp=U/'records'/f"{r['clip_id']}.json";assert sha(rp)==check['record_hashes'][r['clip_id']];rec=read(rp)
  if rec['status']!='OK':bad.append(r);continue
  assert sha(rec['channel_path'])==rec['channel_sha256'];z=np.load(rec['channel_path'],allow_pickle=False)
  rows.append(dict(id=r['clip_id'],role=r['role'],person=r['person'],y=r['label_index'],word=r['word_id'],
   full=z['full'],upper_mask=z['upper_mask'],parity=int(hashlib.sha256(r['clip_id'].encode()).hexdigest(),16)%2))
 return rows,bad
def summary(pred):
 return dict(total=len(pred),top1=sum(r['top1_correct'] for r in pred),top3=sum(r['top3_correct'] for r in pred),input_failures=sum(r['logits'] is None for r in pred),
  by_person={str(p):dict(total=sum(r['person']==p for r in pred),top1=sum(r['person']==p and r['top1_correct'] for r in pred),top3=sum(r['person']==p and r['top3_correct'] for r in pred)) for p in [11,12,13]})
def run(condition,seed,rows,bad,frozen):
 d=U/'tcn_runs'/f'{condition}_{seed}';d.mkdir(parents=True,exist_ok=True)
 if (d/'COMPLETE.json').exists():
  r=read(d/'COMPLETE.json');assert r['frozen_sha256']==sha(U/'FROZEN.json');assert sha(d/'model.pt')==r['checkpoint_sha256'];return r
 assert not (d/'last.pt').exists(),'Review interrupted epoch checkpoint before continuing'
 cfg=dict(frozen['tcn']['model']);cfg.pop('type');torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
 model=MaskedSignTCN(**cfg);basepath=O/'word100_body_boundary_20261017/training_C_20261013_best.pt'
 if condition=='finetune_C':
  assert sha(basepath)==frozen['pins'][str(basepath)];ck=torch.load(basepath,map_location='cpu',weights_only=True)
  assert ck['word_codes']==frozen['classes'];model.load_state_dict(ck['state_dict'],strict=True)
 initial={k:v.clone() for k,v in model.state_dict().items()};t=frozen['tcn'];fit=[r for r in rows if r['role']=='fit'];dev=[r for r in rows if r['role']=='development']
 assert len(fit)+sum(r['role']=='fit' for r in bad)==899;assert set(r['y'] for r in fit)==set(range(100))
 opt=torch.optim.AdamW(model.parameters(),lr=t['lr'][condition],weight_decay=t['weight_decay']);start=time.perf_counter();history=[]
 for epoch in range(1,t['epochs']+1):
  model.train();order=np.random.default_rng(seed+epoch).permutation(len(fit));loss_sum=0.
  for lo in range(0,len(order),t['batch_size']):
   rr=[fit[i] for i in order[lo:lo+t['batch_size']]];x,p,y=batch(rr,epoch=epoch);opt.zero_grad(set_to_none=True);logit=model(x,p)
   loss=torch.nn.functional.cross_entropy(logit,y);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),t['gradient_clip']);opt.step();loss_sum+=float(loss.detach())*len(rr)
  history.append(dict(epoch=epoch,train_ce=loss_sum/len(fit),elapsed_seconds=time.perf_counter()-start))
  torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,torch_rng=torch.get_rng_state(),history=history,frozen_sha256=sha(U/'FROZEN.json')),d/'last.pt')
  write(U/'TCN_PROGRESS.json',dict(status='TRAINING',condition=condition,seed=seed,**history[-1]))
  if epoch%10==0:event('tcn_epoch',condition=condition,seed=seed,**history[-1])
 training_seconds=time.perf_counter()-start;model.eval();metrics={};pred_paths={}
 with torch.inference_mode():
  for view in ['full','upper_mask']:
   pred=[]
   for lo in range(0,len(dev),t['batch_size']):
    rr=dev[lo:lo+t['batch_size']];x,p,y=batch(rr,view);logits=model(x,p).numpy();assert np.isfinite(logits).all()
    for r,l in zip(rr,logits):
     order=np.argsort(-l,kind='stable');pred.append(dict(clip_id=r['id'],person=r['person'],y=r['y'],logits=l.tolist(),top1=int(order[0]),top3=order[:3].tolist(),top1_correct=bool(order[0]==r['y']),top3_correct=bool(r['y'] in order[:3])))
   for r in bad:
    if r['role']=='development':pred.append(dict(clip_id=r['clip_id'],person=r['person'],y=r['label_index'],logits=None,top1=None,top3=[],top1_correct=False,top3_correct=False))
   assert len(pred)==300;pred.sort(key=lambda r:r['clip_id']);path=d/f'PREDICTIONS_{view}.json';write(path,pred);pred_paths[view]=str(path);metrics[view]=summary(pred)
 torch.save(dict(state_dict=model.state_dict(),model_config=cfg,classes=frozen['labels'],word_codes=frozen['classes'],input_schema=frozen['input'],condition=condition,seed=seed,epoch=t['epochs']),d/'model.pt')
 delta=sum(float((model.state_dict()[k]-v).float().square().sum()) for k,v in initial.items())**.5
 result=dict(status='FIXED_FINAL_EPOCH_DEVELOPMENT_ONLY',condition=condition,seed=seed,epochs=t['epochs'],training_seconds=training_seconds,metrics=metrics,
  parameter_count=sum(p.numel() for p in model.parameters()),weight_delta_l2=delta,checkpoint_sha256=sha(d/'model.pt'),frozen_sha256=sha(U/'FROZEN.json'),prediction_paths=pred_paths,history=history)
 write(d/'COMPLETE.json',result);event('tcn_trained',condition=condition,seed=seed,metrics=metrics);return result
def main():
 frozen=read(U/'FROZEN.json');assert not (U/'TCN_TRAINING_COMPLETE.json').exists()
 pin=read(U/'TCN_CODE_CHECK.json');assert pin['status']=='PASS';assert all(sha(p)==h for p,h in pin['pins'].items())
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 rows,bad=dataset();runs=[]
 for condition in frozen['tcn']['conditions']:
  for seed in frozen['tcn']['seeds']:runs.append(run(condition,seed,rows,bad,frozen))
 write(U/'TCN_TRAINING_COMPLETE.json',dict(status='SIX_RUNS_COMPLETE_PENDING_COMPARISON',runs=runs))
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'TCN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
