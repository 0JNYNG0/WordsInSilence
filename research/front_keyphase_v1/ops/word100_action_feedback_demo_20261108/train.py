from task import *
import numpy as np,torch,random,traceback
from model import EvidenceHeads

def collate(rows,view):
 n=len(rows);T=max(len(r['raw_x']) for r in rows);a={}
 for key,size in [('h',64),('p',7),('y',7),('ym',7)]:
  ar=np.zeros((n,T,size),np.float32)
  for i,r in enumerate(rows):v=r[key+'_'+view] if key in ['h','p'] else r[key];ar[i,:len(v)]=v
  a[key]=torch.from_numpy(ar)
 for key,fill,dtype in [('stage_y',-1,np.int64),('stage_w',0,np.float32)]:
  ar=np.full((n,T),fill,dtype)
  for i,r in enumerate(rows):ar[i,:len(r[key])]=r[key]
  a[key]=torch.from_numpy(ar)
 a['valid']=torch.arange(T)[None]<torch.tensor([len(r['raw_x']) for r in rows])[:,None];a['word']=torch.tensor([r['word'] for r in rows]);return a

def main():
 assert not (U/'TRAINING_COMPLETE.json').exists();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);f=read(U/'FROZEN.json');prep=read(U/'PREPARED.json');rows=[]
 for r in prep['rows']:
  assert sha(r['path'])==r['sha256'];z=np.load(r['path']);rows.append({**r,**{k:z[k] for k in z.files}})
 fit=[r for r in rows if r['role']=='fit'];dev=[r for r in rows if r['role']=='development'];assert {r['person'] for r in fit}.isdisjoint({r['person'] for r in dev})
 cfg=f['new_heads'];counts=np.zeros(14)
 for r in fit:
  for k in range(14):counts[k]+=np.sum(r['stage_w'][r['stage_y']==k])
 weights=np.sqrt(counts.sum()/np.maximum(counts,1));weights/=weights.mean();cw=torch.tensor(weights,dtype=torch.float32);done=[]
 write(U/'FIT_CLASS_WEIGHTS.json',dict(counts=counts.tolist(),weights=weights.tolist(),fit_people=list(range(1,10)),synthetic_error_labels='For difference head, flip expected physical feature values at existing reviewed FIT centres. This is target-pair supervision, not real learner-error labels.'))
 for seed in cfg['seeds']:
  d=U/'runs'/str(seed);d.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);m=EvidenceHeads(prep['guide_words']);initial={k:v.clone() for k,v in m.state_dict().items()};opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);history=[];t=time.perf_counter()
  for epoch in range(cfg['epochs']):
   order=np.random.default_rng(seed+epoch).permutation(len(fit));m.train();total=0.
   for k in range(0,len(order),cfg['batch_size']):
    rr=[fit[i] for i in order[k:k+cfg['batch_size']]];b=collate(rr,'full' if epoch%2==0 else 'upper');expected=torch.randint(0,2,b['y'].shape).float();truth=(b['y']!=expected).float();ph,dif=m(b['h'],b['p'],b['word'],b['valid'],expected)
    ce=torch.nn.functional.cross_entropy(ph.transpose(1,2),b['stage_y'],ignore_index=-1,weight=cw,reduction='none');pl=(ce*b['stage_w']).sum()/b['stage_w'].sum().clamp_min(1)
    dl=(torch.nn.functional.binary_cross_entropy_with_logits(dif,truth,reduction='none')*b['ym']).sum()/b['ym'].sum().clamp_min(1);loss=pl+.5*dl
    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2);opt.step();total+=float(loss.detach())
   history.append(dict(epoch=epoch+1,loss=total/6))
   if (epoch+1)%25==0:event('head_training',seed=seed,epoch=epoch+1,loss=history[-1]['loss'])
  m.eval();seconds=time.perf_counter()-t;preds=[]
  with torch.inference_mode():
   for r in rows:
    b=collate([r],'full');ph,_=m(b['h'],b['p'],b['word'],b['valid']);ds=[]
    for expected in [torch.zeros_like(b['y']),torch.ones_like(b['y'])]:ds.append(m(b['h'],b['p'],b['word'],b['valid'],expected)[1][0].sigmoid().numpy())
    p=d/'predictions'/f"{r['clip_id']}.npz";p.parent.mkdir(exist_ok=True);np.savez_compressed(p,phase=ph[0].softmax(-1).numpy(),different_if0=ds[0],different_if1=ds[1],frame_id=r['frame_id'],timestamp_us=r['timestamp_us']);preds.append(dict(clip_id=r['clip_id'],role=r['role'],path=str(p),sha256=sha(p)))
  torch.save(m.state_dict(),d/'model.pt');result=dict(seed=seed,epochs=cfg['epochs'],training_seconds=seconds,parameter_count=sum(p.numel() for p in m.parameters()),weight_delta=sum(float((m.state_dict()[k]-v).float().square().sum()) for k,v in initial.items())**.5,model_sha256=sha(d/'model.pt'),history=history,predictions=preds)
  write(d/'COMPLETE.json',result);done.append(result);event('head_trained',seed=seed,seconds=seconds)
 write(U/'TRAINING_COMPLETE.json',dict(runs=done,backbone_changed=False,old_models_changed=False,real_learner_error_gold=False))
 write(U/'TRAIN_CODE_PINS.json',{str(U/n):sha(U/n) for n in ['model.py','train.py','prepare.py']})
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
