"""Controlled identity diagnostic: preserve local observed relations, remove learned global context.

This is an ablation, not a new runtime architecture or an adoption decision.
Same source labels, matching initialization, loss, seeds and optimizer steps.
"""
from pathlib import Path
import sys,importlib.util,hashlib,time,traceback
import numpy as np,torch
W=Path(__file__).resolve().parent; U=W.parent; B=U/'episode_stage_contrast'; P=U/'episode_stage'
sys.path.insert(0,str(U))
from graph import read,write,stamp
s=importlib.util.spec_from_file_location('stage_base',P/'run.py');prior=importlib.util.module_from_spec(s);s.loader.exec_module(prior)
F=prior.FAMILIES
BaseNet=prior.StageNet

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class RelationsNet(BaseNet):
 def __init__(self,seed,supported):
  super().__init__(seed,supported)
  mask=torch.zeros(684)
  for block in range(3):mask[block*228+64:block*228+164]=1
  self.register_buffer('relation_mask',mask)
 def forward(self,x):return super().forward(x*self.relation_mask)

def freeze():
 assert not (W/'FROZEN.json').exists()
 old=read(B/'FROZEN.json');meta=read(B/'INPUTS_COMPLETE.json')
 for p,h in meta['pins'].items():assert sha(p)==h
 for r in meta['rows']:assert sha(r['path'])==r['sha256'] and sha(r['cache'])==r['cache_sha256']
 assert read(B/'VERIFICATION.json')['status']=='PASS_SAVED_ARITHMETIC_G2_REWORK'
 cfg={k:old[k] for k in ['seeds','steps','batch','lr','weight_decay','gradient_clip','supported','train_counts']}
 paths=[Path(__file__),P/'run.py',B/'FROZEN.json',B/'INPUTS_COMPLETE.json',B/'TRAINING_COMPLETE.json',B/'EVALUATION_COMPLETE.json',B/'BINDING_V2_SCORES.json',B/'CONTEXT_DIAGNOSTIC.json',U/'action_linkage_plan/COVERAGE_AUDIT.json']
 cfg.update(at=stamp(),pins={str(p):sha(p) for p in paths},input='Keep normalized relation100 including observation flags in each before/current/after228 block. Mask learned context64 and frozen phase64. Local133ms before/after context is retained. Input684 shape kept for identical copied initialization; no new word/frame/time input.',hypothesis='Same witnessed source pose retains current relations after block reorder but changes global learned context. This ablation tests whether relying on local relations preserves identity across reorder. It does not prove global context is always harmful.',scope='Use identical16train/8head-excluded public clips and962/468 query labels. Upstream has seen probe people; not fresh. No new pair relation labels or semantic correctness labels.',loss=old['loss'],selection='Final step1200 only. No checkpoint/threshold search. Compare existing saved baseline output. One inference on each of82 exposed public scenarios. Final8 untouched.',pass_requirement='Retain old normal/speed/shape/direction evidence and reduce cross-action contamination in each seed, without merely suppressing all output. G2/G3 remain unpassed until integrated checks.',web_changed=False)
 write(W/'FROZEN.json',cfg)
 # Pin the unchanged inputs under the local interface expected by shared evaluator.
 write(W/'INPUTS_COMPLETE.json',meta)
 print('Frozen matched local-relation ablation; no relation-pair training.')

def train():
 assert not (W/'TRAINING_COMPLETE.json').exists();cfg=read(W/'FROZEN.json')
 for p,h in cfg['pins'].items():assert sha(p)==h
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rows=read(W/'INPUTS_COMPLETE.json')['rows'];sets={k:[r for r in rows if r['split']==k] for k in ['train','probe']}
 def load(k,key):return np.concatenate([np.load(r['path'])[key] for r in sets[k]])
 x=torch.tensor(load('train','x'));y=torch.tensor(load('train','y'));px=torch.tensor(load('probe','x'));py=load('probe','y')
 counts=np.array(cfg['train_counts']);weights=np.where(counts>0,counts.sum(1,keepdims=True)/(np.maximum(counts,1)*np.maximum((counts>0).sum(1,keepdims=True),1)),0);weights=torch.tensor(weights,dtype=torch.float32);done=[]
 for seed in cfg['seeds']:
  dst=W/'runs'/str(seed);assert not dst.exists();torch.manual_seed(seed);net=RelationsNet(seed,cfg['supported']);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);start=time.perf_counter();hist=[];step=0;epoch=0
  while step<cfg['steps']:
   order=np.random.default_rng(seed*1000+epoch).permutation(len(x));epoch+=1
   for a in range(0,len(order),cfg['batch']):
    ii=order[a:a+cfg['batch']];logit=net(x[ii]);target=y[ii];mask=target>=0;safe=target.clamp_min(0);raw=torch.nn.functional.cross_entropy(logit.reshape(-1,3),safe.reshape(-1),reduction='none').reshape(-1,5);cw=weights[torch.arange(5)[None],safe];loss=(raw*cw*mask).sum()/(cw*mask).sum();assert torch.isfinite(loss)
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip']);opt.step();step+=1
    if step%100==0:hist.append(dict(step=step,loss=float(loss.detach())))
    if step==cfg['steps']:break
  dst.mkdir(parents=True);torch.save(net.state_dict(),dst/'model.pt');net.eval()
  with torch.inference_mode():p=net(px).softmax(-1).numpy()
  basepath=B/'runs'/str(seed)/'PROBE.npz';baseline=np.load(basepath);assert np.array_equal(py,baseline['y']);bp=baseline['p'];oldmask=baseline['old_mask']
  np.savez_compressed(dst/'PROBE.npz',y=py,p=p,baseline_p=bp,old_mask=oldmask)
  result={}
  for name,mask in [('old',oldmask),('added',(py>=0)&~oldmask),('all',py>=0)]:
   target=np.where(mask,py,-1);result[name]=dict(baseline=prior.metrics(target,bp),candidate=prior.metrics(target,p))
  rec=dict(seed=seed,path=str(dst/'model.pt'),sha256=sha(dst/'model.pt'),seconds=time.perf_counter()-start,steps=step,history=hist,probe=result,baseline_path=str(basepath),baseline_sha256=sha(basepath),baseline_inference_calls=0)
  write(dst/'TRAINING.json',rec);done.append(rec);print(seed,{k:(v['baseline']['correct'],v['candidate']['correct'],v['candidate']['total']) for k,v in result.items()},flush=True)
 write(W/'TRAINING_COMPLETE.json',dict(at=stamp(),runs=done,web_changed=False,private_used=False))

def evaluate():
 prior.W=W;prior.StageNet=RelationsNet;prior.evaluate()

def bind():
 s=importlib.util.spec_from_file_location('same_binding',B/'bind_action_span.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.W=W;m.main()

if __name__=='__main__':
 try:{'freeze':freeze,'train':train,'evaluate':evaluate,'bind':bind}[sys.argv[1]]()
 except BaseException:
  (W/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
