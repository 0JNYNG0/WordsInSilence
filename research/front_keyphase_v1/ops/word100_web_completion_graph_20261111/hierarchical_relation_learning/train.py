from core import *
from collections import Counter
import time,traceback

def main():
 assert not (W/'TRAINING_COMPLETE.json').exists();cfg=read(W/'FROZEN_V2.json')
 for p,h in cfg['pins'].items():assert sha(p)==h
 torch.set_num_threads(4);torch.set_num_interop_threads(1);d=np.load(W/'pairs.npz');rows=read(W/'INPUTS_COMPLETE.json')['rows'];supervision=np.load(W/'SUPERVISION.npz')['mask'];use=d['train']&d['eligible']&(supervision.sum(1)>0);pr=~d['train'];xx=d['x'];y=d['y'];mask=d['mask'];mean=xx[use].mean(0);std=np.maximum(xx[use].std(0),cfg['std_floor']);weights=np.zeros_like(y)
 for j in range(2):
  selected=np.flatnonzero(use&(supervision[:,j]>0));cnt=Counter((rows[i]['word_id'],int(y[i,j]),rows[i]['clip_id']) for i in selected);clips=Counter((w,c) for w,c,cid in cnt);words={rows[i]['word_id'] for i in selected}
  for i in selected:
   r=rows[i];c=int(y[i,j]);weights[i,j]=1/(len(words)*2*clips[r['word_id'],c]*cnt[r['word_id'],c,r['clip_id']])
 assert np.allclose(weights.sum(0),1)
 tx=torch.tensor(xx[use]);ty=torch.tensor(y[use]);tw=torch.tensor(weights[use]);px=torch.tensor(xx[pr]);done=[]
 for seed in cfg['seeds']:
  dst=W/'runs'/str(seed);assert not dst.exists();torch.manual_seed(seed);net=RelationHead(mean,std);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);hist=[];start=time.perf_counter()
  for step in range(1,cfg['steps']+1):
   raw=torch.nn.functional.binary_cross_entropy_with_logits(net(tx),ty,reduction='none');loss=(raw*tw).sum()/2;assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip']);opt.step()
   if step%100==0:hist.append(dict(step=step,loss=float(loss.detach())))
  net.eval();dst.mkdir(parents=True);torch.save(net.state_dict(),dst/'model.pt')
  with torch.inference_mode():p=net.probabilities(px).numpy();tp=net.probabilities(tx).numpy()
  np.savez_compressed(dst/'PROBE.npz',p=p,y=y[pr],mask=mask[pr],eligible=d['eligible'][pr]);np.savez_compressed(dst/'TRAIN.npz',p=tp,y=y[use],mask=mask[use])
  views=np.array([r['view'] for r in rows])[pr];scores={v:metrics(y[pr][views==v],mask[pr][views==v],p[views==v],d['eligible'][pr][views==v]) for v in ['full','upper']}
  wordids=np.array([r['word_id'] for r in rows])[pr];byword={w:metrics(y[pr][(wordids==w)&(views=='full')],mask[pr][(wordids==w)&(views=='full')],p[(wordids==w)&(views=='full')]) for w in sorted(set(wordids))};primary={name:float(np.mean([byword[w][name]['balanced_accuracy'] for w in cfg['supervision'][j]['mixed_train_words']])) for j,name in enumerate(['parent','atomic_given_parent'])}
  rec=dict(seed=seed,model=str(dst/'model.pt'),sha256=sha(dst/'model.pt'),seconds=time.perf_counter()-start,parameters=sum(a.numel() for a in net.parameters()),history=hist,probe=scores,full_probe_by_word=byword,primary_macro_word_balanced=primary,train=metrics(y[use],supervision[use],tp));write(dst/'TRAINING.json',rec);done.append(rec);print('B seed',seed,'primary',primary,'full',scores['full'],flush=True)
 write(W/'TRAINING_COMPLETE.json',dict(at=stamp(),runs=done,upstream_changed=False,web_changed=False,private_used=False,final8_read=False))

if __name__=='__main__':
 try:main()
 except BaseException:(W/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
