"""B loop: do not drop valid parent-same supervision for repeated actions."""
from core import *
from collections import Counter,defaultdict
import shutil,time,traceback
V=W/'parent_consistency_repair'

def freeze():
 assert not V.exists();V.mkdir();d=np.load(W/'pairs.npz');rows=read(W/'INPUTS_COMPLETE.json')['rows'];old=np.load(W/'SUPERVISION.npz')['mask'];sup=old.copy();sup[:,0]=d['mask'][:,0];weights=np.zeros_like(sup);tr=d['train']&d['eligible'];mixed=set(read(W/'FROZEN_V2.json')['supervision'][0]['mixed_train_words'])
 # Equal parent contrast vs positive-only preservation groups, not score-chosen coefficients.
 for head in [0,1]:
  strata=[lambda r:r['word_id'] in mixed,lambda r:r['word_id'] not in mixed] if head==0 else [lambda r:True]
  for accept in strata:
   ids=[i for i,r in enumerate(rows) if tr[i] and sup[i,head] and accept(r)];count=Counter((rows[i]['word_id'],int(d['y'][i,head]),rows[i]['clip_id']) for i in ids);clips=Counter((w,c) for w,c,cid in count);classes=defaultdict(set)
   for w,c,cid in count:classes[w].add(c)
   for i in ids:
    r=rows[i];c=int(d['y'][i,head]);weights[i,head]=1/(len(strata)*len(classes)*len(classes[r['word_id']])*clips[r['word_id'],c]*count[r['word_id'],c,r['clip_id']])
 assert np.allclose(weights.sum(0),1)
 np.savez_compressed(V/'SUPERVISION.npz',mask=sup,weights=weights)
 for name in ['pairs.npz','INPUTS_COMPLETE.json']:shutil.copyfile(W/name,V/name)
 train=read(W/'TRAINING_COMPLETE.json');diagnosis=dict(at=stamp(),trigger='All3 models falsely split reviewed same-parent repeated actions; original parent loss intentionally excluded all-positive words, despite valid source same-parent labels.',evidence=[dict(seed=r['seed'],word1567=r['full_probe_by_word']['WORD1567']['parent']) for r in train['runs']],repair='Restore valid parent positives as balanced preservation objective. Keep within-word contrast half of parent loss, repeated/other positive-only words half. Never invent negative labels or use probe predictions as targets.',train_data_unchanged=True,model_architecture_unchanged=True,atomic_scope_unchanged=True,comparison='Fixed last1200-step same initialization seeds and all saved82 scenarios; retain failed predecessors. No evaluation threshold sweep.')
 write(V/'CAUSE_AND_REPAIR.json',diagnosis);cfg=read(W/'FROZEN_V2.json');cfg['at']=stamp();cfg['parent_loss_repair']=diagnosis;cfg['pins'].update({str(p):sha(p) for p in [Path(__file__),V/'SUPERVISION.npz',V/'pairs.npz',V/'CAUSE_AND_REPAIR.json',W/'TRAINING_COMPLETE.json',W/'COMPARISON.json']});write(V/'FROZEN_V2.json',cfg)

def train():
 assert not (V/'TRAINING_COMPLETE.json').exists();cfg=read(V/'FROZEN_V2.json')
 for p,h in cfg['pins'].items():assert sha(p)==h
 torch.set_num_threads(4);torch.set_num_interop_threads(1);d=np.load(V/'pairs.npz');sd=np.load(V/'SUPERVISION.npz');rows=read(V/'INPUTS_COMPLETE.json')['rows'];use=d['train']&d['eligible'];pr=~d['train'];xx=d['x'];y=d['y'];mask=d['mask'];mean=xx[use].mean(0);std=np.maximum(xx[use].std(0),cfg['std_floor']);tx=torch.tensor(xx[use]);ty=torch.tensor(y[use]);tw=torch.tensor(sd['weights'][use]);px=torch.tensor(xx[pr]);done=[]
 for seed in cfg['seeds']:
  dst=V/'runs'/str(seed);assert not dst.exists();torch.manual_seed(seed);net=RelationHead(mean,std);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);start=time.perf_counter();history=[]
  for step in range(1,cfg['steps']+1):
   raw=torch.nn.functional.binary_cross_entropy_with_logits(net(tx),ty,reduction='none');loss=(raw*tw).sum()/2;assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip']);opt.step()
   if step%100==0:history.append(dict(step=step,loss=float(loss.detach())))
  net.eval();dst.mkdir(parents=True);torch.save(net.state_dict(),dst/'model.pt')
  with torch.inference_mode():p=net.probabilities(px).numpy();tp=net.probabilities(tx).numpy()
  np.savez_compressed(dst/'PROBE.npz',p=p,y=y[pr],mask=mask[pr],eligible=d['eligible'][pr]);np.savez_compressed(dst/'TRAIN.npz',p=tp,y=y[use],mask=sd['mask'][use]);views=np.array([r['view'] for r in rows])[pr];wordids=np.array([r['word_id'] for r in rows])[pr]
  scores={v:metrics(y[pr][views==v],mask[pr][views==v],p[views==v],d['eligible'][pr][views==v]) for v in ['full','upper']};byword={w:metrics(y[pr][(wordids==w)&(views=='full')],mask[pr][(wordids==w)&(views=='full')],p[(wordids==w)&(views=='full')]) for w in sorted(set(wordids))};primary={name:float(np.mean([byword[w][name]['balanced_accuracy'] for w in cfg['supervision'][j]['mixed_train_words']])) for j,name in enumerate(['parent','atomic_given_parent'])}
  rec=dict(seed=seed,model=str(dst/'model.pt'),sha256=sha(dst/'model.pt'),seconds=time.perf_counter()-start,parameters=sum(a.numel() for a in net.parameters()),history=history,probe=scores,full_probe_by_word=byword,primary_macro_word_balanced=primary,train=metrics(y[use],sd['mask'][use],tp));write(dst/'TRAINING.json',rec);done.append(rec);print('B repair seed',seed,'primary',primary,'fullparent',scores['full']['parent'],flush=True)
 write(V/'TRAINING_COMPLETE.json',dict(at=stamp(),runs=done,upstream_changed=False,web_changed=False,private_used=False,final8_read=False))

def compare_freeze():
 assert not(V/'C_FROZEN.json').exists();cfg=read(W/'C_FROZEN.json');cfg['at']=stamp();cfg['repair']=str(V/'CAUSE_AND_REPAIR.json');cfg['pins'].update(read(V/'FROZEN_V2.json')['pins']);cfg['pins'].update({str(V/'TRAINING_COMPLETE.json'):sha(V/'TRAINING_COMPLETE.json')});write(V/'C_FROZEN.json',cfg)

def compare():
 m=load('_parent_compare',W/'compare.py');m.W=V;m.compare()

if __name__=='__main__':
 try:{'freeze':freeze,'train':train,'compare_freeze':compare_freeze,'compare':compare}[sys.argv[1]]()
 except BaseException:(W/f'REPAIR_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
