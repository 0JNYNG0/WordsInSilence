from core import *
import itertools
meta=read(W/'INPUTS_COMPLETE.json');a=np.load(meta['arrays']);y=a['y'];mask=a['mask'];rows=meta['rows'];split=a['train'];full=np.array([r['view']=='full' for r in rows]);metrics=[]
for run in read(W/'TRAINING_COMPLETE.json')['rows']:
 assert sha(run['model'])==run['model_sha256'] and sha(run['predictions'])==run['predictions_sha256'];p=np.load(run['predictions'])['p'];pred=p>=.5
 for role,which in [('train',split&full),('probe',~split&full)]:
  for j,f in enumerate(F):
   for h in range(3):
    use=which&(mask[:,j,h]>0);truth=y[:,j,h][use].astype(int);pr=pred[:,j,h][use];cm=[[int(np.sum((truth==u)&(pr==v))) for v in [0,1]] for u in [0,1]];rec=[cm[i][i]/sum(cm[i]) if sum(cm[i]) else None for i in [0,1]];metrics.append(dict(seed=run['seed'],split=role,family=f,head=h,confusion=cm,balanced_accuracy=float(np.mean(rec)) if all(v is not None for v in rec) else None,correct=int((truth==pr).sum()),n=len(truth),unique_source_clips=len({r['clip_id'] for r,b in zip(rows,use) if b})))
c=read(W/'COMPARISON.json');assert len(c['rows'])==246;checks=[];diagnosis=[]
for s in c['summary']:
 rr=[r for r in c['rows'] if r['seed']==s['seed'] and r['kind']==s['kind']];cc=[v for r in rr for v in r['conditions'][s['condition']]['cards']];truth=[v for v in cc if 'truth' in v]
 hit=0;wrong=0
 for v in truth:
  obs=set(v['chosen']['body_observed']) if v['chosen'] else set();hit+=set(v['truth']['witness'])<=obs;wrong+=len(set(v['truth']['wrong_witness'])&obs)
 assert hit==s['all_witness'] and wrong==s['wrong_points'] and sum(v['chosen'] is None for v in cc)==s['missing']
for r in c['rows']:
 if r['kind']!='dynamic_complete_blocks':continue
 for mode,v in r['conditions'].items():
  if mode=='baseline':continue
  for card in v['cards']:
   if 'truth' in card and not card['truth']['all_witness']:
    diagnosis.append(dict(seed=r['seed'],clip_id=r['clip_id'],condition=mode,card=card['card'],family=card['family'],chosen=card['chosen'],truth=card['truth']))
protect=read(U/'PROTECTIONS.json');bad=[p for p,h in protect.items() if not Path(p).exists() or sha(p)!=h];assert not bad
v=dict(at=stamp(),status='VERIFIED_REWORK',metrics=metrics,failed_witness_choices=diagnosis,protected_hashes=len(protect),protected_unchanged=True,summary_recomputed=True,source_split='16train8probe public clips; full/upper correlated',reviewer='root separate arithmetic, not independent human',next_cause='Direct candidate supervision removes mixed other-action witnesses but can choose preparation/partial windows after action reordering. Need separate candidate availability from learned ranking and contextual-feature shift, not tune thresholds on these scores.')
write(W/'VERIFICATION.json',v)
for seed in [42,43,44]:
 for head in range(3):
  rr=[r for r in metrics if r['seed']==seed and r['split']=='probe' and r['head']==head];print(seed,head,'correct',sum(r['correct'] for r in rr),'/',sum(r['n'] for r in rr),'macro_balanced',np.mean([r['balanced_accuracy'] for r in rr if r['balanced_accuracy'] is not None]))
print('protected',len(protect),'failed_choices',len(diagnosis))
