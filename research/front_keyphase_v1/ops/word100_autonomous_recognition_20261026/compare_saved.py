from common import *
from collections import defaultdict
def metrics(rows):
 return dict(n=len(rows),correct=sum(r['correct'] for r in rows),top5=sum(r['top5_correct'] for r in rows),accuracy=100*sum(r['correct'] for r in rows)/len(rows),by_signer={str(s):dict(n=sum(r['signer_id']==s for r in rows),correct=sum(r['correct'] for r in rows if r['signer_id']==s)) for s in sorted({r['signer_id'] for r in rows})})
def pairs(base,new):
 gains=[r['clip_id'] for r,q in zip(new,base) if r['correct'] and not q['correct']];losses=[r['clip_id'] for r,q in zip(new,base) if not r['correct'] and q['correct']]
 return dict(gains=gains,losses=losses,delta_pp=100*(len(gains)-len(losses))/len(base))
def main():
 f=read(U/'FROZEN.json')
 for p,h in f['pins'].items():assert sha(p)==h,p
 out={};allrows={}
 for split,n in [('dev',300),('known200',200)]:
  sources={str(s):read(f['saved_scores'][split][str(s)]['path'])['predictions'] for s in SEEDS}
  c=read(B/'training_C_20261013_RESULT.json' if split=='dev' else E/'C_20261013_SCORES.json')['predictions'];out[split]={};allrows[split]={}
  for view in VIEWS:
   pp={s:sorted(q[view],key=lambda x:x['clip_id']) for s,q in sources.items()};base=pp[str(SEEDS[0])];cc=sorted(c[view],key=lambda x:x['clip_id']);ensemble=[]
   assert len(base)==n and len({x['clip_id'] for x in base})==n
   for i,r in enumerate(base):
    rr=[pp[str(s)][i] for s in SEEDS];assert all((x['clip_id'],x['label_index'],x['signer_id'])==(r['clip_id'],r['label_index'],r['signer_id']) for x in rr);assert cc[i]['clip_id']==r['clip_id']
    logits=[q['logits'] for q in rr];ok=all(z is not None for z in logits)
    prob=combine(logits) if ok else None;rank=ranking(prob) if ok else []
    ensemble.append(dict(clip_id=r['clip_id'],label_index=r['label_index'],signer_id=r['signer_id'],probabilities=prob.tolist() if ok else None,prediction=rank[0] if ok else None,top5=rank[:5],correct=bool(ok and rank[0]==r['label_index']),top5_correct=r['label_index'] in rank[:5],status='OK' if ok else 'INPUT_UNAVAILABLE'))
   out[split][view]={'C13':metrics(cc),'T3F13':metrics(base),'E3':metrics(ensemble),'individual_T3F':{s:metrics(pp[str(s)]) for s in SEEDS},'E3_vs_T3F13':pairs(base,ensemble),'E3_vs_C13':pairs(cc,ensemble)};allrows[split][view]=ensemble
 dev=out['dev'];checks={}
 checks['sensor_improves']=dev['sensor']['E3']['correct']>dev['sensor']['T3F13']['correct']
 checks['masked_nondecreasing']=dev['masked']['E3']['correct']>=dev['masked']['T3F13']['correct']
 checks['top5_nondecreasing']=all(dev[v]['E3']['top5']>=dev[v]['T3F13']['top5'] for v in VIEWS)
 checks['each_signer_nondecreasing']=all(dev[v]['E3']['by_signer'][s]['correct']>=dev[v]['T3F13']['by_signer'][s]['correct'] for v in VIEWS for s in dev[v]['E3']['by_signer'])
 write(U/'SAVED_ENSEMBLE_ROWS.json',allrows);write(U/'SAVED_COMPARISON.json',dict(at=now(),status='SAVED_SCORE_COMPARISON_COMPLETE',frozen_sha256=sha(U/'FROZEN.json'),scores=out,development_checks=checks,development_pass=all(checks.values()),new_inference=0,training=0))
 print(json.dumps(dict(counts={s:{v:{k:d[k]['correct'] for k in ['C13','T3F13','E3']} for v,d in q.items()} for s,q in out.items()},checks=checks),ensure_ascii=False))
if __name__=='__main__':main()
