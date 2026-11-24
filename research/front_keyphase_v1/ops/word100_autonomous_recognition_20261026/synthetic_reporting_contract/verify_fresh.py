"""Scalar recount from source logits; no numpy/model/aggregation imports."""
import json,math,hashlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;F=U/'fresh199';SEEDS=(20261013,20261014,20261015)
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def probability(z):
 assert len(z)==100 and all(math.isfinite(x) for x in z)
 e=[math.exp(float(x)-max(z)) for x in z];t=math.fsum(e);return [x/t for x in e]
def decision(z):
 if z is None:return [],False
 return sorted(range(100),key=lambda i:(-float(z[i]),i)),True
def metric(rows):
 out={'n':len(rows),'correct':sum(r[2] for r in rows),'top5':sum(r[3] for r in rows)}
 out['accuracy']=100*out['correct']/len(rows)
 out['by_signer']={str(s):{'n':sum(r[1]==s for r in rows),'correct':sum(r[2] for r in rows if r[1]==s)} for s in sorted({r[1] for r in rows})}
 return out
def main():
 frozen=read(U/'FROZEN.json');ctx=read(F/'EVAL_CONTEXT.json');out=read(F/'COMPARISON.json');ens=read(F/'ENSEMBLE_ROWS.json')
 for pins in [frozen['pins'],frozen.get('protected',{}),ctx['pins'],out['raw_score_pins']]:
  for p,h in pins.items():assert sha(p)==h,p
 assert read(F/'INPUTS_VERIFIED.json')['status']=='PASS'
 docs={f'{a}{s}':read(F/f'{a}_{s}_SCORES.json') for a,ss in [('C',(20261013,)),('T3F',SEEDS)] for s in ss}
 manifest={r['clip_id']:r for r in ctx['rows']};count=raw_count=0
 for view in ('sensor','masked'):
  allrows={k:{r['clip_id']:r for r in d['predictions'][view]} for k,d in docs.items()};stored={r['clip_id']:r for r in ens[view]};checked={k:[] for k in allrows};checked['E3']=[]
  assert set(stored)==set(manifest) and len(stored)==199
  for k,rows in allrows.items():
   assert set(rows)==set(manifest)
   for cid,r in sorted(rows.items()):
    target=manifest[cid]['label_index'];assert r['label_index']==target and r['signer_id']==manifest[cid]['signer_id']
    order,ok=decision(r['logits']);right=bool(ok and order[0]==target);top5=bool(target in order[:5])
    assert r['correct']==right and r['top5_correct']==top5
    checked[k].append((cid,r['signer_id'],right,top5));raw_count+=1
  for cid in sorted(manifest):
   r=stored[cid];target=manifest[cid]['label_index'];z=[allrows[f'T3F{s}'][cid]['logits'] for s in SEEDS]
   p=None
   if all(q is not None for q in z):
    probs=[probability(q) for q in z];p=[math.fsum(a[i] for a in probs)/3 for i in range(100)]
    assert max(abs(a-b) for a,b in zip(p,r['probabilities']))<1e-12
   else:assert r['probabilities'] is None
   order,ok=decision(p);right=bool(ok and order[0]==target);top5=target in order[:5]
   assert r['prediction']==(order[0] if ok else None) and r['top5']==order[:5] and r['correct']==right and r['top5_correct']==top5
   checked['E3'].append((cid,r['signer_id'],right,top5));count+=1
  for reportkey,rowskey in [('C13','C20261013'),('T3F13','T3F20261013'),('E3','E3')]:assert metric(checked[rowskey])==out['scores'][view][reportkey]
  for s in SEEDS:assert metric(checked[f'T3F{s}'])==out['scores'][view]['individual_T3F'][str(s)]
  for reportkey,rowskey in [('E3_vs_T3F13','T3F20261013'),('E3_vs_C13','C20261013')]:
   pairs=list(zip(checked[rowskey],checked['E3']));g=[b[0] for a,b in pairs if b[2] and not a[2]];l=[a[0] for a,b in pairs if a[2] and not b[2]]
   assert out['scores'][view][reportkey]=={'gains':g,'losses':l,'delta_pp':100*(len(g)-len(l))/199}
 result=dict(status='PASS',at=datetime.now(timezone(timedelta(hours=9))).isoformat(),ensemble_rows=count,raw_prediction_rows=raw_count,denominator_per_view=199,verification='Separate scalar arithmetic, hashes and manifest pairing; same Codex implementer, not separate human ground truth',inputs_unchanged=True,model_unchanged=True,training=0,new_inference=0,pins={str(p):sha(p) for p in [Path(__file__),U/'aggregate_fresh.py',F/'COMPARISON.json',F/'ENSEMBLE_ROWS.json']})
 with (F/'SCORES_VERIFIED.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
