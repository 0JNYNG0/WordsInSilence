from common import *
import math
def sp(z):
 m=max(z);e=[math.exp(float(a)-m) for a in z];total=math.fsum(e);return [x/total for x in e]
def main():
 f=read(U/'FROZEN.json');rows=read(U/'SAVED_ENSEMBLE_ROWS.json');report=read(U/'SAVED_COMPARISON.json');checks=0
 for p,h in f['pins'].items():assert sha(p)==h
 for p,h in f['protected'].items():assert sha(p)==h
 for split,n in [('dev',300),('known200',200)]:
  src={s:read(f['saved_scores'][split][str(s)]['path'])['predictions'] for s in SEEDS}
  for view in VIEWS:
   maps={s:{x['clip_id']:x for x in src[s][view]} for s in SEEDS};out=rows[split][view];assert len(out)==n
   correct=top5=0;gains=losses=0
   for a in out:
    rr=[maps[s][a['clip_id']] for s in SEEDS];assert len({(q['label_index'],q['signer_id']) for q in rr})==1
    if all(q['logits'] is not None for q in rr):
     ss=[sp(q['logits']) for q in rr];prob=[math.fsum(q[i] for q in ss)/3 for i in range(100)]
     assert max(abs(x-y) for x,y in zip(prob,a['probabilities']))<1e-12
     order=sorted(range(100),key=lambda i:(-prob[i],i));assert a['prediction']==order[0] and a['top5']==order[:5]
     good=order[0]==a['label_index'];hit=a['label_index'] in order[:5]
    else:assert a['probabilities'] is None;good=hit=False
    assert good==a['correct'] and hit==a['top5_correct'];correct+=good;top5+=hit;gains+=good and not rr[0]['correct'];losses+=not good and rr[0]['correct'];checks+=1
   q=report['scores'][split][view];assert correct==q['E3']['correct'] and top5==q['E3']['top5'];assert gains==len(q['E3_vs_T3F13']['gains']) and losses==len(q['E3_vs_T3F13']['losses'])
 # Synthetic contracts do not read user recordings or fit a coefficient.
 z=np.zeros((3,100));p=combine(z);assert ranking(p)[0]==0 and abs(sum(p)-1)<1e-12
 z=np.random.default_rng(9).normal(size=(3,100));assert np.allclose(combine(z),combine(z[[2,0,1]]));assert np.allclose(combine(z+np.array([100,-30,8])[:,None]),combine(z))
 write(U/'SAVED_VERIFICATION.json',dict(status='PASS',probability_rows=checks,scalar_arithmetic=True,source_files_unchanged=True,protected_files=len(f['protected']),synthetic_contracts=['stable_ties','member_order','per_member_logit_offset'],operator='Codex separate scalar code, not independent person'))
 print('PASS:',checks,'saved probability/rank/paired rows')
if __name__=='__main__':main()
