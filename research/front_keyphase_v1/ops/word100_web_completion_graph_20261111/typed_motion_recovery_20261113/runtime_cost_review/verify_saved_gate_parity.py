"""Verify exact selector parity using saved logits only; never load a model."""
from pathlib import Path
import hashlib,json,numpy as np
U=Path(__file__).resolve().parent;H=U.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
 dest=U/'SAVED_GATE_PARITY.json';assert not dest.exists()
 path=H/'splice_rank_learning/evaluation/COMPARISON.json';d=read(path)
 rm={(r['seed'],r['clip_id'],r['kind']):r for r in d['rows']};pins={str(path):sha(path),str(Path(__file__)):sha(__file__)}
 checked=[]
 for p in d['predictions']:
  r=rm[p['seed'],p['clip_id'],p['kind']]
  for condition,name,hashname in [('local_full','original_prediction','original_prediction_sha256'),('splice_rank','path','sha256')]:
   assert sha(p[name])==p[hashname];pins[p[name]]=p[hashname];z=np.load(p[name])
   for i,c in enumerate(r['conditions'][condition]['cards']):
    sp=z[f'spans_{i}'];presence=z[f'presence_{i}'];logits=z[f'{condition}_logits_{i}']
    assert np.isfinite(presence).all()and np.isfinite(logits).all()
    keep=np.flatnonzero(presence>=0);original=np.where(presence>=0,logits,-np.inf)
    a=int(original.argmax())if len(original)and np.isfinite(original).any()else None
    b=int(keep[np.argmax(logits[keep])])if len(keep)else None
    assert a==b
    expected=c['chosen']['span']if c['chosen']else None
    assert (sp[a].tolist()if a is not None else None)==expected
    if a is not None:assert float(original[a])==float(logits[b])
    checked.append(dict(seed=p['seed'],clip_id=p['clip_id'],kind=p['kind'],condition=condition,card=c['card'],family=c['family'],
                        enumerated=len(sp),admitted=len(keep),chosen_index=a,chosen_span=expected,
                        tied_admitted_best=int(np.sum(logits[keep]==logits[b]))if b is not None else 0))
 for p,h in pins.items():assert sha(p)==h
 result=dict(status='SAVED_LOGIT_ARGMAX_PARITY_PASS_NO_INFERENCE',checks=len(checked),all_exact=True,
             scope='Original local_full and J across all 82 scenarios and 3 seeds. Same per-card ordering; presence>=0 unchanged. This proves selector algebra for saved scores, not batch numeric parity of a future pruned encoder.',
             no_training=True,no_inference=True,no_benchmark=True,rows=checked,pins=pins)
 dest.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 print('Exact saved selector parity',len(checked),'cards;',len(pins),'pins unchanged')
if __name__=='__main__':main()
