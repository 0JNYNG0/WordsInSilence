"""Independent recount of saved diagnostics; never constructs a model."""
import json,hashlib,math
from pathlib import Path
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
 assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
 ctx=read(U/'CONTEXT_v2.json');ss=read(U/'SENSITIVITY.json');fs=read(U/'FEATURE_STATS.json');assert not (U/'RUNNING.json').exists()
 for p,h in {**ctx['pins'],**ss['logit_file_pins']}.items():assert sha(p)==h
 groups={'position':(0,6),'extent':(6,3),'intersection':(9,3),'proximity':(12,3)}
 pc={(r['clip_id'],r['view']):r for r in fs['perclip']};counter={};count=0;disagree={k:[0,0] for k in groups}
 for r in ctx['rows']:
  two={}
  for v,p in r['views'].items():
   with np.load(p,allow_pickle=False) as z:
    xx=z['x_whole'];features=xx[:,280:];st=np.empty((len(xx),2,4),np.int8)
    for side in (0,1):
     for j,(g,(off,sz)) in enumerate(groups.items()):
      a=features[:,15*side+off:15*side+off+sz];assert np.logical_or(a==0,a==1).all() and np.all(a.sum(1)==1);st[:,side,j]=a.argmax(1)
    two[v]=(z['whole_frame_id'].copy(),z['whole_ts'].copy(),st)
   for j,(g,(_,sz)) in enumerate(groups.items()):
    n=np.bincount(st[:,:,j].ravel(),minlength=sz);key=r['role'],v,g
    counter.setdefault(key,np.zeros(sz,np.int64));counter[key]+=n
    assert np.allclose(n/n.sum(),pc[r['clip_id'],v]['groups'][g]['fraction'],rtol=0,atol=1e-14)
    count+=st[:,:,j].size
  aa,bb=two['masked'],two['sensor'];bidx={int(f):i for i,f in enumerate(bb[0])}
  for i,f in enumerate(aa[0]):
   if int(f) not in bidx:continue
   k=bidx[int(f)];assert aa[1][i]==bb[1][k]
   for j,g in enumerate(groups):disagree[g][0]+=int(np.count_nonzero(aa[2][i,:,j]!=bb[2][k,:,j]));disagree[g][1]+=2
 for (role,v,g),counts in counter.items():assert counts.tolist()==fs['distribution'][role][v][g]['hand_frame_counts']
 for g,(n,total) in disagree.items():assert [n,total]==[fs['same_clip_spatial_view_disagreement'][g]['different'],fs['same_clip_spatial_view_disagreement'][g]['total']]
 nr=0
 for run in ss['runs']:
  z=np.load(U/f"FROZEN_TRAIN_{run['seed']}.npz",allow_pickle=False);y=z['labels'];a=z['logits'].astype(float)
  assert a.shape==(2,11,899,100) and len(set(z['clip_ids']))==899 and np.isfinite(a).all()
  for vi,v in enumerate(z['views']):
   bp=a[vi,0].argmax(1);truth=np.eye(100,dtype=bool)[y];bm=a[vi,0][truth]-np.where(truth,-np.inf,a[vi,0]).max(1)
   expected={}
   for ci,cond in enumerate(z['conditions']):
    logits=a[vi,ci];pred=np.argmax(logits,axis=1);maximum=logits.max(1);ce=float(np.mean(maximum+np.log(np.exp(logits-maximum[:,None]).sum(1))-logits[np.arange(899),y]));margin=logits[truth]-np.where(truth,-np.inf,logits).max(1)
    q=run['views'][v][cond];assert q['correct']==int(sum(pred==y));assert abs(q['ce']-ce)<1e-11
    assert q['prediction_changed']==int(sum(pred!=bp));assert q['gained']==int(sum((bp!=y)&(pred==y)));assert q['lost']==int(sum((bp==y)&(pred!=y)));assert q['net_correct']==q['gained']-q['lost']
    assert abs(q['mean_abs_true_margin_change']-float(abs(margin-bm).mean()))<1e-11
    expected[cond]=ce;nr+=899
   for cond,ce in expected.items():assert abs(run['views'][v][cond]['delta_ce']-(ce-expected['original']))<1e-11
 out={'status':'PASS','training':0,'new_DEV_inference':0,'frozen_TRAIN_prediction_rows_recounted':nr,'feature_state_values_recounted':count,'aligned_state_disagreement_verified':True,'protected_file_hashes_unchanged':True,'operator':'Codex local separate code; not independent human/agent','saved_diagnostic_rows_are_not_independent_test_samples':True}
 save(U/'INDEPENDENT.json',out);print(out)
if __name__=='__main__':main()
