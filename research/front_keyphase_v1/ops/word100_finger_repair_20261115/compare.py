from common import *
import collections
assert (U/'TRAINED.json').exists()
z=np.load(U/'INPUTS.npz');rows=read(U/'INPUTS.json')['rows'];y=z['y'];mask=(y>=0)&z['availability'];di=np.array([i for i,r in enumerate(rows)if r['role']=='development']);out={};errors=[]
def metrics(pred,ids):
 mm=mask[ids];p=pred[ids];yy=y[ids];base=z['baseline'][ids]
 return dict(known=int(mm.sum()),answered=int(((p>=0)&mm).sum()),correct=int(((p==yy)&mm).sum()),wrong=int(((p>=0)&(p!=yy)&mm).sum()),baseline_correct=int(((base==yy)&mm).sum()),old_correct_to_wrong=int(((base==yy)&(p>=0)&(p!=yy)&mm).sum()),baseline_unknown_recovered=int(((base<0)&(p==yy)&mm).sum()))
allp={}
for kind in ['expanded_whole_hand','shared_finger_chain']:
 pp=np.stack([np.load(U/kind/f'{seed}_predictions.npz')['predictions']for seed in [42,43,44]]);pred=np.where((pp==pp[0]).all(0),pp[0],-1);allp[kind]=pred
 old=np.array([i for i in di if not rows[i]['new']]);new=np.array([i for i in di if rows[i]['new']]);out[kind]=dict(old=metrics(pred,old),new=metrics(pred,new))
 for i in di:
  for f in np.flatnonzero(mask[i]&(pred[i]>=0)&(pred[i]!=y[i])):errors.append(dict(kind=kind,index=int(i),clip_id=rows[i]['clip_id'],frame=rows[i]['frame'],side=rows[i]['side'],finger=int(f),expected=int(y[i,f]),predicted=int(pred[i,f]),baseline=int(z['baseline'][i,f])))
write(U/'ENSEMBLE_COMPARISON.json',dict(metrics=out,errors=errors))
print(json.dumps(dict(metrics=out,errors=errors),ensure_ascii=False))
