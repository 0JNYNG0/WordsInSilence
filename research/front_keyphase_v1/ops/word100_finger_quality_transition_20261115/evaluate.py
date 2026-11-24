from quality_common import *
import torch
from quality_runtime import QualityObserver,FingerObserver,gate
torch.set_num_threads(2);role=sys.argv[1]if len(sys.argv)>1 else'diagnostic';assert role in ['diagnostic','confirmation_reserved']
if role=='confirmation_reserved':assert(U/'CANDIDATE_FROZEN.json').exists()
labelpath=U/('DIAGNOSTIC_LABELS.json'if role=='diagnostic'else'CONFIRMATION_LABELS.json');rows=read(labelpath)['rows'];qnet=QualityObserver();oldnet=FingerObserver();cache={};rr=[]
def metric(y,p,old=None):
 k=y>=0;out=dict(known=int(k.sum()),correct=int(((p==y)&k).sum()),wrong=int(((p>=0)&(p!=y)&k).sum()),unknown=int(((p<0)&k).sum()))
 if old is not None:out.update(old_correct=int(((old==y)&k).sum()),old_correct_lost=int(((old==y)&(p<0)&k).sum()),old_correct_to_wrong=int(((old==y)&(p>=0)&(p!=y)&k).sum()))
 return out
def binary(y,p):
 k=y>=0;return dict(tp=int(((y==1)&(p==1)).sum()),fp=int(((y==0)&(p==1)).sum()),fn=int(((y==1)&(p!=1)).sum()),tn=int(((y==0)&(p==0)).sum()),**metric(y,p))
for r in rows:
 key=r['clip_id'],r['hand']
 if key not in cache:
  z=np.load(r['channel_path']);x=z['tcn280'];cache[key]=(oldnet.infer(x,r['hand']),qnet.infer(x,r['hand']))
 old,q=cache[key];t=r['frame'];v=dict(**r,old=old['states'][t].tolist(),quality_phase=q['states'][t].tolist())
 for mode in ['quality_only','phase_only','combined']:v[mode]=gate(old['states'],q,mode)[0][t].tolist()
 rr.append(v)
y=np.array([r['posture']for r in rr]);old=np.array([r['old']for r in rr]);summary={mode:metric(y,np.array([r[mode]for r in rr]),old)for mode in ['old','quality_only','phase_only','combined']};heads={}
for j,name in enumerate(['quality_bad','phase_changing']):
 truth=np.array([r['quality']if j==0 else r['phase']for r in rr]);truth=np.where(truth>=0,1-truth,-1)if j==0 else truth;pred=np.array([r['quality_phase']for r in rr])[:,:,j];heads[name]=dict(model=binary(truth,pred),constant_negative=binary(truth,np.zeros_like(pred)))
write(U/f'{role}_RESULTS.json',dict(summary=summary,heads=heads,rows=rr,labels_sha256=sha(labelpath),selected_before_confirmation=read(U/'CANDIDATE_FROZEN.json')['selected']if role=='confirmation_reserved'else None));print(json.dumps(dict(summary=summary,heads=heads)))
