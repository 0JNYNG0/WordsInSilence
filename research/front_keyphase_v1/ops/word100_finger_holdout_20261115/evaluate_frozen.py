from holdout_common import *
import torch,time,collections
sys.path[:0]=[str(P),str(D)]
from finger_runtime import FingerObserver
torch.set_num_threads(2)
role=sys.argv[1]if len(sys.argv)>1 else'diagnostic'
assert role in ['diagnostic','confirmation_reserved']
if role=='confirmation_reserved':assert(U/'CANDIDATE_FROZEN.json').exists(),'Reserve not released yet'
out=U/f'{role}_old';out.mkdir(exist_ok=True);assert not(out/'RESULTS.json').exists()
for p,h in read(U/'FROZEN.json')['model_pins'].items():assert sha(p)==h
labels=read(U/'LABELS_FROZEN.json')['rows'];sources=read(U/'EXTRACTED.json')['rows'];observer=FingerObserver();reports=[]
for r in sources:
 if r['role']!=role:continue
 x=np.load(r['channel_path'])['tcn280'];rr=[v for v in labels if v['clip_id']==r['clip_id']];h=rr[0]['hand'];t=time.perf_counter();q=observer.infer(x,h);elapsed=time.perf_counter()-t
 np.savez_compressed(out/f"{r['clip_id']}.npz",states=q['states'],conflicts=q['conflicts'],available=q['available'],**{k:v for k,v in q['probabilities'].items()})
 for v in rr:
  i=v['frame'];truth=np.array(v['y']);pred=q['states'][i];known=truth>=0
  reports.append(dict(**v,pred=pred.tolist(),available=q['available'][i].tolist(),wrong=np.flatnonzero(known&(pred>=0)&(truth!=pred)).tolist(),unknown=np.flatnonzero(known&(pred<0)).tolist(),correct=np.flatnonzero(known&(truth==pred)).tolist()))
def metrics(rr):return dict(known=sum(sum(v>=0 for v in r['y'])for r in rr),correct=sum(len(r['correct'])for r in rr),wrong=sum(len(r['wrong'])for r in rr),unknown=sum(len(r['unknown'])for r in rr))
summary=dict(all=metrics(reports),left=metrics([r for r in reports if r['hand']=='L']),front=metrics([r for r in reports if r['view']=='front_pointing_index']),different_ring_little=metrics([r for r in reports if r['y'][2]>=0 and r['y'][3]>=0 and r['y'][2]!=r['y'][3]]))
write(out/'RESULTS.json',dict(role=role,summary=summary,rows=reports,labels_sha256=sha(U/'LABELS_FROZEN.json')))
print(json.dumps(summary));print('errors',[(r['clip_id'],r['frame'],r['wrong'])for r in reports if r['wrong']])
