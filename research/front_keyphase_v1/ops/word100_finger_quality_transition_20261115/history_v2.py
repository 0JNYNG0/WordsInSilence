from quality_common import *
from quality_runtime_v2 import QualityObserver,gate
import torch,time
torch.set_num_threads(2);data=np.load(P/'INPUTS.npz');rows=read(P/'INPUTS.json')['rows'];ys=data['y'];mask=(ys>=0)&data['availability'];old=np.load(P/'SUPPLEMENT_PREDICTIONS.npz')['pred'];q=QualityObserver();pred={k:np.full(ys.shape,-1)for k in ['quality_only','phase_only','combined']};cache={};latencies=[]
extras={r['clip_id']:r for r in read(O/'word100_feature_expansion_20261113/INPUTS_NEW.json')['rows']+read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']}
for i,r in enumerate(rows):
 if r['role']!='development':continue
 key=r['clip_id'],r['side']
 if key not in cache:
  if r['clip_id']in extras:z=np.load(extras[r['clip_id']]['channels']);x=z['tcn280']
  else:z=np.load(read(J/'records'/f"{r['clip_id']}.json")['channel_path']);x=z['feedback']if'feedback'in z.files else z['full']
  start=__import__('time').perf_counter();v=q.infer(x,r['side']);latencies.append(__import__('time').perf_counter()-start);cache[key]=(v,z['frame_id'])
 v,fid=cache[key];t=int(np.flatnonzero(fid==r['frame'])[0]);qq=dict(states=v['states'][t:t+1])
 for mode in pred:pred[mode][i]=gate(old[i:i+1],qq,mode)[0][0]
di=np.array([i for i,r in enumerate(rows)if r['role']=='development']);m=mask[di];y=ys[di];b=old[di];summ={}
for mode,p in dict(old=old,**pred).items():
 p=p[di];summ[mode]=dict(known=int(m.sum()),correct=int(((p==y)&m).sum()),wrong=int(((p>=0)&(p!=y)&m).sum()),unknown=int(((p<0)&m).sum()),old_correct_lost=int(((b==y)&(p<0)&m).sum()),old_correct_to_wrong=int(((b==y)&(p>=0)&(p!=y)&m).sum()))
np.savez_compressed(U/'HISTORIC_PRED_V2.npz',**pred)
write(U/'HISTORIC_RESULTS_V2.json',dict(summary=summ,clips=len(set(rows[i]['clip_id']for i in di)),anchors=len(di),timing=dict(median=float(np.median(latencies)),p95=float(np.percentile(latencies,95)),unit='additional quality/change inference per full hand sequence CPU; excludes all previous stages')))
print(json.dumps(summ))
