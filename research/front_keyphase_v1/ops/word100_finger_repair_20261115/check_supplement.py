from common import *
import torch,time,collections
from finger_runtime import FingerObserver
torch.set_num_threads(2);cfg=read(U/'SUPPLEMENT_FROZEN.json')
for p,h in cfg['pins'].items():assert sha(p)==h
data=np.load(U/'INPUTS.npz');rows=read(U/'INPUTS.json')['rows'];ys=data['y'];mask=(ys>=0)&data['availability'];model=FingerObserver();pred=np.full(ys.shape,-1);conf=np.zeros(ys.shape,bool);cache={};latencies=[]
extras={r['clip_id']:r for r in read(E/'INPUTS_NEW.json')['rows']+read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']}
for i,r in enumerate(rows):
    if r['role']!='development':continue
    key=r['clip_id'],r['side']
    if key not in cache:
        if r['clip_id']in extras:
            z=np.load(extras[r['clip_id']]['channels']);x=z['tcn280']
        else:
            z=np.load(read(J/'records'/f"{r['clip_id']}.json")['channel_path']);x=z['feedback']if'feedback'in z.files else z['full']
        t=time.perf_counter();v=model.infer(x,r['side']);latencies.append(time.perf_counter()-t);cache[key]=(v,z['frame_id'])
    v,fid=cache[key];j=int(np.flatnonzero(fid==r['frame'])[0]);pred[i]=v['states'][j];conf[i]=v['conflicts'][j]
def metric(ix):
    m=mask[ix];yy=ys[ix];p=pred[ix];base=data['baseline'][ix];return dict(known=int(m.sum()),correct=int(((p==yy)&m).sum()),wrong=int(((p>=0)&(p!=yy)&m).sum()),answered=int(((p>=0)&m).sum()),baseline_correct=int(((base==yy)&m).sum()),old_correct_to_wrong=int(((base==yy)&(p>=0)&(p!=yy)&m).sum()),conflicts=int((conf[ix]&m).sum()))
di=[i for i,r in enumerate(rows)if r['role']=='development'];old=[i for i in di if not rows[i]['new']];new=[i for i in di if rows[i]['new']];ring=[i for i in new if ys[i,2]!=ys[i,3]]
metrics=dict(old=metric(old),added=metric(new),ring_little=metric(ring));errors=[dict(index=i,clip_id=rows[i]['clip_id'],side=rows[i]['side'],frame=rows[i]['frame'],finger=j,truth=int(ys[i,j]),predicted=int(pred[i,j]))for i in di for j in range(4)if mask[i,j]and pred[i,j]>=0 and pred[i,j]!=ys[i,j]]
np.savez_compressed(U/'SUPPLEMENT_PREDICTIONS.npz',pred=pred,conflict=conf)
write(U/'SUPPLEMENT_POINT_CHECK.json',dict(metrics=metrics,errors=errors,timing=dict(unit='one complete hand sequence; all9smallmodels; excludesRTMW/base/DTW',median=float(np.median(latencies)),p95=float(np.percentile(latencies,95))),finger_template_output_enabled=False))
print(json.dumps(dict(metrics=metrics,errors=errors),ensure_ascii=False))
