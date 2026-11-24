from common import *
import torch,time,statistics
sys.path[:0]=[str(ROOT/'research/sign_model_pilot_v1/src'),str(T),str(B)]
from frozen_temporal import FrozenCTemporal
from training_core import MODEL
def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);f=read(U/'FROZEN.json');ctx=read(B/'training_CONTEXT.json')
 probes=[r for r in ctx['ordered_rows'] if r['role']=='train_candidate'][:16]
 models=[]
 for seed in SEEDS:
  info=f['models'][str(seed)];assert sha(info['path'])==info['sha256'];ck=torch.load(info['path'],map_location='cpu',weights_only=True);m=FrozenCTemporal(**MODEL);m.load_state_dict(ck['state_dict']);m.eval();models.append(m)
 inputs=[]
 for r in probes:
  cp=r['views']['sensor']['channel_file'];assert sha(cp)==r['views']['sensor']['sha256'];a=np.load(cp)['x_whole'];inputs.append((torch.from_numpy(a[None]),torch.ones((1,len(a)))))
 raw={'T3F13':[],'E3':[]}
 with torch.inference_mode():
  for _ in range(3):
   for m in models:m(*inputs[0])
  for repeat in range(5):
   for x,pad in inputs:
    for tag in (['T3F13','E3'] if repeat%2==0 else ['E3','T3F13']):
     start=time.perf_counter_ns()
     if tag=='T3F13':a=torch.softmax(models[0](x,pad),-1)
     else:a=torch.stack([torch.softmax(m(x,pad),-1) for m in models]).mean(0)
     raw[tag].append((time.perf_counter_ns()-start)/1e6);assert torch.isfinite(a).all()
 for info in f['models'].values():assert sha(info['path'])==info['sha256']
 result=dict(status='PASS_TIMING_ONLY_TRAIN_PROBES',at=now(),cpu_threads=4,torch=str(torch.__version__),clips=[r['clip_id'] for r in probes],repeat=5,measurements_ms=raw,summary={k:dict(median_ms=statistics.median(v),p95_ms=float(np.percentile(v,95)),mean_ms=statistics.mean(v)) for k,v in raw.items()},excludes='MediaPipe, video decoding, DTW, RGB/HaMeR, browser, transport; not end-to-end FPS',new_training=0,public_eval_rescored=0,user_data_access=0)
 write(U/'TIMING.json',result);print(json.dumps(result['summary']))
if __name__=='__main__':main()
