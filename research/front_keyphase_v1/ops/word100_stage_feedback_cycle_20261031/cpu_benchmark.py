"""Separate CPU-only pose routes; no SignRep inference or accuracy rescoring."""
from runtime_benchmark import *
def main():
 torch.set_num_threads(4);t=time.perf_counter();info,arrays,cases=load_cases();stats=read(U/'NORMALIZATION.json');head=Head(len(info['targets']));head.load_state_dict(torch.load(U/'runs/pose_capacity_matched_42/head.pt',weights_only=True));head.eval()
 prep=module('cpu_only_capture',OPS/'word100_roi_training_20261017/prep.py');sys.path.insert(0,str(OPS/'word100_signrep_temporal_head_20261029'));from prepare_inputs import read_observation,extract_features
 contract=read(LAB/'config/contract.json');setup=time.perf_counter()-t;rows={r['alias']:r for r in info['rows']};out=[]
 save('CPU_TIMING_FREEZE.json',dict(aliases=['P005','P043','P079'],repeats=2,seed=42,no_gpu_inference=True,accuracy_not_rescored=True,reason='Separate isolated CPU pose path, because initial timing shared detector work with GPU SignRep. No threshold/model change.'))
 for rep in [1,2]:
  for alias in ['P005','P043','P079']:
   r=rows[alias];d=U/'cpu_runtime'/f'{alias}_{rep}';d.parent.mkdir(exist_ok=True);start=time.perf_counter();cap=prep.capture(Path(r['source']),r['clip_id'],d,contract,box=None);cap_s=time.perf_counter()-start
   t=time.perf_counter();raw=read_observation(cap['candidate']);b=r['box'];xy=raw['xy_px'];raw['usable']&=(xy[:,:,0]>=b[0])&(xy[:,:,0]<b[2])&(xy[:,:,1]>=b[1])&(xy[:,:,1]<b[3]);f=extract_features(raw);arr=dict(x=np.nan_to_num(f['x']).astype('float32'),mask=f['mask'],frame_id=raw['frame_id'],timestamp_us=raw['timestamp_us']);ob=roi_observation(cap['candidate'],b);nr=dict(r,observation=cap['candidate']);px=np.where(arr['mask'],(arr['x']-stats['pose_mean'])/np.array(stats['pose_std']),0);x=torch.tensor(np.concatenate([px,arr['mask']],1)[None],dtype=torch.float32);v=torch.ones((1,len(px)));aux=x@torch.from_numpy(projection());fe_s=time.perf_counter()-t
   for c in CONDITIONS[:2]:
    t=time.perf_counter();p=None
    if c!='legacy_dtw':
     with torch.no_grad():p=head(x,aux,v).sigmoid().numpy()[0]
    q=prepare_case(nr,arr,p,ob);refs=[r2 for r2 in info['rows'] if r2['role']=='fit' and r2['word_id']==r['word_id']];result=aggregate(alias,[dict(reference=r2['alias'],**pair(q,cases[c][r2['alias']],c)) for r2 in refs],c);payload=json.dumps(result,ensure_ascii=False,allow_nan=False).encode();(d/f'{c}.json').write_bytes(payload);post=time.perf_counter()-t
    out.append(dict(alias=alias,repeat=rep,condition=c,detector_seconds=cap_s,features_seconds=fe_s,head_dtw_serialize_seconds=post,total_seconds=cap_s+fe_s+post,source_sha256=r['source_sha256']));print(json.dumps(out[-1]),flush=True)
 summary=[]
 for c in CONDITIONS[:2]:
  v=[r['total_seconds'] for r in out if r['condition']==c];summary.append(dict(condition=c,median=float(np.median(v)),p95=float(np.quantile(v,.95)),range=[min(v),max(v)],n=len(v)))
 save('CPU_RUNTIME_RESULT.json',dict(summary=summary,rows=out,setup_seconds=setup,no_gpu_inference=True,quality='Same fixed CPU head as the main saved accuracy evaluation; these new detector runs are timing only.',scope='Full public raw video, fixed ROI, warm loaded head/reference, CPU MediaPipe instance startup included. Feature and detector stages shared across two route component sums. Browser separately measured. CPU-only SignRep path untested.'))
if __name__=='__main__':main()
