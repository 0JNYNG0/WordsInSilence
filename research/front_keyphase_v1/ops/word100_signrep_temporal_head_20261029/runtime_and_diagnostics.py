from train_compare import *
from prepare_inputs import compile_gpu,encode,read_observation,extract_features
def main():
 torch.set_num_threads(4);info=read(U/'INPUTS.json');rows=info['rows'];stats=read(U/'NORMALIZATION.json');batch,_,arrays=assemble(rows,stats);x,z,y,m,v=batch
 # Seed42 chosen by protocol order, never by evaluation result; speed is not
 # claimed to differ meaningfully among equal-shape seeds.
 heads={}
 for condition in ['pose_only','pose_plus_signrep']:
  h=Head(condition=='pose_plus_signrep');h.load_state_dict(torch.load(U/'runs'/f'{condition}_42/head.pt',map_location='cpu',weights_only=True));h.eval();heads[condition]=h
 t=time.perf_counter();comp=compile_gpu();setup=time.perf_counter()-t;timings=[]
 with torch.no_grad():
  for rep in range(2):
   for alias in ['V01','V07','V12']:
    i=next(i for i,r in enumerate(rows) if r['alias']==alias);r=rows[i];n=r['frames'];t=time.perf_counter()
    obs=read_observation(r['observation']);xy=obs['xy_px'];b=r['box'];obs['usable'] &= (xy[:,:,0]>=b[0])&(xy[:,:,0]<b[2])&(xy[:,:,1]>=b[1])&(xy[:,:,1]<b[3]);pose=extract_features(obs)
    px=np.where(pose['mask'],(np.nan_to_num(pose['x'])-np.array(stats['pose_mean']))/np.array(stats['pose_std']),0)
    tx=torch.from_numpy(np.concatenate([px,pose['mask']],axis=1).astype(np.float32)[None]);tv=torch.ones((1,n));pose_seconds=time.perf_counter()-t
    f,ts,ss,sec=encode(r['source'],r['box'],n,comp);assert np.array_equal(f,arrays[i]['window_features'])
    mid=(ts[np.array(ss)]+ts[np.array(ss)+15])/2;features=np.stack([np.interp(ts,mid,f[:,j]) for j in range(768)],1)
    tz=torch.from_numpy(((features-np.array(stats['rgb_mean']))/np.array(stats['rgb_std'])).astype(np.float32)[None]);ht=time.perf_counter();p=heads['pose_plus_signrep'](tx,tz,tv).sigmoid().numpy();head_seconds=time.perf_counter()-ht;total=time.perf_counter()-t
    ref=np.load(U/'runs/pose_plus_signrep_42/probabilities.npy',allow_pickle=False)[i,:n];assert np.allclose(p[0],ref,atol=2e-6,rtol=2e-5)
    timings.append(dict(alias=alias,repeat=rep,frames=n,rgb_decode_encoder_seconds=sec,cached_pose_read_calculate_seconds=pose_seconds,head_seconds=head_seconds,total_seconds=total,feature_bitwise_equal=True,prediction_close=True));print(json.dumps(timings[-1]),flush=True)
  # Isolate new tiny head cost on exactly the same clips, CPU, warmed.
  head_times={}
  for cond,h in heads.items():
   result=[]
   for i,r in enumerate(rows):
    n=r['frames'];t=time.perf_counter()
    for _ in range(10):h(x[i:i+1,:n],z[i:i+1,:n],v[i:i+1,:n])
    result.append((time.perf_counter()-t)/10)
   head_times[cond]=dict(median_seconds=float(np.median(result)),range_seconds=[min(result),max(result)])
 save('RUNTIME.json',dict(setup_compile_warmup_seconds=setup,rows=timings,median_total_seconds=float(np.median([r['total_seconds'] for r in timings])),range_total_seconds=[min(r['total_seconds'] for r in timings),max(r['total_seconds'] for r in timings)],head_only=head_times,scope='Raw public RGB decode + fixed-ROI SignRep GPU FP16 + previously saved MediaPipe observation read/features + CPU head; MediaPipe extraction, DTW, browser transfer/render NOT included; warmed engine',seed_for_timing=42))
 # No new accuracy selection: small fixed representation-level speed checks.
 # This does not recreate RGB at another speed and is not an end-to-end speed benchmark.
 diagnostic=[]
 with torch.no_grad():
  for cond,h in heads.items():
   for speed in [.75,1.25]:
    changed=total=0;known_correct=0
    for i,r in enumerate(rows):
     if r['role']=='fit_candidate':continue
     n=r['frames'];newn=round(n/speed);grid=np.linspace(0,n-1,newn)
     xx=torch.from_numpy(np.stack([np.interp(grid,np.arange(n),x[i,:n,j]) for j in range(138)],1).astype(np.float32)[None]);zz=torch.from_numpy(np.stack([np.interp(grid,np.arange(n),z[i,:n,j]) for j in range(768)],1).astype(np.float32)[None]);new=h(xx,zz,torch.ones((1,newn))).sigmoid().numpy()[0]
     p=np.stack([np.interp(np.arange(n),grid,new[:,j]) for j in range(5)],1);base=np.load(U/'runs'/f'{cond}_42/probabilities.npy',allow_pickle=False)[i,:n];mask=arrays[i]['target_mask'];yy=arrays[i]['y']
     changed+=int((((p>=.5)!=(base>=.5))&mask).sum());total+=int(mask.sum());known_correct+=int((((p>=.5)==yy)&mask).sum())
    diagnostic.append(dict(condition=cond,speed=speed,known_cells=total,changed_decisions=changed,correct_known_cells=known_correct))
 save('DIAGNOSTICS.json',dict(representation_time_resampling=diagnostic,warning='Cached pose+SignRep representations resampled, not raw RGB reenconding or true user speed test. No claim of real learner-error recognition.',missing_policy='All unknown physical labels remain outside correctness scoring; model confidence cannot convert absent visual evidence into a verified feedback claim.'))
if __name__=='__main__':main()
