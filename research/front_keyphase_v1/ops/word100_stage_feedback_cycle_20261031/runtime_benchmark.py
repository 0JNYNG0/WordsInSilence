"""Raw public video timing, never a second accuracy evaluation or user capture."""
from feedback import *
import importlib.util,concurrent.futures
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
 torch.set_num_threads(4)
 t=time.perf_counter();info,arrays,cases=load_cases();stats=read(U/'NORMALIZATION.json');k=len(info['targets']);heads={}
 for c in CONDITIONS[1:]:
  h=Head(k);h.load_state_dict(torch.load(U/'runs'/f'{c}_42/head.pt',weights_only=True));h.eval();heads[c]=h
 prep=module('latency_original_capture',OPS/'word100_roi_training_20261017/prep.py')
 sys.path.insert(0,str(OPS/'word100_signrep_temporal_head_20261029'))
 from prepare_inputs import compile_gpu,encode,read_observation,extract_features
 comp=compile_gpu();setup=time.perf_counter()-t;contract=read(LAB/'config/contract.json');rows={r['alias']:r for r in info['rows']};out=[]
 assert read(U/'RUNTIME_PROTOCOL.json')['aliases']==['P005','P043','P079']
 def capture(r,d):
  t=time.perf_counter();ret=prep.capture(Path(r['source']),r['clip_id'],d,contract,box=None);return ret,time.perf_counter()-t
 def encode_timed(r):
  return encode(r['source'],r['box'],r['frames'],comp)
 for repeat in [1,2]:
  for alias in ['P005','P043','P079']:
   r=rows[alias];d=U/'runtime'/f'{alias}_{repeat}';d.parent.mkdir(exist_ok=True);start=time.perf_counter()
   with concurrent.futures.ThreadPoolExecutor(2) as pool:
    fcap=pool.submit(capture,r,d);frgb=pool.submit(encode_timed,r);cap,cap_seconds=fcap.result();features,ts,starts,rgb_seconds=frgb.result()
   parallel_seconds=time.perf_counter()-start
   t=time.perf_counter();raw=read_observation(cap['candidate']);xy=raw['xy_px'];b=r['box'];raw['usable']&=(xy[:,:,0]>=b[0])&(xy[:,:,0]<b[2])&(xy[:,:,1]>=b[1])&(xy[:,:,1]<b[3]);feat=extract_features(raw)
   assert np.max(np.abs(ts-raw['timestamp_us']))<=1
   mid=(ts[np.array(starts)]+ts[np.array(starts)+15])/2;z=np.stack([np.interp(ts,mid,features[:,j]) for j in range(768)],1).astype('float32')
   arr=dict(x=np.nan_to_num(feat['x']).astype('float32'),mask=feat['mask'],rgb=z,frame_id=raw['frame_id'],timestamp_us=raw['timestamp_us'])
   nr=dict(r,observation=cap['candidate']);ob=roi_observation(cap['candidate'],b)
   px=np.where(arr['mask'],(arr['x']-stats['pose_mean'])/np.array(stats['pose_std']),0);x=torch.tensor(np.concatenate([px,arr['mask']],1)[None],dtype=torch.float32);v=torch.ones((1,len(px)));zinput=torch.tensor((z-stats['rgb_mean'])/np.array(stats['rgb_std']),dtype=torch.float32)[None]
   feature_seconds=time.perf_counter()-t
   for c in CONDITIONS:
    t=time.perf_counter();p=None
    if c!='legacy_dtw':
     aux=x@torch.from_numpy(projection()) if c=='pose_capacity_matched' else zinput
     with torch.no_grad():p=heads[c](x,aux,v).sigmoid().numpy()[0]
    head_seconds=time.perf_counter()-t;t=time.perf_counter();q=prepare_case(nr,arr,p,ob)
    refs=[rr for rr in info['rows'] if rr['role']=='fit' and rr['word_id']==r['word_id']]
    pairs=[dict(reference=rr['alias'],**pair(q,cases[c][rr['alias']],c)) for rr in refs];result=aggregate(alias,pairs,c);compare_seconds=time.perf_counter()-t
    t=time.perf_counter();payload=json.dumps(result,ensure_ascii=False,allow_nan=False).encode('utf8');(d/f'{c}_response.json').write_bytes(payload);serialize_seconds=time.perf_counter()-t
    # Same measured detector operation shared; route totals are component sums.
    front=parallel_seconds if c=='pose_plus_signrep' else cap_seconds
    resultrow=dict(alias=alias,repeat=repeat,condition=c,frames=len(px),duration_seconds=float(ts[-1]/1e6),mediapipe_decode_assignment_save_seconds=cap_seconds,rgb_decode_gpu_seconds=rgb_seconds,parallel_frontend_seconds=parallel_seconds,feature_processing_seconds=feature_seconds,head_seconds=head_seconds,dtw_three_references_seconds=compare_seconds,response_serialization_seconds=serialize_seconds,total_component_seconds=front+feature_seconds+head_seconds+compare_seconds+serialize_seconds,response_bytes=len(payload),response=str(d/f'{c}_response.json'),capture_pins=cap['pins'],accuracy_not_rescored=True)
    out.append(resultrow);print(json.dumps({kk:resultrow[kk] for kk in ['alias','repeat','condition','total_component_seconds','mediapipe_decode_assignment_save_seconds','rgb_decode_gpu_seconds']},ensure_ascii=False),flush=True)
 summary=[]
 for c in CONDITIONS:
  a=[r['total_component_seconds'] for r in out if r['condition']==c];summary.append(dict(condition=c,n=len(a),median=float(np.median(a)),p95=float(np.quantile(a,.95)),range=[min(a),max(a)]))
 save('RUNTIME_RESULT.json',dict(setup_seconds=setup,rows=out,summary=summary,device='MediaPipe CPU + Intel Arc140V OpenVINO GPU SignRep + CPU head/DTW',scope='Warm loaded engines and precomputed references; full original video decoded after simulated recording end. Detector startup, assignment, disk evidence saves included. Fixed public ROI supplied. Component-sum route estimate shares one measured detector operation; not live webcam end-to-end. Browser transport/render measured separately.',no_expression_or_contact_model=True,cpu_only_signrep_not_measured=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/'RUNTIME_FIRST_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
