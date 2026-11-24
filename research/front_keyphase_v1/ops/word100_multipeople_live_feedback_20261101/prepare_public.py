"""Build only the prospectively selected public input bank; no private inputs."""
import sys,json,hashlib,time,traceback
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent; O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
 p=U/n;p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def main():
 assert not (U/'PUBLIC_INPUTS.json').exists(),'Completed input bank must not be rebuilt'
 f=read(U/'FROZEN.json'); inv=read(U/'data_audit/PUBLIC_PEOPLE_INVENTORY.json')
 rows=[r for r in inv['four_expression_inventory'] if r['recorded_person_id'] in f['fit_person_ids']+f['development_person_ids']]
 assert len(rows)==48 and {r['word_id'] for r in rows}==set(f['word_ids'])
 sys.path.insert(0,str(O/'word100_signrep_temporal_head_20261029'))
 from prepare_inputs import encode,compile_gpu,read_observation,extract_features
 comp=None; out=[]; t=time.perf_counter()
 for r in rows:
  cid=r['clip_id']; rec=read(r['native_record']['path'])
  obsfile=r['native_observation']['path'];assert sha(obsfile)==r['native_observation']['sha256']
  sources=[a['path'] for a in r['source_video_caches'] if a.get('matches_recorded_sha256')]
  sources.append(str(U/'data_audit/videos'/f'{cid}.mp4'))
  source=next((Path(p) for p in sources if Path(p).is_file() and sha(p)==rec['source']['sha256']),None)
  assert source is not None,f'Wait for verified public source: {cid}'
  box=rec['calibration']['box_source_pixels']; cached=r.get('stage_rgb69_input')
  if cached and cached.get('exists') and cached.get('matches_recorded_sha256'):
   dest=Path(cached['path']); secs=0; reused=True
  else:
   dest=U/'public_inputs'/f'{cid}.npz';dest.parent.mkdir(exist_ok=True)
   if not dest.exists():
    obs=read_observation(obsfile); xy=obs['xy_px'];x0,y0,x1,y1=box
    obs['usable'] &= (xy[:,:,0]>=x0)&(xy[:,:,0]<x1)&(xy[:,:,1]>=y0)&(xy[:,:,1]<y1)
    feat=extract_features(obs)
    if comp is None:comp=compile_gpu()
    rgb,ts,seq,secs=encode(source,box,len(xy),comp)
    assert np.max(np.abs(ts-obs['timestamp_us']))<=1
    mid=(ts[np.array(seq)]+ts[np.array(seq)+15])/2
    z=np.stack([np.interp(ts,mid,rgb[:,j]) for j in range(768)],1).astype(np.float32)
    np.savez_compressed(dest,x=np.nan_to_num(feat['x']).astype(np.float32),mask=feat['mask'],rgb=z,frame_id=obs['frame_id'],timestamp_us=ts)
   else:secs=None
   reused=False
  with np.load(dest,allow_pickle=False) as a:
   assert a['x'].shape[1]==69 and a['rgb'].shape[1]==768 and np.isfinite(a['rgb']).all()
   n=len(a['x']); assert np.all(np.diff(a['timestamp_us'])>0)
  out.append(dict(clip_id=cid,word_id=r['word_id'],label=r['label'],person=r['recorded_person_id'],role='fit' if r['recorded_person_id']<=9 else 'development',source=str(source),source_sha256=sha(source),observation=obsfile,observation_sha256=sha(obsfile),input=str(dest),input_sha256=sha(dest),box=box,intervals=r['word_interval_seconds'],frames=n,reused_input=reused,rgb_seconds=secs))
  save('PUBLIC_INPUT_PROGRESS.json',dict(done=len(out),total=len(rows),last=cid))
  print(cid,'reused' if reused else 'new',flush=True)
 names=read(O/'word100_semantic_feature_cycle_20261101/DATASET.json')['feature_names']
 save('PUBLIC_INPUTS.json',dict(rows=out,feature_names=names,seconds=time.perf_counter()-t,frozen_sha256=sha(U/'FROZEN.json'),source='public only; no user inputs',new_rgb_clips=sum(not r['reused_input'] for r in out)))
if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/'PREP_FIRST_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
