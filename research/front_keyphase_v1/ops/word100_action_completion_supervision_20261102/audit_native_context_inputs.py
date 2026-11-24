"""Compare saved features at the exact same original pixels/times; no inference.

Fresh bounded-context observation and full-native observation can differ despite
sharing source pixels. This audit measures that difference, not causal accuracy.
"""
from pathlib import Path
from collections import defaultdict
import json,hashlib,importlib.util
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def stats(values):
 a=np.asarray(values,float)
 return dict(count=len(a),mean=float(a.mean()) if len(a) else None,
  median=float(np.median(a)) if len(a) else None,p95=float(np.quantile(a,.95)) if len(a) else None,max=float(a.max()) if len(a) else None)
def main():
 out=U/'NATIVE_CONTEXT_INPUT_AUDIT.json';assert not out.exists()
 path=O/'word100_guide_video_feedback_20261028/motion_observer.py'
 spec=importlib.util.spec_from_file_location('plain_features',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 dummy=dict(xy_px=np.zeros((1,75,2)),observed=np.zeros((1,75),bool),usable=np.zeros((1,75),bool),frame_id=np.array([0]),timestamp_us=np.array([0]))
 names=mod.extract_features(dummy)['feature_names'][:59];assert len(names)==59
 full={r['query_id']:r for r in read(U/'PROCESS_AUTO_NATIVE_MANIFEST.json')['rows']}
 result=[];pins={str(path):sha(path)};aggregate=defaultdict(lambda:defaultdict(list))
 for role,tname,iname in [('fit','PROCESS_VIEW_TARGETS.json','PROCESS_INPUT_MANIFEST.json'),('development','DEV_PROCESS_VIEW_TARGETS.json','DEV_PROCESS_INPUT_MANIFEST.json')]:
  targets=read(U/tname)['rows'];inputs={r['pixel_view_key']:r for r in read(U/iname)['rows']}
  for n in [tname,iname]:pins[str(U/n)]=sha(U/n)
  for r in targets:
   if r['construction_kind_audit_only']!='native':continue
   f=full[r['query_id']];b=inputs[r['pixel_view_key']]
   assert f['clip_id']==b['clip_id']==r['clip_id'] and f['source_sha256']==b['source_sha256']==r['source_sha256']
   for meta in [f,b]:assert sha(meta['input'])==meta['input_sha256'];pins[meta['input']]=meta['input_sha256']
   with np.load(f['input'],allow_pickle=False) as z:a={k:z[k].copy() for k in ['x','mask','rgb','frame_id','timestamp_us','image_available']}
   with np.load(b['input'],allow_pickle=False) as z:c={k:z[k].copy() for k in ['x','mask','rgb','frame_id','timestamp_us','image_available']}
   by={int(v):i for i,v in enumerate(a['frame_id'])};ids=[e['source_native_frame'] for e in r['ordered_frame_map']];ix=[by[v] for v in ids]
   assert a['timestamp_us'][ix].tolist()==[e['source_native_timestamp_us'] for e in r['ordered_frame_map']]
   assert c['timestamp_us'].tolist()==[e['view_timestamp_us'] for e in r['ordered_frame_map']]
   assert len(c['x'])==len(ids) and np.array_equal(a['image_available'][ix],c['image_available'])
   ax=a['x'][ix,:59];cx=c['x'][:,:59];am=a['mask'][ix,:59];cm=c['mask'][:,:59]
   both=am&cm&np.isfinite(ax)&np.isfinite(cx)
   per=[]
   for j,name in enumerate(names):
    differences=np.abs(ax[both[:,j],j]-cx[both[:,j],j]);key=role+':'+name
    aggregate[key]['differences'].extend(differences.tolist())
    aggregate[key]['mask_different'].extend((am[:,j]!=cm[:,j]).tolist())
    per.append(dict(name=name,common_observed=int(both[:,j].sum()),bounded_only=int((cm[:,j]&~am[:,j]).sum()),
     full_only=int((am[:,j]&~cm[:,j]).sum()),absolute_difference=stats(differences)))
   ar=a['rgb'][ix].astype(float);cr=c['rgb'].astype(float);denom=np.linalg.norm(ar,axis=1)*np.linalg.norm(cr,axis=1)
   assert (denom>0).all()
   cosine=(ar*cr).sum(1)/denom
   interior=np.arange(len(ids));interior=(interior>=8)&(interior<len(ids)-8)
   aggregate[role+':RGB']['cosine'].extend(cosine.tolist())
   result.append(dict(query_id=r['query_id'],role=role,clip_id=r['clip_id'],person=r['recorded_person_id'],
    physical_query=r['physical_query'],frames=len(ids),source_frames=ids,source_timestamps_us=a['timestamp_us'][ix].tolist(),
    full_native_indices=ix,bounded_input=b['input'],full_input=f['input'],
    mask_cell_differences=int((am!=cm).sum()),mask_cells=len(ids)*59,
    common_observed_cells=int(both.sum()),geometry=per,
    RGB_cosine=stats(cosine),RGB_cosine_interior_at_least8frames=stats(cosine[interior]),
    exact_pixel_identity='Previously lossless-media verified; this audit compares their saved features only',
    source_frame_time_exact=True))
 assert len(result)==60
 sums={}
 for role in ['fit','development']:
  rr=[r for r in result if r['role']==role]
  sums[role]=dict(query_views=len(rr),unique_sources=len({r['clip_id'] for r in rr}),
   compared_rows=sum(r['frames'] for r in rr),mask_cell_differences=sum(r['mask_cell_differences'] for r in rr),
   mask_cells=sum(r['mask_cells'] for r in rr),RGB_cosine=stats(aggregate[role+':RGB']['cosine']),
   feature_differences={name:dict(absolute_difference=stats(aggregate[role+':'+name]['differences']),
    mask_changes=sum(aggregate[role+':'+name]['mask_different'])) for name in names})
 pins[str(U/'PROCESS_AUTO_NATIVE_MANIFEST.json')]=sha(U/'PROCESS_AUTO_NATIVE_MANIFEST.json')
 out.write_text(json.dumps(dict(status='SAVED_SAME_SOURCE_FRAME_CONTEXT_DIFFERENCES_MEASURED',rows=result,summaries=sums,pins=pins,
  interpretation=['This is same-pixel/same-source-time feature drift, not detector accuracy or an isolated cause of model failure',
   'Geometry uses per-frame shoulder center/width, so this is not automatic evidence of a changed whole-clip hand-size statistic',
   'MediaPipe temporal state/initialization and RGB window context differ; separating their causes would require a new justified experiment',
   'Multiple overlapping query views share original sources; no independent statistical replication claim'],
  model_calls=0,source_files_modified=False,new_training=0,old_scores_recomputed=0),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
 print(json.dumps({role:{k:v for k,v in s.items() if k!='feature_differences'} for role,s in sums.items()}))
if __name__=='__main__':main()
