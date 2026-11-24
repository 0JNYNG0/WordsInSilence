"""Stored evidence only; compare first-upload vs import-preloaded fresh upload."""
from benchmark import read,save,sha,compare,U,B
import numpy as np
def job(prefix):
 info=read(U/f'{prefix}_PUBLIC_UPLOAD_STARTED.json');matches=list((U/'sessions').glob('*/'+info['job_id']));assert len(matches)==1
 return info,matches[0]
def main(prefix='PRELOAD'):
 first,p=job('CANDIDATE');after,q=job(prefix);assert first['sha256']==after['sha256']==sha(after['source'])
 changes=[];checked=0;arrays=[];source_path_only_metadata=[]
 for fn in ['analysis.npz','rgb_windows.npz','observations.npz','baseline.npz','raw.npz','face.npz','assignment.npz']:
  with np.load(p/'adapter'/fn,allow_pickle=False) as a,np.load(q/'adapter'/fn,allow_pickle=False) as b:
   assert a.files==b.files
   for k in a.files:
    checked+=1;x,y=a[k],b[k];assert x.dtype==y.dtype,(fn,k,'dtype')
    if x.dtype.kind in 'SU' and x.shape==():
     try:
      j1,j2=__import__('json').loads(x.item()),__import__('json').loads(y.item())
      for field in ['source_path']:
       if field in j1 and field in j2:
        source_path_only_metadata.append(dict(file=fn,key=k,field=field,before=j1.pop(field),after=j2.pop(field)))
      eq=j1==j2
     except (ValueError,TypeError):eq=np.array_equal(x,y)
    else:eq=np.array_equal(x,y,equal_nan=True) if x.dtype.kind in 'fciu' else np.array_equal(x,y)
    arrays.append(dict(file=fn,key=k,equal=bool(eq),shape=list(x.shape)))
    if not eq:changes.append([fn,k])
 for fn in ['CALIBRATION.json','assignment_trace.json']:
  checked+=1
  if read(p/'adapter'/fn)!=read(q/'adapter'/fn):changes.append(fn)
 for fn in ['observations.sidecar.json','baseline.sidecar.json']:
  a,b=read(p/'adapter'/fn),read(q/'adapter'/fn);a.pop('source_path',None);b.pop('source_path',None);checked+=1
  if a!=b:changes.append(fn)
 for fn in ['FINAL_FEEDBACK.json','RESEARCH_FEEDBACK.json','DETAIL_FEEDBACK.json']:
  a,b=read(p/fn),read(q/fn)
  for key in ['stages','references','selected_whole_references','evidence_guard']:
   checked+=1
   if a[key]!=b[key]:changes.append([fn,key])
 for d in [p,q]:
  m=read(d/'adapter'/'COMPLETE.json');checked+=1
  assert m['source_sha256']==first['sha256'] and not m['training'] and not m['cached_predictions_used']
 ui=read(U/f'{prefix}_LIVE_UI_VERIFICATION.json');assert ui['status']=='PASS'
 # Exact mathematical feature trace preserved; startup moves library imports only.
 result=dict(status='PASS' if not changes else 'FAIL',checks=checked,changes=changes,arrays=arrays,
  first_job=str(p),after_job=str(q),public_source=first['source'],source_sha256=first['sha256'],
  source_path_only_metadata=source_path_only_metadata,
  before_request=read(p/'REQUEST_TIMING.json'),after_request=read(q/'REQUEST_TIMING.json'),
  after_startup=read(q.parent/'STARTUP.json'),new_training=False,user_inputs=0,
  source_pins={str(U/n):sha(U/n) for n in ['candidate_adapter.py','run_server.py','app.js','detail_enrichment.py',f'{prefix}_LIVE_UI_VERIFICATION.json']})
 save(U/('LIVE_ROLLBACK_VERIFICATION.json' if prefix=='ROLLBACK' else 'LIVE_FINAL_VERIFICATION.json'),result)
 print(result['status'],'checks',checked,'changes',changes)
if __name__=='__main__':
 import sys
 main('ROLLBACK' if '--rollback' in sys.argv else 'PRELOAD')
