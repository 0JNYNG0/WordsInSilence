from common import *
from criteria import compare,variant,CONDITIONS,assess_tokens
import time,traceback,os

def main():
 assert not (U/'COMPLETE.json').exists() and not (U/'RUNNING.json').exists()
 scope=read(U/'SCOPE.json');check(scope['pins'])
 write(U/'EXECUTION_CONTEXT.json',{'scope_sha256':sha(U/'SCOPE.json'),'code_sha256':sha(__file__),'pid':os.getpid(),'at':now()})
 write(U/'RUNNING.json',{'pid':os.getpid(),'at':now()})
 ctxsha=sha(U/'EXECUTION_CONTEXT.json');start=time.monotonic();records=[];samples={}
 refs={r['word_id']:r for r in scope['references']}
 ordered=sorted(scope['rows'],key=lambda r:r['clip_id'])
 paths={}
 for r in ordered:
  rec=read(B/'data/records'/f"{r['clip_id']}.json")
  paths[r['clip_id']]={v:rec['views'][v]['channel_file'] for v in scope['views']}
 def load(cid,v):
  with np.load(paths[cid][v],allow_pickle=False) as z:return z['x_whole']
 for word in sorted(refs):
  ref=refs[word]
  for view in scope['views']:
   x=load(ref['clip_id'],view)
   # All edited conditions use exactly the same reference; no semantic labels.
   for name in CONDITIONS:
    y,indices=variant(x,name)
    row={'reference':ref['clip_id'],'query':ref['clip_id'],'word_code':word,'label':ref['label'],'view':view,'condition':name,
      'source_frame_indices':indices.tolist(),'reference_seconds':float(x[:,279].sum()),'query_seconds':float(y[:,279].sum()),
      'semantic_label':'UNREVIEWED_EDIT' if name in ('omit_middle','swap_first_two','repeat_middle','mask_middle') else 'TECHNICAL_ORDER_PRESERVING_CONTROL',
      'metrics':{axis:compare(x,y,axis) for axis in ('global','local')}}
    records.append(row)
   others=[r for r in ordered if r['word_code']==word and r['clip_id']!=ref['clip_id']]
   for r in others:
    y=load(r['clip_id'],view)
    records.append({'reference':ref['clip_id'],'query':r['clip_id'],'word_code':word,'label':ref['label'],'view':view,
      'condition':'other_signer_same_word','reference_seconds':float(x[:,279].sum()),'query_seconds':float(y[:,279].sum()),
      'semantic_label':'DATASET_SAME_WORD_NO_PHASE_GROUND_TRUTH','metrics':{axis:compare(x,y,axis) for axis in ('global','local')}})
   if len(samples)<12:
    samples[ref['clip_id']+'_'+view]=x
  if len(records)%100<40:print(f'Geometry pairs {len(records)}/3398; {time.monotonic()-start:.1f}s',flush=True)
 assert len(records)==3398
 write(U/'GEOMETRY.json',{'context_sha256':ctxsha,'rows':records,'seconds':time.monotonic()-start,'model_inference':0,'training':0,'user_data_read':0})
 np.savez_compressed(U/'NUMERIC_SAMPLES.npz',**samples)
 check(scope['pins'])
 write(U/'COMPLETE.json',{'status':'GEOMETRY_COMPLETE','at':now(),'exit_code':0,'pairs':len(records),'metrics':len(records)*2,'seconds':time.monotonic()-start,
  'geometry_sha256':sha(U/'GEOMETRY.json'),'samples_sha256':sha(U/'NUMERIC_SAMPLES.npz'),'context_sha256':ctxsha})
 (U/'RUNNING.json').unlink()
 print('COMPLETE saved3398pair/view comparisons; no semantic decisions or model training/inference.',flush=True)

if __name__=='__main__':
 try:main()
 except BaseException:
  if not (U/'FIRST_EXECUTION_ERROR.json').exists():write(U/'FIRST_EXECUTION_ERROR.json',{'at':now(),'traceback':traceback.format_exc()})
  raise
