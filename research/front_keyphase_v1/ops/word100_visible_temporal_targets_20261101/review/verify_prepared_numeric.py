"""Root numeric check, separate from input implementation; no model inference."""
from pathlib import Path
import json,hashlib,importlib.util,sys,numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent.parent;O=U.parent
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('previous_readonly_scalar_audit',O/'word100_stage_feedback_cycle_20261031/review/verify_inputs.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
def main():
 inp=read(U/'INPUTS.json');sel=read(U/'SELECTION.json');vi=read(U/'VISUAL_INDEX.json');rows=inp['rows'];out=[]
 assert len(rows)==56 and len({r['clip_id'] for r in rows})==56
 assert {r['person'] for r in rows if r['role']=='fit'}=={1,2,3}
 assert {r['person'] for r in rows if r['role']=='development'}=={4}
 assert {r['person'] for r in rows if r['role']=='new_feature_expression_evaluation'}=={7,8}
 assert not ({r['word_id'] for r in rows if r['role']=='fit'} & {r['word_id'] for r in rows if r['role']=='new_feature_expression_evaluation'})
 old=read(O/'word100_stage_feedback_cycle_20261031/LEARNING_INPUTS.json')
 oldrows=old['rows'];oldby={r['clip_id']:r for r in oldrows}
 for r in rows:
  for key in ['input','observation','record','source']:assert sha(r[key])==r[key+'_sha256']
  rec=read(r['record']);assert rec['role']=='train_candidate' and rec['clip_id']==r['clip_id']
  assert rec['source']['sha256']==r['source_sha256']
  data=dict(np.load(r['input'],allow_pickle=False));obs=dict(np.load(r['observation'],allow_pickle=False));n=len(data['frame_id'])
  assert n==r['frames'] and np.array_equal(data['frame_id'],np.arange(n))
  assert np.array_equal(data['timestamp_us'],obs['timestamp_us'])
  assert np.diff(data['timestamp_us']).min()>0
  assert data['x'].shape==(n,69) and data['rgb'].shape==(n,768) and data['mask'].dtype==bool
  assert np.isfinite(data['x']).all() and np.isfinite(data['rgb']).all() and (data['x'][~data['mask']]==0).all()
  scalar=None
  if r['reused_input']:
   oldr=oldby[r['clip_id']];assert str(Path(oldr['input']))==str(Path(r['input']))
   assert sha(oldr['input'])==r['input_sha256']
  else:
   raw={k:obs[k].copy() for k in ['xy_px','observed','usable','frame_id','timestamp_us']};x0,y0,x1,y1=r['box'];xy=raw['xy_px']
   inside=(xy[:,:,0]>=x0)&(xy[:,:,0]<x1)&(xy[:,:,1]>=y0)&(xy[:,:,1]<y1)
   raw['usable'] &= inside
   exp=audit.observer.extract_features(raw)
   assert exp['feature_names']==r['feature_names']
   assert np.array_equal(np.nan_to_num(exp['x']).astype(np.float32),data['x'])
   assert np.array_equal(exp['mask'],data['mask'])
   scalar=audit.independent_scalar_check(raw,exp)
   alter={k:v.copy() for k,v in raw.items()};alter['xy_px'][~inside]=[1e6,-1e6]
   pert=audit.observer.extract_features(alter)
   assert np.array_equal(pert['mask'],exp['mask']) and np.array_equal(pert['x'][exp['mask']],exp['x'][exp['mask']])
   starts=list(range(0,n-15,2))
   if starts[-1]!=n-16:starts.append(n-16)
   assert np.array_equal(data['window_starts'],starts)
   mid=(data['timestamp_us'][starts]+data['timestamp_us'][np.array(starts)+15])/2
   assert np.array_equal(mid,data['window_center_us'])
   interp=np.stack([np.interp(data['timestamp_us'],mid,data['window_features'][:,j]) for j in range(768)],1).astype(np.float32)
   assert np.array_equal(interp,data['rgb'])
  samples=[s for s in vi['samples'] if s['alias']==r['alias']];assert len(samples)==3
  for s in samples:
   ids=s['frame_ids'];assert len(ids)==13 and ids==list(range(ids[0],ids[0]+13)) and ids[6]==s['center_frame']
   assert data['timestamp_us'][ids].tolist()==s['timestamps_us'] and s['timestamps_us'][6]==s['center_timestamp_us']
  out.append(dict(alias=r['alias'],frames=n,reused_input=bool(r['reused_input']),scalar=scalar))
 assert sum(r['reused_input'] for r in out)==48
 assert all(sha(p)==h for p,h in sel['protected'].items())
 result=dict(status='PASS_ROOT_NUMERIC_VERIFICATION',clips=len(out),original_frames=sum(r['frames'] for r in out),windows=len(vi['samples']),old_inputs_hash_reused=48,new_inputs_scalar_checked=8,scalar_cells=sum(r['scalar']['compared_cells'] for r in out if r['scalar']),protected_pins=len(sel['protected']),rows=out,not_visual_or_linguistic_validity=True,reviewer='Root Codex separate numeric calculation, same system as implementer',input_sha256=sha(U/'INPUTS.json'),selection_sha256=sha(U/'SELECTION.json'))
 with (U/'review/INPUT_NUMERIC_VERIFICATION.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
 print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
