"""Expose the existing SignRep pre-linear feature, public allowlist only.

This supports real low-rank fine-tuning of pretrained fc.1.weight, avoiding
repeated backbone training. Original model/checkpoint/IR are read-only.
"""
from pathlib import Path
import sys,json,time,hashlib,importlib.util,traceback
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;S=O/'word100_signrep_feasibility_20261029'
sys.path.insert(0,str(S));import common
from prepare import read,sha,save
def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def main():
 assert not (U/'PROJECTION_PREPARED.json').exists()
 manifest=read(U/'INPUT_MANIFEST.json')
 import openvino as ov,torch
 core=ov.Core(); model=core.read_model(S/'converted/signrep_features_fp32.xml')
 last=model.output(0).get_node().input_value(0).get_node()
 assert last.get_type_name()=='MatMul' and 'fc.1' in last.get_friendly_name()
 weight=np.array(last.input_value(1).get_node().get_data(),copy=True)
 ckpt=torch.load(S/'assets/ckpt.pt',weights_only=True,map_location='cpu')['model']
 assert np.array_equal(weight,ckpt['fc.1.weight'].numpy()) and weight.shape==(768,768)
 model.add_outputs([last.input_value(0)]);model.reshape([1,3,16,224,224])
 assert any(d.startswith('GPU') for d in core.available_devices)
 t=time.perf_counter();compiled=core.compile_model(model,'GPU',{'PERFORMANCE_HINT':'LATENCY','INFERENCE_PRECISION_HINT':'f16'})
 device=list(compiled.get_property('EXECUTION_DEVICES'));assert all('GPU' in x for x in device)
 compile_seconds=time.perf_counter()-t
 if not (U/'PRETRAINED_PROJECTION.npy').exists():
  with (U/'PRETRAINED_PROJECTION.npy').open('xb') as f:np.save(f,weight,allow_pickle=False)
 adapter=load_module('_learned_guide_public_encoder',O/'word100_multipeople_live_feedback_20261101/adapter.py')
 done=[]
 class Tap:
  def __init__(self):self.pre=[]
  def __call__(self,inputs):
   out=compiled(inputs);self.pre.append(np.array(out[1],np.float32,copy=True).reshape(1,768))
   return {0:np.array(out[0],np.float32,copy=True)}
 for r in manifest['rows']:
  dest=U/'projection_inputs'/f"{r['clip_id']}.npz";meta=dest.with_suffix('.json');dest.parent.mkdir(exist_ok=True)
  if dest.exists():
   assert meta.exists(),'Partial output requires audit, not blind rerun'
   record=read(meta);assert sha(dest)==record['sha256'];done.append(record);continue
  assert sha(Path(r['source']))==r['source_sha256']
  a=np.load(r['input'],allow_pickle=False);tap=Tap();start=time.perf_counter()
  rgb,features,starts,centers=adapter._encode(r['source'],r['box'],a['frame_id'],a['timestamp_us'],tap,None)
  pre=np.concatenate(tap.pre)
  pre_frame=np.stack([np.interp(a['timestamp_us'],centers,pre[:,j]) for j in range(768)],1).astype(np.float32)
  parity=common.diff(features,pre@weight.T)
  cached=common.diff(a['rgb'],rgb)
  # Fixed numerical checks, not downstream accuracy thresholds.
  assert parity['max_relative_l2']<.005 and cached['max_relative_l2']<.005,(r['clip_id'],parity,cached)
  with dest.open('xb') as f:np.savez_compressed(f,prelinear=pre_frame,rgb=rgb,window_prelinear=pre,window_features=features,
   window_starts=starts,frame_id=a['frame_id'],timestamp_us=a['timestamp_us'])
  record=dict(clip_id=r['clip_id'],path=str(dest),sha256=sha(dest),seconds=time.perf_counter()-start,
    projection_parity=parity,cached_feature_parity=cached,source_sha256=r['source_sha256'])
  with meta.open('x',encoding='utf8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
  done.append(record)
  (U/'PROJECTION_PROGRESS.json').write_text(json.dumps(dict(done=len(done),total=48,last=r['clip_id'],device=device)),encoding='utf8')
  print(r['clip_id'],len(done),'/',48,round(record['seconds'],3),flush=True)
 save('PROJECTION_PREPARED.json',dict(status='PASS_PRETRAINED_PROJECTION_INPUTS',rows=done,device=device,compile_seconds=compile_seconds,
  trainable_signrep_weight='fc.1.weight via rank-limited additive update',backbone_frozen=True,
  checkpoint_sha256=sha(S/'assets/ckpt.pt'),projection_sha256=sha(U/'PRETRAINED_PROJECTION.npy'),
  new_public_feature_extractions=len(done),new_training=0,private_inputs=0))
if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/'PROJECTION_FIRST_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
