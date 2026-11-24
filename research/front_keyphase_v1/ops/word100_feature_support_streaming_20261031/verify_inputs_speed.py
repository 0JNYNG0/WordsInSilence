"""Parent independent arithmetic/data check, no detection/model inference."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
U=Path(__file__).resolve().parent;OPS=U.parent
sys.path.insert(0,str(OPS/'word100_guide_video_feedback_20261028'))
from motion_observer import read_observation,extract_features
from evidence_adapter import selftest
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 count=0
 def check(value):
  nonlocal count
  assert value;count+=1
 fresh=read(U/'fresh/INPUTS.json');original=read(OPS/'word100_stage_feedback_cycle_20261031/INPUTS.json');olds={r['source_sha256'] for r in original['rows']};actual=[]
 for r in fresh['rows']:
  for path,expected in [(r['input'],r['input_sha256']),(r['source'],r['source_sha256']),(r['record'],r['record_sha256'])]:check(sha(path)==expected)
  actual.append(r['source_sha256']);check(r['source_sha256'] not in olds);check(r['person'] in [7,8]);ob=read_observation(r['observation']);b=r['box'];xy=ob['xy_px'];ob['usable']&=(xy[:,:,0]>=b[0])&(xy[:,:,0]<b[2])&(xy[:,:,1]>=b[1])&(xy[:,:,1]<b[3]);p=extract_features(ob);z=np.load(r['input'],allow_pickle=False)
  check(np.array_equal(np.nan_to_num(p['x']).astype('float32'),z['x']));check(np.array_equal(p['mask'],z['mask']));check(np.array_equal(p['timestamp_us'],z['timestamp_us']));check(np.array_equal(p['frame_id'],z['frame_id']));check(z['rgb'].shape==(len(p['x']),768));check(np.isfinite(z['rgb']).all())
 check(len(set(actual))==32);s=read(U/'streaming/RESULT.json');calc={}
 for mode in {r['condition'] for r in s['rows']}:
  rr=[r for r in s['rows'] if r['condition']==mode];check(len(rr)==6);dur=[]
  for r in rr:
   trace=read(Path(r['directory'])/'TIMING_TRACE.json');check(r['source_frames_dropped']==0);check(r['online_vs_frozen_batch_assignment_exact']);check(r['old_completed_cpu_capture_comparison']['exact']);check(r['source_frames']==r['frames']);check(abs(r['after_capture_saved_tail_seconds']-max(0,(r['saved_artifacts_ns']-r['capture_end_ns'])/1e9))<1e-10);dur.append(r['after_capture_saved_tail_seconds'])
  calc[mode]=dict(n=len(rr),median=float(np.median(dur)),p95=float(np.quantile(dur,.95)),range=[min(dur),max(dur)])
 check(sum(r['frames'] for r in s['rows'])==1884)
 checks=selftest();check(all(checks.values()))
 oldunit=OPS/'word100_stage_feedback_cycle_20261031'
 for p,h in read(oldunit/'PROTOCOL.json')['protected'].items():check(sha(p)==h)
 out=dict(status='PASS_PARENT_INDEPENDENT_INPUT_AND_SPEED_ARITHMETIC',checks=count,clips=32,frames=sum(r['frames'] for r in fresh['rows']),same_source_overlap=0,speed=calc,evidence_guards=checks,note='Different code from preparer/streaming implementer; not independent human truth. No repeated inference.')
 dest=U/'review/INPUT_SPEED_INDEPENDENT.json';assert not dest.exists();dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
