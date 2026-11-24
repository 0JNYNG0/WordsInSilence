"""Single new execution condition, frozen detectors/weights, parallel graph scheduling."""
from native_replay import U,Engine,run
from backend import T
import json,time,traceback,hashlib
from pathlib import Path
def main():
 if (U/'NATIVE_PARALLEL_COMPLETE.json').exists():raise RuntimeError('Do not repeat completed condition')
 plan=json.loads((U/'FROZEN.json').read_text(encoding='utf-8'))
 with (U/'NATIVE_PARALLEL_SCOPE.json').open('x',encoding='utf-8') as f:json.dump(dict(change='Independent MP pose and hand graphs run concurrently; frozen CandidateLandmarker consumes results unchanged. TCN warmed before capture. File source has at most 2 predecoded frames, not a whole-clip cache.',rows=[r['clip_id'] for r in plan['rows']],require_same_mp_arrays=True,require_same_tcn_scores=True,no_loss=True,time_budget_ms=1000,no_camera=True,no_training=True),f,indent=2)
 for p,h in plan['protected'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 e=Engine();t=time.perf_counter();status=T.identity();warm=time.perf_counter()-t;assert status['model_status']=='READY'
 results=[run(e,row,True,parallel_mp=True,prefetch=True) for row in plan['rows']]
 (U/'NATIVE_PARALLEL_COMPLETE.json').write_text(json.dumps(dict(status='EXECUTED_NEEDS_AUDIT',tcns_prepare_seconds=warm,engine_compile_seconds=e.compile_seconds,engine_warm_seconds=e.warm_seconds,runs=results),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/('NATIVE_PARALLEL_FAILURE_'+str(time.time_ns())+'.txt')).write_text(traceback.format_exc(),encoding='utf-8');raise
