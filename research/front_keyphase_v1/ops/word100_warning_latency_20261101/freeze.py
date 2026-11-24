from pathlib import Path
import json,hashlib,datetime
U=Path(__file__).resolve().parent;B=U.parent/'word100_multipeople_live_feedback_20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
rows=read(B/'PUBLIC_INPUTS.json')['rows']
clips=[r for r in rows if (r['word_id'],r['person']) in [('WORD0739',12),('WORD2523',13)]]
assert len(clips)==2
d=dict(status='FROZEN_BEFORE_NEW_PERFORMANCE_EXECUTION',created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 authorization='Current user requests public warning7/latency causes, improvement and regression validation; no private training or protected model replacement.',
 public_clips=clips,order=['serial','candidate','candidate','serial'],
 timing='2 clips x 2 conditions x 2 repetitions; warmup separately; one root GPU worker, no concurrent benchmarks; entire adapter measured. No failed/timed run dropped.',
 equality='All frame_id, PTS, masks, raw inputs and model outputs compared. Streaming encoder must be bit-exact for identical geometry/input. No omitted windows/native frames or changed precision. Repeat detector variability must be separately identified, never counted as optimization benefit.',
 warning='Use saved paths/posteriors and fixed FIT thresholds; no new thresholds or per-word rules. Annotate provenance and full contiguous supra-threshold interval durations, preserve original warning, peak, missing support, stages. No claim of educational truth.',
 prohibit=['user or mentor input','training/tuning/selection on user data','C/T3F or production replacement','download','external transmission','Git','old result overwrite'],
 source_pins={str(B/n):sha(B/n) for n in ['adapter.py','engine.py','evidence_guard.py','live_engine.py','PUBLIC_INPUTS.json','REFERENCE_BANK.json','NORMAL_RANGES.json','EVALUATION_TASKS.json','FEEDBACK_EVALUATION.json','GUARD_REPORT.json']})
with (U/'FROZEN.json').open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
print('Frozen',len(clips),'public clips')
