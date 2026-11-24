"""Read existing timing/counts only. No model imports, inference or benchmark."""
from pathlib import Path
import hashlib,json,time
import numpy as np
U=Path(__file__).resolve().parent;H=U.parent;G=H.parent;O=G.parent;W=G/'candidate_description_supervision'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(xs):
 a=np.asarray(xs,float)
 return dict(n=len(a),minimum=float(a.min()),median=float(np.median(a)),maximum=float(a.max()))
def main():
 out=U/'SAVED_COST_EVIDENCE.json';assert not out.exists()
 paths=[H/'splice_rank_learning/evaluation/COMPARISON.json',H/'splice_rank_learning/evaluation/FROZEN.json',
        H/'null_candidate_learning/evaluation/COMPARISON.json',H/'splice_rank_learning/INPUTS_COMPLETE.json',
        H/'temporal_rate_rank_learning/PREPARATION_INPUTS_COMPLETE.json',H/'temporal_rate_rank_learning/LAUNCH_STARTED.json',
        H/'temporal_rate_rank_learning/LAUNCH_FINISHED.json',W/'transfer_local_timefix_core.py',W/'source_boundary_core.py',
        W/'evaluate_source_boundary.py',O/'word100_signrep_feasibility_20261029/RUNTIME.json',
        O/'word100_guide_video_feedback_20261028/HARDWARE_CHECK.json',
        O/'word100_action_feedback_demo_20261108/consumer.py',O/'word100_action_feedback_demo_20261108/capture.py',
        O/'word100_action_feedback_demo_20261108/INTEGRATION_COMPLETE.json',H/'G3_CALL_PATH_REVIEW.txt',Path(__file__)]
 pins={str(p):sha(p)for p in paths};jc=read(paths[0]);sc=read(paths[1])['scenarios'];nt=read(paths[2])['timing']
 jm={(r['seed'],r['clip_id'],r['kind']):r for r in jc['rows']};rows=[];chosen_checks=0
 for r in jc['predictions']:
  p=Path(r['original_prediction']);assert sha(p)==r['original_prediction_sha256'];pins[str(p)]=r['original_prediction_sha256']
  z=np.load(p);rr=jm[r['seed'],r['clip_id'],r['kind']];cards=rr['conditions']['local_full']['cards']
  union=set();eligible=set();family23=set()
  for i,c in enumerate(cards):
   sp=z[f'spans_{i}'];gate=z[f'presence_{i}']>=0
   assert len(sp)==len(gate)
   allsp={tuple(v)for v in sp};good={tuple(v)for v in sp[gate]};union|=allsp;eligible|=good
   if c['family']in[2,3]:family23|=good
   for cond in rr['conditions'].values():
    chosen=cond['cards'][i]['chosen']
    # Baseline uses a different selector. Check only original local_full and J.
    if cond is rr['conditions']['baseline']:continue
    if chosen and chosen.get('source') in ('local_full','splice_rank'):
     assert tuple(chosen['span'])in good;chosen_checks+=1
  expected=next(s['candidates']for s in sc if(s['clip_id'],s['kind'])==(r['clip_id'],r['kind']))
  assert len(union)==expected
  rows.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],enumerated_unique=len(union),
                   after_existing_presence=len(eligible),family23_after_presence=len(family23),
                   fraction_retained=len(eligible)/len(union)if union else 0))
 j=read(paths[3]);k=read(paths[4]);ks=read(paths[5]);ke=read(paths[6])
 normal=[r for r in rows if r['kind']=='full_normal']
 result=dict(status='READ_ONLY_SAVED_COST_AUDIT_NO_INFERENCE',at=time.time(),
  candidates_82=stats([s['candidates']for s in sc]),normal_candidates=stats([s['candidates']for s in sc if s['kind']=='full_normal']),
  cache_build_seconds=stats([r['seconds']for r in nt]),normal_cache_build_seconds=stats([r['seconds']for r in nt if r['kind']=='full_normal']),
  cache_build_scope='Saved null_candidate run: source NPZ/hash/read, candidate union, CPU4 per-window encoder, compressed feature write, three new heads, result assembly. Excludes initial model load, RTMW, raw guard/adapter, fresh presence head and UI. Not production latency.',
  gate_counts_246=stats([r['after_existing_presence']for r in rows]),gate_retention_246=stats([r['fraction_retained']for r in rows]),
  gate_normal_retention=stats([r['fraction_retained']for r in normal]),
  gate_claim='The existing presence<0 candidates are already impossible to win. Computing the same gate before encoding can omit only those candidates, preserving ordering and qualifiers. Count savings are arithmetic, not measured speed or implemented numeric parity.',
  chosen_gate_membership_checks=chosen_checks,gate_rows=rows,
  J_preparation=dict(sequences=len(j['sequences']),encoded_candidates=sum(s['encoded_candidates']for s in j['sequences']),
                     duration_seconds=None,duration_note='No start/stop duration in INPUTS_COMPLETE; do not infer from freeze/failure timestamps.'),
  K_preparation=dict(sequences=len(k['sequences']),encoded_candidates=sum(s['encoded_candidates']for s in k['sequences']),
                     wall_seconds=ke['at']-ks['at'],scope='Whole offline launcher including pins/imports/canonical mapping/label selection/encoder/file compression; capped FIT bags, not runtime exhaustive grid.'),
  hardware_saved=read(paths[10]),pins=pins,no_training=True,no_inference=True,no_benchmark=True,no_camera=True)
 for p,h in pins.items():assert sha(p)==h
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items()if k not in ('pins','gate_rows','hardware_saved')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
