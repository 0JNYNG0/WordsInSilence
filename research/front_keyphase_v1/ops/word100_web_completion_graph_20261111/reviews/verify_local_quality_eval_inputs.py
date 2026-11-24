"""Saved diagnostic inputs only: no model construction/forward/evaluation."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime
import sys,json,hashlib,importlib.util,ast
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parents[1];O=U.parent;W=U/'local_motion_quality';R=O/'word100_candidate_role_learning_20261111'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ar=load('_quality_input_arithmetic',U/'reviews/verify_local_quality_inputs.py')

def main():
 manifest=read(W/'EVAL_INPUTS_COMPLETE.json');rows=manifest['rows']
 sources=read(R/'EVALUATION_INPUTS.json')['rows']+[dict(r,kind='dynamic_complete_blocks') for r in read(U/'dynamic_blocks/INPUTS_COMPLETE.json')['rows']]
 assert len(sources)==82 and len(rows)==2926
 boundary={(r['clip_id'],r['kind']):r for r in read(U/'boundary_candidate_probe/RESULTS.json')}
 phase={(r['clip_id'],r['kind']):r for r in read(U/'candidate_quality_probe/RESULTS.json')}
 expected=[];sourcearrays={};sourcechecks=[]
 for s in sources:
  key=s['clip_id'],s['kind'];assert sha(s['path'])==s['sha256']
  with np.load(s['path'],allow_pickle=False) as f:
   z=f['z'].copy();x=f['x'].copy();ts=f['ts'].copy();fh=f['fh'].copy()
  assert z.shape==(len(x),684) and np.array_equal(fh,z[:,392:456]) and np.all(np.diff(ts)>0)
  sourcearrays[key]=dict(z=z,x=x,ts=ts,series={})
  for c in boundary[key]['candidates']:expected.append((s,dict(c,origin='boundary',span=[c['a'],c['b']])))
  for c in phase[key]['candidates']:expected.append((s,dict(c,origin='phase')))
  sourcechecks.append(dict(clip_id=key[0],kind=key[1],frames=len(x),source_sha256=s['sha256'],fh_exactly_matches_cached_slice=True))
 assert len(expected)==len(rows)
 counts=Counter();coverage=Counter();unique=defaultdict(list);maxsummary=0.;maxseq=0.;details=[]
 for row,(source,c) in zip(rows,expected):
  key=source['clip_id'],source['kind'];d=sourcearrays[key];a,b=c['span'];axis,side=c['axis'],c['moving'];k=axis*2+(side=='L')
  assert row['clip_id']==key[0] and row['kind']==key[1] and row['origin']==c['origin'] and row['span']==[a,b]
  assert row['axis']==axis and row['moving']==side and row['old_quality']==c['quality'] and row['old_quality_class']==c['quality_class']
  for field in ['card','was_selected','role_peak']:assert row[field]==c.get(field)
  assert row['source_input']==source['path'] and row['source_sha256']==source['sha256']
  if k not in d['series']:d['series'][k]=ar.geom.relation(d['x'],axis,side)
  v,g=d['series'][k];expectedseq=ar.pack_sequence(v,g,d['ts'],a,b);expectedsummary=ar.summary(d['z'],a,b,k)
  assert sha(row['path'])==row['sha256']
  with np.load(row['path'],allow_pickle=False) as z:
   assert set(z.files)=={'summary','sequence'}
   se=float(np.max(np.abs(z['summary']-expectedsummary)));qe=float(np.max(np.abs(z['sequence']-expectedseq)))
   assert np.allclose(z['summary'],expectedsummary,atol=3e-6,rtol=1e-6),(row['id'],se)
   assert np.array_equal(z['sequence'],expectedseq),(row['id'],qe)
   assert np.isfinite(z['summary']).all() and np.isfinite(z['sequence']).all()
  observed=int(g[a:b+1].sum());assert observed==row['observed_frames'] and b-a+1==row['frames']
  state='none' if observed==0 else 'all' if observed==b-a+1 else 'partial'
  coverage[c['origin']+'/'+state]+=1;counts[c['origin']]+=1;counts['hand_'+side]+=1
  maxsummary=max(maxsummary,se);maxseq=max(maxseq,qe);unique[(key,a,b,k)].append(row['id'])
  details.append(dict(id=row['id'],summary_error=se,sequence_error=qe,observation_state=state))
 assert counts['boundary']==2802 and counts['phase']==124
 tree=ast.parse((W/'evaluate.py').read_text('utf-8'))
 subkeys={n.slice.value for n in ast.walk(tree) if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant) and isinstance(n.slice.value,str)}
 assert not subkeys & {'source_indices','query_to_source_frame_ids','expected_order','gold','label','target','edit_points'}
 report=dict(status='PASS_SAVED_DIAGNOSTIC_INPUTS_NO_MODEL_REEVALUATION',at=datetime.now().astimezone().isoformat(),
 reviewer='Separate Codex deterministic saved-array audit; not independent human target validation',
 counts=dict(source_scenarios=82,candidate_rows=2926,origins=dict(counts),observation_coverage=dict(coverage),unique_source_span_relation=len(unique),duplicated_candidate_rows=sum(len(v)-1 for v in unique.values())),
 arithmetic=dict(all_source_and_input_hashes_match=True,all_source_fh_slices_equal=True,summary_max_abs_error=maxsummary,sequence_max_abs_error=maxseq,all_old_A_probabilities_reused_exactly=True,all_existing_boundary_and_phase_candidates_retained=True,no_source_indices_or_target_access_in_eval_code=True),
 interpretation=['Previously exposed public development diagnostic; not fresh evaluation.','2926 rows include repeated intervals via separate boundary/phase provenance; not2926 independent motions.','Partial/missing observation candidates are retained for honest denominators; they are outside complete-observation training. Diagnostic logits must not become confident user feedback without an observation gate.','Old A is saved R42 quality output. B/C have seed-matched initialization; A is not a separately rerun matched6-seed control.','Training narrows class3 to multiple observed events; old A also used movement+nonmotion mixtures. Compare meanings explicitly rather than treating every argmax change as corrected error.','There are no new educational correctness targets or end-to-end selection changes in this candidate diagnostic.','Manifest source_sha256 names the cached source NPZ hash, not the raw movie hash.'],
 evaluator_preconditions=['Freeze evaluation code/input manifest/training manifest and resulting checkpoint hashes before evaluate.','Run new B/C checkpoints once; never recompute A or use these outputs to adjust epochs/thresholds.','Use saved probability outputs for arithmetic verification afterwards.'],
 inputs_sha256=sha(W/'EVAL_INPUTS_COMPLETE.json'),source_checks=sourcechecks,sample_checks=details,
 scope=dict(model_instantiation=0,neural_forward=0,training=0,pose_or_encoder=0,new_raw_video=0,private=0,stored_exposed_public_arrays_read=82),
 pins={str(p):sha(p) for p in [W/'evaluate.py',W/'EVAL_INPUTS_COMPLETE.json',U/'boundary_candidate_probe/RESULTS.json',U/'candidate_quality_probe/RESULTS.json',Path(__file__)]})
 (U/'reviews/local_quality_evaluation_preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({k:report[k] for k in ['status','counts','arithmetic','inputs_sha256']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
