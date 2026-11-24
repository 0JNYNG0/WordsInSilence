"""Read-only exact-time coverage audit against existing physical DTW stages.

Never propagate centre labels across a window or rename a gap as unimportant.
This measures readiness to attach a learned feature, not feedback accuracy.
"""
from pathlib import Path
import json,hashlib
from collections import Counter
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 data=read(U/'DATASET.json');source=O/'word100_description_evidence_feedback_20261101/REPORT_DATA.json';report=read(source)
 sel=read(O/'word100_stage_feedback_cycle_20261031/SELECTION.json');clips={r['alias']:r['clip_id'] for r in sel['rows']}
 by={};all_ids=set();attached=set()
 for r in data['rows']:
  by.setdefault(r['clip_id'],{})[r['center_frame']]=r;all_ids.add(r['sample_id'])
 cases=[];counts=Counter();original_hash=sha(source)
 for case in report['cases']:
  qclip=clips[case['query']];rclip=clips[case['reference']];q=by.get(qclip,{});r=by.get(rclip,{})
  stages=[]
  for stage in case['stages']:
   pairs=[];qseen=set();rseen=set()
   for qf,rf in stage['path']:
    qs=q.get(qf);rs=r.get(rf)
    if qs:qseen.add(qs['sample_id']);attached.add(qs['sample_id'])
    if rs:rseen.add(rs['sample_id']);attached.add(rs['sample_id'])
    if qs and rs:
     assert abs(qs['center_timestamp_us']/1e6-case['query_pts'][qf])<=1.1e-6
     assert abs(rs['center_timestamp_us']/1e6-case['reference_pts'][rf])<=1.1e-6
     pairs.append(dict(query_sample=qs['sample_id'],reference_sample=rs['sample_id'],query_frame=qf,reference_frame=rf,
       query_window=qs['frame_ids'],reference_window=rs['frame_ids'],
       static_same_frame_pair_only=True,motion_window_alignment_proven=False,
       shared_direct_static_targets=[n for j,n in enumerate(data['targets']) if '_motion_' not in n and qs['target_mask'][j] and rs['target_mask'][j]],
       shared_window_motion_context_targets=[n for j,n in enumerate(data['targets']) if '_motion_' in n and qs['target_mask'][j] and rs['target_mask'][j]]))
   counts['stages']+=1;counts['stages_with_exact_centre_pair']+=bool(pairs);counts['exact_centre_pairs']+=len(pairs)
   stages.append(dict(id=stage['id'],reference_core=stage['reference_core'],original_path_pairs=len(stage['path']),original_path_sha256=hashlib.sha256(json.dumps(stage['path'],separators=(',',':')).encode()).hexdigest(),
    query_centre_samples=sorted(qseen),reference_centre_samples=sorted(rseen),exact_pairs=pairs,
    original_alignment_flags=stage['alignment'],original_source_cards=stage['source_cards'],
    original_measurement_denominators={g:{k:v['saved_metric'].get(k) for k in ['expected','compared','reference_possible','reference_unobserved','path_missing_cells']} for g,v in stage['groups'].items()},
    educational_correctness=None,importance_of_missing_detail=None,readiness='partial_exact_time_support' if pairs else 'no_exact_two_sided_label_pair',
    no_nearest_frame_substitution=True,no_sparse_to_dense_label_propagation=True))
  cases.append(dict(id=case['id'],query_clip=qclip,reference_clip=rclip,stages=stages))
 assert len(cases)==36 and counts['stages']==117 and sha(source)==original_hash
 out=dict(status='PASS_EXACT_TIME_LINK_AUDIT_NOT_FEEDBACK_VALIDATION',source=str(source),source_sha256=original_hash,
  dataset_sha256=sha(U/'DATASET.json'),counts=dict(counts),label_windows_attached=len(attached),label_windows_without_existing_stage=sum(x not in attached for x in all_ids),
  official_interval_or_word_not_semantic_phase_truth=True,original_dtw_recomputed=False,original_denominators_preserved=True,
  new_prediction_attachment_requires_validated_model=True,new_feature_expression_guide_truth_not_created=True,cases=cases)
 with (U/'EVIDENCE_LINK_AUDIT.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
 print(json.dumps({k:v for k,v in out.items() if k!='cases'},ensure_ascii=False))
if __name__=='__main__':main()
