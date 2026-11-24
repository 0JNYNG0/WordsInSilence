from common import *
import re
assert not (F/'LINEAGE_AUDIT.json').exists()
names=['word100_rtmw_joint_learning_20261108/SOURCES.json','word100_shared_motion_learning_20261109/READY.json','word100_learned_guide_feedback_20261102/INPUT_MANIFEST.json','word100_rtmw_joint_learning_20261108/feedback_viewfix/FEEDBACK_READY.json','word100_action_feedback_demo_20261108/PREPARED.json','word100_event_completion_20261109/FROZEN.json']
rows=[];candidates=read(F/'FRESH_REVIEW.json')['rows']
for name in names:
 p=O/name;t=p.read_text(encoding='utf-8-sig')
 rows.append(dict(path=str(p),sha256=sha(p),recorded_ids=sorted(set(re.findall(r'REAL(\d\d)_F',t))),candidate_id_hits=[r['clip_id'] for r in candidates if r['clip_id'] in t],candidate_byte_hash_hits=[r['clip_id'] for r in candidates if r['source_sha256'] in t]))
for name in ['word100_body_boundary_20261017/data/records','word100_t3f_heldout_20261019/data/records','word100_autonomous_recognition_20261026/fresh199/data/records']:
 p=O/name;files=list(p.glob('NIA_SL_WORD*_REAL*_F.json'));ids=[f.stem for f in files]
 rows.append(dict(path=str(p),record_count=len(ids),record_list_sha256=hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(),candidate_id_hits=[r['clip_id'] for r in candidates if r['clip_id'] in ids]))
assert not any(r['candidate_id_hits'] or r.get('candidate_byte_hash_hits') for r in rows)
write(F/'LINEAGE_AUDIT.json',dict(rows=rows,eligible=[r['clip_id'] for r in candidates],verdict='New to audited current model lineage and named earlier public evaluations',limits=['Not a proof of no historical visual exposure anywhere in project','Recorded ID10, not biometric proof of a distinct human','Four clips from one recorded ID; not large independent population'],byte_duplicate_audit=str(F/'EXPOSURE_AUDIT.json'),weights_or_threshold_change=False))
print('LINEAGE_PASS',len(candidates))
