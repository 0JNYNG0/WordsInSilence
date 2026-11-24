"""Preserve initial record; reconcile inherited preparation status and verify files."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
doc=json.loads((P/'REVIEW.json').read_text(encoding='utf-8'))
assert not(P/'REVIEW_COMPLETE.json').exists()
checks=[]
for r in doc['rows']:
    rv=r['review'];assert rv['all_frames_inspected']
    assert rv['inspected_frames']==list(range(min(r['frames']),max(r['frames'])+1))
    assert sorted(rv['hold_frames']+rv['uncertain_frames'])==r['frames']
    assert r['role']=='fit' and r['person'] in [2,3]
    for key,hkey in [('source','source_sha256'),('source_annotation','source_annotation_sha256'),('contact_sheet','contact_sheet_sha256')]:
        assert sha(r[key])==r[hkey];checks.append(r[key])
    for f in r['frame_records']:assert sha(f['path'])==f['sha256'];checks.append(f['path'])
    r['preparation_visual_inspection_completed']=r['visual_inspection_completed']
    r['visual_inspection_completed']=True
doc['metadata_correction']={'original':str(P/'REVIEW.json'),'original_sha256':sha(P/'REVIEW.json'),'reason':'Inherited preparation flag was false while review.all_frames_inspected was true. Mark the final review flag true; labels/images/source unchanged. Initial record preserved.'}
doc['at']=datetime.now(timezone.utc).isoformat()
(P/'REVIEW_COMPLETE.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
verification={'status':'ARITHMETIC_AND_HASH_PASS_VISUAL_ROOT_REVIEW_PENDING','files_checked':len(checks),'distinct_files':len(set(checks)),'clips':len(doc['rows']),'inspected_frames':sum(len(r['frames'])for r in doc['rows']),'hold_frames':sum(len(r['review']['hold_frames'])for r in doc['rows']),'unknown_frames':sum(len(r['review']['uncertain_frames'])for r in doc['rows']),'annotation':str(P/'REVIEW_COMPLETE.json'),'annotation_sha256':sha(P/'REVIEW_COMPLETE.json'),'semantic_completion_all_null':all(r['review']['semantic_completion']is None for r in doc['rows']),'same_agent_arithmetic_not_human_confirmation':True,'original_annotations_preserved':True}
assert (verification['inspected_frames'],verification['hold_frames'],verification['unknown_frames'])==(94,90,4)
(P/'VERIFICATION.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(verification))
