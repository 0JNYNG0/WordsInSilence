from pathlib import Path
import json,hashlib,collections
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
data=read(U/'GUIDE_FINGER_BINDINGS.json');labels=read(U.parent/'annotation_review/ORDINARY_LABELS.json')['rows']
for src in data['sources']:assert sha(src['path'])==src['sha256']
keys=set()
for b in data['bindings']:
    assert b['binding_id'] not in keys;keys.add(b['binding_id'])
    assert b['reference_state'] in ['E','B','F'] and not b['educational_tolerance_verified']
    stage=b['stage_interval_provenance'];a,z=stage['native_frame_interval_inclusive'];t0,t1=stage['native_time_interval_us_inclusive']
    for anchor in b['reference_anchors']:
        assert a<=anchor['frame']<=z and t0<=anchor['native_time_us']<=t1
        assert sha(anchor['image'])==anchor['image_sha256'];assert sha(anchor['annotation_file'])==anchor['annotation_sha256']
        source=next(r for r in labels if r['clip_id']==anchor['clip_id'] and r['frame']==anchor['frame'] and r['side']==b['hand'])
        assert source['states'][['thumb','index','middle','ring','little'].index(b['finger'])]==b['reference_state']
        assert source['source_sha256']==anchor['source_sha256'] and source['pts_us']==anchor['native_time_us']
    if b['source_directly_names_finger_state']:
        assert b['word_id']=='WORD1567' and b['finger']=='thumb' and b['reference_state']=='E' and b['clause_ids']==['ancestor_thumbs']
    else:assert b['clause_ids']==[]
for c in data['official_clauses']:
    assert c['context_only'] and not c['source_action_verified'] and not c['full_span_semantic_verified']
tests=read(U/'CONTRACT_TESTS.json');assert tests['status']=='PASS' and tests['formatter_sha256']==sha(U/'formatter.py')
report=dict(status='PASS_SOURCE_PROVENANCE_AND_OUTPUT_CONTRACT_NOT_MODEL_PERFORMANCE',bindings=len(keys),official_clauses=len(data['official_clauses']),sources_sha_verified=len(data['sources']),synthetic_contract_tests=tests['tests'],unbound_preserved=data['unbound_anchors'],model_inference=0,training=0,web_edits=0,files_sha256={p.name:sha(p) for p in U.iterdir() if p.is_file() and p.suffix in ['.py','.json','.md'] and p.name!='VERIFICATION.json'})
(U/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf-8');print(report['status'],len(keys),tests['tests'])
