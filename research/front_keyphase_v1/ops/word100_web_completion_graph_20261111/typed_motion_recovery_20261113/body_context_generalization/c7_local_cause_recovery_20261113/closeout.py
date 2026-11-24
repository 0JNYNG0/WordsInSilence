from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for child in [U,U/'conditioned_context']:
    c=read(child/'TRAINING_FROZEN.json');v=read(child/'SAVED_VERIFICATION.json')
    assert v['status']=='PASS'and v['comparison_sha256']==sha(child/'COMPARISON.json')
    for p,h in {**c['pins'],**c['protected_pins']}.items():assert sha(p)==h,p
comp=read(U/'conditioned_context/COMPARISON.json')
lost=sorted({(r['clip_id'],r['source_frame'],r['family'])for r in comp['changed_observations']if r['kind']=='regression'and r['code']==1})
assert len(lost)==11
resume=dict(status='NEXT_CAUSE_AUDIT_NOT_EXECUTED',records=[dict(clip_id=c,frame=f,family=k)for c,f,k in lost],
    all_original_labels_preserved=True,no_prior_models_rerun=True,no_threshold_selection=True)
(U/'NEXT_RESIDUALS.json').write_text(json.dumps(resume,ensure_ascii=False,indent=2),'utf8')
paths=['RESULT_KO.txt','GRAPH_PATCH.json','COMPARISON.json','SAVED_VERIFICATION.json','CAUSE_AUDIT.json','RGB_REVIEW.json','NEXT_RESIDUALS.json','CONDITIONING_VERIFICATION.json','conditioned_context/COMPARISON.json','conditioned_context/SAVED_VERIFICATION.json']
for p in U.rglob('*.json'):read(p)
report=dict(status='COMPLETED_BOUNDED_COMPARISONS_NOT_ADOPTED',trained_new_models=36,C6_completed36_rerun=False,
    protected_files=len(c['protected_pins']),protected_hashes_unchanged=True,training_running=False,
    root_docs_or_graph_or_STATE_modified=False,operational_web_changed=False,G2_pass=False,
    output_pins={p:sha(U/p)for p in paths},
    verification='Same Codex independent arithmetic path; parent may independently inspect. Not KSL expert ground truth.',
    discovery_errors='Read-only discovery looked for absent NEXT_PLAN/core paths and Windows wildcards; corrected to actual files. No new training failure or repeated completed run.',
    next='Review eleven residual FIT witnesses and sequence/conditioning separability before new contract. No DEV/final8/private data.')
(U/'CLOSEOUT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
print(report['status'],report['protected_files'],'protected')
