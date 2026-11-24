"""Separate arithmetic/provenance check; deliberately does not call label maker."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;G=O/'word100_web_completion_graph_20261111';N=G/'typed_motion_recovery_20261113/null_candidate_learning'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tr=read(U/'SOURCE_TRACE_TARGETS.json');bags=read(U/'ABSENCE_BAGS.json');old=read(N/'TRAINING_INPUT_INDEX.json')['rows'];masks=read(N/'local_bag_evaluation/RULES_FROZEN.json')['partial_masks']
ann={r['clip_id']:r for p in [G/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',G/'candidate_description_supervision/additional_source_review/SOURCE_EPISODES_ADDED.json'] for r in read(p)['rows']}
checks=0
for p,h in {**tr['pins'],**bags['pins']}.items():assert sha(Path(p))==h;checks+=1
assert sha(Path(bags['arrays']))==bags['arrays_sha256'];data=np.load(bags['arrays']);cache={}
for r in bags['rows']:
    b=old[r['original_bag_index']];s=ann[r['clip_id']];m=masks[r['original_bag_index']]
    assert b['clip_id']==r['clip_id'] and b['family']==r['family'];assert (r['split']=='train')==(r['person'] in range(2,8))
    if b['path'] not in cache:cache[b['path']]=np.load(b['path'])
    z=cache[b['path']];idx=np.array(r['candidate_indices']);sp=z['spans'][idx]
    assert len(set(idx))==len(idx) and len(idx)>0
    assert np.array_equal(data[f"x_{r['index']}"],z['local_full'][idx])
    assert np.array_equal(data[f"y_{r['index']}"],z['y'][idx])
    if r['kind'] in ['partial_source_crop','other_reviewed_action_crop']:
        assert not z['y'][idx].any();a,bound=r['extent'];assert (sp[:,0]>=a).all() and (sp[:,1]<=bound).all()
        if r['kind']=='partial_source_crop':
            assert any(c['transformed_span']==r['extent'] and c['groups']==r['basis'] for c in m['candidates'])
            assert all(len(x['witnessed_frames'])>=2 and x['omitted_witnesses'] for x in r['basis'])
        else:
            own=[k for e in s['events'] if e['family']==r['family'] for k in e['progression_witness_frames']]
            if r['reverse']:own=[m['source_length']-1-k for k in own]
            assert not any(a<=k<=bound for k in own),('overlapping_own_progression',r)
            assert r['basis']['other_family']!=r['family']
    assert r['semantic_completion'] is None;checks+=8
for c in tr['cases']:
    assert c['linguistic_completion'] is None and c['educational_error'] is None and not c['model_prediction']
    if c['kind']=='mask_finish':assert c['source_trace_state']=='observation_unknown'
    if c['kind']=='endpoint_only':assert c['source_trace_state']=='progression_removed_between_endpoints'
    if c['kind']=='reverse':assert c['source_trace_state']=='source_order_changed'
    if c['kind'].startswith(('prefix_','suffix_')):assert c['source_trace_state']=='partial_reviewed_source_trace'
    if c['kind'] in ('full','with_context','duplicate_each2'):assert c['source_trace_state']=='full_reviewed_source_trace'
    if c['kind']=='subsample_every2':assert c['source_trace_state']=='sampled_witnesses_unavailable'
    checks+=3
# Different recording IDs and raw video hashes, not a visually inferred identity.
fit={r['source_video_sha256']for r in bags['rows'] if r['split']=='train'}
probe={r['source_video_sha256']for r in bags['rows'] if r['split']=='probe'}
assert len(fit)==24 and len(probe)==8 and not fit&probe
parents=read(N/'TRAINING_COMPLETE.json')['rows']
for parent in parents:
    p=np.load(parent['predictions']);seed=str(parent['seed'])
    for r in bags['rows']:
        ii=r['original_bag_index'];j=[2,3,4,5,7].index(r['family']);ids=r['candidate_indices']
        score=float(np.max(p[f'logits_{ii}'][ids,j]));null=float(p['null_logits'][j]);base=r['baseline'][seed]
        assert base==dict(null_selected=score<=null,max_extent_logit=score,null_logit=null);checks+=1
out=dict(passed=True,checks=checks,source_trace_cases=len(tr['cases']),bags=len(bags['rows']),fit_source_clips=24,probe_source_clips=8,
    other_action_bags_have_no_own_reviewed_progression=True,no_feature_reencoding=True,
    no_education_error_labels=True,train_justified=any(r['kind']=='other_reviewed_action_crop' and r['null_selected']<r['bags'] for r in bags['fit_baseline']),
    actor='root Codex separate verifier; not independent KSL expert',pins={str(p):sha(p)for p in [U/'SOURCE_TRACE_TARGETS.json',U/'ABSENCE_BAGS.json',U/'source_trace_contract.py',U/'prepare_absence_bags.py']})
with (U/'TARGETS_VERIFIED.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in out.items() if k!='pins'},ensure_ascii=False))
