"""Describe residual FIT-fold failures from saved results; no new scoring."""
from pathlib import Path
import collections,hashlib,json
import numpy as np
U=Path(__file__).resolve().parent;G=U.parent.parent
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
c=read(U/'COMPARISON.json');idx=read(U/'INDEX.json')['rows']
base=G/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json'
extra=G/'candidate_description_supervision/additional_source_review/SOURCE_EPISODES_ADDED.json'
src={r['clip_id']:r for p in [base,extra]for r in read(p)['rows']if r['person']in range(2,8)}
reg=[r for r in c['changed_observations']if r['code']==1 and r['kind']=='regression']
groups=collections.defaultdict(list)
for r in reg:groups[r['clip_id'],r['source_frame'],r['family']].append(r)
with np.load(U/'INPUTS.npz',allow_pickle=False)as z:d={k:z[k].copy()for k in z.files}
records=[]
for (cid,f,family),rr in sorted(groups.items()):
    source=src[cid];gg=next(g for g in source['groups']if g['family']==family)
    si=next(r for r in idx if r['clip_id']==cid and r['view']=='full'and not r['reverse'])
    ii=si['start']+f;assert int(d['source_frame_id'][ii])==f
    x=d['absolute_context'][ii].reshape(3,10,5)
    axis={2:0,3:1,4:2,5:4,7:3}[family]*2
    coords=x[:,axis,:4];av=x[:,axis,4]
    near_boundary=any(a<=f<=b for a,b in [gg['onset'],gg['offset']])
    records.append(dict(clip_id=cid,source_frame=f,family=family,
        regression_seeds=sorted({r['seed']for r in rr}),correlated_observations=len(rr),
        views=sorted({r['view']for r in rr}),reverse_states=sorted({r['reverse']for r in rr}),
        inside_existing_boundary_uncertainty_band=near_boundary,
        full_view_context_availability=av.tolist(),
        full_view_before_current_distance=float(np.linalg.norm(coords[0]-coords[1])),
        full_view_after_current_distance=float(np.linalg.norm(coords[2]-coords[1])),
        interpretation='Directly reviewed movement point is preserved even inside uncertain onset/offset band; proximity alone does not invalidate its label.'))
fits=[]
for seed in [42,43,44]:
    row={'seed':seed}
    for code in [1,3]:
        rr=[r for r in c['fit']if r['seed']==seed and r['code']==code]
        row[str(code)]={k:sum(r[k]for r in rr)for k in ['count','baseline_correct','relative_correct','regressions','recoveries']}
    fits.append(row)
out=dict(status='SAVED_RESIDUAL_AUDIT_COMPLETE',
    lost_motion_unique_source_points=len(records),records=records,training_scope= fits,
    changed_family_failures=[r for r in c['by_family']if r['categories']['1']['relative_correct']<r['categories']['1']['baseline_correct']],
    causal_limit='Feature reparameterization improves aggregate head diagnostic but does not establish true 3D motion, annotation correctness or a universal solution. Source-boundary proximity and feature magnitude are descriptions, not proven causal explanations.',
    next='Review held-out FIT source witnesses and body-relative paths at the preserved sparse positive points. Separate weak/incorrect body observation from correct-but-small movement and model generalization before proposing further supervision. Keep all original labels/thresholds; no word-specific exception or DEV retuning.',
    no_inference=True,no_retraining=True,no_labels_changed=True,
    pins={str(p):sha(p)for p in [U/'COMPARISON.json',U/'INPUTS.npz',base,extra]})
with (U/'RESIDUAL_AUDIT.json').open('x',encoding='utf-8')as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'unique_points':len(records),'boundary_points':sum(r['inside_existing_boundary_uncertainty_band']for r in records),'fit':fits},ensure_ascii=False))
