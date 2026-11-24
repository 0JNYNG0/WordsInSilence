"""Boundary and cache identity audit without re-extracting or re-encoding data."""
from pathlib import Path
import hashlib
import json
import re
import sys
import numpy as np
import torch

H=Path(__file__).resolve().parent;L=H.parents[1]
sys.path[:0]=[str(L/'src'),str(L.parent/'sign_model_pilot_v1/src')]
from wis_lab.service.neural import PROJECT_18
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=read(H/'feature_plan.json');prep=read(H/'FEATURE_PREPARATION.json');meta=read(H/'feature_metadata.json')
checks=[]
def check(name,ok):
    checks.append(dict(name=name,pass_=bool(ok)))
    if not ok:raise AssertionError(name)
check('new_cache_and_metadata_pins',sha(H/'public_features.npz')==prep['arrays_sha256'] and sha(H/'feature_metadata.json')==prep['metadata_sha256'] and sha(H/'feature_plan.json')==prep['plan_sha256'])
check('all_source_pins_unchanged',all(sha(p)==h for p,h in plan['source_pins'].items()))
check('frozen_parent',sha(plan['parent']['path'])==plan['parent']['sha256'])
ck=torch.load(plan['parent']['path'],map_location='cpu',weights_only=True)
with np.load(H/'public_features.npz',allow_pickle=False) as a:
    c=a['classes'].tolist();check('class_order',c==ck['classes'] and len(c)==93)
    for key in a.files:
        if key=='classes':continue
        check(key+'_finite',bool(np.isfinite(a[key]).all()))
    for name,n in [('base',2122),('annotation_dev',699),('pair_context',224),('whole_dev',74),('public',32),('mp',14)]:
        check(name+'_row_alignment',a[name+'_x'].shape==(n,64) and len(meta[name])==n and a[name+'_y'].shape==(n,) and all(c[int(y)]==r.get('label',r.get('true',r.get('reference_label'))) for y,r in zip(a[name+'_y'],meta[name])))
    def signers(rows):return {int(re.search(r'_REAL(\d+)_',r['id']).group(1)) for r in rows if '_REAL' in r['id']}
    check('all_train_signers_and_all93_classes_retained',signers(meta['base'])==set(range(1,10)) and set(a['base_y'])==set(range(93)))
    check('all_development_signers_retained',signers(meta['annotation_dev'])=={11,12,13} and signers(meta['whole_dev'])=={11,12,13})
    check('paired_train_signers',signers(meta['pair_context'])==set(range(1,10)))
    trainids={r['id'].split('_a')[0] for r in meta['base']};devids={r['id'].split('_a')[0] for r in meta['annotation_dev']}
    check('no_source_clip_leakage',not trainids&devids)
    indices=a['pair_base_indices']
    check('paired_control_exact_base_rows',np.array_equal(a['pair_annotation_x'],a['base_x'][indices]) and np.array_equal(a['pair_context_y'],a['base_y'][indices]) and all(meta['base'][int(i)]['id']==r['original_id'] for i,r in zip(indices,meta['pair_context'])))
    check('paired16_classes_unmodified',len(set(a['pair_context_y']))==16)
    with np.load(L/'ops/public_anchor_strength_20260921/public_features_scores.npz',allow_pickle=False) as old:
        check('public32_reused_exactly',np.array_equal(a['public_x'].astype(np.float64),old['features']))
        oldmap={id:old['features'][i] for i,id in enumerate(old['ids'].tolist())}
        check('training27_reused_exactly',all(np.array_equal(a['base_x'][i].astype(np.float64),oldmap[r['cache_id']]) for i,r in enumerate(meta['base']) if r.get('method')=='reuse_public'))
    with np.load(L/'ops/native_context_baseline_20260921/features_and_scores.npz',allow_pickle=False) as old:
        ix={id:i for i,id in enumerate(old['clip_ids'].tolist())}
        check('full_development74_reused_exactly',all(np.array_equal(a['whole_dev_x'][i].astype(np.float64),old['features'][ix[r['id']]]) for i,r in enumerate(meta['whole_dev'])))
    with np.load(L/'ops/public_window_sensitivity_20260921/public_scores.npz',allow_pickle=False) as old,np.load(L/'ops/public_cached_domain_probe_20260921/features_and_scores.npz',allow_pickle=False) as old2:
        rows=read(L/'ops/public_window_sensitivity_20260921/result.json')['rows']
        for i,r in enumerate(rows):
            f=old[r['id']+'__'+r['condition']+'__features'] if r['status']=='EVALUATED' else old2['features'][old2['conditions'].tolist().index(r['prior_condition'])]
            check('mp_reuse_'+str(i),np.array_equal(a['mp_x'][i].astype(np.float64),f))
    # A separate affine calculation ties assembled rows to existing frozen-parent predictions.
    w=ck['state_dict']['head.weight'].numpy().astype(float);b=ck['state_dict']['head.bias'].numpy().astype(float)
    ti=[i for i,label in enumerate(c) if label in set(PROJECT_18.values())]
    summary={}
    for name in ['annotation_dev','whole_dev','public','mp']:
        score=np.einsum('nd,cd->nc',a[name+'_x'].astype(float),w)+b
        preds=np.argmax(score,axis=1);ys=a[name+'_y'];app=np.isin(ys,ti)
        limited=np.asarray(ti)[np.argmax(score[:,ti],axis=1)]
        summary[name]=dict(full_correct=int((preds==ys).sum()),full_n=len(ys),project18_correct=int(((limited==ys)&app).sum()),project18_n=int(app.sum()))
    check('parent_all_native_counts_match',summary['annotation_dev']==dict(full_correct=690,full_n=699,project18_correct=32,project18_n=33) and summary['whole_dev']==dict(full_correct=17,full_n=74,project18_correct=16,project18_n=30))
    check('parent_public_baseline',summary['public']['full_correct']==32 and summary['public']['project18_correct']==32 and summary['mp']['full_correct']==12 and summary['mp']['project18_correct']==12)
out=dict(status='PASS',feature_plan_sha256=sha(H/'feature_plan.json'),feature_cache_sha256=sha(H/'public_features.npz'),checks=checks,parent_reused_baselines=summary,new_encoder_forwards=0,user_arrays_read=0,reserved_arrays_read=0)
with (H/'FEATURE_AUDIT.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(status=out['status'],checks=len(checks),summary=summary)))
