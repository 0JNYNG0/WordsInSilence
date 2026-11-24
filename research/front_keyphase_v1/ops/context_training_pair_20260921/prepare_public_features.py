"""Local operational cache assembly for the predeclared public-only comparison."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
sys.path[:0] = [str(LAB), str(LAB/'src'), str(LAB.parent/'sign_model_pilot_v1/src')]
import tools_prep_union93_candidate as u
from pilot.data import collate

read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(name, value):
    with (HERE/name).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def main():
    started = time.time()
    assert not (HERE/'feature_plan.json').exists()
    policy = read(HERE/'comparison_policy.json')
    for p,h in policy['evidence'].items(): assert sha(p)==h, p
    parent = policy['parent']; assert sha(parent['path'])==parent['sha256']
    payload = torch.load(parent['path'], map_location='cpu', weights_only=True)
    previous = LAB/'runs/visibility_candidate_v1/visibility_final.pt'
    assert sha(previous)=='258f23e1ad515a25b50e6affac8ac4e284ba8f6ae6fac3372c1e7a31ea6848a8'
    old = torch.load(previous, map_location='cpu', weights_only=True)
    classes = list(payload['classes']); assert len(classes)==93 and classes==old['classes']
    nonhead = [k for k in payload['state_dict'] if not k.startswith('head.')]
    assert all(torch.equal(payload['state_dict'][k],old['state_dict'][k]) for k in nonhead)
    sources = {
        'base': LAB/'runs/union93_candidate_v1/inputs.json',
        'apology': LAB/'evidence/sentence_target_catalog_20260919/apology_native_prep.json',
        'semantic_review': LAB/'ops/native_data_pair_20260919/semantic_review_user_confirmed.json',
        'coordinates': LAB/'ops/native_context_readiness_20260921/coordinates.json',
        'eligibility': LAB/'ops/native_context_readiness_20260921/eligibility.json',
        'public_report': LAB/'ops/public_anchor_strength_20260921/PUBLIC_EVALUATION.json',
        'public_cache': LAB/'ops/public_anchor_strength_20260921/public_features_scores.npz',
        'whole_report': LAB/'ops/native_context_baseline_20260921/result.json',
        'whole_cache': LAB/'ops/native_context_baseline_20260921/features_and_scores.npz',
        'mp_report': LAB/'ops/public_window_sensitivity_20260921/result.json',
        'mp_cache': LAB/'ops/public_window_sensitivity_20260921/public_scores.npz',
        'mp2_report': LAB/'ops/public_cached_domain_probe_20260921/result.json',
        'mp2_cache': LAB/'ops/public_cached_domain_probe_20260921/features_and_scores.npz',
        'old_predictions': LAB/'runs/public_anchor_0p01_pair_v1/results.json',
        'channels_code': LAB.parent/'sign_model_pilot_v1/src/pilot/data.py',
        'model_code': LAB.parent/'sign_model_pilot_v1/src/pilot/model_masked_v1.py',
        'loader_code': LAB/'tools_prep_union93_candidate.py',
        'prior_encoder': previous,
    }
    pins={str(p):sha(p) for p in sources.values()}
    semantic=read(sources['semantic_review'])
    assert semantic['status']=='approved' and semantic['approved_label_mapping']=={'미안합니다':'죄송하다'}
    inputs=read(sources['base']); apoptosis=read(sources['apology'])['rows']
    coordinates=read(sources['coordinates'])['rows']
    eligible={r['clip_id']:r for r in read(sources['eligibility'])['rows'] if r['status']=='ELIGIBLE_BY_SOURCE_ANNOTATION'}
    public=read(sources['public_report']); whole=read(sources['whole_report'])
    mp=read(sources['mp_report']); mp2=read(sources['mp2_report'])
    for key, expected in [('public_cache',public['features_scores_sha256']),('whole_cache',whole['arrays_sha256']),('mp_cache',mp['arrays_sha256']),('mp2_cache',mp2['features_sha256'])]:
        assert pins[str(sources[key])]==expected,key
    def native_row(row,role):
        ident=row.get('id',row.get('clip_id'))
        signer=int(re.search(r'_REAL(\d+)_',ident).group(1))
        assert signer in (range(1,10) if role=='train' else (11,12,13))
        p=Path(row['npz'])
        if not p.is_absolute(): p=LAB.parent/'sign_model_pilot_v1'/p
        return dict(id=ident,label=row.get('label','죄송하다'),role=role,signer_id=signer,
                    path=str(p),sha256=row['npz_sha256'],method='native_annotation_forward')
    base=[native_row(r,'train') for r in inputs['source_train']]
    base += [dict(id=r['id'],label=r['label'],role='train',method='reuse_public',cache_id=r['id']) for r in inputs['nikl']]
    base += [dict(**native_row(r,'train'),cache_id=r['clip_id']) for r in apoptosis if r['split_role']=='train_candidate']
    for r in base[-9:]:r['method']='reuse_public'
    dev=[native_row(r,'validation') for r in inputs['source_validation']]
    dev += [native_row(r,'validation') for r in apoptosis if r['split_role']=='development_only']
    assert len(base)==2122 and len(dev)==699
    bmap={r['id']:i for i,r in enumerate(base)}
    assert len(bmap)==2122 and {r['label'] for r in base}==set(classes)
    pair=[]; whole_dev=[]
    for r in coordinates:
        assert r['status']=='PREPARED'
        original=eligible[r['clip_id']]['source_samples'][0]['sample_id']
        if r['clip_id'].startswith('NIA_SL_SEN0356_'):original=r['clip_id']
        q=dict(id=r['clip_id'],label=r['reference_label'],signer_id=r['signer_id'],role=r['split_role'],
               path=r['npz'],sha256=r['npz_sha256'],frames=r['frames'],original_id=original)
        if r['split_role']=='train':
            q['base_index']=bmap[original]
            assert base[q['base_index']]['label']==q['label'] and base[q['base_index']]['signer_id']==q['signer_id']
            assert base[q['base_index']]['sha256']==r['original_annotation_sha256']
            pair.append(q)
        else:whole_dev.append(q)
    assert len(pair)==224 and len(whole_dev)==74 and len({r['label'] for r in pair})==16
    plan=dict(policy_sha256=sha(HERE/'comparison_policy.json'),parent=parent,source_pins=pins,
              encoder_tensors_identical_to_reused_cache_parent=len(nonhead),base=base,development=dev,
              paired_train=pair,whole_development=whole_dev,public_rows=[{k:v for k,v in r.items() if k!='models'} for r in public['rows']],
              mp_rows=[{k:r[k] for k in ('id','condition','reference_label','status')} for r in mp['rows']],
              new_encoder_forwards_planned=3018,user_arrays_read=0,reserved_arrays_read=0)
    save('feature_plan.json',plan)
    model,_=u.rebuild_from_payload(payload); model.eval().requires_grad_(False);torch.set_num_threads(2)
    captured=[]
    hook=model.head.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().numpy().copy()[0]))
    def forward(row,whole_input=False):
        assert sha(row['path'])==row['sha256'],row['id']
        if whole_input:
            with np.load(row['path'],allow_pickle=False) as z:x=z['x'].copy()
            assert x.shape==(row['frames'],280)
        else:x=u._channels_from_source(row['path'],row['sha256'])['x']
        assert np.isfinite(x).all()
        bx,mask,_=collate([dict(x=x,frames=len(x),label=0)])
        with torch.inference_mode():model(torch.from_numpy(bx),torch.from_numpy(mask))
        return captured[-1]
    with np.load(sources['public_cache'],allow_pickle=False) as z:
        assert z['classes'].tolist()==classes
        public_x=z['features'].copy();public_ids=z['ids'].tolist()
        assert public_ids==[r['id'] for r in public['rows']]
        public_map={id:public_x[i] for i,id in enumerate(public_ids)}
    matrices={};meta={}
    try:
        for name,rows in [('base',base),('annotation_dev',dev),('pair_context',pair)]:
            fs=[]
            for i,row in enumerate(rows):
                fs.append(public_map[row['cache_id']] if row.get('method')=='reuse_public' else forward(row,name=='pair_context'))
                if (i+1)%500==0:print(name,i+1,'/',len(rows),flush=True)
            matrices[name+'_x']=np.asarray(fs,dtype=np.float32)
            matrices[name+'_y']=np.array([classes.index(r['label']) for r in rows],dtype=np.int64)
            meta[name]=rows
    finally:hook.remove()
    assert len(captured)==3018
    matrices['pair_base_indices']=np.array([r['base_index'] for r in pair],dtype=np.int64)
    matrices['pair_annotation_x']=matrices['base_x'][matrices['pair_base_indices']].copy()
    assert np.array_equal(matrices['base_y'][matrices['pair_base_indices']],matrices['pair_context_y'])
    with np.load(sources['whole_cache'],allow_pickle=False) as z:
        assert z['classes'].tolist()==classes
        ix={s:i for i,s in enumerate(z['clip_ids'].tolist())}
        matrices['whole_dev_x']=np.stack([z['features'][ix[r['id']]] for r in whole_dev]).astype(np.float32)
    matrices['whole_dev_y']=np.array([classes.index(r['label']) for r in whole_dev],dtype=np.int64);meta['whole_dev']=whole_dev
    assert np.array_equal(public_x.astype(np.float32).astype(np.float64),public_x)
    matrices['public_x']=public_x.astype(np.float32)
    matrices['public_y']=np.array([classes.index(r['true']) for r in public['rows']],dtype=np.int64);meta['public']=plan['public_rows']
    mpfs=[]
    with np.load(sources['mp_cache'],allow_pickle=False) as a,np.load(sources['mp2_cache'],allow_pickle=False) as b:
        assert a['classes'].tolist()==classes==b['classes'].tolist()
        for row in mp['rows']:
            if row['status']=='EVALUATED':f=a[row['id']+'__'+row['condition']+'__features']
            else:f=b['features'][b['conditions'].tolist().index(row['prior_condition'])]
            assert np.array_equal(f.astype(np.float32).astype(np.float64),f)
            mpfs.append(f)
    matrices['mp_x']=np.array(mpfs,dtype=np.float32)
    matrices['mp_y']=np.array([classes.index(r['reference_label']) for r in mp['rows']],dtype=np.int64);meta['mp']=plan['mp_rows']
    matrices['classes']=np.array(classes)
    # Match every old annotation prediction: new caching must preserve deployed input construction.
    old_rows=read(sources['old_predictions'])['arms']['treatment']['evaluations']['full']['rows']
    old_map={r['id']:r for r in old_rows};assert len(old_map)==699
    score=matrices['annotation_dev_x'].astype(np.float64)@payload['state_dict']['head.weight'].numpy().astype(np.float64).T+payload['state_dict']['head.bias'].numpy().astype(np.float64)
    predictions=[classes[i] for i in score.argmax(axis=1)]
    assert all(old_map[r['id']]['true']==r['label'] and old_map[r['id']]['pred']==pred for r,pred in zip(dev,predictions))
    assert sum(r['label']==pred for r,pred in zip(dev,predictions))==690
    for p,h in pins.items():assert sha(p)==h,p
    assert sha(parent['path'])==parent['sha256']
    with (HERE/'public_features.npz').open('xb') as f:np.savez_compressed(f,**matrices)
    save('feature_metadata.json',meta)
    result=dict(status='PREPARED_PENDING_INDEPENDENT_BOUNDARY_AUDIT',plan_sha256=sha(HERE/'feature_plan.json'),
                script_sha256=sha(__file__),arrays_sha256=sha(HERE/'public_features.npz'),metadata_sha256=sha(HERE/'feature_metadata.json'),
                shapes={k:list(v.shape) for k,v in matrices.items()},new_encoder_forwards=len(captured),
                reused_feature_rows=120,paired_annotation_reuses=224,old_annotation_predictions_matched=699,
                parent_annotation_correct=690,seconds=round(time.time()-started,3),user_arrays_read=0,reserved_arrays_read=0)
    save('FEATURE_PREPARATION.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='shapes'}),flush=True)

if __name__=='__main__':main()
