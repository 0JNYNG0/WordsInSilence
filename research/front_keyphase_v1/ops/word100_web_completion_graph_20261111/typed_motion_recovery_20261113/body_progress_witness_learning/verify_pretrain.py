"""Independent saved-input audit. No prepare, training or model forward calls."""
from pathlib import Path
import collections, datetime, hashlib, json
import numpy as np

U=Path(__file__).resolve().parent; G=U.parent.parent; W=G/'candidate_description_supervision'
F=[2,3,4,5,7]; AX={2:0,3:1,4:2,5:4,7:3}
def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pins_match(pins):
    for p,h in pins.items(): assert sha(p)==h,p

def main():
    meta=read(U/'INPUTS_COMPLETE.json'); frozen=read(U/'FROZEN.json')
    assert sha(U/'FROZEN.json')==meta['frozen_sha256']
    assert sha(U/'LABEL_AUDIT.json')==meta['label_audit_sha256']
    assert frozen['pins']==meta['pins']; pins_match(meta['pins'])
    protected=meta['protected_pins']
    assert protected==frozen['protected_pins']==read(G/'PROTECTIONS.json') and len(protected)==477
    pins_match(protected)
    base=read(G/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json')
    added=read(W/'additional_source_review/SOURCE_EPISODES_ADDED.json')
    assert base['training_allowed'] is False and added['training_allowed'] is False
    sources={r['clip_id']:r for r in base['rows'] if r['person'] in [2,4,5,6]}
    sources.update({r['clip_id']:r for r in added['rows']})
    admission=read(U/'SOURCE_ADMISSION.json')
    assert admission['training_allowed'] is True and admission['scope']=='sparse_body_progress_only'
    assert set(admission['approved_clip_ids'])==set(sources) and len(sources)==24
    assert sha(U/'SOURCE_ADMISSION.json')==meta['source_admission']['sha256']
    rows=meta['sequences'];probe=meta['probe_metadata']
    assert len(rows)==144 and len(probe)==48
    assert {r['person'] for r in rows}=={2,3,4,5,6,7}
    assert {r['person'] for r in probe}=={8,9}
    assert all(r['split']=='train' and r['rate']==1 for r in rows)
    for key in ['clip_id','source_sha256']:
        assert not ({r[key] for r in rows}&{r[key] for r in probe})
    for cid in sources:
        rr=[r for r in rows if r['clip_id']==cid]
        assert len(rr)==6 and {(r['view'],r['reverse']) for r in rr}=={
            (v,b) for v in ['full','upper','actual_crop'] for b in [False,True]}
    raw={r['clip_id']:r for r in read(G.parent/'word100_shared_motion_learning_20261109/READY.json')['rows']
         if r['role']=='fit' and r['clip_id'] in sources}
    times={}
    for cid,r in raw.items():
        assert sha(r['path'])==r['sha256']
        with np.load(r['path'],allow_pickle=False) as z: times[cid]=(z['timestamp_us'].copy(),z['frame_id'].copy())
    data=[];codes=collections.Counter();removed=0;positive=0;source_counts=collections.defaultdict(collections.Counter)
    for r in rows:
        assert sha(r['path'])==r['sha256'] and sha(r['original_path'])==r['original_sha256']
        with np.load(r['path'],allow_pickle=False) as z: d={k:z[k].copy() for k in z.files}
        with np.load(r['original_path'],allow_pickle=False) as z: old={k:z[k].copy() for k in ['x','y','mask']}
        n=r['length'];s=sources[r['clip_id']]
        assert r['person']==s['person'] and r['source_sha256']==s['source_sha256']
        assert d['x'].shape==(n,150) and np.array_equal(d['x'],old['x']) and np.isfinite(d['x']).all()
        observed=np.stack([d['x'][:,50:100].reshape(n,10,5)[:,AX[f]*2,4]>.5 for f in F],axis=1)
        assert np.array_equal(d['observed'],observed) and np.array_equal(d['valid'],observed)
        bands={i for e in s['events'] for k in ['onset','offset'] for i in range(e[k][0],e[k][1]+1)}
        query_bands=np.array([n-1-i if r['reverse'] else i for i in sorted(bands)],dtype=int)
        yy=old['y'].copy();mm=old['mask'].astype(bool).copy()
        removal=np.zeros_like(mm);removal[query_bands]=(yy[query_bands]==0)&mm[query_bands]
        removed+=int(removal.sum());mm[removal]=False
        pp=np.where(mm,np.where(yy>0,1,2),0).astype(np.int8)
        last=max(s['groups'],key=lambda g:g['offset'][1])['family']
        for frame in s['hold_sample_frames']:
            assert any(a<=frame<=b for a,b in s['holds'])
            i=n-1-frame if r['reverse'] else frame
            for family in {g['family'] for g in s['groups']}:
                j=F.index(family)
                if frame not in bands and observed[i,j]:
                    assert not (old['mask'][i,j]>0 and old['y'][i,j]>0)
                    yy[i,j]=0;mm[i,j]=True;pp[i,j]=3 if family==last else 4
        assert np.array_equal(d['y'],yy) and np.array_equal(d['mask'],mm)
        assert np.array_equal(d['label_provenance'],pp)
        assert np.array_equal((old['mask']>0)&(old['y']>0),(d['mask']>0)&(d['y']>0))
        assert not (mm&~observed).any() and not (mm[query_bands]&(yy[query_bands]==0)).any()
        assert np.isfinite(d['weights']).all() and (d['weights']>=0).all()
        assert np.all(d['weights'][~mm]==0) and np.all(d['weights'][mm]>0)
        ts,fi=times[r['clip_id']]
        assert np.array_equal(fi,np.arange(n)) and np.array_equal(d['timestamp_us'],ts[-1]-ts[::-1] if r['reverse'] else ts)
        assert np.array_equal(d['source_frame_id'],fi[::-1] if r['reverse'] else fi)
        assert (np.diff(d['timestamp_us'])>0).all()
        counts=collections.Counter(map(int,pp.flat));codes.update(counts);positive+=int(((yy>0)&mm).sum())
        source_counts[r['clip_id']].update(counts);data.append(d)
    class_weights=[]
    for j,family in enumerate(F):
        for value in [0,1]:
            eligible={r['clip_id'] for r,d in zip(rows,data) if ((d['mask'][:,j]>0)&(d['y'][:,j]==value)).any()}
            total=0.
            for cid in eligible:
                ids=[i for i,(r,d) in enumerate(zip(rows,data)) if r['clip_id']==cid and ((d['mask'][:,j]>0)&(d['y'][:,j]==value)).any()]
                for i in ids:
                    d=data[i];take=(d['mask'][:,j]>0)&(d['y'][:,j]==value)
                    expected=1./(10*len(eligible)*len(ids)*int(take.sum()))
                    assert np.allclose(d['weights'][take,j],expected,rtol=0,atol=1e-15)
                    total+=float(d['weights'][take,j].sum())
            assert np.isclose(total,.1,rtol=0,atol=1e-12)
            class_weights.append(dict(family=family,label=value,total=total,eligible_sources=len(eligible)))
    assert positive==822 and removed==102 and codes[3]==522 and codes[4]==126 and codes[2]==1308
    assert {str(k):v for k,v in codes.items()}==meta['counts']['supervised_codes']
    assert np.isclose(sum(d['weights'].sum() for d in data),1.,rtol=0,atol=1e-12)
    pins_match(meta['pins']);pins_match(protected)
    result=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
        status='PASS_SAVED_FIT_INPUTS_VERIFIED_BEFORE_TRAINING',reviewer='Codex progress_input_review_resume separate saved-array arithmetic; not independent human/KSL gold',
        method='Independent label reconstruction from old saved y/mask and immutable source holds/bands; no prepare.targets call',
        fit_sources=24,fit_people=[2,3,4,5,6,7],fit_sequences=144,probe_metadata_only=48,probe_array_payload_reads=0,
        original_positive_unchanged=positive,original_negative_boundary_removed=removed,supervised_codes=dict(codes),
        own_terminal_hold_observations=522,other_family_terminal_hold_observations=126,newly_supervised_observations=522,
        negative_observations=1956,unknown_zero_weight_observations=codes[0],counts_are_not_video_or_people_counts=True,
        exact_feature_copy=True,original_timestamp_and_frame_ids_verified=True,all_family_class_source_view_frame_weights_verified=True,
        family_class_weights=class_weights,source_counts={k:dict(v) for k,v in source_counts.items()},protected_hashes_verified=477,
        original_training_false_preserved=True,training_executed=False,new_inference=False,web_changed=False,
        inputs_sha256=sha(U/'INPUTS_COMPLETE.json'),frozen_sha256=sha(U/'FROZEN.json'),admission_sha256=sha(U/'SOURCE_ADMISSION.json'),verifier_sha256=sha(__file__))
    with (U/'PRETRAIN_VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ['status','fit_sequences','original_positive_unchanged','negative_observations','unknown_zero_weight_observations','protected_hashes_verified']}),flush=True)

if __name__=='__main__': main()
