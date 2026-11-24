"""Separate body-progress readout labels, using immutable public FIT observations.

No observation/model forward, no original label/model/output edits. Own terminal
body holds are negative ONLY for local family progression, never for sign quality,
action existence, hand articulation, source extent or required completion.
"""
from pathlib import Path
import collections, hashlib, json, time, traceback
import numpy as np
U=Path(__file__).resolve().parent;H=U.parent;G=H.parent;O=G.parent;W=G/'candidate_description_supervision'
R=W/'source_rank_learning';F=[2,3,4,5,7];AX={2:0,3:1,4:2,5:4,7:3}
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
def resolved_pins(pins):
    assert isinstance(pins,dict)and pins
    result={}
    for p,h in pins.items():
        assert isinstance(p,str)and isinstance(h,str)and len(h)==64
        assert all(c in '0123456789abcdef'for c in h)
        key=Path(p).resolve();assert key not in result,p
        result[key]=h
    return result
def source_admission(path,base_path,added_path,base,additional,sources):
    """A new scope-limited admission never changes historical source flags."""
    admission=read(path)
    assert admission['status']=='QUALIFIED_FOR_SPARSE_BODY_PROGRESS'
    assert admission['training_allowed']is True
    assert admission['scope']=='sparse_body_progress_only'
    assert admission['original_flags_preserved']is True
    assert base['training_allowed']is False and additional['training_allowed']is False
    annotations=resolved_pins(admission['source_annotations'])
    assert annotations=={base_path.resolve():sha(base_path),added_path.resolve():sha(added_path)}
    approved=admission['approved_clip_ids']
    assert isinstance(approved,list)and len(approved)==24
    assert all(isinstance(cid,str)for cid in approved)
    assert len(set(approved))==24 and set(approved)==set(sources)
    evidence=resolved_pins(admission['review_evidence'])
    prior_review=(H/'observed_action_anchor_design/CORRECTION_BOUNDARY_HOLD.json').resolve()
    assert prior_review in evidence and len(evidence)>=2
    check(evidence)
    prior_rows=read(prior_review)['rows']
    assert len(prior_rows)==16 and {r['clip_id']for r in prior_rows}=={
        cid for cid,s in sources.items()if s['person']in[2,4,5,6]}
    return {str(p):h for p,h in evidence.items()}
def stop():
    for d in[U,H,G]:
        for name in['STOP','STOP.txt','STOP_REQUESTED','STOP_REQUESTED.json']:assert not(d/name).exists()

def targets(source,original,reverse):
    """Strong sparse labels only; no interval filling or false exact boundary."""
    x=original['x'];n=len(x);y=original['y'].copy();mask=original['mask'].astype(bool).copy()
    assert x.shape==(n,150)and y.shape==mask.shape==(n,5)
    original_positive=mask&(y>0)
    # Current body block: ten side/axis entries of 4 coordinates + mask.
    current=x[:,50:100].reshape(n,10,5)
    observed=np.stack([current[:,AX[f]*2,4]>.5 for f in F],axis=1)
    assert not(mask&~observed).any()
    bands=np.zeros(n,bool)
    for event in source['events']:
        for a,b in[event['onset'],event['offset']]:
            assert 0<=a<=b<n;bands[a:b+1]=True
    if reverse:bands=bands[::-1].copy()
    removed_negative=mask&(y==0)&bands[:,None]
    mask[removed_negative]=False
    provenance=np.where(mask,np.where(y>0,1,2),0).astype(np.int8)
    last=max(source['groups'],key=lambda g:g['offset'][1])['family']
    families={g['family']for g in source['groups']}
    holds=[]
    for f in source['hold_sample_frames']:
        assert any(a<=f<=b for a,b in source['holds'])
        i=n-1-f if reverse else f
        for family in sorted(families):
            j=F.index(family)
            status='uncertain_boundary'if bands[i]else'body_unobserved'if not observed[i,j]else'eligible_explicit_body_hold'
            if status=='eligible_explicit_body_hold':
                assert not original_positive[i,j],('motion/hold contradiction',source['clip_id'],f,family)
                y[i,j]=0.;mask[i,j]=True;provenance[i,j]=3 if family==last else 4
            holds.append(dict(source_frame_id=f,query_frame_id=i,family=family,status=status,own_terminal_family=family==last))
    assert np.array_equal(original_positive,mask&(y>0))
    assert not(mask&~observed).any()and not(mask&(y==0)&bands[:,None]).any()
    return dict(y=y,mask=mask.astype(np.float32),observed=observed,valid=observed.copy(),label_provenance=provenance),dict(holds=holds,removed_boundary_negative=int(removed_negative.sum()),positive_unchanged=int(original_positive.sum()))

def main():
    stop();assert not(U/'FROZEN.json').exists()and not(U/'INPUTS_COMPLETE.json').exists()
    meta_path=R/'INPUTS_COMPLETE.json';train_path=R/'TRAINING_COMPLETE.json'
    base_path=G/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json'
    added_path=W/'additional_source_review/SOURCE_EPISODES_ADDED.json'
    admission_path=U/'SOURCE_ADMISSION.json';protection_path=G/'PROTECTIONS.json'
    ready_path=O/'word100_shared_motion_learning_20261109/READY.json'
    files=[Path(__file__),meta_path,train_path,base_path,added_path,ready_path,admission_path,protection_path,
           W/'prepare_source_boundary.py',W/'prepare_rank_learning.py',W/'source_boundary_core.py',W/'core.py',
           G/'episode_stage_contrast/RGB_CONTRAST_LABELS.json',W/'short_action_review/REVIEW.json',W/'short_action_review/FOREARM_REVIEW.json']
    pins={str(p):sha(p)for p in files};protections=read(protection_path);assert len(protections)==477;check(protections)
    meta=read(meta_path);tr=read(train_path)
    base=read(base_path);base_rows=[r for r in base['rows']if r['person']in[2,4,5,6]]
    sources={r['clip_id']:r for r in base_rows}
    assert len(base_rows)==len(sources)==16
    additional=read(added_path);assert len(additional['rows'])==8
    assert {r['person']for r in additional['rows']}=={3,7}
    for r in additional['rows']:assert r['clip_id']not in sources;sources[r['clip_id']]=r
    assert len(sources)==24 and {r['person']for r in sources.values()}=={2,3,4,5,6,7}
    assert all(r['role']=='fit'for r in sources.values())
    pins.update(source_admission(admission_path,base_path,added_path,base,additional,sources))
    fits=[(i,r)for i,r in enumerate(meta['sequences'])if r['split']=='train']
    probe=[dict(original_sequence_index=i,**r)for i,r in enumerate(meta['sequences'])if r['split']=='probe']
    assert len(fits)==144 and {r['clip_id']for i,r in fits}==set(sources)
    assert len(probe)==48 and {r['person']for r in probe}=={8,9}
    assert all(r['person']==sources[r['clip_id']]['person']for i,r in fits)
    assert not({r['clip_id']for i,r in fits}&{r['clip_id']for r in probe})
    assert not({r['source_sha256']for i,r in fits}&{r['source_sha256']for r in probe})
    for cid in sources:
        variants=[(r['view'],r['reverse'])for i,r in fits if r['clip_id']==cid]
        assert len(variants)==6 and set(variants)=={
            (view,reverse)for view in['full','upper','actual_crop']for reverse in[False,True]}
    parents=[r for r in tr['rows']if r['condition']=='boundary_extended']
    assert {r['seed']for r in parents}=={42,43,44}and len(parents)==3
    for r in parents:
        pins[r['model']]=r['model_sha256'];pins[r['predictions']]=r['predictions_sha256']
    raw={r['clip_id']:r for r in read(ready_path)['rows']if r['role']=='fit'and r['clip_id']in sources}
    assert set(raw)==set(sources)
    time_cache={}
    for cid,r in raw.items():
        assert sha(r['path'])==r['sha256'];pins[r['path']]=r['sha256']
        with np.load(r['path'],allow_pickle=False)as z:ts=z['timestamp_us'].copy();fi=z['frame_id'].copy()
        assert np.array_equal(fi,np.arange(len(ts)))and(np.diff(ts)>0).all()
        time_cache[cid]=(ts,fi)
    for i,r in fits:assert sha(r['path'])==r['sha256'];pins[r['path']]=r['sha256']
    check(pins)
    write(U/'FROZEN.json',dict(status='BODY_PROGRESS_WITNESS_INPUT_CONTRACT_NO_TRAINING',family_order=F,
        purpose='A separate observed body-family progression-vs-maintained-posture readout. Existing extent/presence/shape/direction/TCN remain immutable.',
        input='Existing 150-value body_context only; no fingers, latent, word, guide order, absolute clock or source IDs as model input. Rate1 only for isolated supervision change.',
        label_codes={'0':'unknown/masked','1':'unchanged direct progression witness','2':'old reviewed non-target observation outside all inclusive event bands','3':'explicit same-family terminal body hold outside bands','4':'explicit other-family terminal body hold outside bands'},
        hold_contract='Sparse declared hold_sample_frames inside reviewed holds only. All inclusive onset/offset intersections withheld. Not a handshape/finger-change label, no mandatory hold, no incorrect-sign or absence label.',
        positive_contract='Keep all old direct motion positives even inside onset/offset bands; uncertainty in exact boundary does not cancel an explicit observed motion witness.',
        loss='Weighted binary cross entropy, 1/5 per family and 1/2 per class; each eligible source equal, each available view/reverse equal within source/class; known frames equal within sequence. Unknown has zero weight.',
        initial='Matching source_rank_learning boundary_extended parent: preserve mean/std, copy 150->96 hidden and output rows [2,5,8,11,14]. Original parent never modified.',
        training_proposed=dict(seeds=[42,43,44],steps=1200,lr=.001,weight_decay=.001,final_only=True),
        split=dict(fit_people=[2,3,4,5,6,7],fit_sources=24,fit_sequences=144,probe_people=[8,9],probe_metadata_only=48),
        source_admission=dict(path=str(admission_path),sha256=sha(admission_path),scope='sparse_body_progress_only',original_flags_preserved=True),
        no_pose_or_encoder=True,no_model_forward=True,no_probe_array_access=True,no_private=True,no_final8=True,no_web=True,
        no_original_label_edit=True,source_extent_completion=None,semantic_completion=None,pins=pins,protected_pins=protections))
    data=[];rows=[];audit=[]
    for i,r in fits:
        with np.load(r['path'],allow_pickle=False)as z:old={k:z[k].copy()for k in['x','y','mask','bands','valid']}
        assert np.isfinite(old['x']).all();src=sources[r['clip_id']]
        assert src['source_sha256']==r['source_sha256']
        d,a=targets(src,old,r['reverse']);ts,fi=time_cache[r['clip_id']]
        assert len(ts)==len(old['x'])
        d.update(x=old['x'],weights=np.zeros_like(d['y'],dtype=np.float64),timestamp_us=ts[-1]-ts[::-1]if r['reverse']else ts.copy(),source_frame_id=fi[::-1].copy()if r['reverse']else fi.copy())
        data.append(d);rows.append(dict(original_sequence_index=i,clip_id=r['clip_id'],person=r['person'],view=r['view'],reverse=r['reverse'],rate=1.,split='train',length=len(ts),original_path=r['path'],original_sha256=r['sha256'],source_sha256=r['source_sha256']))
        audit.append(dict(clip_id=r['clip_id'],view=r['view'],reverse=r['reverse'],**a))
    for j,family in enumerate(F):
        for value in[0,1]:
            clips=sorted({r['clip_id']for r,d in zip(rows,data)if ((d['mask'][:,j]>0)&(d['y'][:,j]==value)).any()})
            assert clips,('missing class',family,value)
            for cid in clips:
                ids=[i for i,(r,d)in enumerate(zip(rows,data))if r['clip_id']==cid and ((d['mask'][:,j]>0)&(d['y'][:,j]==value)).any()]
                for i in ids:
                    d=data[i];take=(d['mask'][:,j]>0)&(d['y'][:,j]==value)
                    d['weights'][take,j]=1/(len(F)*2*len(clips)*len(ids)*int(take.sum()))
    weights=np.array([[sum(float(d['weights'][(d['mask'][:,j]>0)&(d['y'][:,j]==v),j].sum())for d in data)for v in[0,1]]for j in range(5)])
    assert np.allclose(weights,.1,atol=1e-12)and np.isclose(weights.sum(),1.)
    for i,(r,d)in enumerate(zip(rows,data)):
        assert np.isfinite(d['weights']).all()and not(d['weights'][d['mask']==0]!=0).any()
        dest=U/'sequences'/f'{i}.npz';dest.parent.mkdir(exist_ok=True);assert not dest.exists();np.savez_compressed(dest,**d)
        r.update(path=str(dest),sha256=sha(dest))
    counts=dict(fit_sequences=len(rows),fit_sources=len(sources),positive_observations=sum(a['positive_unchanged']for a in audit),removed_old_negative_at_boundary=sum(a['removed_boundary_negative']for a in audit),
        hold_family_observation_status=dict(collections.Counter(q['status']for a in audit for q in a['holds'])),
        supervised_codes=dict(collections.Counter(str(int(v))for d in data for v in d['label_provenance'].flat)),weights_by_family_and_class=weights.tolist())
    stop();check(pins);check(protections)
    write(U/'LABEL_AUDIT.json',dict(counts=counts,source_review='Existing Codex visual review drafts, not independent human/KSL gold.',rows=audit,no_original_labels_changed=True))
    write(U/'INPUTS_COMPLETE.json',dict(status='BODY_PROGRESS_WITNESS_FIT_INPUTS_COMPLETE_NOT_TRAINED',sequences=rows,probe_metadata=probe,parent_models=parents,family_order=F,counts=counts,
        weights_contract='Global total1, each family/class=.1, eligible source/view/reverse/frame hierarchy; no class or frame-count imbalance gain.',
        normalization='Retain matching parent mean/std; no new normalization fit.',initial_predictions='Use parent saved predictions logits_{original_sequence_index}[:,:,2] only.',
        source_admission=dict(path=str(admission_path),sha256=sha(admission_path)),
        frozen_sha256=sha(U/'FROZEN.json'),label_audit_sha256=sha(U/'LABEL_AUDIT.json'),pins=pins,protected_pins=protections))
    print(json.dumps(counts),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        with (U/f'FIRST_FAILURE_{time.time_ns()}.txt').open('x',encoding='utf-8')as f:f.write(traceback.format_exc())
        raise
