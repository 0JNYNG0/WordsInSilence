"""New no-full-extent bags from reviewed partial/other-action public windows.

All feature windows were previously encoded locally, with no outside frames.
This adds bag-level supervision absent from the old hold-only null experiment.
"""
from pathlib import Path
import json,hashlib,collections
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;G=O/'word100_web_completion_graph_20261111';H=G/'typed_motion_recovery_20261113';N=H/'null_candidate_learning'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta=read(N/'TRAINING_INPUT_INDEX.json');masks=read(N/'local_bag_evaluation/RULES_FROZEN.json')['partial_masks']
source={r['clip_id']:r for p in [G/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',G/'candidate_description_supervision/additional_source_review/SOURCE_EPISODES_ADDED.json'] for r in read(p)['rows']}
rows=[];features=[];parents=read(N/'TRAINING_COMPLETE.json')['rows'];pred={p['seed']:np.load(p['predictions']) for p in parents}
pins={str(p):sha(p) for p in [N/'TRAINING_INPUT_INDEX.json',N/'local_bag_evaluation/RULES_FROZEN.json',N/'TRAINING_COMPLETE.json',U/'SOURCE_TRACE_TARGETS.json']}
for p in parents:
    assert sha(Path(p['predictions']))==p['predictions_sha256'];assert sha(Path(p['model']))==p['model_sha256']
    pins[p['predictions']]=p['predictions_sha256'];pins[p['model']]=p['model_sha256']
for bi,(r,m) in enumerate(zip(meta['rows'],masks)):
    assert m['bag_index']==bi and m['clip_id']==r['clip_id'] and m['family']==r['family']
    assert (r['split']=='train')==(r['person']<=7)
    assert sha(Path(r['path']))==r['sha256'];pins[r['path']]=r['sha256']
    z=np.load(r['path']);sp=z['spans'];x=z['local_full'];full=z['y'].astype(bool)
    def add(kind,ids,extent=None,basis=None):
        ids=np.asarray(ids,dtype=int);assert len(ids)>0
        if kind not in ('positive_source_extent',):assert not full[ids].any()
        scores={}
        family_index=[2,3,4,5,7].index(r['family'])
        for seed,p in pred.items():
            logits=p[f'logits_{bi}'][ids,family_index];null=float(p['null_logits'][family_index])
            scores[str(seed)]=dict(null_selected=bool(logits.max()<=null),max_extent_logit=float(logits.max()),null_logit=null)
        row=dict(index=len(rows),original_bag_index=bi,clip_id=r['clip_id'],person=r['person'],split=r['split'],
            view=r['view'],reverse=r['reverse'],family=r['family'],kind=kind,source_path=r['path'],source_sha256=r['sha256'],
            source_video_sha256=r['source_sha256'],candidate_indices=ids.tolist(),extent=extent,basis=basis,
            positive_candidates=int(full[ids].sum()),baseline=scores,semantic_completion=None)
        rows.append(row);features.append(dict(x=x[ids].copy(),y=full[ids].copy()))
    add(r['origin'],np.arange(len(full)))
    if r['origin']!='positive_source_extent':continue
    # One source partial crop per unique original reviewed witness subset. This
    # avoids treating hundreds of nearby windows as independent new examples.
    seen=set()
    for c in m['candidates']:
        a,b=c['transformed_span'];key=tuple((g['group_id'],tuple(g['witnessed_frames']),tuple(g['omitted_witnesses']))for g in c['groups'])
        if key in seen:continue
        ids=np.flatnonzero((sp[:,0]>=a)&(sp[:,1]<=b));assert len(ids)>0 and not full[ids].any()
        seen.add(key);add('partial_source_crop',ids,[int(a),int(b)],c['groups'])
    # Only explicitly annotated, distinct source actions in the SAME clip.
    # Different word names do not justify cross-class negative labels.
    s=source[r['clip_id']];length=m['source_length']
    for group in s['groups']:
        if group['family']==r['family']:continue
        a,b=group['onset'][0],group['offset'][1]
        pts=sorted(set(k for e in s['events'] if e['event_id'] in group['event_ids']for k in e['progression_witness_frames']))
        if r['reverse']:a,b=length-1-b,length-1-a;pts=sorted(length-1-k for k in pts)
        ids=np.flatnonzero((sp[:,0]>=a)&(sp[:,1]<=b))
        if len(ids)==0:continue
        # Some candidate must actually include the other moving action's
        # directly observed progression, rather than just a one-frame pose.
        if not any(sum(aa<=k<=bb for k in pts)>=2 for aa,bb in sp[ids]):continue
        assert not full[ids].any()
        add('other_reviewed_action_crop',ids,[int(a),int(b)],dict(other_group=group['group_id'],other_family=group['family'],other_progression=pts))
train=[i for i,r in enumerate(rows) if r['split']=='train'];probe=[i for i,r in enumerate(rows) if r['split']=='probe']
assert not ({rows[i]['source_video_sha256']for i in train}&{rows[i]['source_video_sha256']for i in probe})
counts=collections.Counter((r['split'],r['kind'])for r in rows)
baseline=[]
for seed in [42,43,44]:
    for kind in ['positive_source_extent','reviewed_local_hold','partial_source_crop','other_reviewed_action_crop']:
        rr=[r for r in rows if r['split']=='train' and r['kind']==kind]
        baseline.append(dict(seed=seed,kind=kind,bags=len(rr),null_selected=sum(r['baseline'][str(seed)]['null_selected'] for r in rr)))
arrays=U/'absence_bags.npz'
np.savez_compressed(arrays,**{f'{k}_{i}':v for i,d in enumerate(features)for k,v in d.items()})
out=dict(status='PREPARED_NEW_TYPED_NO_FULL_EXTENT_BAGS',rows=rows,train_indices=train,probe_indices=probe,
    arrays=str(arrays),arrays_sha256=sha(arrays),pins=pins,
    summary=[dict(split=k[0],kind=k[1],bags=v) for k,v in sorted(counts.items())],fit_baseline=baseline,
    target='No complete reviewed source extent in this bag. Partial progression remains positive for a separate progression observer; this is NOT a wrong-sign label.',
    common100_changed=False,new_encoder_inference=False,new_parent_model_inference=False,
    source_identity_negative_scope='Only reviewed distinct neighboring action in same source; not absent by word label.',
    no_private=True,no_final8=True)
with (U/'ABSENCE_BAGS.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({'summary':out['summary'],'fit_baseline':baseline},ensure_ascii=False))
