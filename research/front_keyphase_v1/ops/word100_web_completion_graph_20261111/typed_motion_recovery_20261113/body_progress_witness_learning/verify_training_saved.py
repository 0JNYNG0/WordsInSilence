"""Saved FIT outputs only: no model construction, forward or training rerun."""
from pathlib import Path
import collections, datetime, hashlib, json
import numpy as np
import torch

U=Path(__file__).resolve().parent; G=U.parent.parent
F=[2,3,4,5,7]
NAMES={1:'original_direct_motion',2:'old_non_target_outside_bands',3:'own_terminal_body_hold',4:'other_family_terminal_body_hold'}
def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pins_match(pins):
    for p,h in pins.items(): assert sha(p)==h,p
def summarize(old,new,y,take):
    oc=(old>=0)==(y>0); nc=(new>=0)==(y>0)
    return dict(count=int(take.sum()),old_correct=int((take&oc).sum()),new_correct=int((take&nc).sum()),
                old_failures=int((take&~oc).sum()),new_failures=int((take&~nc).sum()),
                regressions=int((take&oc&~nc).sum()),recoveries=int((take&~oc&nc).sum()))
def bce(z,y,w):
    z=z.astype(np.float64);y=y.astype(np.float64)
    return float(((np.maximum(z,0)-z*y+np.log1p(np.exp(-np.abs(z))))*w).sum())

def main():
    completed=read(U/'TRAINING_COMPLETE.json');cfg=read(U/'TRAINING_FROZEN.json');meta=read(U/'INPUTS_COMPLETE.json')
    pre=read(U/'PRETRAIN_VERIFICATION.json')
    assert pre['status']=='PASS_SAVED_FIT_INPUTS_VERIFIED_BEFORE_TRAINING'
    assert pre['inputs_sha256']==sha(U/'INPUTS_COMPLETE.json')
    assert completed['training_frozen_sha256']==sha(U/'TRAINING_FROZEN.json')
    assert cfg['seeds']==[42,43,44] and cfg['steps']==1200
    assert len(completed['rows'])==3 and {r['seed'] for r in completed['rows']}=={42,43,44}
    pins_match(cfg['pins'])
    protected=cfg['protected_pins'];assert protected==meta['protected_pins']==read(G/'PROTECTIONS.json') and len(protected)==477
    pins_match(protected)
    rows=meta['sequences'];data=[]
    for r in rows:
        assert sha(r['path'])==r['sha256'] and r['split']=='train'
        with np.load(r['path'],allow_pickle=False) as z:data.append({k:z[k].copy() for k in ['y','mask','label_provenance','weights','source_frame_id']})
    flat={k:np.concatenate([d[k] for d in data]) for k in ['y','mask','label_provenance','weights']}
    out=[];failures=[];metrics_checks=0
    for run in sorted(completed['rows'],key=lambda r:r['seed']):
        seed=run['seed'];parent=next(p for p in meta['parent_models'] if p['seed']==seed)
        assert run['initial_model']==parent['model'] and run['initial_model_sha256']==parent['model_sha256']
        assert run['steps']==1200 and run['normalization_unchanged'] is True
        for key in ['model','predictions','prediction_index']:assert sha(run[key])==run[key+'_sha256']
        original=torch.load(parent['model'],map_location='cpu',weights_only=True)
        trained=torch.load(run['model'],map_location='cpu',weights_only=True)
        assert set(trained)=={'mean','std','head.0.weight','head.0.bias','head.2.weight','head.2.bias'}
        assert torch.equal(trained['mean'],original['mean']) and torch.equal(trained['std'],original['std'])
        assert trained['head.0.weight'].shape==(96,150) and trained['head.2.weight'].shape==(5,96)
        assert all(torch.isfinite(t).all() for t in trained.values())
        index=read(run['prediction_index']);assert index['source_inputs_sha256']==sha(U/'INPUTS_COMPLETE.json')
        assert len(index['rows'])==len(rows)==144
        oldlist=[];newlist=[];cursor=0;source_parts=collections.defaultdict(list)
        with np.load(parent['predictions'],allow_pickle=False) as previous,np.load(run['predictions'],allow_pickle=False) as current:
            assert set(current.files)=={f'logits_{i}' for i in range(144)}
            for i,(r,d,idx) in enumerate(zip(rows,data,index['rows'])):
                n=r['length'];assert idx==dict(input_index=i,original_sequence_index=r['original_sequence_index'],clip_id=r['clip_id'],view=r['view'],reverse=r['reverse'],start=cursor,length=n)
                a=previous[f"logits_{r['original_sequence_index']}"][:,:,2].copy();b=current[f'logits_{i}'].copy()
                assert a.shape==b.shape==(n,5) and np.isfinite(a).all() and np.isfinite(b).all()
                oldlist.append(a);newlist.append(b);source_parts[r['clip_id']].append((a,b,d))
                oc=(a>=0)==(d['y']>0);nc=(b>=0)==(d['y']>0)
                for t,j in np.argwhere((d['mask']>0)&(~nc)):
                    failures.append(dict(seed=seed,clip_id=r['clip_id'],view=r['view'],reverse=r['reverse'],query_frame_id=int(t),source_frame_id=int(d['source_frame_id'][t]),family=F[j],label_code=int(d['label_provenance'][t,j]),kind='regression' if oc[t,j] else 'persisting_failure',old_logit=float(a[t,j]),new_logit=float(b[t,j])))
                cursor+=n
        old=np.concatenate(oldlist);new=np.concatenate(newlist);y=flat['y']
        summary={NAMES[c]:summarize(old,new,y,flat['label_provenance']==c) for c in NAMES}
        assert summary['original_direct_motion']['count']==822 and summary['own_terminal_body_hold']['count']==522
        assert sum(summary[n]['count'] for n in summary)==2778
        family_rows=[]
        for j,family in enumerate(F):
            for label in [0,1]:
                take=(flat['mask'][:,j]>0)&(y[:,j]==label)
                s=summarize(old[:,j],new[:,j],y[:,j],take)
                recorded=next(r for r in run['FIT'] if r['family']==family and r['label']==label)
                assert recorded==dict(family=family,label=label,count=s['count'],old_correct=s['old_correct'],new_correct=s['new_correct'])
                family_rows.append(dict(family=family,label=label,**s));metrics_checks+=1
        source_rows=[]
        for cid,parts in sorted(source_parts.items()):
            a=np.concatenate([p[0] for p in parts]);b=np.concatenate([p[1] for p in parts]);yy=np.concatenate([p[2]['y'] for p in parts]);pp=np.concatenate([p[2]['label_provenance'] for p in parts])
            source_rows.append(dict(clip_id=cid,by_provenance={NAMES[c]:summarize(a,b,yy,pp==c) for c in NAMES}))
        out.append(dict(seed=seed,model_sha256=run['model_sha256'],predictions_sha256=run['predictions_sha256'],normalization_exactly_unchanged=True,
                        by_provenance=summary,by_family_class=family_rows,by_source=source_rows,
                        weighted_BCE=dict(old=bce(old,y,flat['weights']),new=bce(new,y,flat['weights']))))
    pins_match(cfg['pins']);pins_match(protected)
    result=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),status='SAVED_FIT_METRICS_SEPARATELY_VERIFIED_NOT_GENERALIZATION_OR_ADOPTION',
        reviewer='Codex progress_input_review_resume saved-array recomputation; not independent human/KSL gold',runs=out,new_failure_observations=failures,
        training_report_family_class_checks=metrics_checks,fit_sources=24,fit_people=6,fit_sequences=144,protected_hashes_verified=477,
        observation_counts_include_view_and_reverse=True,source_rows_are_recording_level_not_unseen_person_validation=True,
        new_inference=False,training_rerun=False,probe_arrays_opened=0,completion=None,web_changed=False,
        input_sha256=sha(U/'INPUTS_COMPLETE.json'),training_complete_sha256=sha(U/'TRAINING_COMPLETE.json'),pretrain_sha256=sha(U/'PRETRAIN_VERIFICATION.json'),verifier_sha256=sha(__file__))
    with (U/'SAVED_FIT_VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],runs=[dict(seed=r['seed'],by_provenance=r['by_provenance'],weighted_BCE=r['weighted_BCE']) for r in out]),indent=2),flush=True)

if __name__=='__main__':main()
