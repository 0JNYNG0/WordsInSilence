"""Read-only post-hoc FIT cause audit. No model inference or changed labels."""
from pathlib import Path
import json, hashlib, collections, sys
import numpy as np
sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
O = U.parent.parent
G = O/'word100_web_completion_graph_20261111'
C6 = G/'typed_motion_recovery_20261113/body_context_generalization'
C7 = C6/'c7_local_cause_recovery_20261113'
C9 = C7/'c9_sequence_context'
F = [2, 3, 4, 5, 7]
AX = {2:0,3:2,4:4,5:8,7:6}
def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):
    with (U/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    inputs=[C6/'INDEX.json',C6/'INPUTS.npz',C6/'TRAINING_COMPLETE.json',C9/'TRAINING_COMPLETE.json',C9/'FIT_AUDIT.json',C9/'CAUSE_AUDIT.json',C9/'RGB_REVIEW.json',C7/'NEXT_RESIDUALS.json',Path(__file__)]
    ann=[G/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',G/'candidate_description_supervision/additional_source_review/SOURCE_EPISODES_ADDED.json']
    pins={str(p):sha(p) for p in inputs+ann}
    sources={r['clip_id']:r for p in ann for r in read(p)['rows'] if r['role']=='fit' and r['person'] in range(2,8)}
    idx=read(C6/'INDEX.json')['rows'];z=dict(np.load(C6/'INPUTS.npz'));raw=z['absolute_context'].reshape(-1,3,10,5)[:,1]
    assert len(sources)==24 and {r['person'] for r in idx}==set(range(2,8))
    old=read(C6/'TRAINING_COMPLETE.json')['rows'];new=read(C9/'TRAINING_COMPLETE.json')['rows']
    outputs={}
    for seed in [42,43,44]:
        outputs[seed]={k:np.full(z['y'].shape,np.nan) for k in ['C6','C9_mean','C9_ordered']}
        for person in range(2,8):
            for name,table,condition in [('C6',old,'relative_context'),('C9_mean',new,'mean_control'),('C9_ordered',new,'ordered')]:
                r=next(r for r in table if r['person']==person and r['seed']==seed and r['condition']==condition)
                assert sha(r['prediction'])==r['prediction_sha256'];pins[r['prediction']]=r['prediction_sha256']
                p=np.load(r['prediction'])['logits'];take=z['person']==person;outputs[seed][name][take]=p[take]
        assert all(np.isfinite(a).all() for a in outputs[seed].values())
    write('AUDIT_FROZEN.json',dict(scope='Public FIT saved C6/C9 outputs, prior C8 loss11, matched successes; no new scores/inference/model runs',pins=pins,
        selection='All known motion/own-hold C6-correct/C9-ordered-wrong cells retained. Source-point union separate from view/time-reversal/seed repeats. Panels: per family/label most persistent native full-forward loss + nearest same-source success; no post-audit score selection.',
        protected='Original C6-C9 models/results and labels read-only; no DEV/final8/private/source writes',
        causal_limits='Observed source labels Codex reviewed, not independent KSL gold. Correlated geometry/failure flags are not a proof that one factor caused failure.',
        link_limit='C9 outputs five point-level body movement scores, not automatically selected intervals. Its own interval/model factorial cannot be invented. Historical same-input interval evidence must be separately identified.'))
    metrics=[];regressions=collections.defaultdict(list);native={};cases=[]
    for r in idx:
        if r['view']=='full' and not r['reverse']:
            for i in range(r['start'],r['stop']):
                for j,fam in enumerate(F):
                    code=int(z['label_provenance'][i,j])
                    if code in [1,3]:native[(r['clip_id'],int(z['source_frame_id'][i]),fam,code)]=(i,j,r)
    for seed,out in outputs.items():
        c0=(out['C6']>=0)==(z['y']>0);c1=(out['C9_ordered']>=0)==(z['y']>0)
        for code in [1,2,3,4]:
            take=z['label_provenance']==code
            metrics.append(dict(seed=seed,code=code,n=int(take.sum()),C6=int(c0[take].sum()),C9=int(c1[take].sum()),lost=int((take&c0&~c1).sum()),gained=int((take&~c0&c1).sum())))
        for i,j in zip(*np.where(np.isin(z['label_provenance'],[1,3])&c0&~c1)):
            r=idx[int(z['seq_id'][i])];key=(r['clip_id'],int(z['source_frame_id'][i]),F[j],int(z['label_provenance'][i,j]))
            regressions[key].append(dict(seed=seed,view=r['view'],reverse=r['reverse'],C6_logit=float(out['C6'][i,j]),C9_logit=float(out['C9_ordered'][i,j])))
    def detail(key,kind):
        cid,fr,fam,code=key;i,j,r=native[key];axis=AX[fam];ts=z['timestamp_us'];sel=np.arange(max(r['start'],i-4),min(r['stop'],i+5));a=raw[sel,axis];m=a[:,4]>.5
        events=[e for e in sources[cid]['events'] if e['family']==fam]
        boundary=any(x<=fr<=y for e in events for x,y in [e['onset'],e['offset']])
        steps=np.diff(a[:,:4],axis=0);valid=m[:-1]&m[1:]
        logits={str(seed):{name:float(v[i,j]) for name,v in out.items()} for seed,out in outputs.items()}
        flags=['native_point_classifier_error_not_created_by_interval_selection'] if kind=='regression' else []
        if not m.all():flags.append('local_observation_gap_coexists')
        if boundary:flags.append('existing_uncertain_boundary_band_but_direct_label_kept')
        if fam==4 and float(raw[i,3,2])<.1:flags.append('foreshortened_reference_forearm')
        return dict(clip_id=cid,frame=fr,family=fam,code=code,role=kind,person=r['person'],source=sources[cid]['source'],source_sha256=sources[cid]['source_sha256'],source_review=sources[cid]['reviewer'],
            progression_events=events,hold_intervals=sources[cid]['holds'],in_boundary_band=boundary,context_source_frames=z['source_frame_id'][sel].tolist(),primary_observed=m.tolist(),primary_values=a[:,:4].tolist(),
            observed_adjacent_path4=np.abs(steps[valid]).sum(0).tolist(),current_opposite_forearm_length=float(raw[i,3,2]),saved_native_logits=logits,
            C9_new_regression_cells=regressions.get(key,[]),flags=flags,original_target_unchanged=True)
    for key in sorted(regressions):
        if key in native:cases.append(detail(key,'regression'))
    # A finite explanatory panel set; all errors remain in the table and denominator.
    panel=[];selected=set()
    for fam in F:
        for code in [1,3]:
            options=[k for k in regressions if k in native and k[2:]==(fam,code) and any(not v['reverse'] and v['view']=='full' for v in regressions[k])]
            if not options:continue
            key=min(options,key=lambda k:(-sum(v['view']=='full' and not v['reverse'] for v in regressions[k]),k))
            panel.append(detail(key,'regression'));selected.add(key)
            cid,fr,_,_=key
            success=[]
            for k,(i,j,r) in native.items():
                if k[0]==cid and k[2:]==(fam,code) and all((out['C6'][i,j]>=0)==bool(z['y'][i,j]) and (out['C9_ordered'][i,j]>=0)==bool(z['y'][i,j]) for out in outputs.values()):success.append(k)
            if success:panel.append(detail(min(success,key=lambda k:(abs(k[1]-fr),k)),'matched_success'))
    coverage=[]
    for fam in F:
        for code in [1,3]:
            ks=[k for k in native if k[2:]==(fam,code)]
            coverage.append(dict(family=fam,code=code,unique_native_points=len(ks),clips=sorted({k[0]for k in ks}),words=sorted({sources[k[0]]['word_id']for k in ks}),people=sorted({sources[k[0]]['person']for k in ks})))
    point_summary=collections.Counter((r['family'],r['code']) for r in cases)
    out=dict(status='SAVED_CAUSE_AUDIT_COMPLETE_NO_NEW_MODEL',metrics=metrics,C8_prior_loss11=read(C7/'NEXT_RESIDUALS.json')['records'],C9_unique_source_points=len(cases),
        C9_unique_by_family_code=[dict(family=f,code=c,n=n)for(f,c),n in sorted(point_summary.items())],
        errors_inside_boundary=sum(r['in_boundary_band']for r in cases),errors_with_context_gap=sum(not all(r['primary_observed'])for r in cases),
        family_training_coverage=coverage,records=cases,panel_records=panel,
        evidence='Current native movement/hold errors exist before interval choice. Interval improvement alone cannot repair these point-level errors. Source/posture/observation variation may coexist; point values alone do not prove semantics.',
        limitations='Known rows only; no missing observed point was made normal or deleted. Repeated views/reversal/seeds not independent examples. No wristrotation or full-action completion label in this five-output diagnostic.',
        no_label_changes=True,no_new_inference=True,no_new_training=True)
    write('SAVED_CAUSE_AUDIT.json',out)
    assert all(sha(p)==h for p,h in pins.items())
    print(json.dumps({k:out[k]for k in ['status','metrics','C9_unique_source_points','C9_unique_by_family_code','errors_inside_boundary','errors_with_context_gap']},ensure_ascii=False))
if __name__=='__main__':main()
