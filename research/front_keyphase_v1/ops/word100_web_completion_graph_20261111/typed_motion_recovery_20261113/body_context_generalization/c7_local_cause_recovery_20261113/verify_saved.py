"""Independent arithmetic path over saved tensors; no neural forward/training."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
BASE=Path(__file__).resolve().parent;U=BASE/'conditioned_context' if len(sys.argv)>1 else BASE;P=BASE.parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,d):
    with (U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
checks=[]
def ck(name,yes):assert yes,name;checks.append(name)
def metric(a,b,y,take):
    aa=((a>=0)==(y>0))[take];bb=((b>=0)==(y>0))[take]
    return dict(count=int(aa.size),baseline_correct=int(aa.sum()),local_correct=int(bb.sum()),regressions=int((aa&~bb).sum()),recoveries=int((~aa&bb).sum()))
c=read(U/'TRAINING_FROZEN.json');train=read(U/'TRAINING_COMPLETE.json');z=np.load(P/'INPUTS.npz');d={k:z[k]for k in z.files};idx=read(P/'INDEX.json')['rows'];ca=read(BASE/'CAUSE_AUDIT.json');F=[2,3,4,5,7]
ck('new18only',len(train['rows'])==18)
for p,h in {**c['pins'],**c['protected_pins']}.items():ck('hash '+p,sha(p)==h)
agg=[];families=[];folds=[];changes=[];invariance=[];residual=[]
for seed in c['seeds']:
    old=np.full_like(d['y'],np.nan,dtype=np.float32);new=old.copy()
    for person in range(2,8):
        a=next(r for r in c['baseline_rows']if r['seed']==seed and r['person']==person)
        b=next(r for r in train['rows']if r['seed']==seed and r['person']==person)
        ck('pairedinitial',a['initial_trainable_hash']==b['initial_trainable_hash'])
        for rr in [a,b]:
            ck('predhash',sha(rr['prediction'])==rr['prediction_sha256']);ck('modelhash',sha(rr['model'])==rr['model_sha256'])
        aa=np.load(a['prediction'])['logits'];bb=np.load(b['prediction'])['logits'];keep=d['person']==person
        ck('finiteshape',aa.shape==bb.shape==d['y'].shape and np.isfinite(bb).all())
        old[keep]=aa[keep];new[keep]=bb[keep]
        folds.append(dict(person=person,seed=seed,categories={str(k):metric(aa,bb,d['y'],(d['label_provenance']==k)&keep[:,None])for k in [1,2,3,4]}))
    ck('eachheldIDonce',np.isfinite(old).all()and np.isfinite(new).all())
    agg.append(dict(seed=seed,categories={str(k):metric(old,new,d['y'],d['label_provenance']==k)for k in [1,2,3,4]}))
    for j,f in enumerate(F):families.append(dict(seed=seed,family=f,categories={str(k):metric(old[:,j],new[:,j],d['y'][:,j],d['label_provenance'][:,j]==k)for k in [1,2,3,4]}))
    for row,j in zip(*np.where((d['mask']>0)&(((old>=0)==(d['y']>0))!=((new>=0)==(d['y']>0))))):
        r=idx[int(d['seq_id'][row])];changes.append(dict(seed=seed,clip_id=r['clip_id'],person=r['person'],view=r['view'],reverse=r['reverse'],source_frame=int(d['source_frame_id'][row]),family=F[j],code=int(d['label_provenance'][row,j]),kind='recovery'if(new[row,j]>=0)==(d['y'][row,j]>0)else'regression',old=float(old[row,j]),new=float(new[row,j])))
    by={(r['clip_id'],r['view'],r['reverse']):r for r in idx}
    def at(cid,view,fr,rev=False):
        r=by[cid,view,rev];ii=np.flatnonzero(d['source_frame_id'][r['start']:r['stop']]==fr);ck('oneframe',len(ii)==1);return r['start']+int(ii[0])
    rows=[]
    for r in ca['controls']:
        if not r['primary_identical']:continue
        a=at(r['clip_id'],'full',r['frame']);b=at(r['clip_id'],'upper',r['frame']);j=F.index(r['family'])
        rows.append(dict(**r,old_same_decision=bool((old[a,j]>=0)==(old[b,j]>=0)),new_same_decision=bool((new[a,j]>=0)==(new[b,j]>=0)),new_logit_diff=float(abs(new[a,j]-new[b,j]))))
    invariance.append(dict(seed=seed,count=len(rows),old_inconsistent=sum(not r['old_same_decision']for r in rows),new_inconsistent=sum(not r['new_same_decision']for r in rows),max_new_logit_difference=max(r['new_logit_diff']for r in rows)))
    for r in ca['records']:
        jj=F.index(r['family']);vv=[]
        for view in ['full','upper','actual_crop']:
            for rev in [False,True]:
                k=at(r['clip_id'],view,r['source_frame'],rev)
                vv.append(dict(view=view,reverse=rev,qualified=bool(d['mask'][k,jj]),old_motion=bool(old[k,jj]>=0),new_motion=bool(new[k,jj]>=0)))
        residual.append(dict(seed=seed,clip_id=r['clip_id'],frame=r['source_frame'],family=r['family'],observations=vv))
gate=dict(overall_motion_hold_preserved=all(r['categories'][str(k)]['local_correct']>=r['categories'][str(k)]['baseline_correct']for r in agg for k in [1,3]),
    every_motion_family_preserved=all(r['categories']['1']['local_correct']>=r['categories']['1']['baseline_correct']for r in families),
    no_fold_loses_both=all(not all(r['categories'][str(k)]['local_correct']<r['categories'][str(k)]['baseline_correct']for k in [1,3])for r in folds),
    same_evidence_view_invariance=all(r['new_inconsistent']<=(r['old_inconsistent']if U!=BASE else 0) for r in invariance),
    strict_gain=any(r['categories'][str(k)]['local_correct']>r['categories'][str(k)]['baseline_correct']for r in agg for k in [1,3]))
report=dict(status='DIAGNOSTIC_PASS'if all(gate.values())else'DIAGNOSTIC_REWORK',gate=gate,aggregate=agg,by_family=families,by_person=folds,invariance=invariance,changed_observations=changes,original15_residual_followup=residual,
    G2_pass=False,web_changed=False,no_independent_KSL_expert=True,denominator='Correlated sourcepoint×family×3views×2timeorientations; historicalFITonly, not independentuser accuracy.')
write('COMPARISON.json',report)
# Separate loop recount, not reusing vectorized metric.
for seedrow in agg:
    for code,cell in seedrow['categories'].items():
        foldcells=[r['categories'][code]for r in folds if r['seed']==seedrow['seed']]
        for field in cell:ck('foldrecount',sum(r[field]for r in foldcells)==cell[field])
write('SAVED_VERIFICATION.json',dict(status='PASS',checks=len(checks),comparison_sha256=sha(U/'COMPARISON.json'),training_frozen_sha256=sha(U/'TRAINING_FROZEN.json'),method='Separate saved arithmetic by same Codex; no model re-inference, no independent human ground truth.'))
print(json.dumps(dict(gate=gate,aggregate=agg,invariance=invariance,forearm=[r for r in families if r['family']==4]),ensure_ascii=False))
