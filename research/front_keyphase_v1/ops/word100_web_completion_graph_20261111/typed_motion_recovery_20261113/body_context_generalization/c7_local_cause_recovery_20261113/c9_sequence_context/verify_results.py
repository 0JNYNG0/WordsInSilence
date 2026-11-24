"""Saved prediction arithmetic only; no trained-model forward."""
from pathlib import Path
import json,hashlib
import numpy as np
U=Path(__file__).resolve().parent;C7=U.parent;C6=C7.parent;G=C6.parent.parent;F=[2,3,4,5,7]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):
    with (U/n).open('x',encoding='utf-8')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
checks=[]
def ck(n,b):assert b,n;checks.append(n)
def metric(a,b,y,t):
    aa=((a>=0)==(y>0))[t];bb=((b>=0)==(y>0))[t]
    return dict(count=int(aa.size),before=int(aa.sum()),after=int(bb.sum()),recoveries=int((~aa&bb).sum()),regressions=int((aa&~bb).sum()))
c=read(U/'TRAINING_FROZEN.json');trained=read(U/'TRAINING_COMPLETE.json');d=dict(np.load(C6/'INPUTS.npz'));rows=read(C6/'INDEX.json')['rows'];ck('36new',len(trained['rows'])==36)
for p,h in {**c['pins'],**c['protected_pins']}.items():ck('pin',sha(p)==h)
sources={r['clip_id']:r for p in [G/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',G/'candidate_description_supervision/additional_source_review/SOURCE_EPISODES_ADDED.json']for r in read(p)['rows']if r['role']=='fit'}
boundary=np.zeros_like(d['y'],bool)
for i,row in enumerate(rows):
    idx=np.flatnonzero(d['seq_id']==i);fr=d['source_frame_id'][idx]
    for j,fam in enumerate(F):
        for e in sources[row['clip_id']]['events']:
            if e['family']==fam:
                boundary[idx,j]|=((fr>=e['onset'][0])&(fr<=e['onset'][1]))|((fr>=e['offset'][0])&(fr<=e['offset'][1]))
aggregate=[];family=[];folds=[];short=[];changes=[]
for seed in c['seeds']:
    outputs={k:np.full_like(d['y'],np.nan,dtype=np.float32)for k in ['C6','mean_control','ordered']}
    for person in range(2,8):
        pp=[next(r for r in trained['rows']if r['person']==person and r['seed']==seed and r['condition']==k)for k in ['mean_control','ordered']]
        ck('paired_init',pp[0]['initial_trainable_hash']==pp[1]['initial_trainable_hash'])
        old=next(r for r in c['baseline_rows']if r['person']==person and r['seed']==seed)
        pred={}
        for k,r in [('C6',old),('mean_control',pp[0]),('ordered',pp[1])]:
            ck('saved_hash',sha(r['prediction'])==r['prediction_sha256']);v=np.load(r['prediction'])['logits'];ck('finite_shape',v.shape==d['y'].shape and np.isfinite(v).all());pred[k]=v;take=d['person']==person;outputs[k][take]=v[take]
        for k in ['mean_control','C6']:
            folds.append(dict(seed=seed,person=person,baseline=k,categories={str(code):metric(pred[k],pred['ordered'],d['y'],(d['label_provenance']==code)&take[:,None])for code in [1,2,3,4]}))
    for k,v in outputs.items():ck('allheldonce',np.isfinite(v).all())
    for k in ['mean_control','C6']:
        a,b=outputs[k],outputs['ordered'];aggregate.append(dict(seed=seed,baseline=k,categories={str(code):metric(a,b,d['y'],d['label_provenance']==code)for code in [1,2,3,4]}))
        for j,fam in enumerate(F):family.append(dict(seed=seed,baseline=k,family=fam,categories={str(code):metric(a[:,j],b[:,j],d['y'][:,j],d['label_provenance'][:,j]==code)for code in [1,2,3,4]}))
        short.append(dict(seed=seed,baseline=k,boundary_movement=metric(a,b,d['y'],boundary&(d['label_provenance']==1)),interior_movement=metric(a,b,d['y'],~boundary&(d['label_provenance']==1))))
        for ii,jj in zip(*np.where((d['mask']>0)&(((a>=0)==(d['y']>0))!=((b>=0)==(d['y']>0))))):
            r=rows[int(d['seq_id'][ii])];changes.append(dict(seed=seed,baseline=k,clip_id=r['clip_id'],view=r['view'],reverse=r['reverse'],frame=int(d['source_frame_id'][ii]),family=F[jj],code=int(d['label_provenance'][ii,jj]),kind='recovery'if(b[ii,jj]>=0)==(d['y'][ii,jj]>0)else'regression'))
gates={}
for base in ['mean_control','C6']:
    agg=[r for r in aggregate if r['baseline']==base];fam=[r for r in family if r['baseline']==base];ff=[r for r in folds if r['baseline']==base]
    gates[base]=dict(pooled_preserved=all(r['categories'][str(k)]['after']>=r['categories'][str(k)]['before']for r in agg for k in [1,3]),families_preserved=all(r['categories']['1']['after']>=r['categories']['1']['before']for r in fam),no_double_fold_loss=all(not all(r['categories'][str(k)]['after']<r['categories'][str(k)]['before']for k in [1,3])for r in ff),some_strict_gain=any(r['categories'][str(k)]['after']>r['categories'][str(k)]['before']for r in agg for k in [1,3]))
    for r in agg:
        for code,m in r['categories'].items():
            for field in m:ck('fold_sums',sum(p['categories'][code][field]for p in ff if p['seed']==r['seed'])==m[field])
write('COMPARISON.json',dict(status='DIAGNOSTIC_PASS'if all(all(g.values())for g in gates.values())else'DIAGNOSTIC_REWORK',gates=gates,aggregate=aggregate,by_family=family,by_fold=folds,boundary_subset=short,changed_observations=changes,short_motion_limit='Boundary movement witnesses + pretrain synthetic pulse checks; not independent rapid-sign benchmark.',no_G2_pass=True,no_web_change=True,no_independent_human_truth=True))
write('SAVED_VERIFICATION.json',dict(status='PASS',checks=len(checks),comparison_sha256=sha(U/'COMPARISON.json'),frozen_sha256=sha(U/'TRAINING_FROZEN.json'),method='Separate saved arithmetic; no rerun or modelselection.'))
print(json.dumps(dict(gates=gates,aggregate=aggregate,boundary_subset=short),ensure_ascii=False))
