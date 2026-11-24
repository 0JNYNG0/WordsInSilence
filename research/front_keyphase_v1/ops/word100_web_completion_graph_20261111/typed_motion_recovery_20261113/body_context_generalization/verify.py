"""Separate saved-input/output arithmetic verification. No model forward."""
from pathlib import Path
import hashlib, json, sys, time, traceback
import numpy as np
import torch
U=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
checks=[]
def check(name,ok):
    checks.append(name)
    assert bool(ok),name

def inputs():
    cfg=read(U/'FROZEN.json');idx=read(U/'INDEX.json')['rows']
    for group in ['pins','protected_pins']:
        for p,h in cfg[group].items():check('hash '+p,sha(p)==h)
    with np.load(U/'INPUTS.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    a=d['absolute_context'].reshape(-1,3,10,5);b=d['relative_context'].reshape(-1,3,10,5)
    check('same current',np.array_equal(a[:,1],b[:,0]))
    check('before difference',np.allclose(a[:,0,:,:4]-a[:,1,:,:4],b[:,1,:,:4],atol=1e-7))
    check('after difference',np.allclose(a[:,2,:,:4]-a[:,1,:,:4],b[:,2,:,:4],atol=1e-7))
    check('availability preserved',np.array_equal(a[:,:,:,4][:,[1,0,2]],b[:,:,:,4]))
    check('all finite',all(np.isfinite(v).all() for v in d.values()))
    for i,r in enumerate(idx):
        sl=slice(r['start'],r['stop'])
        check('index '+str(i),np.all(d['seq_id'][sl]==i) and np.all(d['person'][sl]==r['person']))
        with np.load(r['path'],allow_pickle=False) as z:
            for k in ['y','mask','observed','label_provenance','source_frame_id','timestamp_us']:
                check(f'original {i} {k}',np.array_equal(z[k],d[k][sl]))
    check('no unknown label weight',np.array_equal(d['mask']>0,d['label_provenance']>0))
    for fold in cfg['folds']:
        p=fold['person'];tr=d['person']!=p;te=~tr
        check('four held sources '+str(p),len({r['clip_id']for r in idx if r['person']==p})==4)
        check('source separation '+str(p),not ({r['source_sha256'] for r in idx if r['person']==p}&{r['source_sha256']for r in idx if r['person']!=p}))
        with np.load(fold['weights']) as z:w=z['weights']
        check('excluded labels '+str(p),not w[te].any() and not w[d['mask']==0].any())
        check('total loss weight '+str(p),np.isclose(w.sum(),1.))
        for j in range(5):
            for value in [0,1]:
                take=tr&(d['mask'][:,j]>0)&(d['y'][:,j]==value)
                check(f'balanced family class {p} {j} {value}',np.isclose(w[take,j].sum(),.1))
                masses=[]
                for cid in sorted({r['clip_id']for r in idx if r['person']!=p}):
                    ss=[i for i,r in enumerate(idx)if r['clip_id']==cid]
                    q=take&np.isin(d['seq_id'],ss)
                    if q.any():masses.append(w[q,j].sum())
                check(f'balanced sources {p} {j} {value}',np.allclose(masses,masses[0]))
        norm=read(fold['normalization'])
        for c in cfg['conditions']:
            x=d[c][tr].astype(np.float64)
            check('fold mean '+str(p)+c,np.allclose(x.mean(0),norm[c]['mean'],atol=1e-12))
            check('fold std '+str(p)+c,np.allclose(np.maximum(x.std(0),1e-4),norm[c]['std'],atol=1e-12))
    return cfg,idx,d

def metric(old,new,y,take):
    aa=(old>=0)==(y>0);bb=(new>=0)==(y>0)
    return dict(count=int(take.sum()),baseline_correct=int(aa[take].sum()),relative_correct=int(bb[take].sum()),
                regressions=int((aa&~bb&take).sum()),recoveries=int((~aa&bb&take).sum()))

def result():
    cfg,idx,d=inputs();trained=read(U/'TRAINING_COMPLETE.json')
    check('36 fixed new runs',len(trained['rows'])==36)
    aggregate=[];folds=[];byfamily=[];errors=[];fit=[]
    for seed in cfg['seeds']:
        old=np.full_like(d['y'],np.nan);new=old.copy()
        for p in range(2,8):
            pair=[next(r for r in trained['rows']if r['person']==p and r['seed']==seed and r['condition']==c)for c in cfg['conditions']]
            check(f'matched initialization {p} {seed}',pair[0]['initial_trainable_hash']==pair[1]['initial_trainable_hash'])
            pred=[]
            for r in pair:
                check('model hash '+r['model'],sha(r['model'])==r['model_sha256'])
                check('saved logits hash '+r['prediction'],sha(r['prediction'])==r['prediction_sha256'])
                st=torch.load(r['model'],map_location='cpu',weights_only=True)
                check('small architecture '+r['model'],tuple(st['head.0.weight'].shape)==(96,150) and tuple(st['head.2.weight'].shape)==(5,96))
                norm=read(U/'folds'/str(p)/'NORMALIZATION.json')[r['condition']]
                check('checkpoint fold mean '+r['model'],np.array_equal(st['mean'].numpy(),np.asarray(norm['mean'],np.float32)))
                check('checkpoint fold std '+r['model'],np.array_equal(st['std'].numpy(),np.asarray(norm['std'],np.float32)))
                with np.load(r['prediction'],allow_pickle=False) as z:v=z['logits'].copy()
                check('prediction shape finite '+r['prediction'],v.shape==d['y'].shape and np.isfinite(v).all());pred.append(v)
            hold=d['person']==p;old[hold]=pred[0][hold];new[hold]=pred[1][hold]
            rows={}
            for code in [1,2,3,4]:
                take=(d['label_provenance']==code)&hold[:,None]
                rows[str(code)]=metric(*pred,d['y'],take)
                fit.append(dict(person=p,seed=seed,code=code,**metric(*pred,d['y'],(d['label_provenance']==code)&~hold[:,None])))
            folds.append(dict(person=p,seed=seed,categories=rows))
        check('each person evaluated once '+str(seed),np.isfinite(old).all() and np.isfinite(new).all())
        cats={str(code):metric(old,new,d['y'],d['label_provenance']==code)for code in [1,2,3,4]}
        aggregate.append(dict(seed=seed,categories=cats))
        for j,f in enumerate([2,3,4,5,7]):
            catsf={str(code):metric(old[:,j],new[:,j],d['y'][:,j],d['label_provenance'][:,j]==code)for code in [1,2,3,4]}
            byfamily.append(dict(seed=seed,family=f,categories=catsf))
        for row,j in zip(*np.where((d['mask']>0)&(((old>=0)==(d['y']>0))!=((new>=0)==(d['y']>0))))):
            r=idx[int(d['seq_id'][row])]
            errors.append(dict(seed=seed,clip_id=r['clip_id'],person=r['person'],view=r['view'],reverse=r['reverse'],
                source_frame=int(d['source_frame_id'][row]),family=[2,3,4,5,7][j],code=int(d['label_provenance'][row,j]),
                kind='recovery'if ((new[row,j]>=0)==(d['y'][row,j]>0))else'regression',baseline_logit=float(old[row,j]),relative_logit=float(new[row,j])))
    no_overall_loss=all(r['categories'][str(c)]['relative_correct']>=r['categories'][str(c)]['baseline_correct']for r in aggregate for c in [1,3])
    no_family_loss=all(r['categories']['1']['relative_correct']>=r['categories']['1']['baseline_correct']for r in byfamily)
    no_double_fold_loss=all(not all(r['categories'][str(c)]['relative_correct']<r['categories'][str(c)]['baseline_correct']for c in [1,3])for r in folds)
    gain=any(r['categories'][str(c)]['relative_correct']>r['categories'][str(c)]['baseline_correct']for r in aggregate for c in [1,3])
    gates=dict(overall_motion_hold_preserved=no_overall_loss,motion_family_preserved=no_family_loss,no_fold_loses_both=no_double_fold_loss,some_strict_gain=gain)
    save(U/'COMPARISON.json',dict(status='DIAGNOSTIC_PASS'if all(gates.values())else'DIAGNOSTIC_REWORK',gate=gates,
        categories={'1':'direct movement','2':'other reviewed non-target observation','3':'own-family terminal body hold','4':'other-family terminal body hold'},
        aggregate=aggregate,by_family=byfamily,by_person=folds,fit=fit,changed_observations=errors,
        denominators='Views/reversal duplicate sources; each seed 24 public source videos,6 recorded IDs. Not number of people or independent clips.',
        no_G2_pass=True,no_deployment=True,no_new_teacher_truth=True))
    save(U/'SAVED_VERIFICATION.json',dict(status='PASS',checks=len(checks),frozen_sha256=sha(U/'FROZEN.json'),
        comparison_sha256=sha(U/'COMPARISON.json'),method='Separate saved arithmetic by same Codex, no model reruns; not independent human annotation.'))
    print(json.dumps(dict(gates=gates,aggregate=aggregate),ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:
        assert len(sys.argv)==2 and sys.argv[1] in ['pretrain','saved']
        if sys.argv[1]=='pretrain':
            inputs();save(U/'PRETRAIN_VERIFICATION.json',dict(status='PASS',checks=len(checks),frozen_sha256=sha(U/'FROZEN.json'),method='Separate arithmetic and source exclusion audit by same Codex.'));print('Pretrain PASS',len(checks))
        else:result()
    except BaseException:
        (U/f'FIRST_VERIFY_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
